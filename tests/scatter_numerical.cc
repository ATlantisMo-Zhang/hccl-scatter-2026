#include <acl/acl.h>
#include <hccl/hccl.h>
#include <hccl/hccl_rank_graph.h>
#include <mpi.h>
#include <dlfcn.h>
#include <limits.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "custom.h"

namespace {
int rankId = -1;
constexpr uint64_t MIB = 1024ULL * 1024;
constexpr uint64_t TRANSFER_WORDS = MIB / sizeof(float);
constexpr uint32_t GUARD = 0x427b4000;
constexpr uint32_t POISON = 0x42de8000;
const char *CANDIDATE = nullptr;

[[noreturn]] void Fail(const char *what, int line, int code)
{
    std::fprintf(stderr, "FAIL rank=%d line=%d detail=%s code=%d\n", rankId, line, what, code);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, code == 0 ? 99 : code);
    std::abort();
}
#define CHECK(call) do { const int code = static_cast<int>(call); if (code != 0) Fail(#call, __LINE__, code); } while (0)
#define REQUIRE(value) do { if (!(value)) Fail(#value, __LINE__, 99); } while (0)

// All generated bit patterns are positive, finite, normal FP32. Multiplication
// by 17 permutes the 253 * 2^23 valid positive-normal bit patterns, so each
// tested input (below that many elements) has distinct slice/offset values.
uint32_t Pattern(size_t round, uint64_t globalIndex)
{
    constexpr uint64_t finitePatterns = 0x7e800000ULL;
    return 0x00800000u + static_cast<uint32_t>((round * 0x013579bdULL + globalIndex * 17) % finitePatterns);
}

struct Buffer {
    void *raw = nullptr;
    void *data = nullptr;
    uint64_t bytes = 0;
    uint64_t guardBytes = 0;
};

void FillConstant(void *destination, uint64_t bytes, uint32_t bits)
{
    REQUIRE(bytes % sizeof(float) == 0);
    std::vector<uint32_t> staging(std::min<uint64_t>(bytes / 4, TRANSFER_WORDS), bits);
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t len = std::min<uint64_t>(bytes - offset, staging.size() * 4);
        CHECK(aclrtMemcpy(static_cast<char *>(destination) + offset, len, staging.data(), len, ACL_MEMCPY_HOST_TO_DEVICE));
        offset += len;
    }
}

Buffer Allocate(uint64_t bytes, size_t round)
{
    Buffer buffer{};
    buffer.bytes = bytes;
    buffer.guardBytes = 64 + (round % 11) * 4;
    REQUIRE(bytes <= UINT64_MAX - 2 * buffer.guardBytes);
    CHECK(aclrtMalloc(&buffer.raw, bytes + 2 * buffer.guardBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    buffer.data = static_cast<char *>(buffer.raw) + buffer.guardBytes;
    FillConstant(buffer.raw, buffer.guardBytes, GUARD);
    FillConstant(static_cast<char *>(buffer.data) + bytes, buffer.guardBytes, GUARD);
    return buffer;
}

void FillInput(const Buffer &buffer, size_t round)
{
    std::vector<uint32_t> staging(std::min<uint64_t>(buffer.bytes / 4, TRANSFER_WORDS));
    for (uint64_t index = 0; index < buffer.bytes / 4;) {
        const uint64_t words = std::min<uint64_t>(buffer.bytes / 4 - index, staging.size());
        for (uint64_t j = 0; j < words; ++j) staging[j] = Pattern(round, index + j);
        CHECK(aclrtMemcpy(static_cast<char *>(buffer.data) + index * 4, words * 4,
            staging.data(), words * 4, ACL_MEMCPY_HOST_TO_DEVICE));
        index += words;
    }
}

void CheckGuard(const Buffer &buffer, size_t round, const char *name)
{
    std::vector<uint32_t> staging(buffer.guardBytes / 4);
    for (unsigned side = 0; side < 2; ++side) {
        const void *address = side == 0 ? buffer.raw : static_cast<const char *>(buffer.data) + buffer.bytes;
        CHECK(aclrtMemcpy(staging.data(), buffer.guardBytes, address, buffer.guardBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        for (size_t i = 0; i < staging.size(); ++i) {
            if (staging[i] != GUARD) {
                std::fprintf(stderr, "GUARD_FAIL rank=%d round=%zu buffer=%s side=%u word=%zu got=0x%x\n",
                    rankId, round, name, side, i, staging[i]);
                Fail("guard", __LINE__, 98);
            }
        }
    }
}

void CheckValues(const Buffer &buffer, size_t round, uint64_t globalBegin, const char *name)
{
    std::vector<uint32_t> staging(std::min<uint64_t>(buffer.bytes / 4, TRANSFER_WORDS));
    for (uint64_t index = 0; index < buffer.bytes / 4;) {
        const uint64_t words = std::min<uint64_t>(buffer.bytes / 4 - index, staging.size());
        CHECK(aclrtMemcpy(staging.data(), words * 4, static_cast<const char *>(buffer.data) + index * 4,
            words * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        for (uint64_t j = 0; j < words; ++j) {
            const uint32_t expected = Pattern(round, globalBegin + index + j);
            if (staging[j] != expected) {
                std::fprintf(stderr, "VALUE_FAIL rank=%d round=%zu buffer=%s index=%llu got=0x%x expected=0x%x\n",
                    rankId, round, name, static_cast<unsigned long long>(index + j), staging[j], expected);
                Fail("element comparison", __LINE__, 97);
            }
        }
        index += words;
    }
    CheckGuard(buffer, round, name);
}

std::vector<std::vector<uint32_t>> QueryGroups(HcclComm comm, int size, const std::string &topology)
{
    uint32_t *ranks = nullptr, count = 0;
    CHECK(HcclRankGraphGetRanksByLayer(comm, 0, &ranks, &count));
    REQUIRE(ranks != nullptr && count > 0 && count <= static_cast<uint32_t>(size));
    const std::vector<uint32_t> copied(ranks, ranks + count);
    uint32_t localMask = 0;
    for (uint32_t peer : copied) {
        REQUIRE(peer < static_cast<uint32_t>(size));
        REQUIRE((localMask & (1u << peer)) == 0);
        localMask |= 1u << peer;
    }
    REQUIRE((localMask & (1u << rankId)) != 0);
    std::vector<uint32_t> masks(size);
    CHECK(MPI_Allgather(&localMask, 1, MPI_UINT32_T, masks.data(), 1, MPI_UINT32_T, MPI_COMM_WORLD));
    std::vector<std::vector<uint32_t>> groups;
    uint32_t covered = 0;
    for (int r = 0; r < size; ++r) {
        REQUIRE((masks[r] & (1u << r)) != 0);
        REQUIRE((masks[r] >> size) == 0);
        if (covered & (1u << r)) continue;
        REQUIRE((covered & masks[r]) == 0);
        std::vector<uint32_t> group;
        for (int peer = 0; peer < size; ++peer) if (masks[r] & (1u << peer)) {
            REQUIRE(masks[peer] == masks[r]);
            group.push_back(peer);
        }
        covered |= masks[r];
        groups.push_back(group);
    }
    REQUIRE(covered == ((1u << size) - 1));
    std::vector<size_t> lengths;
    for (const auto &group : groups) lengths.push_back(group.size());
    std::sort(lengths.begin(), lengths.end());
    const std::vector<size_t> expected = topology == "2x8" ? std::vector<size_t>{8, 8}
        : topology == "4x1" ? std::vector<size_t>{1, 1, 1, 1}
        : topology == "8plus4" ? std::vector<size_t>{4, 8}
        : topology == "4x3" ? std::vector<size_t>{3, 3, 3, 3}
        : topology == "2x1" ? std::vector<size_t>{1, 1} : std::vector<size_t>{};
    REQUIRE(lengths == expected);
    std::printf("TOPOLOGY rank=%d label=%s size=%d localLayer0Mask=0x%x groups=%zu\n",
        rankId, topology.c_str(), size, localMask, groups.size());
    if (rankId == 0) for (size_t g = 0; g < groups.size(); ++g) {
        uint32_t mask = 0;
        for (uint32_t r : groups[g]) mask |= 1u << r;
        std::printf("GROUP index=%zu mask=0x%x members=%zu first=%u last=%u\n",
            g, mask, groups[g].size(), groups[g].front(), groups[g].back());
    }
    return groups;
}

void CheckResources(HcclComm comm, size_t round, int size, AlgResourceCtx &previous, bool &havePrevious, uint32_t currentRoot, aclrtStream userStream)
{
    void *address = nullptr;
    uint64_t bytes = 0;
    CHECK(HcclEngineCtxGet(comm, "hccl_scatter_postrace_scratch_v1", COMM_ENGINE_CCU, &address, &bytes));
    AlgResourceCtx current{};
    CHECK(AlgResourceCtx::Decode(address, bytes, current));
    REQUIRE(current.myRank == static_cast<uint32_t>(rankId) && current.rankSize == static_cast<uint32_t>(size));
    REQUIRE(current.groupCount <= 2);
    REQUIRE((current.groupCount == 2) == (current.slaveThread != 0));
    if (havePrevious) REQUIRE(std::memcmp(&previous, &current, sizeof(current)) == 0);
    std::printf("CACHE rank=%d round=%zu bytes=%llu groups=%u stable=%u slaveThread=%llu\n",
        rankId, round, static_cast<unsigned long long>(bytes), current.groupCount, havePrevious,
        static_cast<unsigned long long>(current.slaveThread));
    int activeGroup = -1;
    uint32_t matches = 0;
    uint64_t activeChannel = 0;
    for (uint32_t g = 0; g < current.groupCount; ++g) {
        const auto &reg = current.groups[g].registration;
        const bool needSmallPull = current.relayTopology == 0 && reg.channelCount != 0;
        REQUIRE(needSmallPull ? current.groups[g].smallPullKernel != 0 : current.groups[g].smallPullKernel == 0);
        const unsigned missionCount = 1u + (current.relayTopology != 0 ? 1u : 0u) + (needSmallPull ? 1u : 0u);
        REQUIRE(missionCount <= 2);
        std::printf("REGISTERED_LAYOUT rank=%d round=%zu topology=%u group=%u layer=%u die=%u channels=%u needSmallPull=%u missionCount=%u registerEndSuccessFromReadyCache=1\n",
            rankId, round, current.relayTopology, g, current.groups[g].layer, current.groups[g].dieId,
            reg.channelCount, needSmallPull, missionCount);
        std::printf("SMALL_PULL_RESOURCE rank=%d round=%zu group=%u layer=%u die=%u channels=%u smallPullKernel=%llu originalKernel=%llu relayKernel=%llu decoded=1\n",
            rankId, round, g, current.groups[g].layer, current.groups[g].dieId, reg.channelCount,
            static_cast<unsigned long long>(current.groups[g].smallPullKernel),
            static_cast<unsigned long long>(current.groups[g].kernel),
            static_cast<unsigned long long>(current.groups[g].relayKernel));
        for (uint32_t i = 0; i < reg.channelCount; ++i) if (reg.peers[i] == currentRoot) {
            activeGroup = static_cast<int>(g);
            activeChannel = reg.channels[i];
            ++matches;
        }
    }
    REQUIRE(matches == (rankId == static_cast<int>(currentRoot) ? 0u : 1u));
    std::printf("ACTIVE_GROUP rank=%d round=%zu root=%u role=%s activeGroup=%d die=%d kernel=%llu channel=%llu userStream=%p cachedGroups=%u slaveThread=%llu\n",
        rankId, round, currentRoot, rankId == static_cast<int>(currentRoot) ? "root" : "receiver", activeGroup,
        activeGroup < 0 ? -1 : static_cast<int>(current.groups[activeGroup].dieId),
        activeGroup < 0 ? 0ULL : static_cast<unsigned long long>(current.groups[activeGroup].kernel),
        static_cast<unsigned long long>(activeChannel), userStream, current.groupCount,
        static_cast<unsigned long long>(current.slaveThread));
    if (!havePrevious) {
        for (uint32_t g = 0; g < current.groupCount; ++g) {
            const auto &group = current.groups[g];
            const auto &reg = group.registration;
            std::printf("CCU_GROUP rank=%d group=%u die=%u kernel=%llu peers=%u copySelf=%u\n",
                rankId, g, group.dieId, static_cast<unsigned long long>(group.kernel), reg.channelCount, reg.copySelf);
            for (uint32_t i = 0; i < reg.channelCount; ++i) {
                std::printf("CCU_CHANNEL rank=%d group=%u die=%u index=%u peer=%u handle=%llu notifyCount=3\n",
                    rankId, g, group.dieId, i, reg.peers[i], static_cast<unsigned long long>(reg.channels[i]));
            }
        }
        previous = current;
        havePrevious = true;
    }
}

struct Case { uint32_t root; uint64_t count; uint64_t requestedInputBytes; std::string label; };
} // namespace

int main(int argc, char **argv)
{
    CHECK(MPI_Init(&argc, &argv));
    CANDIDATE = std::getenv("SCATTER_TEST_LIBRARY");
    REQUIRE(CANDIDATE != nullptr);
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rankId));
    int size = 0;
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    REQUIRE(size >= 2 && size <= 16);
    REQUIRE(argc >= 3);
    const std::string topology = argv[1], mode = argv[2];
    REQUIRE(mode == "relay_smoke" || mode == "relay_short" || mode == "relay_alias" || mode == "relay_tail" || mode == "relay_4x3" || mode == "fallback"
        || mode == "probe_small" || mode == "small_switch" || mode == "boundary" || mode == "ordinary_large" || mode == "nonrelay_mix" || mode == "fusion_short" || mode == "fusion_tail" || mode == "dual_boundary" || mode == "first_fault" || mode == "a21_large" || mode == "m4x3");
    REQUIRE((topology == "2x8" && size == 16) || (topology == "4x3" && size == 12)
        || (topology == "8plus4" && size == 12) || (topology == "4x1" && size == 4));
    void *library = dlopen(CANDIDATE, RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "DLOPEN_ERROR %s\n", dlerror()); Fail("dlopen candidate", __LINE__, 96); }
    void *implementation = dlsym(library, "HcclScatter");
    Dl_info impl{}, dispatch{};
    REQUIRE(implementation != nullptr && dladdr(implementation, &impl));
    REQUIRE(dladdr(reinterpret_cast<void *>(&HcclScatter), &dispatch));
    char actual[PATH_MAX]{}, wanted[PATH_MAX]{};
    REQUIRE(realpath(impl.dli_fname, actual) && realpath(CANDIDATE, wanted));
    REQUIRE(std::strcmp(actual, wanted) == 0);
    std::printf("IMPLEMENTATION rank=%d symbol=%p path=%s\n", rankId, implementation, actual);
    std::printf("DISPATCH rank=%d symbol=%p path=%s\n", rankId, reinterpret_cast<void *>(&HcclScatter), dispatch.dli_fname);
    // Keep the VM proxy call: it records Scatter's original buffers/op metadata,
    // then uses RTLD_NEXT to reach the separately verified candidate library.
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(rankId));
    HcclRootInfo info{};
    if (rankId == 0) CHECK(HcclGetRootInfo(&info));
    CHECK(MPI_Bcast(&info, sizeof(info), MPI_BYTE, 0, MPI_COMM_WORLD));
    HcclComm comm = nullptr;
    CHECK(HcclCommInitRootInfo(size, &info, rankId, &comm));
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream));
    const auto groups = QueryGroups(comm, size, topology);
    std::vector<uint32_t> roots;
    for (size_t g = 0; g < groups.size(); ++g)
        roots.push_back(g + 1 == groups.size() ? groups[g].back() : groups[g].front());
    REQUIRE(std::find(roots.begin(), roots.end(), size - 1) != roots.end());
    if (rankId == 0) for (size_t g = 0; g < roots.size(); ++g)
        std::printf("ROOT_COVERAGE group=%zu root=%u\n", g, roots[g]);
    std::vector<Case> cases;
    auto add = [&](uint32_t root, uint64_t bytes, const char *label) {
        REQUIRE(root < static_cast<uint32_t>(size) && bytes % sizeof(float) == 0);
        cases.push_back({root, bytes / sizeof(float), bytes * size, label});
    };
    if (mode == "m4x3") {
        REQUIRE(topology == "4x3");
        add(0, 16 * MIB, "m_mesh_root0_threshold");
        add(11, 16 * MIB + 4, "m_mesh_root11_large_new_address");
        add(3, 68, "m_small_between_large");
        add(3, 32 * MIB, "m_mesh_root3_large");
        add(11, 0, "m_zero_after_large");
        add(11, 16 * MIB - 4, "m_below_relay_threshold");
        add(6, 16 * MIB + 8, "m_relay_after_small_root6");
    } else if (mode == "a21_large") {
        REQUIRE(topology == "2x8");
        add(0, 16 * MIB, "a21_large_root0");
        add(15, 16 * MIB + 4, "a21_large_root15_new_address");
    } else if (mode == "first_fault") {
        REQUIRE(topology == "2x8");
        add(0, 4, "dual_whole_small4_root0");
        add(15, 8, "dual_whole_small8_root15");
        add(15, 32768, "dual_small_repeat_root15");
    } else if (mode == "dual_boundary") {
        REQUIRE(topology == "2x8");
        add(0, 4, "dual_whole_small4_root0");
        add(15, 8, "dual_whole_small8_root15");
        add(15, 32768, "dual_small_repeat_root15");
        add(7, 16 * MIB - 4, "dual_pull_threshold_minus4");
        add(0, 16 * MIB, "dual_relay_threshold_exact");
        add(15, 16 * MIB + 4, "dual_relay_threshold_plus4");
        add(7, 0, "dual_zero_after_relay");
        add(0, 32768, "dual_pull_after_relay_zero");
        add(8, 32 * MIB + 4, "balanced_scratch_multichunk_root8");
    } else if (mode == "fusion_short") {
        REQUIRE(topology == "2x8" || topology == "4x3");
        add(0, 32768, "fusion_small_root0");
        add(size - 1, 16 * MIB + 4, "fusion_relay_last_root");
        add(0, 0, "fusion_zero_after_relay");
        add(size - 1, 68, "fusion_small_after_relay");
        add(size - 1, 68, "fusion_small_repeat_new_address_content");
    } else if (mode == "fusion_tail") {
        REQUIRE(topology == "2x8" || topology == "4x3");
        REQUIRE(MAX_DATA_SIZE == 8 * MIB);
        add(0, 16 * MIB + 4, "test_only_two_relay_chunks_tail4_root0");
        add(size - 1, 16 * MIB + 8, "test_only_two_relay_chunks_tail8_last_root");
    } else if (mode == "probe_small") {
        add(size - 1, 32768, "small_ready_push_nonzero_root");

    } else if (mode == "nonrelay_mix") {
        REQUIRE(topology == "8plus4" || topology == "4x1");
        const uint32_t first = groups.front().front(), last = groups.back().back();
        add(first, 68, "small_ready_push_first_group");
        add(last, 32768, "small_last_group_role_switch");
        add(last, 32768, "small_repeat_new_address_content");
        add(first, 16 * MIB - 4, "small_ready_push_threshold_minus4");
        add(first, 16 * MIB, "large_threshold_original_kernel");
        add(last, 4, "small_ready_push_after_large");
        add(last, 0, "zero_after_small");
        add(first, 524288, "small_ready_push_after_zero");
    } else if (mode == "small_switch") {
        add(0, 4, "small_root0");
        add(size - 1, 32768, "small_last_root");
        add(size - 1, 0, "zero_after_small");
        add(size - 1, 32768, "small_repeat_same_root_new_address_content");
        add(0, 524288, "small_root_role_change");
    } else if (mode == "boundary") {
        REQUIRE(topology == "2x8");
        add(7, 16 * MIB - 4, "receiver_threshold_minus4");
        add(15, 16 * MIB, "relay_threshold_exact");
        add(0, 16 * MIB + 4, "relay_threshold_plus4");
    } else if (mode == "ordinary_large") {
        REQUIRE(topology == "4x1");
        add(3, 16 * MIB + 4, "large_original_receiver_kernel");
    } else if (mode == "relay_smoke") {
        REQUIRE(topology == "2x8");
        add(0, 16 * MIB, "multi_neighbor_minimum_root0");
    } else if (mode == "relay_short") {
        REQUIRE(topology == "2x8");
        add(15, 16 * MIB + 4, "multi_neighbor_root15_tail4");
        add(7, 0, "zero_after_relay");
        add(7, 16 * MIB, "multi_neighbor_root7");
        add(0, 12, "small_f6_after_relay");
    } else if (mode == "relay_alias") {
        REQUIRE(topology == "4x3");
        add(11, 16 * MIB, "m4x3_root11_exact_self_alias");
    } else if (mode == "relay_tail") {
        REQUIRE(topology == "2x8");
        add(0, 2 * MIB + 4, "test_only_two_relay_chunks_tail4_root0");
        add(15, 2 * MIB + 4, "test_only_two_relay_chunks_tail4_root15");
        add(0, 2 * MIB + 8, "test_only_two_relay_chunks_tail8_root0");
        add(15, 2 * MIB + 8, "test_only_two_relay_chunks_tail8_root15");
    } else if (mode == "relay_4x3") {
        REQUIRE(topology == "4x3");
        add(11, 16 * MIB + 4, "multi_neighbor_4x3_root11");
        add(0, 32768, "ordinary_small_after_4x3_relay");
    } else {
        REQUIRE(topology == "8plus4" || topology == "4x1");
        add(size - 1, 16 * MIB + 4, "unsupported_topology_f6_fallback");
    }
    AlgResourceCtx cached{};
    bool haveCache = false;
    for (size_t round = 0; round < cases.size(); ++round) {
        const auto &test = cases[round];
        const uint64_t bytes = test.count * sizeof(float), total = bytes * size;
        REQUIRE(total / 4 < 0x7e800000ULL);
        Buffer output = Allocate(bytes, round);
        FillConstant(output.data, bytes, POISON);
        Buffer input{};
        if (rankId == static_cast<int>(test.root) && total > 0) {
            input = Allocate(total, round + 1);
            FillInput(input, round);
        }
        const bool selfAlias = mode == "relay_alias" && rankId == static_cast<int>(test.root);
        void *actualOutput = selfAlias ? static_cast<char *>(input.data) + test.root * bytes : output.data;
        std::printf("CONTRACT rank=%d round=%zu selfAlias=%u userStream=%p priorStreamComplete=1\n", rankId, round, selfAlias, stream);
        // aclrtMemcpy is synchronous. All input/guard writes above complete
        // before this stream receives the Scatter kernels; no MPI barrier.
        std::printf("CASE rank=%d round=%zu topo=%s label=%s root=%u count=%llu B=%llu requestedS=%llu actualS=%llu input=%p output=%p inputRaw=%p outputRaw=%p aggregateUserDeviceBytes=%llu\n",
            rankId, round, topology.c_str(), test.label.c_str(), test.root,
            static_cast<unsigned long long>(test.count), static_cast<unsigned long long>(bytes),
            static_cast<unsigned long long>(test.requestedInputBytes), static_cast<unsigned long long>(total),
            input.data, actualOutput, input.raw, output.raw,
            static_cast<unsigned long long>(2 * total + (size + 1) * 208));
        const auto begin = std::chrono::steady_clock::now();
        CHECK(HcclScatter(input.data, actualOutput, test.count, HCCL_DATA_TYPE_FP32, test.root, comm, stream));
        CHECK(aclrtSynchronizeStream(stream));
        const auto synchronized = std::chrono::steady_clock::now();
        if (selfAlias) {
            Buffer slice{}; slice.raw = actualOutput; slice.data = actualOutput; slice.bytes = bytes;
            CheckValues(slice, round, static_cast<uint64_t>(rankId) * test.count, "self_alias");
            CheckGuard(output, round, "unused_root_allocation");
        } else {
            CheckValues(output, round, static_cast<uint64_t>(rankId) * test.count, "output");
        }
        if (input.raw != nullptr) CheckValues(input, round, 0, "root_input_unchanged");
        CheckResources(comm, round, size, cached, haveCache, test.root, stream);
        const double elapsed = std::chrono::duration<double>(synchronized - begin).count();
        std::printf("PASS rank=%d round=%zu root=%u count=%llu B=%llu actualS=%llu outputWords=%llu outputGuard=%s rootInput=%s rootGuard=%s cache=PASS vmWallSeconds=%.6f\n",
            rankId, round, test.root, static_cast<unsigned long long>(test.count), static_cast<unsigned long long>(bytes),
            static_cast<unsigned long long>(total), static_cast<unsigned long long>(test.count),
            selfAlias ? "NA_SELF_ALIAS" : "PASS", input.raw ? "PASS" : "NA", input.raw ? "PASS" : "NA", elapsed);
        CHECK(aclrtFree(output.raw));
        if (input.raw) CHECK(aclrtFree(input.raw));
        std::printf("RELEASE rank=%d round=%zu afterStreamComplete=1 liveUserAllocations=0\n", rankId, round);
    }
    CHECK(aclrtDestroyStream(stream));
    CHECK(HcclCommDestroy(comm));
    CHECK(aclrtResetDevice(rankId));
    CHECK(aclFinalize());
    std::printf("ALL_PASS rank=%d topology=%s mode=%s rounds=%zu numericalAndGuardOnly=1\n",
        rankId, topology.c_str(), mode.c_str(), cases.size());
    dlclose(library);
    CHECK(MPI_Finalize());
    return 0;
}
