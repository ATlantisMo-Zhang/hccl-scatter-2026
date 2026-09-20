/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>

#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint64_t RELAY_MIN_BYTES = 16ULL << 20;
constexpr uint64_t NO_PEER = UINT64_MAX;
enum RelayPhase : uint64_t { ROOT_MESH = 1, ROOT_CLOS, MESH_RECEIVE, HELPER_SEND, HELPER_FINISH, REMOTE_RECEIVE };

HcclResult RelaySlice(const OpParam &param, const AlgResourceCtx &resources, uint64_t bytes,
    uint64_t offset, uint64_t len, uint64_t relayBytes, uint64_t inputToken, uint64_t outputToken)
{
    const bool isRoot = param.myRank == param.root;
    const bool localRoot = (resources.localMask & (1u << param.root)) != 0;
    uint32_t mesh = resources.groupCount, clos = resources.groupCount;
    for (uint32_t g = 0; g < resources.groupCount; ++g) {
        if (resources.groups[g].layer == 0) { mesh = g; }
        if (resources.groups[g].layer == 1) { clos = g; }
    }
    if (mesh >= resources.groupCount || clos >= resources.groupCount || mesh == clos) { return HCCL_E_PARA; }
    std::vector<uint32_t> helpers, targets;
    if (localRoot || resources.relayTopology == 2) {
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            const bool inRootGroup = ((resources.localMask & (1u << peer)) != 0) == localRoot;
            if (inRootGroup) {
                if (peer != param.root) { helpers.push_back(peer); }
            } else { targets.push_back(peer); }
        }
    }
    uint64_t target = NO_PEER;
    const uint32_t helperCount = resources.relayTopology == 2 ? 7 : 2;
    if (localRoot || resources.relayTopology == 2) {
        if (helpers.size() < helperCount || targets.size() < helperCount) { return HCCL_E_PARA; }
        for (uint32_t i = 0; i < helperCount; ++i) {
            if (localRoot && helpers[i] == param.myRank) { target = targets[i]; }
            if (!localRoot && targets[i] == param.myRank) { target = helpers[i]; }
        }
    }
    const uint64_t input = reinterpret_cast<uintptr_t>(param.inputPtr);
    const uint64_t output = reinterpret_cast<uintptr_t>(param.outputPtr) + offset;
    const uint64_t self = isRoot ? input + param.myRank * bytes + offset : 0;
    auto launch = [&](uint32_t g, uint64_t phase, ThreadHandle thread) -> HcclResult {
        const auto &group = resources.groups[g];
        if (group.relayKernel == 0) { return HCCL_E_PARA; }
        const uint64_t args[] = {inputToken, output, outputToken, len, param.root, self,
            (isRoot && self != output) ? 1ULL : 0ULL, isRoot ? input + offset : 0,
            bytes, relayBytes, len - relayBytes, phase, target};
        return FromCcu(HcommCcuKernelLaunch(thread, group.relayKernel, args, sizeof(args) / sizeof(args[0])));
    };
    HCCL_INFO("[Scatter] rank[%u] root[%u] topology[%u] offset[%llu] len[%llu] R[%llu] helperTarget[%llu]",
        param.myRank, param.root, resources.relayTopology, static_cast<unsigned long long>(offset),
        static_cast<unsigned long long>(len), static_cast<unsigned long long>(relayBytes),
        static_cast<unsigned long long>(target));
    if (isRoot) {
        if (resources.slaveThread == 0 || resources.slaveThread == param.cpuThread) { return HCCL_E_PARA; }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, resources.slaveThread, 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resources.slaveThread, 0, CUSTOM_TIMEOUT)));
        for (uint32_t g = 0; g < resources.groupCount; ++g) {
            CHK_RET(launch(g, g == mesh ? ROOT_MESH : ROOT_CLOS,
                g == 1 ? resources.slaveThread : param.cpuThread));
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resources.slaveThread, param.cpuThread, 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, 0, CUSTOM_TIMEOUT)));
    } else if (localRoot) {
        CHK_RET(launch(mesh, MESH_RECEIVE, param.cpuThread));
        if (target != NO_PEER) {
            CHK_RET(launch(clos, HELPER_SEND, param.cpuThread));
            CHK_RET(launch(mesh, HELPER_FINISH, param.cpuThread));
        }
    } else {
        CHK_RET(launch(clos, REMOTE_RECEIVE, param.cpuThread));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    if (param.count == 0) { return HCCL_SUCCESS; }
    AlgResourceCtx resources{};
    CHK_RET(AlgResourceCtx::Decode(param.resCtx, param.ctxSize, resources));
    if (resources.myRank != param.myRank || resources.rankSize != param.rankSize) { return HCCL_E_PARA; }
    const bool isRoot = param.myRank == param.root;
    uint32_t activeGroup = resources.groupCount;
    if (!isRoot) {
        uint32_t matches = 0;
        for (uint32_t g = 0; g < resources.groupCount; ++g) {
            const auto &reg = resources.groups[g].registration;
            for (uint32_t i = 0; i < reg.channelCount; ++i) {
                if (reg.peers[i] == param.root) {
                    activeGroup = g;
                    ++matches;
                }
            }
        }
        if (matches != 1) { return HCCL_E_PARA; }
    }
    const bool parallel = isRoot && resources.groupCount == 2;
    if (parallel && resources.slaveThread == param.cpuThread) { return HCCL_E_PARA; }
    const uint64_t bytes = param.count * sizeof(float);
    const uint64_t input = reinterpret_cast<uintptr_t>(param.inputPtr);
    const uint64_t output = reinterpret_cast<uintptr_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    if (param.myRank == param.root) {
        CHK_RET(FromCcu(HcommCcuGetMemToken(input, bytes * param.rankSize, &inputToken)));
    }
    CHK_RET(FromCcu(HcommCcuGetMemToken(output, bytes, &outputToken)));
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t len = std::min<uint64_t>(bytes - offset, MAX_DATA_SIZE);
        const uint64_t relayBytes = resources.relayTopology == 2 ? 4 * (len / 10)
            : resources.relayTopology == 4 ? ((len * 3 / 4) / 4) * 4 : 0;
        if (resources.relayTopology != 0 && bytes >= RELAY_MIN_BYTES && relayBytes != 0) {
            CHK_RET(RelaySlice(param, resources, bytes, offset, len, relayBytes, inputToken, outputToken));
            offset += len;
            continue;
        }
        if (parallel) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, resources.slaveThread, 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resources.slaveThread, 0, CUSTOM_TIMEOUT)));
        }
        for (uint32_t g = 0; g < resources.groupCount; ++g) {
            if (!isRoot && g != activeGroup) { continue; }
            const auto &group = resources.groups[g];
            const uint64_t self = param.myRank == param.root ? input + param.myRank * bytes + offset : 0;
            std::vector<uint64_t> args = {inputToken, output + offset, outputToken, len, param.root,
                self, (param.myRank == param.root && self != output + offset) ? 1ULL : 0ULL};
            if (group.registration.channelCount >= 7) {
                args.push_back(param.myRank == param.root
                    ? input + group.registration.peers[0] * bytes + offset : 0);
                args.push_back(bytes);
            } else {
                for (uint32_t i = 0; i < group.registration.channelCount; ++i) {
                    args.push_back(param.myRank == param.root
                        ? input + group.registration.peers[i] * bytes + offset : 0);
                }
            }
            if (resources.relayTopology == 2) {
                args.push_back(bytes < RELAY_MIN_BYTES ? 1ULL : 0ULL);
            }
            const ThreadHandle thread = parallel && g == 1 ? resources.slaveThread : param.cpuThread;
            const bool smallPull = resources.relayTopology == 0 && bytes < RELAY_MIN_BYTES
                && group.registration.channelCount != 0;
            const CcuKernelHandle kernel = smallPull ? group.smallPullKernel : group.kernel;
            CHK_RET(FromCcu(HcommCcuKernelLaunch(thread, kernel, args.data(), args.size())));
        }
        if (parallel) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resources.slaveThread, param.cpuThread, 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, 0, CUSTOM_TIMEOUT)));
        }
        offset += len;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
