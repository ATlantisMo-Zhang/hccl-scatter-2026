/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom.h"
#include "log.h"
#include "exec_op.h"
#include <algorithm>
#include <limits>

namespace ops_hccl {
namespace {
const ChannelInfo &PeerChannel(const AlgResourceCtx &resources, uint32_t rank, uint32_t peer)
{
    return resources.channels[peer < rank ? peer : peer - 1];
}

HcclResult ExecParallelPush(const OpParam &param, const AlgResourceCtx &resources)
{
    const uint64_t bytes = param.count * sizeof(float);
    const ThreadHandle master = resources.aicpuThread;
    auto *output = static_cast<char *>(param.outputPtr);
    if (param.myRank == param.root) {
        const auto *input = static_cast<const char *>(param.inputPtr);
        for (size_t i = 1; i < resources.threads.size(); ++i) {
            CHK_RET(HcommThreadNotifyRecordOnThread(master, resources.threads[i], 0));
        }
        const char *self = input + static_cast<uint64_t>(param.root) * bytes;
        if (self != output) { CHK_RET(HcommLocalCopyOnThread(master, output, self, bytes)); }
        for (size_t i = 1; i < resources.threads.size(); ++i) {
            const ThreadHandle worker = resources.threads[i];
            const auto &channel = resources.channels[i - 1];
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            CHK_RET(HcommWriteOnThread(worker, channel.handle, channel.remoteCclMem.addr,
                input + static_cast<uint64_t>(channel.remoteRank) * bytes, bytes));
            CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel.handle, NOTIFY_IDX_ACK));
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
            CHK_RET(HcommThreadNotifyRecordOnThread(worker, master, static_cast<uint32_t>(i)));
        }
        for (size_t i = 1; i < resources.threads.size(); ++i) {
            CHK_RET(HcommThreadNotifyWaitOnThread(master, static_cast<uint32_t>(i), CUSTOM_TIMEOUT));
        }
    } else {
        const auto &channel = PeerChannel(resources, param.myRank, param.root);
        CHK_RET(HcommChannelNotifyRecordOnThread(master, channel.handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(master, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(master, output, resources.localBuffer.addr, bytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(master, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecQuarterPush(const OpParam &param, const AlgResourceCtx &resources)
{
    const uint64_t bytes = param.count * sizeof(float);
    const uint64_t forwarded = (param.count / 4) * sizeof(float);
    const uint64_t direct = bytes - forwarded;
    const ThreadHandle master = resources.aicpuThread;
    const uint32_t localMask = (resources.relayTopology.groupMasks[0] & (1u << param.root)) != 0
        ? resources.relayTopology.groupMasks[0] : resources.relayTopology.groupMasks[1];
    uint32_t helpers[7]{};
    uint32_t receivers[8]{};
    uint32_t helperCount = 0;
    uint32_t receiverCount = 0;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.root) { continue; }
        if ((localMask & (1u << rank)) != 0) { helpers[helperCount++] = rank; }
        else { receivers[receiverCount++] = rank; }
    }
    auto *output = static_cast<char *>(param.outputPtr);
    if (param.myRank == param.root) {
        const auto *input = static_cast<const char *>(param.inputPtr);
        for (size_t i = 1; i < resources.threads.size(); ++i) {
            CHK_RET(HcommThreadNotifyRecordOnThread(master, resources.threads[i], 0));
        }
        const char *self = input + static_cast<uint64_t>(param.root) * bytes;
        if (self != output) { CHK_RET(HcommLocalCopyOnThread(master, output, self, bytes)); }
        for (size_t i = 1; i < resources.threads.size(); ++i) {
            const ThreadHandle worker = resources.threads[i];
            const auto &channel = resources.channels[i - 1];
            auto *remote = static_cast<char *>(channel.remoteCclMem.addr);
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            if ((localMask & (1u << channel.remoteRank)) != 0) {
                for (uint32_t r = 0; r < receiverCount; ++r) {
                    if (helpers[r % helperCount] != channel.remoteRank) { continue; }
                    const uint64_t slot = bytes + static_cast<uint64_t>(r / helperCount) * forwarded;
                    CHK_RET(HcommWriteOnThread(worker, channel.handle, remote + slot,
                        input + static_cast<uint64_t>(receivers[r]) * bytes + direct, forwarded));
                }
                CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel.handle, NOTIFY_IDX_ACK));
                CHK_RET(HcommWriteOnThread(worker, channel.handle, remote,
                    input + static_cast<uint64_t>(channel.remoteRank) * bytes, bytes));
                CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
            } else {
                CHK_RET(HcommWriteOnThread(worker, channel.handle, remote,
                    input + static_cast<uint64_t>(channel.remoteRank) * bytes, direct));
                CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel.handle, NOTIFY_IDX_ACK));
            }
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
            CHK_RET(HcommThreadNotifyRecordOnThread(worker, master, static_cast<uint32_t>(i)));
        }
        for (size_t i = 1; i < resources.threads.size(); ++i) {
            CHK_RET(HcommThreadNotifyWaitOnThread(master, static_cast<uint32_t>(i), CUSTOM_TIMEOUT));
        }
    } else if ((localMask & (1u << param.myRank)) != 0) {
        const auto &rootChannel = PeerChannel(resources, param.myRank, param.root);
        auto *local = static_cast<char *>(resources.localBuffer.addr);
        uint32_t relayIndices[2]{};
        uint32_t workerIndices[2]{};
        uint32_t relayCount = 0;
        for (uint32_t r = 0; r < receiverCount; ++r) {
            if (helpers[r % helperCount] != param.myRank) { continue; }
            const uint32_t peer = receivers[r];
            relayIndices[relayCount] = r;
            workerIndices[relayCount++] = peer < param.myRank ? peer + 1 : peer;
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(master, rootChannel.handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(master, rootChannel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        for (uint32_t r = 0; r < relayCount; ++r) {
            CHK_RET(HcommThreadNotifyRecordOnThread(master, resources.threads[workerIndices[r]], 0));
        }
        CHK_RET(HcommChannelNotifyWaitOnThread(master, rootChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(master, output, local, bytes));
        for (uint32_t r = 0; r < relayCount; ++r) {
            const uint32_t workerIndex = workerIndices[r];
            const ThreadHandle worker = resources.threads[workerIndex];
            const auto &channel = resources.channels[workerIndex - 1];
            const uint64_t slot = bytes + static_cast<uint64_t>(relayIndices[r] / helperCount) * forwarded;
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            CHK_RET(HcommWriteOnThread(worker, channel.handle,
                static_cast<char *>(channel.remoteCclMem.addr) + direct, local + slot, forwarded));
            CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel.handle, NOTIFY_IDX_ACK));
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
            CHK_RET(HcommThreadNotifyRecordOnThread(worker, master, workerIndex));
        }
        for (uint32_t r = 0; r < relayCount; ++r) {
            CHK_RET(HcommThreadNotifyWaitOnThread(master, workerIndices[r], CUSTOM_TIMEOUT));
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(master, rootChannel.handle, NOTIFY_IDX_DATA_SIGNAL));
    } else {
        const auto &rootChannel = PeerChannel(resources, param.myRank, param.root);
        const uint32_t r = static_cast<uint32_t>(
            std::find(receivers, receivers + receiverCount, param.myRank) - receivers);
        const auto &helperChannel = PeerChannel(resources, param.myRank, helpers[r % helperCount]);
        CHK_RET(HcommChannelNotifyRecordOnThread(master, rootChannel.handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyRecordOnThread(master, helperChannel.handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(master, rootChannel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        CHK_RET(HcommChannelNotifyWaitOnThread(master, helperChannel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(master, output, resources.localBuffer.addr, bytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(master, rootChannel.handle, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyRecordOnThread(master, helperChannel.handle, NOTIFY_IDX_DATA_SIGNAL));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecPartialRelay(const OpParam &param, const AlgResourceCtx &resources, uint64_t chunk)
{
    const uint64_t bytes = param.count * sizeof(float);
    const ThreadHandle thread = resources.aicpuThread;
    const uint32_t localMask = (resources.relayTopology.groupMasks[0] & (1u << param.root)) != 0
        ? resources.relayTopology.groupMasks[0] : resources.relayTopology.groupMasks[1];
    uint32_t helpers[7]{};
    uint32_t receivers[8]{};
    uint32_t helperCount = 0;
    uint32_t receiverCount = 0;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.root) { continue; }
        if ((localMask & (1u << rank)) != 0) { helpers[helperCount++] = rank; }
        else { receivers[receiverCount++] = rank; }
    }
    const uint64_t relaySlot = chunk / 16 * sizeof(float);
    auto *local = static_cast<char *>(resources.localBuffer.addr);
    auto *output = static_cast<char *>(param.outputPtr);
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t len = std::min(bytes - offset, chunk);
        const uint64_t forwarded = len / 16 * sizeof(float);
        const uint64_t direct = len - forwarded;
        if (param.myRank == param.root) {
            const auto *input = static_cast<const char *>(param.inputPtr);
            for (uint32_t h = 0; h < helperCount; ++h) {
                const uint32_t helper = helpers[h];
                CHK_RET(HcommLocalCopyOnThread(thread, local + static_cast<uint64_t>(helper) * chunk,
                    input + static_cast<uint64_t>(helper) * bytes + offset, len));
                for (uint32_t i = h; i < receiverCount; i += helperCount) {
                    const uint32_t peer = receivers[i];
                    CHK_RET(HcommLocalCopyOnThread(thread, local + static_cast<uint64_t>(peer) * chunk,
                        input + static_cast<uint64_t>(peer) * bytes + offset, len));
                }
                CHK_RET(HcommChannelNotifyRecordOnThread(thread,
                    PeerChannel(resources, param.myRank, helper).handle, NOTIFY_IDX_ACK));
                for (uint32_t i = h; i < receiverCount; i += helperCount) {
                    CHK_RET(HcommChannelNotifyRecordOnThread(thread,
                        PeerChannel(resources, param.myRank, receivers[i]).handle, NOTIFY_IDX_ACK));
                }
            }
            for (const auto &channel : resources.channels) {
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
            }
        } else {
            const auto &rootChannel = PeerChannel(resources, param.myRank, param.root);
            const auto *rootCcl = static_cast<const char *>(rootChannel.remoteCclMem.addr);
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, rootChannel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            if ((localMask & (1u << param.myRank)) != 0) {
                CHK_RET(HcommReadOnThread(thread, rootChannel.handle, output + offset,
                    rootCcl + static_cast<uint64_t>(param.myRank) * chunk, len));
                uint32_t h = 0;
                while (helpers[h] != param.myRank) { ++h; }
                if (forwarded != 0) {
                    for (uint32_t i = h; i < receiverCount; i += helperCount) {
                        const uint32_t peer = receivers[i];
                        const uint64_t slot = i / helperCount * relaySlot;
                        CHK_RET(HcommReadOnThread(thread, rootChannel.handle, local + slot,
                            rootCcl + static_cast<uint64_t>(peer) * chunk + direct, forwarded));
                        CHK_RET(HcommChannelNotifyRecordOnThread(thread,
                            PeerChannel(resources, param.myRank, peer).handle, NOTIFY_IDX_ACK));
                    }
                    for (uint32_t i = h; i < receiverCount; i += helperCount) {
                        CHK_RET(HcommChannelNotifyWaitOnThread(thread,
                            PeerChannel(resources, param.myRank, receivers[i]).handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
                    }
                }
            } else {
                uint32_t i = 0;
                while (receivers[i] != param.myRank) { ++i; }
                CHK_RET(HcommReadOnThread(thread, rootChannel.handle, output + offset,
                    rootCcl + static_cast<uint64_t>(param.myRank) * chunk, direct));
                if (forwarded != 0) {
                    const auto &helperChannel = PeerChannel(resources, param.myRank, helpers[i % helperCount]);
                    const auto *helperCcl = static_cast<const char *>(helperChannel.remoteCclMem.addr);
                    const uint64_t slot = i / helperCount * relaySlot;
                    CHK_RET(HcommChannelNotifyWaitOnThread(thread, helperChannel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
                    CHK_RET(HcommReadOnThread(thread, helperChannel.handle, output + offset + direct,
                        helperCcl + slot, forwarded));
                    CHK_RET(HcommChannelNotifyRecordOnThread(thread, helperChannel.handle, NOTIFY_IDX_DATA_SIGNAL));
                }
            }
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, NOTIFY_IDX_DATA_SIGNAL));
        }
        offset += len;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    const uint64_t rankBytes = param.count * sizeof(float);
    const uint64_t totalBytes = rankBytes * param.rankSize;
    const uint64_t maxStageBytes = std::numeric_limits<uint32_t>::max();
    const ThreadHandle thread = resCtx.aicpuThread;
    auto *local = static_cast<char *>(resCtx.localBuffer.addr);
    auto *output = static_cast<char *>(param.outputPtr);
    if (param.rankSize == 16 && rankBytes >= 1024ULL * 1024 && rankBytes <= maxStageBytes) {
        uint64_t minimumCapacity = resCtx.localBuffer.size;
        for (const auto &channel : resCtx.channels) {
            minimumCapacity = std::min(minimumCapacity, channel.remoteCclMem.size);
        }
        if (rankBytes <= minimumCapacity) {
            const uint64_t forwarded = (param.count / 4) * sizeof(float);
            if (resCtx.relayTopology.groupMasks[0] != 0 && rankBytes + 2 * forwarded <= minimumCapacity) {
                return ExecQuarterPush(param, resCtx);
            }
            return ExecParallelPush(param, resCtx);
        }
    }
    uint64_t relayChunk = 0;
    if (rankBytes >= 1024ULL * 1024 && param.rankSize == 16 && resCtx.relayTopology.groupMasks[0] != 0) {
        const uint64_t rootCapacity = param.myRank == param.root ? resCtx.localBuffer.size
            : PeerChannel(resCtx, param.myRank, param.root).remoteCclMem.size;
        uint64_t minimumCapacity = resCtx.localBuffer.size;
        for (const auto &channel : resCtx.channels) {
            minimumCapacity = std::min(minimumCapacity, channel.remoteCclMem.size);
        }
        relayChunk = std::min(std::min(rankBytes, rootCapacity / param.rankSize),
            std::min(minimumCapacity, maxStageBytes)) / sizeof(float) * sizeof(float);
        if (relayChunk < 16) { relayChunk = 0; }
    }

    if (param.myRank == param.root) {
        const auto *input = static_cast<const char *>(param.inputPtr);
        const char *self = input + static_cast<uint64_t>(param.root) * rankBytes;
        const uint64_t selfChunk = resCtx.localBuffer.size / sizeof(float) * sizeof(float);
        if (self != output) {
            for (uint64_t offset = 0; offset < rankBytes;) {
                const uint64_t len = std::min(rankBytes - offset, selfChunk);
                CHK_RET(HcommLocalCopyOnThread(thread, output + offset, self + offset, len));
                offset += len;
            }
        }
        if (relayChunk != 0) { return ExecPartialRelay(param, resCtx, relayChunk); }
        if (totalBytes <= resCtx.localBuffer.size && totalBytes <= maxStageBytes) {
            if (rankBytes >= 1024ULL * 1024) {
                for (const auto &channel : resCtx.channels) {
                    const uint64_t offset = static_cast<uint64_t>(channel.remoteRank) * rankBytes;
                    CHK_RET(HcommLocalCopyOnThread(thread, local + offset, input + offset, rankBytes));
                    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK));
                }
            } else {
                CHK_RET(HcommLocalCopyOnThread(thread, local, input, totalBytes));
                const uint64_t childEnd = std::min<uint64_t>(param.rankSize, 5);
                for (uint64_t child = 1; child < childEnd; ++child) {
                    const uint64_t peer = (child + param.root) % param.rankSize;
                    const auto &channel = resCtx.channels[peer < param.myRank ? peer : peer - 1];
                    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK));
                }
                for (uint64_t child = 1; child < childEnd; ++child) {
                    const uint64_t peer = (child + param.root) % param.rankSize;
                    const auto &channel = resCtx.channels[peer < param.myRank ? peer : peer - 1];
                    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
                }
                return HCCL_SUCCESS;
            }
            for (const auto &channel : resCtx.channels) {
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
            }
            return HCCL_SUCCESS;
        }
        const uint64_t batchRanks = rankBytes <= maxStageBytes
            ? std::min<uint64_t>(param.rankSize, resCtx.localBuffer.size / rankBytes) : 0;
        if (batchRanks >= 2) {
            for (uint64_t peer = 0; peer < param.rankSize; ++peer) {
                if (peer >= batchRanks) {
                    const uint64_t old = peer - batchRanks;
                    if (old != param.root) {
                        const auto &channel = resCtx.channels[old < param.root ? old : old - 1];
                        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
                    }
                }
                if (peer == param.root) { continue; }
                const auto &channel = resCtx.channels[peer < param.root ? peer : peer - 1];
                const char *source = input + peer * rankBytes;
                char *slot = local + (peer % batchRanks) * rankBytes;
                CHK_RET(HcommLocalCopyOnThread(thread, slot, source, rankBytes));
                CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK));
            }
            for (uint64_t peer = param.rankSize - batchRanks; peer < param.rankSize; ++peer) {
                if (peer == param.root) { continue; }
                const auto &channel = resCtx.channels[peer < param.root ? peer : peer - 1];
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
            }
            return HCCL_SUCCESS;
        }
        const uint64_t slotBytes = std::min(resCtx.localBuffer.size / param.rankSize, maxStageBytes)
            / sizeof(float) * sizeof(float);
        if (slotBytes != 0) {
            for (uint64_t offset = 0; offset < rankBytes;) {
                const uint64_t len = std::min(rankBytes - offset, slotBytes);
                for (const auto &channel : resCtx.channels) {
                    const char *source = input + static_cast<uint64_t>(channel.remoteRank) * rankBytes + offset;
                    char *slot = local + static_cast<uint64_t>(channel.remoteRank) * slotBytes;
                    CHK_RET(HcommLocalCopyOnThread(thread, slot, source, len));
                    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK));
                }
                for (const auto &channel : resCtx.channels) {
                    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
                }
                offset += len;
            }
            return HCCL_SUCCESS;
        }
        for (const auto &channel : resCtx.channels) {
            const uint64_t chunk = std::min(resCtx.localBuffer.size, channel.remoteCclMem.size)
                / sizeof(float) * sizeof(float);
            const char *source = input + static_cast<uint64_t>(channel.remoteRank) * rankBytes;
            for (uint64_t offset = 0; offset < rankBytes;) {
                const uint64_t len = std::min(rankBytes - offset, chunk);
                CHK_RET(HcommLocalCopyOnThread(thread, local, source + offset, len));
                CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK));
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
                offset += len;
            }
        }
    } else {
        if (relayChunk != 0) { return ExecPartialRelay(param, resCtx, relayChunk); }
        const auto it = std::find_if(resCtx.channels.begin(), resCtx.channels.end(),
            [&param](const ChannelInfo &channel) { return channel.remoteRank == param.root; });
        if (it == resCtx.channels.end()) {
            return HCCL_E_INTERNAL;
        }
        if (totalBytes <= it->remoteCclMem.size && totalBytes <= maxStageBytes) {
            const char *source = static_cast<const char *>(it->remoteCclMem.addr)
                + static_cast<uint64_t>(param.myRank) * rankBytes;
            if (rankBytes < 1024ULL * 1024) {
                const uint64_t logicalRank = (static_cast<uint64_t>(param.myRank)
                    + param.rankSize - param.root) % param.rankSize;
                const uint64_t parent = ((logicalRank - 1) / 4 + param.root) % param.rankSize;
                const auto &parentChannel = resCtx.channels[parent < param.myRank ? parent : parent - 1];
                const uint64_t firstChild = logicalRank * 4 + 1;
                const uint64_t childEnd = std::min<uint64_t>(param.rankSize, firstChild + 4);
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, parentChannel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
                for (uint64_t child = firstChild; child < childEnd; ++child) {
                    const uint64_t peer = (child + param.root) % param.rankSize;
                    const auto &channel = resCtx.channels[peer < param.myRank ? peer : peer - 1];
                    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK));
                }
                CHK_RET(HcommReadOnThread(thread, it->handle, output, source, rankBytes));
                for (uint64_t child = firstChild; child < childEnd; ++child) {
                    const uint64_t peer = (child + param.root) % param.rankSize;
                    const auto &channel = resCtx.channels[peer < param.myRank ? peer : peer - 1];
                    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
                }
                CHK_RET(HcommChannelNotifyRecordOnThread(thread, parentChannel.handle, NOTIFY_IDX_DATA_SIGNAL));
                return HCCL_SUCCESS;
            }
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, it->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            CHK_RET(HcommReadOnThread(thread, it->handle, output, source, rankBytes));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, it->handle, NOTIFY_IDX_DATA_SIGNAL));
            return HCCL_SUCCESS;
        }
        const uint64_t batchRanks = rankBytes <= maxStageBytes
            ? std::min<uint64_t>(param.rankSize, it->remoteCclMem.size / rankBytes) : 0;
        if (batchRanks >= 2) {
            const char *source = static_cast<const char *>(it->remoteCclMem.addr)
                + (param.myRank % batchRanks) * rankBytes;
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, it->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            CHK_RET(HcommReadOnThread(thread, it->handle, output, source, rankBytes));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, it->handle, NOTIFY_IDX_DATA_SIGNAL));
            return HCCL_SUCCESS;
        }
        const uint64_t slotBytes = std::min(it->remoteCclMem.size / param.rankSize, maxStageBytes)
            / sizeof(float) * sizeof(float);
        if (slotBytes != 0) {
            const char *source = static_cast<const char *>(it->remoteCclMem.addr)
                + static_cast<uint64_t>(param.myRank) * slotBytes;
            for (uint64_t offset = 0; offset < rankBytes;) {
                const uint64_t len = std::min(rankBytes - offset, slotBytes);
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, it->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
                CHK_RET(HcommReadOnThread(thread, it->handle, output + offset, source, len));
                CHK_RET(HcommChannelNotifyRecordOnThread(thread, it->handle, NOTIFY_IDX_DATA_SIGNAL));
                offset += len;
            }
            return HCCL_SUCCESS;
        }
        const uint64_t chunk = std::min(resCtx.localBuffer.size, it->remoteCclMem.size)
            / sizeof(float) * sizeof(float);
        for (uint64_t offset = 0; offset < rankBytes;) {
            const uint64_t len = std::min(rankBytes - offset, chunk);
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, it->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            CHK_RET(HcommReadOnThread(thread, it->handle, output + offset, it->remoteCclMem.addr, len));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, it->handle, NOTIFY_IDX_DATA_SIGNAL));
            offset += len;
        }
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
