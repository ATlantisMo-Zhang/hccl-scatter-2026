/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <cstring>
#include <type_traits>
#include "common.h"
#include "log.h"

constexpr uint64_t SCATTER_CACHE_READY = 0x5343415454455231ULL;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t peers[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
    uint32_t myRank = 0;
    uint32_t rankSize = 0;
    uint32_t copySelf = 0;
    uint32_t localMask = 0;
    uint32_t relayTopology = 0;
};

struct CcuGroup {
    CcuKernelArgBase registration{};
    CcuKernelHandle kernel = 0;
    CcuKernelHandle relayKernel = 0;
    CcuKernelHandle smallPullKernel = 0;
    uint32_t dieId = 0;
    uint32_t layer = 2;
};

struct AlgResourceCtx {
    uint64_t ready = 0;
    uint32_t myRank = 0;
    uint32_t rankSize = 0;
    uint32_t groupCount = 0;
    uint32_t reserved = 0;
    uint32_t localMask = 0;
    uint32_t relayTopology = 0;
    ThreadHandle slaveThread = 0;
    CcuGroup groups[MAX_RANK_SIZE]{};

    static HcclResult Decode(const void *data, uint64_t bytes, AlgResourceCtx &out)
    {
        static_assert(std::is_trivially_copyable<AlgResourceCtx>::value, "fixed cache layout required");
        if (data == nullptr || bytes != sizeof(out)) { return HCCL_E_PARA; }
        std::memcpy(&out, data, sizeof(out));
        if (out.ready != SCATTER_CACHE_READY || out.rankSize == 0 || out.rankSize > MAX_RANK_SIZE
            || out.myRank >= out.rankSize || out.groupCount == 0 || out.groupCount > 2
            || (out.groupCount == 2 ? out.slaveThread == 0 : out.slaveThread != 0)) {
            return HCCL_E_PARA;
        }
        const uint32_t allRanks = (1u << out.rankSize) - 1u;
        if (out.relayTopology == 0) {
            if (out.localMask != 0) { return HCCL_E_PARA; }
        } else {
            if ((out.relayTopology != 2 || out.rankSize != 16)
                && (out.relayTopology != 4 || out.rankSize != 12)) { return HCCL_E_PARA; }
            uint32_t localCount = 0;
            for (uint32_t mask = out.localMask; mask != 0; mask >>= 1) { localCount += mask & 1u; }
            if (out.groupCount != 2 || (out.localMask & ~allRanks) != 0
                || (out.localMask & (1u << out.myRank)) == 0
                || localCount != out.rankSize / out.relayTopology) { return HCCL_E_PARA; }
        }
        uint32_t seen = 0;
        uint32_t seenLayers = 0;
        for (uint32_t g = 0; g < out.groupCount; ++g) {
            const auto &group = out.groups[g];
            const auto &reg = group.registration;
            const bool needSmallPull = out.relayTopology == 0 && reg.channelCount != 0;
            if (group.kernel == 0 || reg.channelCount >= out.rankSize || reg.myRank != out.myRank
                || (needSmallPull ? group.smallPullKernel == 0 : group.smallPullKernel != 0)
                || reg.rankSize != out.rankSize || reg.copySelf != (g == 0 ? 1u : 0u)
                || reg.localMask != out.localMask || reg.relayTopology != out.relayTopology
                || (g != 0 && out.groups[g - 1].dieId >= group.dieId)) { return HCCL_E_PARA; }
            if (out.relayTopology != 0) {
                if (group.relayKernel == 0 || group.layer > 1 || reg.channelCount == 0
                    || (seenLayers & (1u << group.layer)) != 0) { return HCCL_E_PARA; }
                seenLayers |= 1u << group.layer;
            } else if (group.relayKernel != 0) { return HCCL_E_PARA; }
            for (uint32_t i = 0; i < reg.channelCount; ++i) {
                const uint32_t peer = reg.peers[i];
                if (peer >= out.rankSize || peer == out.myRank || reg.channels[i] == 0
                    || (seen & (1u << peer)) != 0 || (i != 0 && reg.peers[i - 1] >= peer)) {
                    return HCCL_E_PARA;
                }
                if (out.relayTopology != 0
                    && group.layer != ((out.localMask & (1u << peer)) != 0 ? 0u : 1u)) {
                    return HCCL_E_PARA;
                }
                seen |= 1u << peer;
            }
        }
        const uint32_t expected = allRanks ^ (1u << out.myRank);
        return seen == expected ? HCCL_SUCCESS : HCCL_E_PARA;
    }
};

inline HcclResult FromCcu(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS: return HCCL_SUCCESS;
        case CCU_E_PARA: return HCCL_E_PARA;
        case CCU_E_PTR: return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT: return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND: return HCCL_E_NOT_FOUND;
        default: return HCCL_E_INTERNAL;
    }
}

#endif // OPS_HCCL_CUSTOM_H
