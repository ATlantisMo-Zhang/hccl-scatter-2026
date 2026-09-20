/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel.h"
#include "custom.h"

namespace ccu = AscendC::ccu;

#define CCU_CHECK(call) \
    do { \
        const CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            HCCL_ERROR("[Scatter] CCU primitive failed: %d", result); \
            return result; \
        } \
    } while (0)

namespace ops_hccl {
CcuResult CcuSmallPullKernel(CcuKernelArg arg)
{
    const auto *reg = static_cast<const CcuKernelArgBase *>(arg);
    if (reg == nullptr || reg->rankSize == 0 || reg->rankSize > MAX_RANK_SIZE
        || reg->myRank >= reg->rankSize || reg->channelCount == 0
        || reg->channelCount >= reg->rankSize) { return CCU_E_PARA; }
    constexpr uint32_t SOURCE_XN = 1;
    constexpr uint32_t TOKEN_XN = 2;
    constexpr uint16_t SOURCE_READY = (1u << SOURCE_XN) | (1u << TOKEN_XN);
    constexpr uint16_t DONE = 1u << 4;
    ccu::Variable inputToken, output, outputToken, len, root, self, doSelf;
    CCU_CHECK(ccu::LoadArg(inputToken, 0));
    CCU_CHECK(ccu::LoadArg(output, 1));
    CCU_CHECK(ccu::LoadArg(outputToken, 2));
    CCU_CHECK(ccu::LoadArg(len, 3));
    CCU_CHECK(ccu::LoadArg(root, 4));
    CCU_CHECK(ccu::LoadArg(self, 5));
    CCU_CHECK(ccu::LoadArg(doSelf, 6));
    std::vector<ccu::Variable> source(reg->channelCount);
    const bool compact = reg->channelCount >= 7;
    std::vector<ccu::Variable> compactArgs(compact ? 2 : 0);
    if (compact) {
        CCU_CHECK(ccu::LoadArg(compactArgs[0], 7));
        CCU_CHECK(ccu::LoadArg(compactArgs[1], 8));
    }
    std::vector<ccu::Variable> peerSource;
    std::vector<ccu::Variable> peerToken;
    for (uint32_t i = 0; i < reg->channelCount; ++i) {
        if (!compact) { CCU_CHECK(ccu::LoadArg(source[i], 7 + i)); }
        peerSource.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], SOURCE_XN));
        peerToken.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], TOKEN_XN));
    }
    ccu::Event received;
    ccu::Event copied;
    CCU_IF(root == reg->myRank)
    {
        if (compact) {
            uint32_t previousPeer = reg->peers[0];
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                if (i == 0) { source[i] = compactArgs[0]; }
                else { source[i] = source[i - 1]; }
                for (uint32_t peer = previousPeer; peer < reg->peers[i]; ++peer) {
                    source[i] += compactArgs[1];
                }
                previousPeer = reg->peers[i];
            }
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], source[i], SOURCE_XN, 0, 1u << SOURCE_XN));
            CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], inputToken, TOKEN_XN, 0, 1u << TOKEN_XN));
        }
        if (reg->copySelf != 0) {
            CCU_IF(doSelf != 0)
            {
                ccu::LocalAddr src;
                ccu::LocalAddr dst;
                src.addr = self;
                src.token = inputToken;
                dst.addr = output;
                dst.token = outputToken;
                CCU_CHECK(ccu::LocalCopy(dst, src, len, copied, 1));
                CCU_CHECK(ccu::EventWait(copied, 1));
            }
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, DONE));
        }
    }
    for (uint32_t i = 0; i < reg->channelCount; ++i) {
        CCU_IF(root == reg->peers[i])
        {
            CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, SOURCE_READY));
            ccu::LocalAddr dst;
            ccu::RemoteAddr src;
            dst.addr = output;
            dst.token = outputToken;
            src.addr = peerSource[i];
            src.token = peerToken[i];
            CCU_CHECK(ccu::Read(reg->channels[i], dst, src, len, received, 1));
            CCU_CHECK(ccu::EventWait(received, 1));
            CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, DONE));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuKernel(CcuKernelArg arg)
{
    const auto *reg = static_cast<const CcuKernelArgBase *>(arg);
    if (reg == nullptr || reg->rankSize == 0 || reg->rankSize > MAX_RANK_SIZE
        || reg->myRank >= reg->rankSize || reg->channelCount >= reg->rankSize) { return CCU_E_PARA; }
    constexpr uint32_t OUTPUT_XN = 1;
    constexpr uint32_t TOKEN_XN = 2;
    constexpr uint16_t READY = (1u << OUTPUT_XN) | (1u << TOKEN_XN);
    constexpr uint16_t WRITTEN = 1u << 3;
    constexpr uint16_t CONSUMED = 1u << 4;
    ccu::Variable inputToken, output, outputToken, len, root, self, doSelf;
    CCU_CHECK(ccu::LoadArg(inputToken, 0));
    CCU_CHECK(ccu::LoadArg(output, 1));
    CCU_CHECK(ccu::LoadArg(outputToken, 2));
    CCU_CHECK(ccu::LoadArg(len, 3));
    CCU_CHECK(ccu::LoadArg(root, 4));
    CCU_CHECK(ccu::LoadArg(self, 5));
    CCU_CHECK(ccu::LoadArg(doSelf, 6));
    std::vector<ccu::Variable> source(reg->channelCount);
    const bool compact = reg->channelCount >= 7;
    std::vector<ccu::Variable> compactArgs(compact ? 2 : 0);
    if (compact) {
        CCU_CHECK(ccu::LoadArg(compactArgs[0], 7));
        CCU_CHECK(ccu::LoadArg(compactArgs[1], 8));
    }
    std::vector<ccu::Variable> peerOutput;
    std::vector<ccu::Variable> peerToken;
    for (uint32_t i = 0; i < reg->channelCount; ++i) {
        if (!compact) { CCU_CHECK(ccu::LoadArg(source[i], 7 + i)); }
        peerOutput.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], OUTPUT_XN));
        peerToken.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], TOKEN_XN));
    }
    ccu::Event sent;
    ccu::Event copied;
    CCU_IF(root == reg->myRank)
    {
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, READY));
        }
        if (compact) {
            uint32_t previousPeer = reg->peers[0];
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                if (i == 0) { source[i] = compactArgs[0]; }
                else { source[i] = source[i - 1]; }
                for (uint32_t peer = previousPeer; peer < reg->peers[i]; ++peer) {
                }
                previousPeer = reg->peers[i];
            }
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            ccu::LocalAddr src;
            ccu::RemoteAddr dst;
            src.addr = source[i];
            src.token = inputToken;
            dst.addr = peerOutput[i];
            dst.token = peerToken[i];
            CCU_CHECK(ccu::Write(reg->channels[i], dst, src, len, sent, 1u << i));
        }
        if (reg->copySelf != 0) {
            CCU_IF(doSelf != 0)
            {
                ccu::LocalAddr src;
                ccu::LocalAddr dst;
                src.addr = self;
                src.token = inputToken;
                dst.addr = output;
                dst.token = outputToken;
                CCU_CHECK(ccu::LocalCopy(dst, src, len, copied, 1));
                CCU_CHECK(ccu::EventWait(copied, 1));
            }
        }
        if (reg->channelCount != 0) {
            CCU_CHECK(ccu::EventWait(sent, (1u << reg->channelCount) - 1u));
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, WRITTEN));
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, CONSUMED));
        }
    }
    for (uint32_t i = 0; i < reg->channelCount; ++i) {
        CCU_IF(root == reg->peers[i])
        {
            CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], output, OUTPUT_XN, 0, 1u << OUTPUT_XN));
            CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], outputToken, TOKEN_XN, 0, 1u << TOKEN_XN));
            CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, WRITTEN));
            CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, CONSUMED));
        }
    }
    return CCU_SUCCESS;
}
CcuResult CcuSmallReadyPushKernel(CcuKernelArg arg)
{
    const auto *reg = static_cast<const CcuKernelArgBase *>(arg);
    if (reg == nullptr || reg->rankSize == 0 || reg->rankSize > MAX_RANK_SIZE
        || reg->myRank >= reg->rankSize || reg->channelCount == 0
        || reg->relayTopology != 0 || reg->channelCount >= reg->rankSize) { return CCU_E_PARA; }
    constexpr uint32_t OUTPUT_XN = 1;
    constexpr uint32_t TOKEN_XN = 2;
    constexpr uint16_t READY = (1u << OUTPUT_XN) | (1u << TOKEN_XN);
    constexpr uint16_t WRITTEN = 1u << 3;
    ccu::Variable inputToken, output, outputToken, len, root, self, doSelf;
    CCU_CHECK(ccu::LoadArg(inputToken, 0));
    CCU_CHECK(ccu::LoadArg(output, 1));
    CCU_CHECK(ccu::LoadArg(outputToken, 2));
    CCU_CHECK(ccu::LoadArg(len, 3));
    CCU_CHECK(ccu::LoadArg(root, 4));
    CCU_CHECK(ccu::LoadArg(self, 5));
    CCU_CHECK(ccu::LoadArg(doSelf, 6));
    std::vector<ccu::Variable> source(reg->channelCount);
    const bool compact = reg->channelCount >= 7;
    std::vector<ccu::Variable> compactArgs(compact ? 2 : 0);
    if (compact) {
        CCU_CHECK(ccu::LoadArg(compactArgs[0], 7));
        CCU_CHECK(ccu::LoadArg(compactArgs[1], 8));
    }
    std::vector<ccu::Variable> peerOutput;
    std::vector<ccu::Variable> peerToken;
    for (uint32_t i = 0; i < reg->channelCount; ++i) {
        if (!compact) { CCU_CHECK(ccu::LoadArg(source[i], 7 + i)); }
        peerOutput.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], OUTPUT_XN));
        peerToken.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], TOKEN_XN));
    }
    ccu::Event sent;
    ccu::Event copied;
    CCU_IF(root == reg->myRank)
    {
        if (compact) {
            uint32_t previousPeer = reg->peers[0];
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                if (i == 0) { source[i] = compactArgs[0]; }
                else { source[i] = source[i - 1]; }
                for (uint32_t peer = previousPeer; peer < reg->peers[i]; ++peer) {
                }
                previousPeer = reg->peers[i];
            }
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, READY));
            ccu::LocalAddr src;
            ccu::RemoteAddr dst;
            src.addr = source[i];
            src.token = inputToken;
            dst.addr = peerOutput[i];
            dst.token = peerToken[i];
            CCU_CHECK(ccu::Write(reg->channels[i], dst, src, len, sent, 1u << i));
        }
        if (reg->copySelf != 0) {
            CCU_IF(doSelf != 0)
            {
                ccu::LocalAddr src;
                ccu::LocalAddr dst;
                src.addr = self;
                src.token = inputToken;
                dst.addr = output;
                dst.token = outputToken;
                CCU_CHECK(ccu::LocalCopy(dst, src, len, copied, 1));
                CCU_CHECK(ccu::EventWait(copied, 1));
            }
        }
        if (reg->channelCount != 0) {
            CCU_CHECK(ccu::EventWait(sent, (1u << reg->channelCount) - 1u));
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, WRITTEN));
        }
    }
    for (uint32_t i = 0; i < reg->channelCount; ++i) {
        CCU_IF(root == reg->peers[i])
        {
            CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], output, OUTPUT_XN, 0, 1u << OUTPUT_XN));
            CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], outputToken, TOKEN_XN, 0, 1u << TOKEN_XN));
            CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, WRITTEN));
        }
    }
    return CCU_SUCCESS;
}
CcuResult CcuDualModeKernel(CcuKernelArg arg)
{
    const auto *reg = static_cast<const CcuKernelArgBase *>(arg);
    if (reg == nullptr || reg->rankSize == 0 || reg->rankSize > MAX_RANK_SIZE
        || reg->myRank >= reg->rankSize || reg->relayTopology != 2
        || reg->channelCount < 7 || reg->channelCount >= reg->rankSize) { return CCU_E_PARA; }
    constexpr uint32_t OUTPUT_XN = 1;
    constexpr uint32_t TOKEN_XN = 2;
    constexpr uint16_t READY = (1u << OUTPUT_XN) | (1u << TOKEN_XN);
    constexpr uint16_t WRITTEN = 1u << 3;
    constexpr uint16_t CONSUMED = 1u << 4;
    ccu::Variable inputToken, output, outputToken, len, root, self, doSelf, pullMode;
    CCU_CHECK(ccu::LoadArg(inputToken, 0));
    CCU_CHECK(ccu::LoadArg(output, 1));
    CCU_CHECK(ccu::LoadArg(outputToken, 2));
    CCU_CHECK(ccu::LoadArg(len, 3));
    CCU_CHECK(ccu::LoadArg(root, 4));
    CCU_CHECK(ccu::LoadArg(self, 5));
    CCU_CHECK(ccu::LoadArg(doSelf, 6));
    CCU_CHECK(ccu::LoadArg(pullMode, 9));
    std::vector<ccu::Variable> source(reg->channelCount);
    const bool compact = reg->channelCount >= 7;
    std::vector<ccu::Variable> compactArgs(compact ? 2 : 0);
    if (compact) {
        CCU_CHECK(ccu::LoadArg(compactArgs[0], 7));
        CCU_CHECK(ccu::LoadArg(compactArgs[1], 8));
    }
    std::vector<ccu::Variable> peerOutput;
    std::vector<ccu::Variable> peerToken;
    for (uint32_t i = 0; i < reg->channelCount; ++i) {
        if (!compact) { CCU_CHECK(ccu::LoadArg(source[i], 7 + i)); }
        peerOutput.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], OUTPUT_XN));
        peerToken.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], TOKEN_XN));
    }
    ccu::Event sent;
    ccu::Event copied;
    ccu::Event received;
    CCU_IF(pullMode == 0)
    {
        CCU_IF(root == reg->myRank)
        {
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, READY));
            }
            if (compact) {
                uint32_t previousPeer = reg->peers[0];
                for (uint32_t i = 0; i < reg->channelCount; ++i) {
                    if (i == 0) { source[i] = compactArgs[0]; }
                    else { source[i] = source[i - 1]; }
                    for (uint32_t peer = previousPeer; peer < reg->peers[i]; ++peer) {
                    }
                    previousPeer = reg->peers[i];
                }
            }
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                ccu::LocalAddr src;
                ccu::RemoteAddr dst;
                src.addr = source[i];
                src.token = inputToken;
                dst.addr = peerOutput[i];
                dst.token = peerToken[i];
                CCU_CHECK(ccu::Write(reg->channels[i], dst, src, len, sent, 1u << i));
            }
            if (reg->copySelf != 0) {
                CCU_IF(doSelf != 0)
                {
                    ccu::LocalAddr src;
                    ccu::LocalAddr dst;
                    src.addr = self;
                    src.token = inputToken;
                    dst.addr = output;
                    dst.token = outputToken;
                    CCU_CHECK(ccu::LocalCopy(dst, src, len, copied, 1));
                    CCU_CHECK(ccu::EventWait(copied, 1));
                }
            }
            if (reg->channelCount != 0) {
                CCU_CHECK(ccu::EventWait(sent, (1u << reg->channelCount) - 1u));
            }
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, WRITTEN));
            }
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, CONSUMED));
            }
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_IF(root == reg->peers[i])
            {
                CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], output, OUTPUT_XN, 0, 1u << OUTPUT_XN));
                CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], outputToken, TOKEN_XN, 0, 1u << TOKEN_XN));
                CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, WRITTEN));
                CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, CONSUMED));
            }
        }
    }
    CCU_IF(pullMode == 1)
    {
        CCU_IF(root == reg->myRank)
        {
            if (compact) {
                uint32_t previousPeer = reg->peers[0];
                for (uint32_t i = 0; i < reg->channelCount; ++i) {
                    if (i == 0) { source[i] = compactArgs[0]; }
                    else { source[i] = source[i - 1]; }
                    for (uint32_t peer = previousPeer; peer < reg->peers[i]; ++peer) {
                        source[i] += compactArgs[1];
                    }
                    previousPeer = reg->peers[i];
                }
            }
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], source[i], OUTPUT_XN, 0, 1u << OUTPUT_XN));
                CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], inputToken, TOKEN_XN, 0, 1u << TOKEN_XN));
            }
            if (reg->copySelf != 0) {
                CCU_IF(doSelf != 0)
                {
                    ccu::LocalAddr src;
                    ccu::LocalAddr dst;
                    src.addr = self;
                    src.token = inputToken;
                    dst.addr = output;
                    dst.token = outputToken;
                    CCU_CHECK(ccu::LocalCopy(dst, src, len, copied, 1));
                    CCU_CHECK(ccu::EventWait(copied, 1));
                }
            }
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, CONSUMED));
            }
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_IF(root == reg->peers[i])
            {
                CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, READY));
                ccu::LocalAddr dst;
                ccu::RemoteAddr src;
                dst.addr = output;
                dst.token = outputToken;
                src.addr = peerOutput[i];
                src.token = peerToken[i];
                CCU_CHECK(ccu::Read(reg->channels[i], dst, src, len, received, 1));
                CCU_CHECK(ccu::EventWait(received, 1));
                CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, CONSUMED));
            }
        }
    }
    return CCU_SUCCESS;
}
} // namespace ops_hccl

namespace ops_hccl {
namespace {
constexpr uint16_t RELAY_READY = (1u << 1) | (1u << 2);
constexpr uint16_t RELAY_WRITTEN = 1u << 3;
constexpr uint16_t RELAY_CONSUMED = 1u << 4;
constexpr uint16_t RELAY_STAGED = 1u << 5;
constexpr uint16_t RELAY_FORWARDED = 1u << 6;
constexpr uint16_t RELAY_DONE = 1u << 7;
constexpr uint16_t RELAY_DONE_ACK = 1u << 8;
constexpr uint16_t RELAY_PLAN = (1u << 9) | (1u << 10);
constexpr uint64_t NO_HELPER = UINT64_MAX;

CcuResult RelayWrite(ChannelHandle channel, ccu::Variable &dstAddress, ccu::Variable &dstToken,
    ccu::Variable &srcAddress, ccu::Variable &srcToken, ccu::Variable &bytes, ccu::Event &event, uint16_t mask)
{
    ccu::LocalAddr src;
    ccu::RemoteAddr dst;
    src.addr = srcAddress;
    src.token = srcToken;
    dst.addr = dstAddress;
    dst.token = dstToken;
    return ccu::Write(channel, dst, src, bytes, event, mask);
}

CcuResult RelayReady(ChannelHandle channel, ccu::Variable &address, ccu::Variable &token)
{
    CCU_CHECK(ccu::WriteVariableWithNotify(channel, address, 1, 0, 1u << 1));
    CCU_CHECK(ccu::WriteVariableWithNotify(channel, token, 2, 0, 1u << 2));
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuRelayKernel(CcuKernelArg arg)
{
    const auto *reg = static_cast<const CcuKernelArgBase *>(arg);
    if (reg == nullptr || (reg->relayTopology != 2 && reg->relayTopology != 4)
        || reg->rankSize > MAX_RANK_SIZE || reg->myRank >= reg->rankSize
        || reg->channelCount == 0 || reg->channelCount >= reg->rankSize) { return CCU_E_PARA; }
    ccu::Variable inputToken, output, outputToken, len, root, self, doSelf;
    ccu::Variable inputBase, stride, relayBytes, directBytes, phase, target;
    CCU_CHECK(ccu::LoadArg(inputToken, 0));
    CCU_CHECK(ccu::LoadArg(output, 1));
    CCU_CHECK(ccu::LoadArg(outputToken, 2));
    CCU_CHECK(ccu::LoadArg(len, 3));
    CCU_CHECK(ccu::LoadArg(root, 4));
    CCU_CHECK(ccu::LoadArg(self, 5));
    CCU_CHECK(ccu::LoadArg(doSelf, 6));
    CCU_CHECK(ccu::LoadArg(inputBase, 7));
    CCU_CHECK(ccu::LoadArg(stride, 8));
    CCU_CHECK(ccu::LoadArg(relayBytes, 9));
    CCU_CHECK(ccu::LoadArg(directBytes, 10));
    CCU_CHECK(ccu::LoadArg(phase, 11));
    CCU_CHECK(ccu::LoadArg(target, 12));
    std::vector<ccu::Variable> peerOutput, peerToken, peerPlan, peerPlanBytes;
    bool mesh = true;
    for (uint32_t i = 0; i < reg->channelCount; ++i) {
        const bool local = (reg->localMask & (1u << reg->peers[i])) != 0;
        if (i == 0) { mesh = local; }
        if (local != mesh) { return CCU_E_PARA; }
        peerOutput.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], 1));
        peerToken.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], 2));
        peerPlan.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], 3));
        peerPlanBytes.push_back(ccu::GetResByChannel<ccu::Variable>(reg->channels[i], 4));
    }
    std::vector<uint32_t> helpers, targets;
    for (uint32_t peer = 0; peer < reg->rankSize; ++peer) {
        if ((reg->localMask & (1u << peer)) != 0) {
            if (peer != reg->myRank) { helpers.push_back(peer); }
        } else { targets.push_back(peer); }
    }
    const uint32_t m = reg->relayTopology == 2 ? 7 : 2;
    if (helpers.size() < m || targets.size() < m) { return CCU_E_PARA; }
    helpers.resize(m);
    targets.resize(m);
    std::vector<ccu::Variable> source(reg->rankSize);
    ccu::Event copied, staged, sent, restored;
    CCU_IF(root == reg->myRank)
    {
        source[0] = inputBase;
        for (uint32_t peer = 1; peer < reg->rankSize; ++peer) {
            source[peer] = source[peer - 1];
            source[peer] += stride;
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, RELAY_READY));
        }
        if (mesh) {
            uint16_t helperMask = 0;
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                for (uint32_t h = 0; h < m; ++h) {
                    if (reg->peers[i] == helpers[h]) {
                        helperMask |= 1u << i;
                    }
                }
            }
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                if ((helperMask & (1u << i)) == 0) {
                    CCU_CHECK(RelayWrite(reg->channels[i], peerOutput[i], peerToken[i],
                        source[reg->peers[i]], inputToken, len, sent, 1u << i));
                }
            }
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                for (uint32_t h = 0; h < m; ++h) {
                    if (reg->peers[i] != helpers[h]) { continue; }
                    ccu::Variable tailSource;
                    tailSource = source[targets[h]];
                    tailSource += directBytes;
                    CCU_CHECK(RelayWrite(reg->channels[i], peerOutput[i], peerToken[i],
                        tailSource, inputToken, relayBytes, staged, 1u << i));
                }
            }
            CCU_CHECK(ccu::EventWait(staged, helperMask));
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                if ((helperMask & (1u << i)) != 0) {
                    CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, RELAY_STAGED));
                }
            }
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                if ((helperMask & (1u << i)) != 0) {
                    ccu::Variable suffixSource, suffixOutput;
                    suffixSource = source[reg->peers[i]];
                    suffixSource += relayBytes;
                    suffixOutput = peerOutput[i];
                    suffixOutput += relayBytes;
                    CCU_CHECK(RelayWrite(reg->channels[i], suffixOutput, peerToken[i],
                        suffixSource, inputToken, directBytes, sent, 1u << i));
                }
            }
            if (reg->relayTopology != 2) {
                CCU_CHECK(ccu::EventWait(sent, (1u << reg->channelCount) - 1u));
            }
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                if ((helperMask & (1u << i)) == 0) { continue; }
                CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, RELAY_FORWARDED));
                CCU_CHECK(RelayWrite(reg->channels[i], peerOutput[i], peerToken[i],
                    source[reg->peers[i]], inputToken, relayBytes, restored, 1u << i));
            }
            if (reg->relayTopology == 2) {
                CCU_CHECK(ccu::EventWait(sent, (1u << reg->channelCount) - 1u));
            }
            CCU_CHECK(ccu::EventWait(restored, helperMask));
        } else {
            std::vector<ccu::Variable> plan(reg->channelCount);
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                uint64_t helper = NO_HELPER;
                for (uint32_t h = 0; h < m; ++h) {
                    if (reg->peers[i] == targets[h]) { helper = helpers[h]; }
                }
                if (reg->relayTopology != 2) {
                    plan[i] = helper;
                    CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], plan[i], 3, 0, 1u << 9));
                    CCU_CHECK(ccu::WriteVariableWithNotify(reg->channels[i], relayBytes, 4, 0, 1u << 10));
                }
                CCU_CHECK(RelayWrite(reg->channels[i], peerOutput[i], peerToken[i],
                    source[reg->peers[i]], inputToken, helper == NO_HELPER ? len : directBytes, sent, 1u << i));
            }
            CCU_CHECK(ccu::EventWait(sent, (1u << reg->channelCount) - 1u));
        }
        if (reg->relayTopology == 4 ? mesh : (reg->copySelf != 0)) {
            CCU_IF(doSelf != 0)
            {
                ccu::LocalAddr src, dst;
                src.addr = self;
                src.token = inputToken;
                dst.addr = output;
                dst.token = outputToken;
                CCU_CHECK(ccu::LocalCopy(dst, src, len, copied, 1));
                CCU_CHECK(ccu::EventWait(copied, 1));
            }
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, RELAY_WRITTEN));
        }
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, RELAY_CONSUMED));
        }
    }
    if (mesh) {
        for (uint32_t i = 0; i < reg->channelCount; ++i) {
            CCU_IF(root == reg->peers[i])
            {
                CCU_IF(phase == 3)
                {
                    CCU_CHECK(RelayReady(reg->channels[i], output, outputToken));
                    CCU_IF(target != NO_HELPER)
                    {
                        CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, RELAY_STAGED));
                    }
                    CCU_IF(target == NO_HELPER)
                    {
                        CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, RELAY_WRITTEN));
                        CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, RELAY_CONSUMED));
                    }
                }
                CCU_IF(phase == 5)
                {
                    CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, RELAY_FORWARDED));
                    CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, RELAY_WRITTEN));
                    CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, RELAY_CONSUMED));
                }
            }
        }
    } else {
        CCU_IF(phase == 4)
        {
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                CCU_IF(target == reg->peers[i])
                {
                    CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, RELAY_READY));
                    CCU_CHECK(RelayWrite(reg->channels[i], peerOutput[i], peerToken[i],
                        output, outputToken, relayBytes, sent, 1));
                    CCU_CHECK(ccu::EventWait(sent, 1));
                    CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, RELAY_DONE));
                    CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, RELAY_DONE_ACK));
                }
            }
        }
        CCU_IF(phase == 6)
        {
            for (uint32_t i = 0; i < reg->channelCount; ++i) {
                CCU_IF(root == reg->peers[i])
                {
                    CCU_CHECK(RelayReady(reg->channels[i], output, outputToken));
                    if (reg->relayTopology != 2) {
                        CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, RELAY_PLAN));
                    }
                    ccu::Variable &helper = reg->relayTopology == 2 ? target : peerPlan[i];
                    ccu::Variable tailOutput;
                    tailOutput = output;
                    tailOutput += directBytes;
                    for (uint32_t h = 0; h < reg->channelCount; ++h) {
                        if (h == i) { continue; }
                        CCU_IF(helper == reg->peers[h])
                        {
                            CCU_CHECK(RelayReady(reg->channels[h], tailOutput, outputToken));
                            CCU_CHECK(ccu::NotifyWait(reg->channels[h], 0, RELAY_DONE));
                            CCU_CHECK(ccu::NotifyRecord(reg->channels[h], 0, RELAY_DONE_ACK));
                        }
                    }
                    CCU_CHECK(ccu::NotifyWait(reg->channels[i], 0, RELAY_WRITTEN));
                    CCU_CHECK(ccu::NotifyRecord(reg->channels[i], 0, RELAY_CONSUMED));
                }
            }
        }
    }
    return CCU_SUCCESS;
}
} // namespace ops_hccl
