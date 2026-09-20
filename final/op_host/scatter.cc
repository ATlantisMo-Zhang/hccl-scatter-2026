/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <cstdio>
#include <limits>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_launch.h>

#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

static void AppendU32(std::vector<uint8_t> &key, uint32_t value)
{
    key.push_back(static_cast<uint8_t>(value >> 24));
    key.push_back(static_cast<uint8_t>(value >> 16));
    key.push_back(static_cast<uint8_t>(value >> 8));
    key.push_back(static_cast<uint8_t>(value));
}

static bool AppendEndpointKey(std::vector<uint8_t> &key, const EndpointDesc &endpoint)
{
    if (endpoint.protocol != COMM_PROTOCOL_UBC_CTP
        || endpoint.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        return false;
    }
    AppendU32(key, static_cast<uint32_t>(endpoint.protocol));
    AppendU32(key, static_cast<uint32_t>(endpoint.commAddr.type));
    const uint8_t *address = nullptr;
    size_t size = 0;
    switch (endpoint.commAddr.type) {
        case COMM_ADDR_TYPE_IP_V4:
            address = reinterpret_cast<const uint8_t *>(&endpoint.commAddr.addr);
            size = sizeof(endpoint.commAddr.addr);
            break;
        case COMM_ADDR_TYPE_IP_V6:
            address = reinterpret_cast<const uint8_t *>(&endpoint.commAddr.addr6);
            size = sizeof(endpoint.commAddr.addr6);
            break;
        case COMM_ADDR_TYPE_EID:
            address = endpoint.commAddr.eid;
            size = sizeof(endpoint.commAddr.eid);
            break;
        default:
            return false;
    }
    key.insert(key.end(), address, address + size);
    AppendU32(key, static_cast<uint32_t>(endpoint.loc.locType));
    return true;
}

static HcclResult SelectLink(HcclComm comm, uint32_t myRank, uint32_t peer,
    CommLink &chosen, uint32_t &layer, uint32_t &die)
{
    if (myRank == peer) { return HCCL_E_PARA; }
    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    HcclResult ret = HcclRankGraphGetLayers(comm, &layerData, &layerCount);
    if (ret != HCCL_SUCCESS) { return ret; }
    if (layerCount == 0 || layerData == nullptr) { return HCCL_E_NOT_FOUND; }
    std::vector<uint32_t> layers(layerData, layerData + layerCount);
    std::sort(layers.begin(), layers.end());
    layers.erase(std::unique(layers.begin(), layers.end()), layers.end());

    const uint32_t lowRank = std::min(myRank, peer);
    const uint32_t highRank = std::max(myRank, peer);
    bool found = false;
    CommLink best{};
    uint32_t bestLayer = 0;
    std::vector<uint8_t> bestKey;
    for (const uint32_t candidateLayer : layers) {
        if (candidateLayer > 1) { continue; }
        CommLink *linkData = nullptr;
        uint32_t linkCount = 0;
        ret = HcclRankGraphGetLinks(comm, candidateLayer, lowRank, highRank, &linkData, &linkCount);
        if (ret != HCCL_SUCCESS) { return ret; }
        if (linkCount == 0) { continue; }
        if (linkData == nullptr) { return HCCL_E_PTR; }
        const std::vector<CommLink> links(linkData, linkData + linkCount);
        for (const auto &link : links) {
            if (link.linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) { continue; }
            std::vector<uint8_t> key;
            AppendU32(key, candidateLayer);
            AppendU32(key, static_cast<uint32_t>(link.linkAttr.linkProtocol));
            if (!AppendEndpointKey(key, link.srcEndpointDesc)
                || !AppendEndpointKey(key, link.dstEndpointDesc)) {
                continue;
            }
            if (!found || key < bestKey) {
                found = true;
                best = link;
                bestLayer = candidateLayer;
                bestKey = std::move(key);
            }
        }
    }
    if (!found) { return HCCL_E_NOT_SUPPORT; }
    if (myRank != lowRank) { std::swap(best.srcEndpointDesc, best.dstEndpointDesc); }
    EndpointAttrDieId localDie = std::numeric_limits<uint32_t>::max();
    ret = HcclRankGraphGetEndpointInfo(comm, myRank, &best.srcEndpointDesc,
        ENDPOINT_ATTR_DIE_ID, sizeof(localDie), &localDie);
    if (ret != HCCL_SUCCESS) { return ret; }
    if (localDie == std::numeric_limits<uint32_t>::max()) { return HCCL_E_PARA; }
    chosen = best;
    layer = bestLayer;
    die = localDie;
    return HCCL_SUCCESS;
}

namespace {
constexpr CommEngine ENGINE = COMM_ENGINE_CCU;

bool ValidRange(const void *ptr, uint64_t bytes)
{
    return ptr != nullptr && bytes <= std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(ptr);
}

HcclResult CheckParam(const OpParam &param)
{
    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE
        || param.myRank >= param.rankSize || param.root >= param.rankSize
        || param.count > std::numeric_limits<uint64_t>::max() / sizeof(float)) { return HCCL_E_PARA; }
    const uint64_t bytes = param.count * sizeof(float);
    if (bytes > std::numeric_limits<uint64_t>::max() / param.rankSize) { return HCCL_E_PARA; }
    if (bytes == 0) { return HCCL_SUCCESS; }
    if (!ValidRange(param.outputPtr, bytes)) { return HCCL_E_PARA; }
    if (param.myRank == param.root) {
        const uint64_t total = bytes * param.rankSize;
        if (!ValidRange(param.inputPtr, total)) { return HCCL_E_PARA; }
        const uintptr_t input = reinterpret_cast<uintptr_t>(param.inputPtr);
        const uintptr_t output = reinterpret_cast<uintptr_t>(param.outputPtr);
        if (output < input + total && input < output + bytes
            && output != input + param.root * bytes) { return HCCL_E_PARA; }
    }
    return HCCL_SUCCESS;
}

HcclResult QueryRelayTopology(HcclComm comm, const OpParam &param, AlgResourceCtx &resources)
{
    uint32_t *data = nullptr;
    uint32_t count = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &data, &count));
    if (count == 0 || data == nullptr) { return HCCL_E_PARA; }
    const std::vector<uint32_t> layers(data, data + count);
    if (std::find(layers.begin(), layers.end(), 0u) == layers.end()
        || std::find(layers.begin(), layers.end(), 1u) == layers.end()) { return HCCL_SUCCESS; }

    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &data, &count));
    if (count == 0 || count > param.rankSize || data == nullptr) { return HCCL_E_PARA; }
    std::vector<uint32_t> sizes(data, data + count);
    uint64_t total = 0;
    for (const auto size : sizes) {
        if (size == 0 || size > param.rankSize) { return HCCL_E_PARA; }
        total += size;
    }
    if (total != param.rankSize) { return HCCL_E_PARA; }
    std::sort(sizes.begin(), sizes.end());
    uint32_t topology = 0;
    if (sizes == std::vector<uint32_t>{8, 8}) { topology = 2; }
    else if (sizes == std::vector<uint32_t>{3, 3, 3, 3}) { topology = 4; }
    else { return HCCL_SUCCESS; }

    uint32_t localMask = 0;
    for (uint32_t layer = 0; layer < 2; ++layer) {
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &data, &count));
        if (count == 0 || count > param.rankSize || data == nullptr) { return HCCL_E_PARA; }
        std::vector<uint32_t> ranks(data, data + count);
        std::sort(ranks.begin(), ranks.end());
        const auto uniqueEnd = std::unique(ranks.begin(), ranks.end());
        if (uniqueEnd != ranks.end()) { return HCCL_E_PARA; }
        uint32_t mask = 0;
        for (const auto rank : ranks) {
            if (rank >= param.rankSize) { return HCCL_E_PARA; }
            mask |= 1u << rank;
        }
        if ((mask & (1u << param.myRank)) == 0) { return HCCL_E_PARA; }
        if (layer == 0) {
            if (ranks.size() != param.rankSize / topology) { return HCCL_E_PARA; }
            localMask = mask;
        } else if (mask != (1u << param.rankSize) - 1u) { return HCCL_E_PARA; }
    }
    resources.localMask = localMask;
    resources.relayTopology = topology;
    return HCCL_SUCCESS;
}

HcclResult CreateResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resources)
{
    struct Edge { CommLink link; uint32_t peer; uint32_t layer; };
    std::map<uint32_t, std::vector<Edge>> byDie;
    std::vector<HcclChannelDesc> descriptors(param.rankSize - 1);
    std::vector<ChannelHandle> handles(param.rankSize - 1);
    CHK_RET(QueryRelayTopology(comm, param, resources));
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) { continue; }
        CommLink link{};
        uint32_t layer = 0, die = 0;
        CHK_RET(SelectLink(comm, param.myRank, peer, link, layer, die));
        if (resources.relayTopology != 0
            && layer != ((resources.localMask & (1u << peer)) != 0 ? 0u : 1u)) {
            return HCCL_E_NOT_SUPPORT;
        }
        byDie[die].push_back({link, peer, layer});
        auto &desc = descriptors[peer < param.myRank ? peer : peer - 1];
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = peer;
        desc.notifyNum = 3;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint = link.srcEndpointDesc;
        desc.remoteEndpoint = link.dstEndpointDesc;
    }
    if (byDie.empty()) { byDie[0] = {}; }
    if (byDie.size() > 2) { return HCCL_E_NOT_SUPPORT; }
    if (resources.relayTopology != 0) {
        if (byDie.size() != 2) { return HCCL_E_NOT_SUPPORT; }
        uint32_t seenLayers = 0;
        for (const auto &entry : byDie) {
            const uint32_t layer = entry.second.front().layer;
            if (layer > 1 || (seenLayers & (1u << layer)) != 0) { return HCCL_E_NOT_SUPPORT; }
            seenLayers |= 1u << layer;
            for (const auto &edge : entry.second) {
                if (edge.layer != layer) { return HCCL_E_NOT_SUPPORT; }
            }
        }
    }
    CcuInsHandle instance = 0;
    uint32_t count = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &count));
    if (count != 1 || instance == 0 || HcommCcuKernelRegisterStart == nullptr
        || HcommCcuKernelRegister == nullptr || HcommCcuKernelRegisterEnd == nullptr
        || HcommCcuKernelLaunch == nullptr) { return HCCL_E_NOT_SUPPORT; }
    resources.myRank = param.myRank;
    resources.rankSize = param.rankSize;
    resources.groupCount = byDie.size();
    if (resources.groupCount == 2) {
        CHK_RET(HcclThreadAcquire(comm, ENGINE, 1, 1, &resources.slaveThread));
        if (resources.slaveThread == 0 || resources.slaveThread == param.cpuThread) { return HCCL_E_PARA; }
    }
    if (!descriptors.empty()) {
        CHK_RET(HcclChannelAcquire(comm, ENGINE, descriptors.data(), descriptors.size(), handles.data()));
    }
    uint32_t g = 0;
    for (const auto &entry : byDie) {
        auto &group = resources.groups[g];
        auto &reg = group.registration;
        group.dieId = entry.first;
        reg.myRank = param.myRank;
        reg.rankSize = param.rankSize;
        reg.copySelf = g == 0 ? 1u : 0u;
        reg.localMask = resources.localMask;
        reg.relayTopology = resources.relayTopology;
        reg.channelCount = entry.second.size();
        if (resources.relayTopology != 0) { group.layer = entry.second.front().layer; }
        uint32_t i = 0;
        for (const auto &edge : entry.second) {
            reg.channels[i] = handles[edge.peer < param.myRank ? edge.peer : edge.peer - 1];
            reg.peers[i] = edge.peer;
            HCCL_INFO("[Scatter] rank[%u] peer[%u] layer[%u] localDie[%u] channel[%llu]",
                param.myRank, edge.peer, edge.layer, entry.first, static_cast<unsigned long long>(reg.channels[i]));
            ++i;
        }
        ++g;
    }
    const CcuResult startResult = HcommCcuKernelRegisterStart(instance);
    if (startResult != CCU_SUCCESS) {
        HCCL_ERROR("[Scatter] register start rank[%u] ccuRet[%d]", param.myRank, startResult);
        return FromCcu(startResult);
    }
    for (uint32_t index = 0; index < resources.groupCount; ++index) {
        auto &group = resources.groups[index];
        const void *args[] = {&group.registration};
        const bool dualMode = resources.relayTopology == 2;
        const CcuResult result = HcommCcuKernelRegister(instance, group.dieId,
            dualMode ? "CcuDualMode" : "CcuScatter",
            dualMode ? reinterpret_cast<void *>(ops_hccl::CcuDualModeKernel)
                     : reinterpret_cast<void *>(ops_hccl::CcuKernel), args, 1, &group.kernel);
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("[Scatter] register rank[%u] group[%u] die[%u] ccuRet[%d]",
                param.myRank, index, group.dieId, result);
            return FromCcu(result);
        }
        if (resources.relayTopology == 0 && group.registration.channelCount != 0) {
            const CcuResult smallPullResult = HcommCcuKernelRegister(instance, group.dieId,
                "CcuSmallReadyPush", reinterpret_cast<void *>(ops_hccl::CcuSmallReadyPushKernel),
                args, 1, &group.smallPullKernel);
            if (smallPullResult != CCU_SUCCESS) {
                HCCL_ERROR("[Scatter] register smallReadyPush rank[%u] group[%u] die[%u] ccuRet[%d]",
                    param.myRank, index, group.dieId, smallPullResult);
                return FromCcu(smallPullResult);
            }
        }
        if (resources.relayTopology != 0) {
            const CcuResult relayResult = HcommCcuKernelRegister(instance, group.dieId,
                "CcuMultiNeighborRelay", reinterpret_cast<void *>(ops_hccl::CcuRelayKernel),
                args, 1, &group.relayKernel);
            if (relayResult != CCU_SUCCESS) {
                HCCL_ERROR("[Scatter] register relay rank[%u] group[%u] die[%u] ccuRet[%d]",
                    param.myRank, index, group.dieId, relayResult);
                return FromCcu(relayResult);
            }
        }
    }
    const CcuResult endResult = HcommCcuKernelRegisterEnd(instance);
    if (endResult != CCU_SUCCESS) {
        HCCL_ERROR("[Scatter] register end rank[%u] ccuRet[%d]", param.myRank, endResult);
        return FromCcu(endResult);
    }
    resources.ready = SCATTER_CACHE_READY;
    return HCCL_SUCCESS;
}

HcclResult GetResources(HcclComm comm, OpParam &param)
{
    HcclResult result = HcclEngineCtxGet(comm, param.tag, ENGINE, &param.resCtx, &param.ctxSize);
    if (result == HCCL_E_NOT_FOUND) {
        AlgResourceCtx resources{};
        param.ctxSize = sizeof(resources);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ENGINE, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ENGINE, param.tag, &resources, sizeof(resources), 0));
        CHK_RET(CreateResources(comm, param, resources));
        CHK_RET(HcclEngineCtxCopy(comm, ENGINE, param.tag, &resources, sizeof(resources), 0));
    } else { CHK_RET(result); }
    AlgResourceCtx checked{};
    CHK_RET(AlgResourceCtx::Decode(param.resCtx, param.ctxSize, checked));
    if (checked.myRank != param.myRank || checked.rankSize != param.rankSize) { return HCCL_E_PARA; }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    OpParam param{};
    std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_scatter_ccu");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_RET(CheckParam(param));
    CHK_RET(HcclThreadAcquireWithStream(comm, ENGINE, stream, 1, &param.cpuThread));
    HcclDfxOpInfo dfx{};
    dfx.opType = param.opType;
    dfx.dataType = dataType;
    dfx.outputType = dataType;
    dfx.dataCount = recvCount;
    dfx.root = root;
    dfx.engine = ENGINE;
    dfx.cpuTsThread = param.cpuThread;
    dfx.inputMemAddr = param.myRank == root ? reinterpret_cast<uintptr_t>(sendBuf) : 0;
    dfx.inputMemSize = param.myRank == root ? recvCount * sizeof(float) * param.rankSize : 0;
    dfx.outputMemAddr = reinterpret_cast<uintptr_t>(recvBuf);
    dfx.outputMemSize = recvCount * sizeof(float);
    std::snprintf(dfx.algTag, sizeof(dfx.algTag), "%s", param.tag);
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, &dfx));
    if (recvCount == 0) { return HCCL_SUCCESS; }
    CHK_RET(GetResources(comm, param));
    return ops_hccl::ExecOp(param);
}
