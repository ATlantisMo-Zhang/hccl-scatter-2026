/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <set>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint64_t CACHE_READY = 0x5343415454455230ULL;
constexpr CommEngine DEVICE_ENGINE = COMM_ENGINE_AICPU_TS;
constexpr CommEngine HOST_ENGINE = COMM_ENGINE_CPU_TS;

bool ValidRange(const void *address, uint64_t bytes)
{
    return address != nullptr && bytes <= std::numeric_limits<uintptr_t>::max()
        - reinterpret_cast<uintptr_t>(address);
}

HcclResult CheckParam(const OpParam &param)
{
    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.rankSize == 0
        || param.root >= param.rankSize || param.myRank >= param.rankSize
        || param.count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return HCCL_E_PARA;
    }
    const uint64_t bytes = param.count * sizeof(float);
    if (bytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        return HCCL_E_PARA;
    }
    if (bytes == 0) {
        return HCCL_SUCCESS;
    }
    if (!ValidRange(param.outputPtr, bytes)) {
        return HCCL_E_PARA;
    }
    if (param.myRank == param.root) {
        const uint64_t total = bytes * param.rankSize;
        if (!ValidRange(param.inputPtr, total)) {
            return HCCL_E_PARA;
        }
        const uintptr_t input = reinterpret_cast<uintptr_t>(param.inputPtr);
        const uintptr_t output = reinterpret_cast<uintptr_t>(param.outputPtr);
        if (output < input + total && input < output + bytes
            && output != input + static_cast<uint64_t>(param.root) * bytes) {
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

bool ReadProtocol(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBC_TP
        || protocol == COMM_PROTOCOL_UBOE;
}

bool AppendEndpointKey(const EndpointDesc &endpoint, std::vector<uint32_t> &key)
{
    if (!ReadProtocol(endpoint.protocol) || endpoint.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        return false;
    }
    key.push_back(static_cast<uint32_t>(endpoint.protocol));
    key.push_back(static_cast<uint32_t>(endpoint.commAddr.type));
    const CommAddr &address = endpoint.commAddr;
    switch (address.type) {
        case COMM_ADDR_TYPE_IP_V4:
            key.push_back(address.addr.s_addr);
            break;
        case COMM_ADDR_TYPE_IP_V6:
            for (const auto byte : address.addr6.s6_addr) { key.push_back(byte); }
            break;
        case COMM_ADDR_TYPE_ID:
            key.push_back(address.id);
            break;
        case COMM_ADDR_TYPE_EID:
            for (const auto byte : address.eid) { key.push_back(byte); }
            break;
        default:
            return false;
    }
    key.push_back(static_cast<uint32_t>(endpoint.loc.locType));
    key.push_back(endpoint.loc.device.devPhyId);
    key.push_back(endpoint.loc.device.superDevId);
    key.push_back(endpoint.loc.device.serverIdx);
    key.push_back(endpoint.loc.device.superPodIdx);
    return true;
}

HcclResult SelectLink(HcclComm comm, const std::vector<uint32_t> &layers,
    uint32_t rank, uint32_t peer, HcclChannelDesc &desc, uint32_t *selectedLayer = nullptr)
{
    const uint32_t low = std::min(rank, peer);
    const uint32_t high = std::max(rank, peer);
    for (const uint32_t layer : layers) {
        CommLink *links = nullptr;
        uint32_t count = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layer, low, high, &links, &count));
        if (count == 0) { continue; }
        CHK_PTR_NULL(links);
        const std::vector<CommLink> copied(links, links + count);
        std::vector<uint32_t> bestKey;
        const CommLink *best = nullptr;
        for (const auto &link : copied) {
            if (!ReadProtocol(link.linkAttr.linkProtocol)) { continue; }
            std::vector<uint32_t> key{static_cast<uint32_t>(link.linkAttr.linkProtocol)};
            if (!AppendEndpointKey(link.srcEndpointDesc, key)
                || !AppendEndpointKey(link.dstEndpointDesc, key)) { continue; }
            if (best == nullptr || key < bestKey) {
                best = &link;
                bestKey = key;
            }
        }
        if (best != nullptr) {
            if (selectedLayer != nullptr) { *selectedLayer = layer; }
            desc.remoteRank = peer;
            desc.channelProtocol = best->linkAttr.linkProtocol;
            desc.localEndpoint = rank == low ? best->srcEndpointDesc : best->dstEndpointDesc;
            desc.remoteEndpoint = rank == low ? best->dstEndpointDesc : best->srcEndpointDesc;
            desc.notifyNum = 2;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("No supported Read link for rank %u peer %u", rank, peer);
    return HCCL_E_NOT_SUPPORT;
}

bool ValidRelayTopology(const RelayTopology &topology, uint32_t rankSize)
{
    const uint32_t a = topology.groupMasks[0];
    const uint32_t b = topology.groupMasks[1];
    if (a == 0 && b == 0) { return true; }
    if (rankSize != 16 || a >= b || (a & b) != 0 || (a | b) != 0xFFFFu) { return false; }
    uint32_t members = 0;
    for (uint32_t rank = 0; rank < 16; ++rank) { members += (a >> rank) & 1u; }
    return members == 8;
}

HcclResult QueryRelayTopology(HcclComm comm, const OpParam &param, RelayTopology &topology)
{
    topology = RelayTopology{};
    if (param.rankSize != 16) { return HCCL_SUCCESS; }
    uint32_t *data = nullptr;
    uint32_t count = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &data, &count));
    if (count != 2) { return HCCL_SUCCESS; }
    CHK_PTR_NULL(data);
    const std::vector<uint32_t> sizes0(data, data + count);
    if (sizes0[0] != 8 || sizes0[1] != 8) { return HCCL_SUCCESS; }
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 1, &data, &count));
    if (count != 1) { return HCCL_SUCCESS; }
    CHK_PTR_NULL(data);
    const uint32_t size1 = data[0];
    if (size1 != param.rankSize) { return HCCL_SUCCESS; }
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &data, &count));
    if (count != 8) { return HCCL_E_PARA; }
    CHK_PTR_NULL(data);
    const std::vector<uint32_t> localRanks(data, data + count);
    uint32_t localMask = 0;
    for (const uint32_t rank : localRanks) {
        if (rank >= param.rankSize || (localMask & (1u << rank)) != 0) { return HCCL_E_PARA; }
        localMask |= 1u << rank;
    }
    if ((localMask & (1u << param.myRank)) == 0) { return HCCL_E_PARA; }
    const uint32_t otherMask = 0xFFFFu ^ localMask;
    topology.groupMasks[0] = std::min(localMask, otherMask);
    topology.groupMasks[1] = std::max(localMask, otherMask);
    return ValidRelayTopology(topology, param.rankSize) ? HCCL_SUCCESS : HCCL_E_PARA;
}

HcclResult CheckResources(const OpParam &param, const AlgResourceCtx &resources)
{
    const uint32_t expectedThreads = param.rankSize == 16 ? 16 : 1;
    if (resources.threads.size() != expectedThreads || resources.aicpuThread == 0
        || resources.aicpuThread != resources.threads[0]
        || resources.channels.size() != param.rankSize - 1
        || resources.localBuffer.size < sizeof(float)
        || !ValidRange(resources.localBuffer.addr, resources.localBuffer.size)) {
        return HCCL_E_PARA;
    }
    std::set<ThreadHandle> threadHandles;
    for (const auto thread : resources.threads) {
        if (thread == 0 || !threadHandles.insert(thread).second) { return HCCL_E_PARA; }
    }
    if (!ValidRelayTopology(resources.relayTopology, param.rankSize)) { return HCCL_E_PARA; }
    std::set<ChannelHandle> handles;
    size_t index = 0;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) { continue; }
        const auto &channel = resources.channels[index++];
        if (channel.remoteRank != peer || channel.notifyNum != 2 || channel.handle == 0
            || !handles.insert(channel.handle).second || channel.remoteCclMem.size < sizeof(float)
            || !ValidRange(channel.remoteCclMem.addr, channel.remoteCclMem.size)) {
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CreateResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resources)
{
    CHK_RET(HcclGetHcclBuffer(comm, &resources.localBuffer.addr, &resources.localBuffer.size));
    const uint32_t threadCount = param.rankSize == 16 ? 16 : 1;
    resources.threads.resize(threadCount);
    CHK_RET(HcclThreadAcquire(comm, DEVICE_ENGINE, threadCount, threadCount, resources.threads.data()));
    resources.aicpuThread = resources.threads[0];
    if (param.rankSize == 1) { return CheckResources(param, resources); }

    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    CHK_PTR_NULL(layerData);
    std::vector<uint32_t> layers(layerData, layerData + layerCount);
    layers.erase(std::remove_if(layers.begin(), layers.end(),
        [](uint32_t layer) { return layer > 1; }), layers.end());
    std::sort(layers.begin(), layers.end());
    layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
    CHK_RET(QueryRelayTopology(comm, param, resources.relayTopology));
    std::vector<HcclChannelDesc> descs(param.rankSize - 1);
    CHK_RET(HcclChannelDescInit(descs.data(), descs.size()));
    size_t index = 0;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank) {
            uint32_t selectedLayer = 0;
            CHK_RET(SelectLink(comm, layers, param.myRank, peer, descs[index++], &selectedLayer));
            if (resources.relayTopology.groupMasks[0] != 0) {
                const uint32_t localMask = (resources.relayTopology.groupMasks[0] & (1u << param.myRank)) != 0
                    ? resources.relayTopology.groupMasks[0] : resources.relayTopology.groupMasks[1];
                const uint32_t expectedLayer = (localMask & (1u << peer)) != 0 ? 0 : 1;
                if (selectedLayer != expectedLayer) { return HCCL_E_NOT_SUPPORT; }
            }
        }
    }
    std::vector<ChannelHandle> handles(descs.size());
    CHK_RET(HcclChannelAcquire(comm, DEVICE_ENGINE, descs.data(), descs.size(), handles.data()));
    resources.channels.resize(descs.size());
    for (size_t i = 0; i < descs.size(); ++i) {
        auto &channel = resources.channels[i];
        channel.remoteRank = descs[i].remoteRank;
        channel.notifyNum = descs[i].notifyNum;
        channel.handle = handles[i];
        CHK_RET(HcclChannelGetHcclBuffer(comm, channel.handle,
            &channel.remoteCclMem.addr, &channel.remoteCclMem.size));
    }
    return CheckResources(param, resources);
}

HcclResult CacheResources(HcclComm comm, OpParam &param, AlgResourceCtx &resources)
{
    auto bytes = resources.Serialize();
    std::vector<char> hostBytes(sizeof(CACHE_READY) + bytes.size(), 0);
    std::copy(bytes.begin(), bytes.end(), hostBytes.begin() + sizeof(CACHE_READY));
    void *hostCtx = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, HOST_ENGINE, hostBytes.size(), &hostCtx));
    HcclResult result = HcclEngineCtxCopy(comm, HOST_ENGINE, param.tag, hostBytes.data(), hostBytes.size(), 0);
    bool deviceCreated = false;
    if (result == HCCL_SUCCESS) {
        param.ctxSize = bytes.size();
        result = HcclEngineCtxCreate(comm, param.tag, DEVICE_ENGINE, param.ctxSize, &param.resCtx);
        deviceCreated = result == HCCL_SUCCESS;
    }
    if (result == HCCL_SUCCESS) {
        result = HcclEngineCtxCopy(comm, DEVICE_ENGINE, param.tag, bytes.data(), bytes.size(), 0);
    }
    if (result == HCCL_SUCCESS) {
        result = HcclEngineCtxCopy(comm, HOST_ENGINE, param.tag, &CACHE_READY, sizeof(CACHE_READY), 0);
    }
    if (result != HCCL_SUCCESS) {
        if (deviceCreated) {
            const HcclResult cleanup = HcclEngineCtxDestroy(comm, param.tag, DEVICE_ENGINE);
            if (cleanup != HCCL_SUCCESS) { HCCL_ERROR("Device context cleanup failed: %d", cleanup); }
        }
        const HcclResult cleanup = HcclEngineCtxDestroy(comm, param.tag, HOST_ENGINE);
        if (cleanup != HCCL_SUCCESS) { HCCL_ERROR("Host context cleanup failed: %d", cleanup); }
    }
    return result;
}

HcclResult GetResources(HcclComm comm, OpParam &param, AlgResourceCtx &resources)
{
    void *hostCtx = nullptr;
    uint64_t hostSize = 0;
    const HcclResult hostResult = HcclEngineCtxGet(comm, param.tag, HOST_ENGINE, &hostCtx, &hostSize);
    const HcclResult deviceResult = HcclEngineCtxGet(comm, param.tag, DEVICE_ENGINE, &param.resCtx, &param.ctxSize);
    if (hostResult == HCCL_E_NOT_FOUND && deviceResult == HCCL_E_NOT_FOUND) {
        CHK_RET(CreateResources(comm, param, resources));
        return CacheResources(comm, param, resources);
    }
    CHK_RET(hostResult);
    CHK_RET(deviceResult);
    const uint64_t expectedThreads = param.rankSize == 16 ? 16 : 1;
    const uint64_t expected = sizeof(ThreadHandle) + sizeof(CommBuffer) + 2 * sizeof(size_t)
        + expectedThreads * sizeof(ThreadHandle) + static_cast<uint64_t>(param.rankSize - 1) * sizeof(ChannelInfo) + sizeof(RelayTopology);
    if (hostCtx == nullptr || param.resCtx == nullptr || param.ctxSize != expected
        || hostSize != expected + sizeof(CACHE_READY)) { return HCCL_E_PARA; }
    uint64_t ready = 0;
    std::memcpy(&ready, hostCtx, sizeof(ready));
    if (ready != CACHE_READY) { return HCCL_E_PARA; }
    const char *start = static_cast<const char *>(hostCtx) + sizeof(ready);
    size_t threadCount = 0;
    size_t channelCount = 0;
    std::memcpy(&threadCount, start + sizeof(ThreadHandle) + sizeof(CommBuffer), sizeof(size_t));
    if (threadCount != expectedThreads) { return HCCL_E_PARA; }
    const uint64_t channelCountOffset = sizeof(ThreadHandle) + sizeof(CommBuffer) + sizeof(size_t)
        + expectedThreads * sizeof(ThreadHandle);
    std::memcpy(&channelCount, start + channelCountOffset, sizeof(size_t));
    if (channelCount != param.rankSize - 1) { return HCCL_E_PARA; }
    std::vector<char> bytes(start, start + expected);
    resources.DeSerialize(bytes);
    return CheckResources(param, resources);
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    OpParam param{};
    std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_scatter");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_RET(CheckParam(param));

    CHK_RET(HcclThreadAcquireWithStream(comm, HOST_ENGINE, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, DEVICE_ENGINE, &param.cpuThreadOnAicpu));
    HcclDfxOpInfo dfxInfo{};
    dfxInfo.opType = param.opType;
    dfxInfo.dataType = dataType;
    dfxInfo.outputType = dataType;
    dfxInfo.dataCount = recvCount;
    dfxInfo.root = root;
    std::snprintf(dfxInfo.algTag, sizeof(dfxInfo.algTag), "%s", param.tag);
    dfxInfo.engine = DEVICE_ENGINE;
    dfxInfo.cpuTsThread = param.cpuThread;
    dfxInfo.cpuWaitAicpuNotifyIdx = 0;
    dfxInfo.inputMemAddr = reinterpret_cast<uintptr_t>(sendBuf);
    dfxInfo.inputMemSize = param.myRank == root ? recvCount * sizeof(float) * param.rankSize : 0;
    dfxInfo.outputMemAddr = reinterpret_cast<uintptr_t>(recvBuf);
    dfxInfo.outputMemSize = recvCount * sizeof(float);
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, &dfxInfo));

    AlgResourceCtx resources{};
    CHK_RET(GetResources(comm, param, resources));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resources.aicpuThread, HOST_ENGINE, &param.aicpuThreadOnCpu));
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
