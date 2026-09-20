/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_COMMON_H
#define OPS_HCCL_COMMON_H

#include <unordered_map>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>
#include <hccl/hcomm_primitives.h>
#include <acl/acl_rt.h>

constexpr uint32_t NOTIFY_IDX_ACK = 0;
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr uint32_t CUSTOM_TIMEOUT = 1800;

constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;
constexpr uint32_t INVALID_VALUE_RANKID = 0xFFFFFFFF;

struct OpParam {
    char tag[TAG_LENGTH];
    void *inputPtr = nullptr;
    void *outputPtr = nullptr;
    uint64_t count = 0;
    uint32_t root = 0;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    ThreadHandle cpuThread;
    ThreadHandle cpuThreadOnAicpu;
    ThreadHandle aicpuThreadOnCpu; ///< AICPU_TS通信引擎上的thread资源
    void *resCtx = nullptr;        ///< 通信引擎上下文中的资源信息，存放 custom.h 中 AlgResourceCtx 序列化后的内容
    uint64_t ctxSize = 0;
};

const std::unordered_map<HcclDataType, uint32_t> SIZE_TABLE = {{HCCL_DATA_TYPE_FP32, sizeof(float)}};

#endif // OPS_HCCL_COMMON_H
