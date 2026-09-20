/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

namespace ops_hccl {

// CCU Kernel 函数
CcuResult CcuKernel(CcuKernelArg arg);
CcuResult CcuDualModeKernel(CcuKernelArg arg);
CcuResult CcuSmallPullKernel(CcuKernelArg arg);
CcuResult CcuSmallReadyPushKernel(CcuKernelArg arg);
CcuResult CcuRelayKernel(CcuKernelArg arg);

// TODO: 可编写多个 CCU Kernel 函数，以最大化性能
// CcuResult CcuKernel2(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
