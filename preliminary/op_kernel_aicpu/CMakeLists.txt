 # -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(hccl_device)

# ==================================================
# Device 侧动态链接库
# ==================================================
add_library(hccl_device SHARED)

target_sources(hccl_device PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/exec_op.cc
)

target_include_directories(hccl_device PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    # CANN Toolkit头文件
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
)

target_compile_options(hccl_device PRIVATE
    -std=c++17
    -Werror
    -Wall
    -fno-common
    -fno-strict-aliasing
    -pipe
    -fstack-protector-all
    -U_FORTIFY_SOURCE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O2>
)

target_link_directories(hccl_device PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/devlib/device
)

target_link_libraries(hccl_device PRIVATE
    ccl_kernel
)

install(TARGETS hccl_device LIBRARY
    DESTINATION lib64
)
