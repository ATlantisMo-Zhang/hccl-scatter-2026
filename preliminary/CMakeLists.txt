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
 project(hccl_scatter_aicpu)

message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")

set(HCC_TOOLCHAIN_DIR "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc")
set(HCC_CXX_COMPILER  "${HCC_TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++")
set(HCC_C_COMPILER    "${HCC_TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc")
set(HCC_C_AR          "${HCC_TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-ar")

# 编译 Host 侧链接库
add_subdirectory(op_host)

# 编译 Device 侧链接库
include(ExternalProject)
ExternalProject_Add(hccl_device
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/op_kernel_aicpu
    BINARY_DIR ${CMAKE_BINARY_DIR}/device_build
    CMAKE_ARGS
        -DTOOLCHAIN_DIR=${HCC_TOOLCHAIN_DIR}
        -DCMAKE_C_COMPILER=${HCC_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${HCC_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_C_AR=${HCC_C_AR}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
    INSTALL_COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR}/device_build --prefix ${CMAKE_INSTALL_PREFIX}
    BUILD_ALWAYS TRUE
)

# 安装
install(FILES include/hccl.h
    DESTINATION include
)
