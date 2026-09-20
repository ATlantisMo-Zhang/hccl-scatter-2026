# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BUILD_TYPE="Release"
ENABLE_FORMAT="OFF"

CPU_NUM="$(nproc)"
ASCEND_CANN_PACKAGE_PATH="/usr/local/Ascend/cann"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --debug     编译 Debug 版本
    --format    格式化代码
    -h, --help  显示帮助信息
EOF
    exit 0
}

parse_args() {
    local opts
    opts=$(getopt -o h -l debug,format,help -- "$@") || usage
    eval set -- "${opts}"

    for arg; do
        case "${arg}" in
            --debug) BUILD_TYPE="Debug" ;;
            --format) ENABLE_FORMAT="ON" ;;
            -h|--help) usage ;;
        esac
    done
}

parse_cann_path() {
    if [[ -z "${ASCEND_HOME_PATH}" ]]; then
        printf "ERROR: ASCEND_HOME_PATH is not set.\n" >&2
        printf "Please ensure CANN-Toolkit is properly installed and source environment variables by running:\n" >&2
        printf "  source /path/to/Ascend/cann/set_env.sh\n" >&2
        exit 1
    fi

    # 使用 local 避免污染全局（如需要全局可去掉 local）
    ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
    return 0
}

build() {
    # 创建构建目录
    cd "${PROJECT_DIR}"
    mkdir -p "${BUILD_DIR}"

    # 配置
    cmake -S . -B "${BUILD_DIR}" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}" \
        -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}"

    # 编译
    cmake --build "${BUILD_DIR}" -j${CPU_NUM}

    # 安装
    cmake --install "${BUILD_DIR}"
}

format() {
    cd "${PROJECT_DIR}"
    find ./include -type f -regex '.*\.\(h\|hpp\|cpp\|cc\)$' -exec clang-format -i {} \;
    find ./op_host -type f -regex '.*\.\(h\|hpp\|cpp\|cc\)$' -exec clang-format -i {} \;
    find ./op_kernel_ccu -type f -regex '.*\.\(h\|hpp\|cpp\|cc\)$' -exec clang-format -i {} \;
}

main() {
    # 解析参数
    parse_args "$@"
    # 解析 CANN-Toolkit 路径
    parse_cann_path

    if [[ "${ENABLE_FORMAT}" == "ON" ]]; then
        # 格式化代码
        format
    else
        # 编译
        build
    fi
}

main "$@"
