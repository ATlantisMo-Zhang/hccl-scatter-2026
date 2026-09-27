#!/usr/bin/env bash
# Build only. Invoke the driver inside an already isolated HCCL-VM session.
set -eo pipefail
: "${ASCEND_HOME_PATH:?Source the matching CANN set_env.sh first}"
repo=$(cd "$(dirname "$0")/.." && pwd)
candidate=${1:-"$repo/final/build/lib64/libhccl.so"}
candidate=$(realpath "$candidate")
test -f "$candidate"
mkdir -p "$repo/tests/build"
mpicxx -std=c++17 -O2 -Wall -Wextra -DOMPI_SKIP_MPICXX -D_GLIBCXX_USE_CXX11_ABI=0 \
    "$repo/tests/scatter_numerical.cc" -I "$repo/final/include" \
    -I "$ASCEND_HOME_PATH/include" -I "$ASCEND_HOME_PATH/include/hcomm" \
    -I "$ASCEND_HOME_PATH/include/hcomm/ccu" -I "$ASCEND_HOME_PATH/pkg_inc" \
    -I "$ASCEND_HOME_PATH/pkg_inc/hcomm" -I "$ASCEND_HOME_PATH/pkg_inc/hcomm/ccu" \
    -I "$ASCEND_HOME_PATH/include/hccl" -I "$ASCEND_HOME_PATH/pkg_inc/hccl" \
    -L "$(dirname "$candidate")" -L "$ASCEND_HOME_PATH/lib64" \
    -lhccl -lhcomm -lascendcl -ldl -o "$repo/tests/build/scatter_numerical"
sha256sum "$candidate" "$repo/tests/scatter_numerical.cc" "$repo/tests/build/scatter_numerical"
