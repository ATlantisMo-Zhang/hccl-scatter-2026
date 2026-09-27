# 本地验证

这些测试检查数值、内存边界、资源复用和 Checker 依赖图，**不测比赛性能**。需要匹配的 CANN 9.1.0、MPI 和 HCCL-VM；普通 CPU 无法独立执行。

## 构建数值驱动

```bash
source /path/to/Ascend/cann/set_env.sh
bash tests/build_numerical.sh /absolute/path/to/libhccl.so
```

驱动通过 `SCATTER_TEST_LIBRARY` 接收待测库绝对路径，并核对 `HcclScatter` 实际解析地址；`LD_LIBRARY_PATH` 必须同时优先指向该库所在目录。驱动使用当前 `final/include/custom.h` 解码资源缓存，**不能原样用于布局不同的 A21**。

## 在已经隔离的 VM 会话内运行

本仓库不自动替换机器上的共享 Runner。实际验证使用独立 VM 副本和 user/mount/IPC/PID namespace，将 VM 副本 bind mount 到安装路径，并隔离 `/dev/shm`、`/dev/mqueue` 和 `/tmp`。在副本中添加 `topo_meta/` 的配置，按对应配置创建 communicator；以下示例只适用于该会话内部：

```bash
hccl-vm mock-comm final_f3_2x8
hccl-vm plugin install @runner
export SCATTER_TEST_LIBRARY=/absolute/path/to/libhccl.so
export LD_LIBRARY_PATH="$(dirname "$SCATTER_TEST_LIBRARY"):$LD_LIBRARY_PATH"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=4
export RANK_TABLE_FILE=/path/to/isolated-vm/data/ranktable.json
mpirun --allow-run-as-root --oversubscribe -np 16 /absolute/path/to/tests/build/scatter_numerical 2x8 dual_boundary
hccl-vm plugin run @checker
```

Checker 异步启动后必须等待其完成，并检查对应全部调用的结果；插件启动命令返回 0 不代表检查通过。实际运行器另外记录 driver/Checker 退出码和加载二进制哈希，并在首错或 240 秒上限时终止本次隔离环境。仓库提供驱动、配置和检查器脚本，不包含 CANN、VM 二进制或一键安装器。

| 拓扑 | MPI ranks | mode | 用例数 |
|---|---:|---|---:|
| 2×8 | 16 | `dual_boundary` | 9 |
| 4×3 | 12 | `m4x3` | 7 |
| 4×1 | 4 | `nonrelay_mix` | 8 |
| 8+4 | 12 | `nonrelay_mix` | 8 |
| 4×3 精确 self alias | 12 | `relay_alias` | 1，单独报告 |

mode 的历史名字仅用于调用驱动，`m4x3` 不表示测试 M 版代码。每轮变更输入数据和分配偏移，全量逐位检查有限 FP32 输出、root 输入不变及前后 guard；stream 同步后立即释放本轮用户缓冲。初始化时 MPI 交换拓扑和 communicator 信息，每次 Scatter 后不加 MPI barrier。

`HCCL_BUFFSIZE=4` 在本次环境实际返回 8 MiB HCCL buffer，不能直接用环境变量数字推算容量。驱动删除了旧版按 A21 比例预测启动数的诊断，实际分片以 Checker trace 为准。

## Runner 环境说明

`runner-zero-init.patch` 是本次私有 Runner 相对原环境的三行修改：在 `ccu_resource_v1.cc` 构造新资源时清零 XN/GSA/CKE。它不属于提交算子，也不代表修复所有 VM 问题。应针对自己的 VM 源码核对后，仅构建到独立验证副本；不要覆盖共享安装。已用二进制身份和已知 self-alias Checker 限制见[验证记录](../docs/validation-postrace.md)。

## 检查已导出的依赖图

```bash
python tests/inspect_vm_graph.py /path/to/isolated-vm/data --output graph-report.json
```

脚本要求 scratch stage 在对应 forward 之前，helper 自身输出与 forward 之间没有互相可达的依赖，并检查输出区间没有缺口或重叠。它解释的是本次 Checker 导出格式，不能代替数值检查、硬件时序或所有协议证明。

当前 DOT 文件名精确到秒，同一秒的调用可能覆盖前一个图；报告只覆盖实际保留下来的图，不能把图数量当成功调用数量。全部调用数值与 Checker 结果必须另外核对。
