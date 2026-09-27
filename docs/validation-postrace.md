# 赛后版验证记录（2026-09-27）

版本：`postrace-20260927`，源码在 [final/](../final/)。同一最终二进制完成四种拓扑 **32 次调用、356 个 rank/调用组合**的数值与 Checker 检查。新版没有官方性能测量。

## 身份与环境

- 算子库 SHA256：`f7996a52dae7eacdc0937837c77e6bfd045120696d34e3d4e6c4da9aeb3832cc`。
- 数值驱动源码 SHA256：`d3e4a7698d42aa9aeffeac255a7ba21f1ccae8f530694ab956c74f72a3634b0f`，对应仓库 `tests/scatter_numerical.cc`；`final/` 文件逐个与实际构建快照匹配。
- CANN 9.1.0，WSL Ubuntu，g++/mpicxx；Release 编译成功。构建路径影响产物身份，重新构建不要求二进制哈希相同。
- 每轮独立 VM 副本，隔离 user/mount/IPC/PID namespace；没有替换共享 Runner。预检后还检查运行中 Runner 进程的实际二进制哈希。
- 私有 Runner：`2d63a1765d3510a63bcbdd0e77b10a00a748d17c2e620981284c6d9b67423416`。相对原环境只修正新建 XN/GSA/CKE 的零初始化，三行补丁在 `tests/runner-zero-init.patch`。
- Checker：`3c6ff13fee42b18d67a64bbb00a6e0c23a23eadaf0c8a319b3e31ea41457d1d5`，未为本次测试修改；环境初始化修正不等于所有 VM 问题均已解决。
- `HCCL_BUFFSIZE=4`，本地 API 实际返回 8 MiB scratch。生产算法按返回容量分片；没有缩小源码中的 256 MiB 常量来冒充生产路径。

机器可读记录：[validation-20260927.json](evidence/validation-20260927.json)，包含各调用 root/长度、源码及日志哈希、driver 与 Checker 退出码。过程墙钟时间只反映本地工具运行，不是性能点耗时。

## 最终二进制的数值和 Checker 矩阵

每条序列共用 communicator，逐次改变内容与分配偏移；逐位检查完整有限 FP32 数据、root 原输入、前后 guard 及缓存稳定性。每次本 rank stream 同步后立即释放用户缓冲，不增加每轮 MPI barrier。Checker 要求该序列的全部 op 编号均报告成功。

| 拓扑 | 调用序列（root, 每 rank 字节数） | 结果 |
|---|---|---|
| 2×8 | (0,4), (15,8), (15,32768), (7,16MiB−4), (0,16MiB), (15,16MiB+4), (7,0), (0,32768), (8,32MiB+4) | 9 次数值 + Checker 全通过 |
| 4×3 | (0,16MiB), (11,16MiB+4), (3,68), (3,32MiB), (11,0), (11,16MiB−4), (6,16MiB+8) | 7 次数值 + Checker 全通过；覆盖四个 root 组 |
| 4×1 | (0,68), (3,32768), (3,32768), (0,16MiB−4), (0,16MiB), (3,4), (3,0), (0,524288) | 8 次数值 + Checker 全通过 |
| 8+4 | (0,68), (11,32768), (11,32768), (0,16MiB−4), (0,16MiB), (11,4), (11,0), (0,524288) | 8 次数值 + Checker 全通过 |

2×8 的 32MiB+4 和 4×3 大消息会因 scratch 容量产生多片，覆盖偏移、均衡尾片和跨片通知复用。这里没有实跑每 rank 数百 MiB / GiB 级官方大点，也没有覆盖所有 rank 作为 root 的排列；大容量最终性能和更广输入空间不能由该矩阵代替。

精确 self alias 单独验证：4×3，root=11，每 rank 16MiB，root 输出恰为输入中自身切片。12 个 rank 的数值、输入及 guard 检查通过（driver exit 0），但 Checker 报 **406**：root 输出累计验证 0 字节，期望 16MiB（checker wait exit 1）。此时 root 无需自拷贝，当前 Checker 没有把原位已有数据算作输出覆盖。该结果与既有环境现象一致，但本次不修改 Checker，也不把它标为全通过；完整 alias 支持尚未获得 Checker 背书。

## 依赖图：验证改动确实改变数据路径

`tests/inspect_vm_graph.py` 检查保留的 Checker DOT：

1. 每个 `INPUT → CCL scratch` stage 必须先于对应 `CCL → OUTPUT` forward。
2. 对应 helper 的完整自身输出与 forward 之间均不存在可达依赖。
3. 每个已写输出按区间覆盖一次，没有借用前缀造成的重复写入和恢复。

报告见 [2×8](evidence/graph-2x8.json)、[4×3](evidence/graph-4x3.json)、[4×1](evidence/graph-4x1.json)、[8+4](evidence/graph-8plus4.json)。其中中转检查只适用于有 scratch 的图，普通图主要检查输出覆盖。`tests/fixtures/` 各保留一个 2×8 / 4×3 多片图，读者可直接运行脚本重验。

DOT 文件名精确到秒，部分短调用会覆盖同秒前一文件，故报告不声称覆盖所有调用的每张图；完整 32 次成功来自独立数值日志和 Checker op 完成记录。

单片 2×8 新图为 29 次网络传输，4×3 为 15 次；不同分片数不能直接横比总任务数。无依赖说明可调度空间，不能直接证明硬件实现并行，更不能换算成 μs。

## 过程中发现并修正的问题

- 第一版 scratch 虽数值通过，图上仍有 helper 自身输出到 forward 的依赖；把 STAGED 发布放到自身输出 Write 之前后重新检查。
- 一版 Host 给专用 ReadyPush 多传 small 参数，4×1 返回 `argsNum=11, loaded=10`。最终版只给需要该参数的通用 / DualMode 内核传参，四拓扑使用最终库重新跑过。
- 旧 GitHub A21 清理版漏掉三处 source stride 递推；本次修复并保存原始 A21。此问题产生于归档，不能解释参赛分数。

## 性能结论的边界

已经验证：最终库可构建，所列数值及 Checker 用例通过，scratch 复用路径得到执行，保留图中的 restore 写入与相关串行依赖被移除。

尚未验证：官方逐点耗时、优化比例、是否达到跑分前三、每 rank 超过 32MiB+4 的执行、持续 kernel 或角色专门化收益。不能用 VM 墙钟时间、前三的成绩或简化流量下界填补这些数据。
