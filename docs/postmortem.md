# 赛后复盘：以 A21 对照跑分前三

更新于 2026-09-27。参赛基线是用户确认的 **A21**；目标是改进代码，不讨论答辩或奖项认定。新版位于 [final/](../final/)，原始 A21 位于 [baselines/a21/](../baselines/a21/)。

## 1. 差距先看数据

本次统一采用 2026-09-27 的[排名快照](evidence/ranking-20260927.json)，不用跨轮分数或不同队伍各自最快点拼成一个“对手”。

| 性能点 | 我们 | yiran | 相对 yiran 多耗时 |
|---|---:|---:|---:|
| 19 | 11 | 9 | 22.2% |
| 20 / 21 | 965 / 757 | 942 / 738 | 2.4% / 2.6% |
| 22 | 12 | 9 | 33.3% |
| 23 / 24 | 2064 / 1615 | 2057 / 1608 | 0.34% / 0.44% |
| 25 | 11 | 9 | 22.2% |
| 26 / 27 | 925 / 726 | 917 / 717 | 0.87% / 1.26% |
| 28 | 13 | 9 | 44.4% |
| 29 / 30 | 1721 / 1347 | 1681 / 1316 | 2.4% / 2.4% |

单位 μs。小消息有稳定的 2–4 μs 差距，大消息主要差距约 2%–3%。旧文“23/24 已持平”只描述当时同屏参照，不能延伸成与最终前三持平。点号与具体拓扑的对应仍来自赛中版本响应推断，平台没有提供逐点 trace，不把它当作已证实映射。

## 2. 实际参考了什么

以下是跑分前三的公开代码，不宣称已核验每份归档与比赛二进制逐字节一致。Bueryy 的源码在尚未合入 master 的 PR 中，因此只看 master 会漏掉第二名。

| 队伍 | 固定来源 | 重点阅读 |
|---|---|---|
| yiran，第 1，389.79 | [master 快照 c2b149e](https://gitcode.com/cann/cann-ops-competitions/tree/c2b149e43613cdcae482a1eb83bcd1afb9643a14/03_university/CANN-HCCL-Northeast-Competition-2026/final/submissions/哈尔滨工业大学+yiran) | `CcuScatterCoarseRelay`、`CcuScatterKernel`，Host 分片与角色缓存 |
| Bueryy，第 2，389.70 | [PR 1861](https://gitcode.com/cann/cann-ops-competitions/merge_requests/1861)，head `99dbcddc8fdf51d462ab33002ee2d3843c146606` | `BuildRoleSendKernel`、`BuildRoleReceiveKernel`、`BuildWriteKernel`，角色和地址发布策略 |
| cc，第 3，389.26 | [master 快照 c2b149e](https://gitcode.com/cann/cann-ops-competitions/tree/c2b149e43613cdcae482a1eb83bcd1afb9643a14/03_university/CANN-HCCL-Northeast-Competition-2026/final/submissions/哈尔滨工业大学+cc) | `CcuScatterRelayRelayMeshKernel/OwnKernel/ClosKernel`，小消息 Pull |

完整文件哈希及路径见[来源清单](evidence/source-manifest.json)。不要求“所有获奖队伍都已上传”这个前提成立；本次只需这三份已取得的代码。已有引用之外也检查过其他队伍，但不拿它们代替前三作为本次目标。

### 主要机制对照

| 问题 | A21 | 前三提供的参照 | 新版处理 |
|---|---|---|---|
| helper 中转区 | 借用自身输出前缀，转发确认后恢复 | yiran/cc 使用独立 HCCL scratch；cc 分开 gather、own、forward | 独立 scratch；一次写完整 helper 输出，去掉 restore |
| 转发与自身数据 | 数据别名强制串行 | yiran/cc 让本地数据处理与另一 die 的转发重叠 | STAGED 在自身输出 Write 前发布；任务图检查两者无互相等待 |
| 自拷贝 | relay 中较晚提交并等待 | 前三多处提前提交，最终等待 | relay 在网络分支前提交，最终仍等完成 |
| 小消息收尾 | 某些路径等待整组 Write、自拷贝、CONSUMED | Bueryy 的角色发送逐通道 EventWait 后通知，复制最后收尾 | 采用逐通道完成通知；4×3 小消息去掉 CONSUMED |
| Host / 分片 | 每次构造 vector，固定上限可能产生极小尾片 | yiran 使用栈参数、均衡 tile、内核持久循环 | 定长参数数组；按实际 scratch 容量均衡分片 |
| 角色和地址交换 | 注册时保留通用 root 分支，每轮交换地址 | yiran/cc 按 root 注册；Bueryy 专门化角色并在地址变化时刷新 | 此版保留通用角色与每轮地址交换，明确仍有控制成本 |

**没有把所有前三代码特性都移植。** yiran/cc 部分小消息 Pull 分支省略 root ACK 和接收 Read 的 EventWait，源码明确依赖运行框架的轮次/stream 同步。我们的测试允许本 rank stream 完成后立刻释放输入或输出，不在每次调用后增加 MPI barrier。没有先证明相同生命周期保证，就不能靠删等待追求 9 μs。Bueryy 某些安全完成分支保留逐 Write EventWait，说明应逐分支检查，不能把前三概括成同一套协议。

## 3. 我之前的判断出了什么问题

这里的“我”指辅助生成、调试和分析代码的 AI。问题不应归咎于使用者没有多跑几次。

**第一，把流量下界误当成了实现极限。** `15B/(11b)` 只描述理想出口吞吐，不包含阶段间依赖、通知、启动和尾片。A21 已经有双 die 和多 helper，但借用输出区增加了 `STAGED → forward → ACK → FORWARDED → restore` 链。既然数据结构能改变依赖图，“剩下只差控制微调”的判断就不成立。

**第二，过多沿用借输出前缀的设计继续重排等待。** A21 优化 sent/restore 顺序确实有官测收益，但没有消除 restore 本身。对照前三后，应先改变缓冲区所有权，再谈局部等待顺序。新版不仅把地址换成 scratch，还保证 scratch 在 FORWARDED 后才复用，避免把性能改动变成跨片数据竞争。

**第三，对小消息的优化不够系统。** 9 μs 对 11–13 μs 的差距不能由大消息带宽模型解释。角色专门化、参数个数、地址交换、ACK 和整组等待都在关键路径上。此前只改其中一两个通知，随后用“每次只有 1 μs”解释收益递减，遗漏了整个控制链的成本。新版先完成不依赖特殊运行框架的改动；角色缓存和持续内核仍是后续可测方向，不能写成已实现。

**第四，未充分区分验证器问题与算法问题。** 旧复盘把 F9/F16A/F17 等本地失败统称为实现可靠性不足，并过早对换 root 错误作协议归因。后续诊断发现私有 Runner 的新建 XN/GSA/CKE 未初始化会影响结论。修正初始化后的私有环境能推进验证，但这也不证明每个历史失败都由 Runner 引起。本次报告分别保留数值、Checker 和运行环境身份，不用单个 FAIL 推导根因。

**第五，归档整理破坏了可运行性。** 旧 GitHub `final/op_kernel_ccu/ccu_kernel.cc` 有三处 source 递推空循环；原始 A21 中的 `source[i] += compactArgs[1];` 与行尾注释一起被漏掉了。这是赛后清理产生的归档错误，不是 A21 比赛失分原因。新版恢复三处递推，并保存 19 文件原始快照和哈希，避免把“可读版”默认当成可运行基线。

窗口管理和最小改动验证仍然重要，但不能替代对算法结构和实验环境的诊断。正确顺序应是：稳定基线与环境 → 比较真实依赖 → 改动 → 数值及任务图检查 → 同条件性能测量。

## 4. 这次具体改了什么

### 独立中转缓冲与完成边界

资源缓存新增 scratch 地址与容量，使用单独缓存 tag 和新版 magic，避免误解码旧布局。Host 取得 scratch 的独立 token，relay 新增两项启动参数。helper 在 Mesh 通道发布 scratch 地址/token；Clos 通道上的 helper/长度计划保留原含义。

root 写 scratch，等待对应写完成后发 STAGED，再提交 helper 的完整自身输出。helper 收到 STAGED 后从 scratch 转发，target 确认后回 FORWARDED。root 同时等自身输出写完和 FORWARDED，才进入下一片。root 自拷贝提前提交，返回前仍等待其完成。没有去掉大消息 ACK，也没有假定所有内存地址共用 token。

按单片完整 relay 路径计，2×8 网络传输任务由 A21 的 `7 stage + 7 suffix + 7 restore + 8 direct + 7 forward = 36` 降为 `7 stage + 7 own + 8 direct + 7 forward = 29`；4×3 由 17 降为 15。A21 数量来自源码路径，新版数量由保留的任务图核对。它描述传输任务数和输出写入结构，**不是同比例的耗时下降**；不同容量导致的片数也要另算。

实现过程中第一版曾先提交 helper output 再发 STAGED。数值测试通过，但本地 Checker 导出的依赖图仍将自身输出关联到转发前置依赖。调整顺序后图检查通过。因此本次只称“在该 Checker 图中解除依赖”，不把图中的每条边直接解释为硬件调度规律，更不把无依赖直接换算为提速百分比。

### 比例与分片

设每 rank 数据量为 B，机内单链带宽 b，root Clos 总出口近似为 4b。沿用一 helper 配一 target 的结构：

- 2×8：`T_mesh=(1+r)B/b`，`T_clos=(8-7r)B/(4b)`，平衡点 `r=4/11`。
- 4×3：`T_mesh=(1+r)B/b`，`T_clos=(9-2r)B/(4b)`，平衡点 `r=5/6`。

这里 r 是每 helper 转发量占 B 的比例。yiran 的多目标拆分安排与我们的配对结构不同，不能仅凭总比例相同就声称调度相同。新版在 FP32 元素边界向下取整，并使每片中转量不超过 HCCL 返回的 scratch 容量。

以实际容量计算片数，再均分元素，避免上限片后只剩 4 字节。保持每片双线程 join 和 scratch 复用屏障，没有实现双缓冲流水或跨片持久 kernel。低容量环境会增加片数，可能抵消一部分优化收益，需要测量。

### 小消息与 Host

普通 Push / ReadyPush 在各通道 Write 的完成事件消费后发布 WRITTEN，自拷贝完成等待放在通知之后。4×3 小消息跳过 CONSUMED，但保留 READY 和数据完成；下一次 READY 确认接收者已经消费前一轮 WRITTEN。2×8 Pull 仍保留 Read 完成及 ACK。Host 用定长数组传参，通用及 DualMode 内核附带 small 参数，专用 ReadyPush 内核不传此参数，严格匹配各内核 LoadArg 数量。一次 4×1 回归曾发现多传参数，修正后使用同一最终二进制重新验证。

## 5. 已经证明与尚未证明

[验证记录](validation-postrace.md)给出具体测试、二进制身份和边界。数值正确、guard 完整、输入未修改、资源缓存稳定，以及图上输出不再重复恢复，属于本次可以检查的结果。

**尚未得到新版官方逐点耗时，不能说已达到前三，也不能给出预计总分。** 新版依然保留部分通用控制路径；持久 kernel、按 root 专门化和地址缓存没有实现。前三代码的成绩不能为我们这一组合改动背书。

下一次可用性能环境应使用同一套 A21/新版二进制、拓扑和输入做交替多次测试，分别记录小消息延迟、大消息吞吐、首次注册和缓存命中，优先检查 4×3 分片数量及 2×8 中转关键路径。每项只有获得对照数据后，才写成性能收益。
