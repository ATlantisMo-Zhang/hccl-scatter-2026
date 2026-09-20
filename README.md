# HCCL Scatter 集合通信算子（2026 HCCL 通信库创新大赛 · 东北赛区）

基于昇腾 CANN 平台的 Scatter 集合通信算子实现与性能优化，覆盖初赛（AICPU + TS 引擎）与决赛（CCU 引擎）两个阶段。决赛排名第 18（三等奖取前 17）。

## 成绩

| 阶段 | 引擎 | 最终提交 | 性能 |
|---|---|---|---|
| 初赛 | AICPU + TS | 255304 | 38μs / 1.13ms / 895μs（固定 1/4 远端分流，大点较基线快 22.6%/22.2%） |
| 决赛 | CCU | A21 | 11/965/757/12/2060/1610/11/925/726/13/1720/1350 μs，12 点距同屏 TBest 平均约 4%（23/24 两点持平 TBest） |

## 这个仓库里有什么

- [preliminary/](preliminary/) — 初赛完整工程（AICPU），选手实现 3 个文件
- [final/](final/) — 决赛完整工程（CCU），选手实现 6 个文件
- [docs/performance-journey.md](docs/performance-journey.md) — 性能优化历程：23 个版本、每个版本的机制与收益账本
- [docs/ccu-programming-notes.md](docs/ccu-programming-notes.md) — CCU 编程踩坑笔记：通知协议、任务图、双 die 约束
- [docs/postmortem.md](docs/postmortem.md) — 赛后复盘：窗口管理教训与方法论沉淀

## 技术要点速览

- **拓扑感知路由**：按 layer0 分组识别 2×8（8/8）、4×3（3/3/3/3）、4×1、8+4 四类拓扑，为每种拓扑与消息规模选择不同的数据流
- **双 die 并行**：root 端两个 IO Die 各自独立推进，Host 侧双线程下发 + 完成 join（决赛最大单档收益：大消息 30%–50%）
- **helper 中转与负载均衡**：2×8 下 root 将 7 条机内链交给 7 个 helper 代转 4/11 数据量，与 Clos 直发 4 条链构成 15B/11b 的理想流量平衡
- **协议按消息大小切换**：小消息 Pull（root 发布 source/token，接收者 Read）、ReadyPush（逐 peer 准备即写、省 CONSUMED 收尾）、大消息 Push/中转
- **借输出前缀中转**：helper 借用接收者输出前缀作中转区，转发完成（FORWARDED）后 root 恢复前缀，suffix 写与 restore 依赖解耦
- **流量建模先行**：每个优化前先建关键路径模型（root 出口容量、中转比例平衡点），模型与官测结果互相校准（F11 模型预测 9.1%、实测 8.7%）

## 代码结构

两份工程均为官方赛题模板的完整可编译工程（CANN Open Software License），选手实现文件已标注：

- 初赛（3 文件）：`custom.h`、`scatter.cc`（Host 资源与链路选择）、`exec_op.cc`（AICPU 内核通信编排）
- 决赛（6 文件）：`custom.h`、`scatter.cc`（Host 链路选择与 kernel 注册）、`exec_op.{h,cc}`（Host 编排）、`ccu_kernel.{h,cc}`（CCU 内核协议）

## 关于比赛

2026 HCCL 通信库创新大赛由哈尔滨工业大学主办、华为赞助，赛题为在昇腾 950 仿真环境实现 Scatter 算子（初赛 AICPU、决赛 CCU），评分覆盖多类拓扑（2×8 / 4×3 / 4×1 / 8+4）与消息规模共 12 个性能点。决赛阶段每轮提交有固定官测窗口，成绩以平台逐点用时为准。
