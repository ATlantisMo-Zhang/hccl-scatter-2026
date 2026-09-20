# 决赛：Scatter（CCU）

决赛阶段的完整工程。赛题为在 CCU 引擎上实现 Scatter 算子，覆盖 2×8 / 4×3 / 4×1 / 8+4 多类拓扑与消息规模，共 12 个性能点。

最终提交（A21）成绩（μs）：**11 / 965 / 757 / 12 / 2060 / 1610 / 11 / 925 / 726 / 13 / 1720 / 1350**。其中 23/24 两点与同屏 TBest 持平。

## 数据流总览

按拓扑与消息规模路由到不同协议：

| 场景 | 协议 | 说明 |
|---|---|---|
| 2×8 大消息（≥16MiB） | 双 die 并行 + 7 helper 中转 | Mesh 组 7 helper 各代转 4/11，Clos 组直发 |
| 4×3 大消息 | 双 die 并行 + 2 helper 中转 + Mesh 自身复制 | root 自身复制放在较空闲的 Mesh 组 |
| 2×8 小消息 | Pull | root 发布 source/token，接收者 Read，省 READY 汇合 |
| 非中转小消息 | ReadyPush | 逐 peer 准备即 Write，省 CONSUMED 收尾 |
| 其余 | 普通 Push / 双模式 | 完整 READY→WRITTEN→CONSUMED 协议 |

## 关键设计

### 双 die 并行（F1，大消息 30%–50%）

2×8/4×3 的 rank 分布在两个 IO Die 上。root 端两个 die 各自独立推进数据路径：Host 用主从双线程分别下发两个 kernel，先全部下发再 join——两个 die 的网络传输互不等待。

### helper 中转与 15B/11b 平衡（F7/F11，大消息再降 22%）

root 必须向其余 15 个 rank 导出 15B 数据，出口容量为 7 条机内链 + 4 倍 Clos（共 11b）。理想时间下界 `15B/(11b)` 要求机内链也满载：7 个 helper 各代转 rB 时，Mesh 项 `(1+r)B/b` 与 Clos 项 `(8-7r)B/(4b)` 在 r=4/11 处相等——实测 20/21 点兑现 8.7%（模型预测 9.1%）。

### 借输出前缀中转与依赖解耦

helper 把转发数据暂存在**接收者的输出前缀区**（借用而非独立 scratch）：root 先发 suffix（不依赖前缀），helper 转发完成后 root 再恢复被借用的前缀。A21 版本进一步把 restore 的发起从"全组 suffix 完成"提前到"逐 helper 转发完成"，suffix 写与 restore 重叠。

### 协议按消息大小切换

- 小消息（<16MiB）延迟主导：Pull/ReadyPush 缩短完成依赖链（22 点 14→12μs，19 点 12→11μs）
- 大消息带宽主导：Push/中转最大化链路利用率
- 16MiB 为切换门槛，大调用的短尾片保留 Push，避免协议切换引入跨片风险

## 版本演进与收益账本

完整 23 个版本的机制与官测数据见 [docs/performance-journey.md](../docs/performance-journey.md)。核心结论：每档收益与机制本质匹配——

```
F1 双die并行   大消息 -30~50%   （结构）
F8 发送提前    20/21 -22%      （消除过早等待）
F11 七helper   20/21 -8.7%     （负载均衡，达流量模型平衡点）
F15 复制迁移   29/30 -2%       （收尾归属）
A21 等待重排   20/21 -0.4%     （链路满载后控制微调收益趋零）
```

## 选手实现文件

- `include/custom.h` — 资源上下文、注册数据、缓存 Decode 校验
- `op_host/scatter.cc` — Host 侧：链路选择（layer 感知）、拓扑识别（8/8 与 3/3/3/3）、kernel 注册（每 die 最多 2 mission）
- `op_host/exec_op.{h,cc}` — Host 编排：拓扑/消息路由、分片循环、双线程下发与 join
- `op_kernel_ccu/ccu_kernel.{h,cc}` — CCU 内核：普通 Push、SmallPull、ReadyPush、双模式、Relay（Mesh/Clos 两分支）

其余文件为官方模板预制逻辑。构建方式见官方模板说明（`bash build.sh`）。
