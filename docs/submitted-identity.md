# 提交包：M / 351048

`submitted/` 是决赛最终提交的原样源码，19 个文件。它既不是赛后重构的 `final/`，也不是提交前的 A21。

## 来源

平台提交 ID 351048，2026-09-18 20:14。本地版本代号 M，目录名 `A21_M_4X3_OVERLAP_SELF`。

源码从冻结包取出：

```
review_materials/experiments/A21_M_4X3_OVERLAP_SELF_20260918-194720/delivery/
  A21_M_4X3_OVERLAP_SELF_source_19_files.zip
```

冻结包 SHA256 是 `c6a683a3…ef69`。包内根目录是 `final/`，解压时去掉了这一层。19 个文件逐个跟同目录的 `SOURCE_SHA256SUMS.txt` 对过，全部一致；入库后的 blob 也跟冻结包逐字节相同，没有经过换行转换（`submitted/.gitattributes` 写的是 `* text=auto eol=lf`）。

当时的验证情况在 `FINAL_IDENTITY.json` 里：31 个生产用例，加 2 个只用于测试的缩片用例，共 33 个独立输出用例。生产库 SHA256 是 `0d86de1a…6aa5`。精确 self-alias 那个用例数值通过、Checker 报 406，当时就在报告里单独列出来了。官方性能一栏是 `NOT_RUN`，这份代码没有跑分。

## 三个版本

这三个目录经常被混着说，区别在这里。

| 目录 | 是什么 | magic | 2×8 / 4×3 比例 | 官方成绩 |
|---|---|---|---|---|
| `submitted/` | 最终提交 M | `A21_M4X3_CACHE_READY` `0x4132314D34330001` | 4/10、3/4 | 有，371.34 分第 18 |
| `baselines/a21/` | 提交前的 A21 | `F18_A21_CACHE_READY` `0x4631384132310001` | 4/10、3/4 | 有，A21 逐点账本 |
| `final/` | 赛后重构 | `SCATTER_CACHE_READY` `0x5343415454455232` | 4/11、5/6 | 无 |

三个 magic 都不一样，cache tag 也不一样。这样旧库留在 EngineCtx 里的条目，不会被新库当成有效缓存。只改 magic 不改 tag，或者反过来，都会留下半边。

行号也各差 1 到 11 行，引代码的时候要说清是哪一份。

## 和 A21 的差别

只动了 3 个文件。

`include/custom.h` 两行：magic 从 `F18_A21_CACHE_READY` 换成 `A21_M4X3_CACHE_READY`，值也换了，`Decode` 里的比较跟着改。

`op_host/scatter.cc` 九行：日志前缀、三个内核注册名、cache tag 全部从 `F18_A21` 换成 `A21_M4X3`。没有逻辑改动。

`op_kernel_ccu/ccu_kernel.cc` 是唯一的功能改动，而且只影响 4×3。A21 里 root 的自拷贝只有一处，放在 mesh 转发之后：

```cpp
if (relayTopology == 4 ? mesh : (copySelf != 0)) { CCU_IF(doSelf) { LocalCopy(...); } EventWait(copied); }
```

M 把 `LocalCopy` 提到转发循环刚结束的位置，4×3 时先提交，等待还留在原处：

```cpp
if (relayTopology == 4) { CCU_IF(doSelf) { LocalCopy(...); } }      // 新增，提前提交
if (relayTopology != 2) { EventWait(sent, ...); }
// ...
if (relayTopology == 4 ? mesh : (copySelf != 0)) {
    CCU_IF(doSelf) { if (relayTopology != 4) { LocalCopy(...); } EventWait(copied); }
}
```

`M_vs_A21_actual_work.json` 数过一遍：17 次 Write、17 个 SQE、1 次 LocalCopy、44 条 metadata、30 次 NotifyRecord、59 个 wait，跟 A21 完全一样，只有复制时机的先后变了。

29 / 30 两点在当时的显示精度里没有变化。赛后总结第 8 节纠正过"这一步能快 2%"的说法，那是没验证过的估计。所以这次改动只能这么说：把 root 自拷贝和转发拆开了，没有测到收益。

## 哈希

```
8ac26e997fe64aa725e9c85ea7313057f01747f7ab6cdbfc5bba386a370469f9  .clang-format
f6b95230e226483126446c24523856e354c162d449d0e7a0a7c2e3a53c19c325  .gitattributes
b90c7179dd89dac61a0bbf040cd3c2baaae1fc101e01a308ddb4b1b71b996ae8  .gitignore
1584dfa3970b4160502e202c40d3a00667f5cb741963de2f4527cf35fd444006  CMakeLists.txt
c5accbbd8546e94c34aed24afe689a617627d18eed5a6c48277e48db57c23851  LICENSE
05e21c24a754982671a21998bdcc3e33f6acbfce906e41582511b9e36a24895e  README.md
642f070e887127212946b5b87f978277fb60e13ad619d28c0ea55452704f454e  build.sh
9bfd50bcbbd577c1ac28304548c4b69513fcf4b38507320abff4e88e21f0a440  include/binary_stream.h
81e9808cce7897f197c25ee327ff5ba7de92c8ae1a7230dd12d57f2f67b47d23  include/common.h
9cacbaabee74bfdff9832721a6c9770269250a3dc63d3cf156a651774951bd9c  include/custom.h
47d08dc9961a3e6bdbe81afd7cbd519423bcb4f2b6b78e87d64fc4f02f676903  include/hccl.h
59dfc121916d11b2d446b14286d1efbf0335a661ea84c42fc0ed0a22ab0828f4  include/log.h
821defcca7a81dd34a09f7d70970a004611eb719d41dbd92dbc3ec8baff6f586  op_host/CMakeLists.txt
3b3f7d0f0e2667661315acde87908fba2967f76fddc2d7dc0325a4d3d35509a1  op_host/exec_op.cc
da1698f5ca99cad87371ffd36dc03553e7117987dd9c28d1246b84808e6c8315  op_host/exec_op.h
1807ec4cbe34f0fad7588f799705f4229d6fd1c25d7d11da4f4a276ba851f98b  op_host/scatter.cc
af1a3bb225862e0318de1708eaa8193a3697e6fefea2b735db7417910aecc07e  op_kernel_ccu/CMakeLists.txt
9f70ab6eae94042ae0b6054911c8b9ecc7afdfd7edb115209112238b734fd611  op_kernel_ccu/ccu_kernel.cc
c371a23c98a6b55615edddcc4de386af1dd4d5acd44828d2266f850965578797  op_kernel_ccu/ccu_kernel.h
```

## 怎么用

```bash
bash submitted/build.sh
```

产物是 `submitted/build/lib64/libhccl.so`。

要复现参赛版本的行为就用它。跑本地数值验证不行：`tests/` 的驱动按 `final/include/custom.h` 的布局解资源缓存，`submitted/` 的 `AlgResourceCtx` 没有 scratch 字段，解不出来（`tests/README.md` 第 12 行写过这一点）。读算法主线还是看 `final/`，注释和结构都更完整。

## 边界

- 32 次调用全过是 `final/` 的数字，不是这份的。
- 4×3 那次改动没有测到收益，不能说快 2%。
- 不能拿 `submitted/` 和 `final/` 逐行相减当赛后优化的收益。两者协议基础不同，一个借输出前缀，一个用独立 scratch。
- 371.34 分和第 18 名对应的是 351048 这个包，不能挂到 `final/` 上。
