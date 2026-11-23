# R23 —— 双输出行垂直复用（未采纳，连续无收益 #2，触发停止）

- 日期：2026-11-23。
- 基线：R22c；其大图与 R21 输出逐字节一致，源码在 `baseline_src/`，旧 A/B 二进制未再改动。
- 用户本次规则：连续 **2 个无正向收益**优化轮次停止。R22 是 #1；R23 未达到 ≥1% 双版本平均 T4 改善，是 #2。因此停止性能探索，R24/R25 不再继续。

## 假设与调度

原每个输出从 5 槽 H 环读 H(y−2)…H(y+2)。将环扩为 6 槽（官方宽度下每线程 245,760→294,912 bytes，仍 ≤2MB 私有 L2），一次处理 y/y+1：

1. 先获得 y 的 5 个 H 行；装入 H(y+3)，新增的一行不会覆盖 y 所需的 H(y−2)；
2. 计算 y+1 时使用前一行的四行 H(y−1)…H(y+2) 加新 H(y+3)；
3. 为下一对行装入 H(y+4)，覆盖不再需要的 H(y−2)；
4. 奇数条带尾行保留 5-tap 单输出。

每个输出按 `k0→k4` 顺序累加，输出交错 BGR 的布局、浮点 FMA 和 `(int)(v+.5)` 饱和路径不变。**理论希望**是每两个输出由 10 次 H 数组读取降到 6 次，不像 R22 那样 x 分块或重复水平 halo。

## 三个实现版本及代码生成排查

### A：内联双输出（`candidate_src/`）

两版 `conv_band` 中直接写双表达式。144 个窄图/小图回归全过；timg 验卷双版本 PASS、36 字节参考差异，且与 R22c 的结果图逐字节相同。

构建报告 `vector_pthread.txt` / `vector_omp.txt`：Pthread 双输出循环 `conv_pthread.c:234` 因 **number of versioning for alias run-time tests exceeds 10** 未向量化，OpenMP 成功。R18 已出现内联上下文影响向量化，不能一边成功就假定另一边同样成功。

### B：局部 `restrict`（`candidate_restrict_src/`）

在循环局部 H0…H5、D0/D1 上声明 `restrict`，仅探索新双输出生成代码所需的安全别名元数据。`vector_restrict_pthread.txt` 证明 Pthread **仍因同一限制未向量化**，局部声明没有提供编译器所需的信息；该版不进入性能测量。

### C：独立双输出函数（`candidate_helper_src/`，实际测速版）

将 V+pack 抽成 `vpass2_pack(const float * restrict h0, …, h5, uint8_t * restrict d0,d1,n3)`，参数 restrict 与实际六槽不相交调用一致；加 `noinline`，避免 Pthread 内联上下文的别名阈值再次退化。每行有一次函数调用，不是所有 channel/n 都调用。

- `build_helper.txt`：Release 构建成功、无新增 warning。
- `vector_helper_pthread.txt` 和 `vector_helper_omp.txt`：双输出主循环两版均 32-byte vectorized，且没有该循环的运行期别名版本报告。
- `asm_helper_*.txt`：官方宽度整除 32 的 AVX2 主循环没有向量值 spill 到栈；向量尾部和 fallback 占文件的一部分，不能笼统把所有 `[rsp]` 都当主循环 spill。
- **理论部分落空**：主循环gcc仍会将共享 H1…H4 再次作为 vmulps/vfmadd 的内存操作数，未完全驻留寄存器实现 10→6 次加载，同时因 -funroll-loops 展开且发出 pack/clamp 指令。故数据来源不能写成“已实现六次 load”。目前更有效的可能是两条独立链的调度/循环结构，H 行多半仍在热 cache 中；实际指令数/重复内存操作数需要靠代码生成解释，不能仅按 C 表达式计数。
- 添加 restrict/noinline 是为了恢复新算法的 AVX2 路径，避免拿“标量错误实现”测判算法，不另申报一个优化轮次。R22/R23 的停止判断针对完整的双输出策略。

## 正确性（测速前）

- `edge_helper/`：**144 cases，0 failures**，覆盖非均匀 w1–12、多个小高度、BGR 与行 padding；其中奇偶条带、宽度小于 SIMD 宽度的 fallback 都有测试。
- `check_round.sh R23helper`：Pthread/OpenMP 均 shape/similarity PASS、MSE=0.00、相似度 100.00%；与 double reference 差异仍为既有 36 字节。
- 两版 R23helper 输出分别与 R22c 的 `timg_blur_{pthread,omp}.bmp` **逐字节相同**。
- 命令与 stdout 保存为 `edge_helper.log`、`check_round_helper.log`。第一版 A 也在 `edge/` 和 `R23/` 中通过相同大图验证。

## 测量

`bash 作业/代码/bench_ab.sh R23 作业/优化日志/_ab_base both 7`。

- timg 4096×2304，P 核 taskset 0 / 0,2 / 0,2,4,6。
- 程序自打印 Total time；每个版本每个配置 7 对交替，best 按既有规则判主结果，median 同时检查。
- Pthread 时间：loadavg 2.37 / 2.74 / 3.10；OpenMP 开始：2.50 / 2.76 / 3.10。所有原始 stdout/stderr 在 `ab_*_raw.txt`；汇总在 `ab_*.txt`。
- median/best 没有出现历史分钟级 2–3× 突发污染。R23base 同码测得 pthread T4 Δbest +2.11%，故即使一边改善约 1%，也不能推掉另一边同窗口的 T4 退化、选择性采纳。

| 版本 | 配置 | base best/median (ms) | candidate best/median (ms) | Δbest | Δmedian |
|---|---|---:|---:|---:|---:|
| Pthread | T1 | 23.733 / 24.521 | 23.542 / 23.984 | −0.80% | −2.19% |
| Pthread | T2 | 14.867 / 14.996 | 14.280 / 14.698 | −3.95% | −1.99% |
| Pthread | T4 | 8.790 / 9.180 | **8.687** / 9.009 | **−1.17%** | −1.86% |
| OpenMP | T1 | 24.284 / 24.533 | 23.614 / 23.781 | −2.76% | −3.07% |
| OpenMP | T2 | 14.419 / 15.153 | 13.535 / 14.623 | −6.13% | −3.50% |
| OpenMP | T4 | 8.750 / 9.229 | **8.999** / 9.265 | **+2.85%** | +0.39% |

**双版本 T4 算术：**

```text
base_avg = (8.790 + 8.750) / 2 = 8.770 ms
cand_avg = (8.687 + 8.999) / 2 = 8.843 ms
改善率   = (8.770 - 8.843) / 8.770 × 100% = −0.83%
```

负数表示本指标实际慢 0.83%；即使按单边 Pthread 的 −1.17% 看，它不是双版本平均改善。OpenMP 是最佳值、典型值均变差。

## 结论与处置

- **未达 ≥1% 正向收益，回退到 R22c**。不为双输出单独额外做 noisy 标签重复：两版的中位数与最优值一致支持没有双版本平均收益，而非 med 远大于 best 的突发故障；继续检查手写 AVX/不同寄存器策略已属于新的探索，服从停止规则。
- R23 虽在 T1/T2 部分有较好看的单版本改善，但作业成绩与采纳目标为 T4 两版平均；不得选择性保留以制造收益。
- 最终 src 从 `baseline_src/` 恢复，R23 A/B/C 源码和完整数据留档；不回滚或者覆盖用户的 `优化日志.md` 原先未提交修改。
- **停止计数：R22 = #1，R23 = #2 → 本轮探索结束。** 最终保留版本为 R22c（即 R21 的宽图数值路径 + 非性能窄图修复）。

## 报告可引用的教训

1. 源码写“共享”不等于生成机器码只 load 一次；看向量化报告后还需看实际汇编内存操作数。
2. 碰到 alias 阈值退化时，局部 restrict 与实际具有独立参数生命周期的不内联函数并不等价。
3. 最优值、典型值、同码噪声、两个实现必须一起考虑，不能从单边或单机配置挑结论。
4. 新局部技巧的 T1/T2 改善不一定带到目标 T4；正确性和向量化只是准入条件，而不是收益证明。
