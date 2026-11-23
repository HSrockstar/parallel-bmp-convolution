# R23base —— 第三轮开工前的 R21 同码复现

- 日期：2026-11-23（以文件内命令输出时间为准）。
- 当前源码与 `R21/src_snapshot/` 逐字节相同，A/B 的两侧均为未改源码的同码 R21 构建。
- 构建/检查：`bash check_round.sh R23base`。
- 正确性：Pthread/OpenMP 均 teacher shape/similarity PASS，MSE=0.00，相似度 100.00%；与 double `reference.bmp` 均维持 36 字节差异。
- 测速：`bash bench_ab.sh R23base ../_ab_base both 7`，P 核 `0/0,2/0,2,4,6`，Total time，各组 7 对。

## 同码测量波动（不是优化收益）

| 版本 | 项目 | base best/median | candidate best/median | Δbest |
|---|---|---:|---:|---:|
| Pthread | T1 | 23.648 / 24.812 | 24.612 / 25.033 | +4.08% |
| Pthread | T2 | 14.634 / 15.086 | 14.450 / 15.287 | −1.26% |
| Pthread | T4 | 8.573 / 9.266 | 8.754 / 9.241 | +2.11% |
| OpenMP | T1 | 24.726 / 24.924 | 24.588 / 24.781 | −0.56% |
| OpenMP | T2 | 14.402 / 14.912 | 14.166 / 14.660 | −1.64% |
| OpenMP | T4 | 8.883 / 9.275 | 8.916 / 9.446 | +0.37% |

## 解释

中位数未出现分钟级负载突发式的成倍膨胀，但低负载噪声仍可让同码 T4 best 相差约 0.4%–2.1%。本轮继续沿用 ≥1% 的既有采纳线；若候选只有一侧、一次刚超过阈值而 T1/T2/中位数不同向，不能认定收益。必须保留原始样本、做双版本同窗口复核。

原始 stdout/stderr 在 `ab_pthread_raw.txt`、`ab_omp_raw.txt`，汇总为 `ab_pthread.txt`、`ab_omp.txt`，两次正式验卷输出在 `verify_*.txt` 和 `run_*.txt`。
