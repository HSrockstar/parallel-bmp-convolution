# 本轮留档索引与命令复现

## 最终采用状态

- `作业/代码/conv_pthread.c`、`conv_omp.c` = `R22c/src_snapshot/`。
- `作业/优化日志/_ab_base/` = 同源码的 R22c 构建，回退后已再次 cmp 确认。
- R23 候选双输出源码仅保留在 `R23/candidate_*_src/`，不进入最终代码。
- 原有 `作业/优化日志.md` 未提交修改已保留；教师 `附件/` 和 `环境验证/test.cpp` 没有修改。

## 人读总记录

| 文件 | 内容 |
|---|---|
| `作业/优化日志.md`（末尾第三轮） | 完整假设、数据、算术、回退和停止结论 |
| `R22c/README.md` | 边界公式、修复前证据、独立行 ASan、重建/复测命令 |
| `R23base/README.md` | R21 同码 A/B 波动与本次基线 |
| `R23/README.md` | 双输出 A/B/C 三实现、向量化和汇编检查、A/B 性能算术 |
| `R23/acceptance.txt` | 最终源码 identity、验收与 ASan 摘要 |
| `R23/final_manifest.txt` | 最终 Makefile、双源码与二进制 SHA-256 |
| `作业/结果/final_verify.txt` | 当前 R22c 的教师 6/6 验卷原文与附加测试摘要 |
| `作业/结果/bench_summary.md` | 本次官方 3 次 window 的 T1/T2/T4 与 S/E |
| `R23final/bench_stats.txt` | 当前保留版 5 次记录、best/median/all |

## 两类实验输出不要混淆

- `R23/ab_*.txt`、`R23/ab_*_raw.txt`：候选 R23 对 R22c，7 对 base/candidate，含 T1/T2/T4，采纳判定的主证据。
- `R23final/`：回退后 R22c 自己的正确性与 `bench_round.sh` 5 次数据。
- `作业/结果/bench_*.txt`、`bench_summary.md`：最后执行官方 `run_bench.sh` 的 3 次数据，不是 R23 的优化收益。
- `R23base/old_official_*_R15*`：改写官方结果之前保存的 R15 历史版本。

## 退回或重现候选时

所有候选都存于 `R23/candidate_*_src/`（不是 `作业/代码` 当前版本）。需要重现时另用独立构建目录或先保存当前源码，不要把这些探索文件直接当成保留版本。

候选正常构建固定：
```bash
gcc-11 -O3 -march=native -funroll-loops -ffp-contract=fast -Wall -std=c11 conv_pthread.c -o conv_pthread -lpthread
gcc-11 -O3 -march=native -funroll-loops -ffp-contract=fast -Wall -std=c11 -fopenmp conv_omp.c -o conv_omp
bash bench_ab.sh <新标签> <基线目录> both 7
```

R22c 的窄图复测、ASan 编译及官方小图检查命令见 `R22c/README.md`。`check_edges.py --work-dir` 与 `check_small.sh` 会要求新目录，以免覆盖先前证据；不要通过删除旧实验目录来绕过。

## 大量小文件的说明

每个边界用例均保留：生成的输入 BMP、输出 BMP、卷积核 txt、像素矩阵 txt、程序 stdout/stderr；`results.tsv` 汇总每项差异与相似度。它们是 144 项 × 两实现 × Release/ASan × 修复前后对照的细粒度原始证据，因此本轮目录文件数较多；在终端推荐先看 `results.tsv`、各轮 README 与上述总记录，无需逐个点开图片。

ASan 可执行文件、R23 汇编 `.txt` 与临时对象文件位于相应实验目录，不能混入 Release 性能测量。`candidate_src` 的第一版也通过 144 边界与大图输出一致性检查；三个版本的差别主要是 GCC 是否能生成 AVX2 主循环。它们是为公平评测同一双输出假说而做的实现修正，不记作三个独立的优化收益轮次。
