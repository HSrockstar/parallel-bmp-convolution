# R22c —— 连续 BGR 窄图边界修复（不计性能轮次）

- 日期：2026-11-23。
- 前置版本：R21。`pre_fix/` 保存修改前的双源码；`src_snapshot/` 保存修复版；`bin_before/` 保留旧 A/B 基线的构建副本。
- 源码 SHA-256、原基线二进制哈希、编译器、系统和负载见 `baseline_manifest.txt`；原始 git 状态见 `git_status_before.txt`。
- **计数规则**：这是独立 correctness 修复，不计为优化收益或失败。之后性能轮次沿用 R22 = 连续无收益 #1；用户本次要求两个连续无收益停止（替代旧三振协议）。

## 静态原因

原 `hpass_row` 内部区从 `2 * KRAD * 3 = 12` 字节起算，左右边缘窗口也各取 12 字节。实际 5-tap 的半径只有两个 BGR 像素，即 `KRAD * 3 = 6` 字节。

- 宽度 5：右侧扩展边缘可以在左缘读 `srow[-3..-1]`；这使普通 BMP 映射读到了相邻行甚至文件头。
- 宽度 5–7：左右处理窗重叠，tap 贡献重复写入。
- 宽度 1–4：走原窄图慢路径，不受影响。
- 宽度 ≥8：窗口不重叠、内部区与边缘之和恰好覆盖整行；修复后应与 R21 输出逐字节一致。

修复是两个 C 文件中六处 `2 * KRAD * 3` 改成 `KRAD * 3`。两个版本、两种内部/边缘路径的源语义相同；官方宽度 4096 的每行 FMA 顺序不变。

## 新回归工具

`作业/代码/check_edges.py`：标准库 Python，不修改教师素材。自行构造含非零行尾 padding 的非对称 BGR 图（宽度 1–12、高度 1/2/3/4/5/8），对两个实现跑 4 线程，计算独立直接 5×5 double 零填充卷积，检查尺寸、padding、相似度 >99% 和单字节差 ≤1。工作目录必须为新目录，`results.tsv` 留全部数值。

`作业/代码/check_edges_asan.c`：把单行源数组精确分配在堆上，直接调用 `hpass_row`，宽度 1–12。编译时以 `-DCONV_SOURCE='"conv_pthread.c"'` 或 `conv_omp.c` 选择实现。它避免 BMP 映射中邻接文件头掩盖越界。

`作业/代码/check_small.sh <新轮次>`：Pure_white/Pure_black × 两版 teacher shape/similarity 检查，保留程序与验卷器原始输出，不覆盖旧轮次。

## 修复前复现（过程含一次测试器修正）

1. 首次 `edge_before` 全部误报：参考数据只含 48 字节像素，而比较函数错误地拿它与 102 字节整幅 BMP 比较长度。这些 144 个 FAIL 是**测试器长度问题，不是程序结果**。保留 `edge_before.log` 与目录作反例；修正后使用全新目录 `edge_before_v2`。
2. 修正后、**未改 C 源码**复测：144 项，36 FAIL，全部集中在 w=5/6/7；其余 w=1–4/8–12 共 108 项全部零像素差异。
   - w5×h3：两版均 27/45 有效字节不同，最大绝对差 108，相似度 73.90069%。
   - w5×h8：两版均 72/120 有效字节不同，最大绝对差 139，相似度 65.83260%。
   - w6×h3：18/45 差异，最大差 100，相似度 80.58380%。
   - w7×h3：9/45 差异，最大差 100，相似度 87.44242%。
   - 详见 `edge_before_v2/results.tsv` 和边缘定位输出 `edge_before_v2.log`。
3. `R23smallbase` 与 `R23base` 的官方 Pure_white、Pure_black、timg 验卷全过。这佐证了均匀/大尺寸官方样本不能替代有内容梯度的小尺寸测试。
4. 整程序 ASan 归档 `asan_edge_before`：数值失败与正常构建相同，但没有报地址错误。原因是一个 `srow` 仅表示 BMP 映射中的一行，负下标读到映射内相邻字节时，不会越过 ASan 的映射对象边界。
5. 独立单行堆探针：Pthread/OpenMP 均返回 1，`asan_row_*_before.txt` 在 `hpass_row` 右边缘读取报告 `heap-buffer-overflow`。这才把逻辑行越界变成 ASan 可见的堆越界。

## 修复后验证

- `edge_after`：普通 Release 构建，**144 cases，0 failures**。
- `asan_edge_after`：整程序 `-O1 -g -fsanitize=address` 构建，**144 cases，0 failures**。
- `asan_row_*_after.txt`：两个独立行探针 exit 0，无 ASan 报告。
- 两张小图 teacher 4/4 PASS：`R22c_small/`，终端汇总 `official_small_after.txt`。
- timg：`check_round.sh R22c`，双版本 shape/similarity PASS，MSE=0.00，相似度 100.00%，仍与 double 参考图有 36 字节平局差；见 `verify_*.txt`、`check_round_after.txt`。
- `R22c/timg_blur_{pthread,omp}.bmp` 分别与 `R23base` 的 R21 输出逐字节相同，不是仅满足相似度阈值。

常规构建命令仍为 `make -C 作业/代码 clean all`，gcc-11、优化参数及 Makefile 未变。修复后 R22c 的二进制进入 `_ab_base` 作为 R23 的正确性基线；旧二进制已由 `bin_before/` 保留，R21 源码快照保持不变。

### 重建与复测命令

在仓库根目录运行；`OUT`、`WORK`、`R22c_repro_small`、`R22c_repro_timg` 应换成不存在的新标签以防覆盖证据。

```bash
CODE=作业/代码
OUT=作业/优化日志/R22c/asan_repro
WORK="$OUT/edge"
mkdir -p "$OUT"
make -C "$CODE" clean all

# 独立单行堆 ASan 探针（需从代码目录可直接定位 CONV_SOURCE）
gcc-11 -I"$CODE" -DCONV_SOURCE='"conv_pthread.c"' \
  -O1 -g -fno-omit-frame-pointer -fsanitize=address -Wall -std=c11 \
  "$CODE/check_edges_asan.c" -o "$OUT/hpass_pthread" -lpthread -lm
gcc-11 -I"$CODE" -DCONV_SOURCE='"conv_omp.c"' \
  -O1 -g -fno-omit-frame-pointer -fsanitize=address -Wall -std=c11 -fopenmp \
  "$CODE/check_edges_asan.c" -o "$OUT/hpass_omp" -lm
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 "$OUT/hpass_pthread"
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 "$OUT/hpass_omp"

# 整程序的 ASan 构建，以及两个构建各自的 144 项结果对照
gcc-11 -O1 -g -fno-omit-frame-pointer -fsanitize=address -Wall -std=c11 \
  "$CODE/conv_pthread.c" -o "$OUT/conv_pthread" -lpthread
gcc-11 -O1 -g -fno-omit-frame-pointer -fsanitize=address -Wall -std=c11 -fopenmp \
  "$CODE/conv_omp.c" -o "$OUT/conv_omp"
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
python3 -I "$CODE/check_edges.py" --pthread "$OUT/conv_pthread" \
  --omp "$OUT/conv_omp" --work-dir "$WORK"
python3 -I "$CODE/check_edges.py" --pthread "$CODE/conv_pthread" \
  --omp "$CODE/conv_omp" --work-dir "$WORK.release"
bash "$CODE/check_small.sh" R22c_repro_small
bash "$CODE/check_round.sh" R22c_repro_timg
```

## 报告可引用的教训

1. **系统通过验收图 ≠ 通用寻址正确**：内容相同的图会掩盖边界公式错误，宽高都可能触发内存问题。
2. **sanitizer 是否报错依赖分配对象粒度**：这里必须独立分配行，而不是在整幅映射中取行指针。
3. **先审计测试器再归因被测程序**：误报原样保留并修正测试器，随后修复前的有效证据与修复后的结果才可比较。
