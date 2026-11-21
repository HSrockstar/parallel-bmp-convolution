# BMP 图像卷积作业 —— 代码目录（最终优化版）

Pthread / OpenMP 两版并行高斯卷积。5×5 核（σ=1.5, r=2）、stride=1、边缘常数 0 填充，
结果与教师验卷器（`filter2D(..., BORDER_CONSTANT)`）比对：**MSE=0.00，相似度 100.00%**
（与 double 基准仅 35/28.3M 字节 ±1 平局差异）。

优化迭代记录（R0–R13，含三次无收益探索的回退）见 [../优化日志.md](../优化日志.md)。

## 编译

```bash
make            # 生成 conv_pthread 和 conv_omp
```

CFLAGS：`-O3 -march=native -funroll-loops -ffp-contract=fast -Wall -std=c11`

## 运行

```bash
./conv_pthread <input.bmp> <output.bmp> [threads] [--dump-full]
./conv_omp     <input.bmp> <output.bmp> [threads] [--dump-full]
```

- `threads`：线程数，默认 4（OpenMP 版也可用 `OMP_NUM_THREADS`）。
- `--dump-full`：把卷积后完整像素矩阵写进 txt（timg 约 28M 像素，文件很大，慎用）。
- 每次运行自动生成 `<output>.kernel.txt`（5×5 卷积核）和 `<output>.matrix.txt`
  （像素矩阵：小图默认全量，timg 默认导出左上角 20×20 供验证）。
- I/O 采用 mmap 零拷贝（读=按需缺页，写=直接写脏页），程序输出逐线程卷积时间
  （缺页开销已含在内）与全程 Total time（成绩口径）。

## 实现要点

1. **分离核两趟卷积**：5×5 高斯核 ≈ 1D 核 `k1` 的外积（对称特征分解最小二乘拟合，
   残差 max 5.4e-9）。水平趟 `H(r) = Σ_kx k1[kx]·src[r][x+kx-2]`（5 行环形缓冲驻 L2，
   源行只读一次），垂直趟 `out = Σ_ky k1[ky]·H(y+ky-2)`。每像素乘加 75→30。
2. **通道平面化**：源行解交织成 B/G/R 平面，水平/垂直趟全为连续 float 访问，
   AVX2 8 宽向量无交织 shuffle；尾趟 pack 回交错 uint8。
3. **mmap 零拷贝 I/O**：输入 `PROT_READ` 映射直读；输出 `ftruncate + MAP_SHARED`
   映射直写（脏页由内核回写）。无显式 read/write。
4. **BMP 免翻转**：核垂直方向对称 ⇒ 对"上下翻转的图"直接卷积 = 卷积结果的上下翻转，
   故按文件原始行序（自下而上）就地处理，写回时同向高度即可。
5. **边界零填充天然处理**：源行越界 → H 置零平面；x 方向逐 kx 取有效区间，无分支。
6. **并行结构**：按行条带静态划分（Pthread：pthread_create/join 逐线程 clock_gettime
   计时；OpenMP：parallel 区内手动等分条带，omp_get_wtime 逐线程计时）。

## 自测与验卷

```bash
make check      # Pure_white.bmp 自测并过教师验卷器
```

正式验收（验卷器在 `环境验证/`，已按附件原样编译）：

```bash
../../环境验证/test ../../附件/.../测试程序/timg.bmp ../结果/timg_blur_pthread.bmp
python3  ../../附件/.../测试程序/test_blur.py <原图> <你的模糊图>
```

最终验卷记录：`../结果/final_verify.txt`（6/6 PASS，MSE=0.00）。

## Benchmark（加速比 / 效率）

```bash
bash run_bench.sh [pthread|omp|both] [input.bmp]   # 默认 both + timg.bmp
make bench
```

- 1 / 2 / 4 核各跑 3 次取最优 `Total time`，输出 S(p)=T(1)/T(p)、E(p)=S(p)/p，
  明细存 `../结果/bench_pthread.txt`、`bench_omp.txt`，汇总 `../结果/bench_summary.md`。
- **taskset 选核注意**：本机 i9-13900K 的 CPU 0/1 是同一物理核的超线程兄弟，
  2 核/4 核实验按物理核计：`taskset -c 0,2` 和 `taskset -c 0,2,4,6`。

优化迭代期间使用 `bench_round.sh <轮次>`（5 次取最优，原始输出逐轮留档到 `../优化日志/`）
与 `check_round.sh <轮次>`（验卷 + 与 double 参考图逐字节比对门禁）。

## 最终数据（timg.bmp，2026-11-21 实测，取 3 次最优）

| 版本 | T(1) ms | T(2) ms | T(4) ms | S(2) | S(4) | E(2) | E(4) |
|------|---------|---------|---------|------|------|------|------|
| Pthread | 38.0 | 21.6 | **12.0** | 1.76 | 3.17 | 0.88 | 0.79 |
| OpenMP  | 39.1 | 21.7 | **12.1** | 1.80 | 3.23 | 0.90 | 0.81 |

baseline（-O2 朴素实现）为 293.2/162.8/95.5 与 290.5/160.3/93.9 —— **4 核成绩口径提速 7.8 倍**。

## 文件

| 文件 | 说明 |
|------|------|
| `conv_pthread.c` | Pthread 版（条带划分 + 分离核平面卷积 + mmap I/O，逐线程计时） |
| `conv_omp.c`     | OpenMP 版（同构实现，omp_get_wtime 计时） |
| `run_bench.sh`   | 正式 1/2/4 核 benchmark + 加速比/效率汇总 |
| `bench_round.sh` | 优化迭代用统一基准（5 次取最优，逐轮留档） |
| `check_round.sh` | 正确性门禁（教师验卷器 + 参考图逐字节比对） |
| `Makefile`       | `make` / `make check` / `make bench` / `make clean` |
