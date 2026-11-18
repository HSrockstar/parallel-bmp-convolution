# BMP 图像卷积作业 —— 代码目录（baseline）

Pthread / OpenMP 两版并行高斯卷积。5×5 核（σ=1.5, r=2）、stride=1、边缘常数 0 填充，
自解析 BMP 头（24 位真彩、自下而上、4 字节对齐 stride），结果与教师验卷器
（`filter2D(..., BORDER_CONSTANT)`）逐像素一致（MSE=0，相似度 100%）。

## 编译

```bash
make            # 生成 conv_pthread 和 conv_omp（gcc -O2）
```

## 运行

```bash
./conv_pthread <input.bmp> <output.bmp> [threads] [--dump-full]
./conv_omp     <input.bmp> <output.bmp> [threads] [--dump-full]
```

- `threads`：线程数，默认 4（OpenMP 版也可用 `OMP_NUM_THREADS`）。
- `--dump-full`：把卷积后完整像素矩阵写进 txt（timg 约 28M 像素，文件很大，慎用）。
- 每次运行自动生成 `<output>.kernel.txt`（5×5 卷积核）和 `<output>.matrix.txt`
  （像素矩阵：512×512 小图默认全量，timg 默认导出左上角 20×20 供验证）。
- 终端输出读图 / 并行卷积 / 写图分段耗时 + **逐线程运算时间**，`Total time` 为
  成绩统计口径的总运行时间。

## 自测与验卷

```bash
make check      # Pure_white.bmp 自测并过教师验卷器
```

正式验收（验卷器在 `环境验证/`，已按附件原样编译）：

```bash
../../环境验证/test ../../附件/.../测试程序/timg.bmp ../结果/timg_blur_pthread.bmp
python3  ../../附件/.../测试程序/test_blur.py <原图> <你的模糊图>
```

## Benchmark（加速比 / 效率）

```bash
bash run_bench.sh [pthread|omp|both] [input.bmp]   # 默认 both + timg.bmp
make bench
```

- 1 / 2 / 4 核各跑 3 次取最优 `Total time`，输出 S(p)=T(1)/T(p)、E(p)=S(p)/p，
  明细存 `../结果/bench_pthread.txt`、`bench_omp.txt`，汇总 `../结果/bench_summary.md`。
- **taskset 选核注意**：本机 i9-13900K 的 CPU 0/1 是同一物理核的超线程兄弟，
  2 核/4 核实验按物理核计：`taskset -c 0,2` 和 `taskset -c 0,2,4,6`。

## baseline 数据（timg.bmp，2026-11-18 实测，取 3 次最优）

| 版本 | T(1) ms | T(2) ms | T(4) ms | S(2) | S(4) | E(2) | E(4) |
|------|---------|---------|---------|------|------|------|------|
| Pthread | 297.2 | 162.5 | 95.0 | 1.83 | 3.13 | 0.91 | 0.78 |
| OpenMP  | 290.6 | 160.4 | 92.8 | 1.81 | 3.13 | 0.91 | 0.78 |

4 核总耗时 ≈95 ms，其中读图+写图约 30 ms（串行），卷积并行区约 65-90 ms。
成绩按 4 核总运行时间计，后续优化方向：减少 I/O 开销、改善访存局部性、
浮点转定点/分离核行列降乘法数等（在优化版中做，baseline 保持朴素实现）。

## 文件

| 文件 | 说明 |
|------|------|
| `conv_pthread.c` | Pthread 版（按行分块，pthread_create/join，clock_gettime 逐线程计时） |
| `conv_omp.c`     | OpenMP 版（`#pragma omp parallel for schedule(static)`，omp_get_wtime 计时） |
| `run_bench.sh`   | 1/2/4 核 benchmark + 加速比/效率汇总 |
| `Makefile`       | `make` / `make check` / `make bench` / `make clean` |
