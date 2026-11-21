# BMP 图像卷积作业 —— 代码目录（最终优化版）

Pthread / OpenMP 两版并行高斯卷积。5×5 核（σ=1.5, r=2）、stride=1、边缘常数 0 填充，
结果与教师验卷器（`filter2D(..., BORDER_CONSTANT)`）比对：**MSE=0.00，相似度 100.00%**
（与 double 基准仅 36/28.3M 字节 ±1 平局差异）。

优化迭代记录（R0–R16 + OpenMP 计时修复，含四次无收益探索的回退）见 [../优化日志.md](../优化日志.md)。

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
2. **单趟融合（R14/R15）**：垂直/水平趟均改为每 x 一次算完 5 tap
   （5 load + 4 FMA + 1 store），免去中间平面 4 次"读-改-写"整平面往返；
   边缘列走逐 tap 路径，窄图（w≤4）保留旧实现。由 RDTSC 分段插桩定位
   （vpass 占纯计算 49%），两轮合计 T4 -20%。
3. **连续 BGR 单布局（R19，取代 R9 通道平面化）**：三通道共用同一 1D 核 ⇒
   每行视作 3w 字节数组，水平 tap 读 n±6/n±3/n（步长 3 字节，恒同通道）；
   H 环形缓冲每槽 3w float 与源行/输出行同序，垂直趟直接舍入饱和写字节——
   解交织、vbuf、pack 交织三个中间布局转换一次性删除，输出逐字节不变。
4. **mmap 零拷贝 I/O**：输入 `PROT_READ` 映射直读；输出 `ftruncate + posix_fallocate
   + MAP_SHARED` 映射直写（R21：fallocate 把 ext4 块分配挪到串行预备段，
   从并行缺页路径卸下 ~7% T4；脏页由内核回写）。无显式 read/write。
5. **BMP 免翻转**：核垂直方向对称 ⇒ 对"上下翻转的图"直接卷积 = 卷积结果的上下翻转，
   故按文件原始行序（自下而上）就地处理，写回时同向高度即可。
6. **边界零填充天然处理**：源行越界 → H 置零平面；x 方向逐 kx 取有效区间，无分支。
7. **并行结构**：按行条带静态划分（Pthread：pthread_create/join 逐线程 clock_gettime
   计时；OpenMP：parallel 区内手动等分条带，omp_get_wtime 逐线程计时）。
   OpenMP 线程数优先级：**显式传参 > `OMP_NUM_THREADS` > OpenMP 默认**，程序打印实际值。

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

优化迭代期间使用 `bench_round.sh <轮次> [pthread|omp|both] [RUNS]`（缺省 5 次取最优，
best+median 留档 `bench_stats.txt`，原始输出逐轮留档到 `../优化日志/`）、
`check_round.sh <轮次>`（验卷 + 与 double 参考图逐字节比对门禁），以及
`bench_ab.sh <轮次> <基线二进制目录>`（基线/候选**交替**对照，抗共享机负载突发；
基线二进制存档于 `../优化日志/_ab_base/`）。三个脚本均内置构建门禁（先 make，失败即止）。

## 最终数据（timg.bmp）

第二轮迭代后（R21 = R19 连续 BGR + R21 fallocate，2026-11-22，A/B 交替安静窗实测）：

| 版本 | T(1) ms | T(2) ms | T(4) ms | S(2) | S(4) | E(2) | E(4) |
|------|---------|---------|---------|------|------|------|------|
| Pthread | 24.5 | 14.6 | **~8.4–8.9** | 1.68 | 2.40 | 0.84 | 0.60 |
| OpenMP  | 24.6 | 14.4 | **~9.2** | 1.70 | 2.68 | 0.85 | 0.67 |

（当日共享机负载突发，官方 run_bench 行偏高，空闲期可复测正式口径。）

第一轮最终（R15 代码，2026-11-21 实测，官方 3 次取最优）：

| 版本 | T(1) ms | T(2) ms | T(4) ms | S(2) | S(4) | E(2) | E(4) |
|------|---------|---------|---------|------|------|------|------|
| Pthread | 29.1 | 16.8 | **9.8** | 1.73 | 2.98 | 0.86 | 0.74 |
| OpenMP  | 28.8 | 16.7 | **10.4** | 1.72 | 2.77 | 0.86 | 0.69 |

baseline（-O2 朴素实现）为 293.2/162.8/95.5 与 290.5/160.3/93.9 —— **4 核成绩口径提速 ~11× / ~10×**。
（2026-11-21 早间 R10 版为 12.0/12.1 ms；R14/R15 单趟融合再 -18.6%/-14.0%；
2026-11-22 第二轮 R19/R21 再约 -12%~-14%。）

## 文件

| 文件 | 说明 |
|------|------|
| `conv_pthread.c` | Pthread 版（条带划分 + 分离核连续 BGR 卷积 + mmap I/O，逐线程计时） |
| `conv_omp.c`     | OpenMP 版（同构实现，omp_get_wtime 计时） |
| `run_bench.sh`   | 正式 1/2/4 核 benchmark + 加速比/效率汇总 |
| `bench_round.sh` | 优化迭代用统一基准（缺省 5 次取最优 + best/median 留档，逐轮留档） |
| `bench_ab.sh`    | 基线/候选交替对照（抗共享机负载突发，best+median） |
| `check_round.sh` | 正确性门禁（教师验卷器 + 参考图逐字节比对） |
| `Makefile`       | `make` / `make check` / `make bench` / `make clean`（CC 钉住 gcc-11） |
