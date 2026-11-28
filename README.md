# Parallel BMP Convolution（Pthread / OpenMP 图像并行卷积）

并行计算课程作业：从 BMP 图片读取像素，用 **5×5 高斯核（σ=1.5）、stride=1、边缘常数 0 填充**
进行模糊卷积，分别以 **Pthread** 与 **OpenMP** 两种方式并行实现，并与教师验卷器
（OpenCV `filter2D(..., BORDER_CONSTANT)`）比对，要求相似度 > 99%（最终实现 MSE=0.00、
相似度 100.00%，与 double 基准逐像素仅 36/28.3M 字节的 ±1 平局差异）。

## 2026-11-24 调研与算法实验

新增 [完整调研、工程清单与实测报告](research/OPTIMIZATION_RESEARCH.md) 和 [可运行代码说明](research/README.md)。包含 17 个 Pthread/OpenMP CPU 方法、17 个 NumPy/SciPy 对照入口、3 个待 CUDA 实机验证的 GPU 方法，以及随机交错基准和自动调参。原有两个作业程序已修复 BMP 校验、非标准像素偏移、同文件覆盖和错误传播问题。

```bash
make -C research
research/conv_lab_pthread --list
research/conv_lab_pthread input.bmp output.bmp --method avx2 --threads 4
python3 research/test_methods.py
```

历史 R0–R23 数据与本次 AMD EPYC / GCC 14.2 的实验分开保存。以下内容描述历史实验；新增实验与限制以调研报告为准。

完整作业要求见 [作业要求.md](作业要求.md)；**优化过程全记录见 [作业/优化日志.md](作业/优化日志.md)**。
现覆盖 R0–R23 与 R22c 正确性修复，含每轮原理、改动、验证、原始数据、保留/回退理由和停止条件。当前性能设计为 R21；R22/R23 是本轮连续两次未达收益的探索，已按用户要求停止。

## 目录结构

```
图片卷积/
├── 作业要求.md       # 作业统一要求说明
├── 附件/             # 课程附件原样解压（教师验卷器 + 测试图，禁止修改）
├── 环境验证/         # test.cpp 编译运行验证（验卷器副本，MD5 与附件一致）
├── 作业/
│   ├── 代码/         # conv_pthread.c / conv_omp.c / Makefile / run_bench.sh
│   │                 #   + bench_ab.sh / check_small.sh / check_edges.py 等验证工具
│   ├── 结果/         # 卷积结果图、kernel/matrix 导出、benchmark 记录、最终验卷记录
│   └── 优化日志/     # 每轮原始基准输出 + 验卷器输出 + 参考图（报告数据源）
└── 作业/优化日志.md   # 优化迭代主日志（R0-R23 + R22c 边界修复）
```

## 快速开始

```bash
cd 作业/代码
make                       # 编译 conv_pthread 和 conv_omp

./conv_pthread <input.bmp> <output.bmp> [threads] [--dump-full]
./conv_omp     <input.bmp> <output.bmp> [threads] [--dump-full]

make check                 # Pure_white.bmp 自测并过教师验卷器
make bench                 # 1/2/4 核 benchmark + 加速比/效率汇总（结果存 ../结果/）
make official-run          # 2026-11-28 测评口径：plain gcc -O2 + 首次运行 + 教师验卷
```

## 基准数据（timg.bmp 4096×2304）

> **2026-11-28 测评标准更新**：老师公告改为“最朴素 O2 编译 + 固定输入输出 + 统一编译运行指令 + 只记录第一次测评”。
> 下表 8.6/8.7 ms 是 `-O3 -march=native`、多次取最优的**优化分析口径**，**不是**新标准下的成绩口径。
> 新标准实测：不绑核约 23–27 ms，绑 4 个物理 P 核约 19 ms（取较快版本），仍快于去年第一 0.059117 s。
> 完整自检、逐条对照与原始数据见 [作业/测评标准对齐.md](作业/测评标准对齐.md)。

**当前保留版（R22c，2026-11-23，`bench_round.sh R23final both 5` 最优 Total time）：**

| 版本 | T(1) ms | T(2) ms | T(4) ms | S(2) | S(4) | E(2) | E(4) |
|------|---------|---------|---------|------|------|------|------|
| Pthread | 23.481 | 14.507 | **8.661** | 1.6186 | 2.7111 | 0.8093 | 0.6778 |
| OpenMP  | 24.279 | 14.145 | **8.725** | 1.7164 | 2.7827 | 0.8582 | 0.6957 |

R22c 保持 R21 的大图数值与性能实现，只修复宽度 5–7 的边缘处理；此项不计性能收益。R23 双输出复用的两版本平均 T4 反而增加 0.83%，已回退；R22+R23 连续两次无正向收益，本轮停止。

本次官方 `run_bench.sh` 三次记录为 Pthread 23.552/14.373/9.398ms、OpenMP 24.398/14.277/9.498ms；与上表是不同运行窗口，详见各 `R23final` 原始日志。历史 R21 第二轮安静窗 T4 为约 8.4–8.9 / 9.2ms。不得把跨窗口最小值拼作优化收益。

第一轮最终版（R15 代码，官方 run_bench.sh，3 次取最优 Total time，2026-11-21）：

| 版本 | T(1) ms | T(2) ms | T(4) ms | S(2) | S(4) | E(2) | E(4) |
|------|---------|---------|---------|------|------|------|------|
| Pthread | 29.1 | 16.8 | **9.8** | 1.73 | 2.98 | 0.86 | 0.74 |
| OpenMP  | 28.8 | 16.7 | **10.4** | 1.72 | 2.77 | 0.86 | 0.69 |

（baseline：Pthread 293.2/162.8/95.5，OpenMP 290.5/160.3/93.9 —— **4 核提速 ~11× / ~10×**。）

测试环境：Ubuntu 22.04，Intel i9-13900K，gcc 11.4（`-O3 -march=native -funroll-loops -ffp-contract=fast`），
taskset 按物理 P 核限核（0,2 / 0,2,4,6）。OpenMP 线程数：显式传参 > `OMP_NUM_THREADS` > 默认。

## 主要优化点（按收益排序，详见优化日志）

1. **分离核两趟卷积**（-50%）：5×5 核最小二乘秩 1 拟合成 1D 核（残差 5.4e-9），
   水平趟 5 行环形缓冲驻 L2 + 垂直趟连续累加，每像素乘加 75→30
2. **mmap 零拷贝 I/O**（-42%）：输入输出均文件映射，卷积直读直写页缓存，免 pread/pwrite 拷贝
3. **vpass/hpass 单趟融合**（R14/R15，合计 -20%）：每 x 一次算完 5 tap（5 load + 4 FMA + 1 store），
   免中间平面 4 次"读-改-写"整平面往返；由 RDTSC 分段插桩定位（vpass 占纯计算 49%）
4. **I/O 与计算重叠**（-20%，后被 mmap 方案吸收）：条带级 pread/conv/pwrite 流水线
5. **float32 + 通道平面化**（合计 -22%）：8 宽 AVX2 向量、全连续访存（平面化布局后被 R19 取代）
6. **编译选项**（合计 -16%）：-O3 -march=native、-ffp-contract=fast（ISO 模式下 FMA 收缩默认关闭）
7. **BMP 免翻转**（-10%）：核垂直对称 ⇒ 翻转图直接卷积等价，省 28MB 翻转拷贝
8. **连续 BGR 单布局**（R19，第二轮 -6~7%/T1 -12~14%）：行=3w 字节、水平 tap 步长 3B 恒同通道，
   H 环形缓冲与源/输出同序，垂直趟直接舍入写字节——一次性删除解交织/vbuf/pack 交织三件布局转换
9. **posix_fallocate 输出预分配**（R21，第二轮 -6.5~8.4%）：ext4 块分配从 mmap 写缺页的
   并行路径挪到串行预备段

## 注意

- `附件/` 目录下的教师验卷程序与素材一律原样使用，不做任何修改（test.cpp MD5 已核对）。
- 编译产物不入库，已在 `.gitignore` 排除。
- 程序计时输出含逐线程卷积时间（作业要求）；mmap 版本的 I/O 以按需缺页形式
  计入各线程卷积时间，输出中有如实标注。
