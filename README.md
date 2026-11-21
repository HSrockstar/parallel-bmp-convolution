# Parallel BMP Convolution（Pthread / OpenMP 图像并行卷积）

并行计算课程作业：从 BMP 图片读取像素，用 **5×5 高斯核（σ=1.5）、stride=1、边缘常数 0 填充**
进行模糊卷积，分别以 **Pthread** 与 **OpenMP** 两种方式并行实现，并与教师验卷器
（OpenCV `filter2D(..., BORDER_CONSTANT)`）比对，要求相似度 > 99%（最终实现 MSE=0.00、
相似度 100.00%，与 double 基准逐像素仅 35/28.3M 字节的 ±1 平局差异）。

完整作业要求见 [作业要求.md](作业要求.md)；**优化过程全记录见 [作业/优化日志.md](作业/优化日志.md)**
（R0–R13 共 14 轮迭代，含每轮原理/改动/验证/数据/结论与三次无收益探索的回退记录）。

## 目录结构

```
图片卷积/
├── 作业要求.md       # 作业统一要求说明
├── 附件/             # 课程附件原样解压（教师验卷器 + 测试图，禁止修改）
├── 环境验证/         # test.cpp 编译运行验证（验卷器副本，MD5 与附件一致）
├── 作业/
│   ├── 代码/         # conv_pthread.c / conv_omp.c / Makefile / run_bench.sh
│   │                 #   + bench_round.sh / check_round.sh（优化迭代用统一基准脚本）
│   ├── 结果/         # 卷积结果图、kernel/matrix 导出、benchmark 记录、最终验卷记录
│   └── 优化日志/     # 每轮原始基准输出 + 验卷器输出 + 参考图（报告数据源）
└── 作业/优化日志.md   # 优化迭代主日志（R0-R13）
```

## 快速开始

```bash
cd 作业/代码
make                       # 编译 conv_pthread 和 conv_omp

./conv_pthread <input.bmp> <output.bmp> [threads] [--dump-full]
./conv_omp     <input.bmp> <output.bmp> [threads] [--dump-full]

make check                 # Pure_white.bmp 自测并过教师验卷器
make bench                 # 1/2/4 核 benchmark + 加速比/效率汇总（结果存 ../结果/）
```

## 基准数据（timg.bmp 4096×2304）

**最终版（官方 run_bench.sh，3 次取最优 Total time）：**

| 版本 | T(1) ms | T(2) ms | T(4) ms | S(2) | S(4) | E(2) | E(4) |
|------|---------|---------|---------|------|------|------|------|
| Pthread | 38.0 | 21.6 | **12.0** | 1.76 | 3.17 | 0.88 | 0.79 |
| OpenMP  | 39.1 | 21.7 | **12.1** | 1.80 | 3.23 | 0.90 | 0.81 |

（baseline：Pthread 293.2/162.8/95.5，OpenMP 290.5/160.3/93.9 —— **4 核提速 7.8 倍**。）

测试环境：Ubuntu 22.04，Intel i9-13900K，gcc 11.4（`-O3 -march=native -funroll-loops -ffp-contract=fast`），
taskset 按物理 P 核限核（0,2 / 0,2,4,6）。

## 主要优化点（按收益排序，详见优化日志）

1. **分离核两趟卷积**（-50%）：5×5 核最小二乘秩 1 拟合成 1D 核（残差 5.4e-9），
   水平趟 5 行环形缓冲驻 L2 + 垂直趟连续累加，每像素乘加 75→30
2. **mmap 零拷贝 I/O**（-42%）：输入输出均文件映射，卷积直读直写页缓存，免 pread/pwrite 拷贝
3. **I/O 与计算重叠**（-20%，后被 mmap 方案吸收）：条带级 pread/conv/pwrite 流水线
4. **float32 + 通道平面化**（合计 -22%）：8 宽 AVX2 向量、全连续访存
5. **编译选项**（合计 -16%）：-O3 -march=native、-ffp-contract=fast（ISO 模式下 FMA 收缩默认关闭）
6. **BMP 免翻转**（-10%）：核垂直对称 ⇒ 翻转图直接卷积等价，省 28MB 翻转拷贝

## 注意

- `附件/` 目录下的教师验卷程序与素材一律原样使用，不做任何修改（test.cpp MD5 已核对）。
- 编译产物不入库，已在 `.gitignore` 排除。
- 程序计时输出含逐线程卷积时间（作业要求）；mmap 版本的 I/O 以按需缺页形式
  计入各线程卷积时间，输出中有如实标注。
