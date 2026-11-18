# Parallel BMP Convolution（Pthread / OpenMP 图像并行卷积）

并行计算课程作业：从 BMP 图片读取像素，用 **5×5 高斯核（σ=1.5）、stride=1、边缘常数 0 填充**
进行模糊卷积，分别以 **Pthread** 与 **OpenMP** 两种方式并行实现，并与教师验卷器
（OpenCV `filter2D(..., BORDER_CONSTANT)`）比对，要求相似度 > 99%（本实现 MSE=0，逐像素一致）。

完整作业要求见 [作业要求.md](作业要求.md)。

## 目录结构

```
图片卷积/
├── 作业要求.md       # 作业统一要求说明
├── 附件/             # 课程附件原样解压（教师验卷器 + 测试图，禁止修改）
├── 环境验证/         # test.cpp 编译运行验证（验卷器副本）
└── 作业/
    ├── 代码/         # conv_pthread.c / conv_omp.c / Makefile / run_bench.sh
    └── 结果/         # 卷积结果图、kernel/matrix 导出、benchmark 记录
```

## 快速开始

```bash
cd 作业/代码
make                       # 编译 conv_pthread 和 conv_omp（gcc -O2）

./conv_pthread <input.bmp> <output.bmp> [threads] [--dump-full]
./conv_omp     <input.bmp> <output.bmp> [threads] [--dump-full]

make check                 # Pure_white.bmp 自测并过教师验卷器
make bench                 # 1/2/4 核 benchmark + 加速比/效率汇总
```

## 基准数据（timg.bmp 4096×2304，3 次取最优）

| 版本 | T(1) ms | T(2) ms | T(4) ms | S(2) | S(4) | E(2) | E(4) |
|------|---------|---------|---------|------|------|------|------|
| Pthread | 297.2 | 162.5 | 95.0 | 1.83 | 3.13 | 0.91 | 0.78 |
| OpenMP  | 290.6 | 160.4 | 92.8 | 1.81 | 3.13 | 0.91 | 0.78 |

测试环境：Ubuntu 22.04，Intel i9-13900K（taskset 按物理核限核）。
详细实现说明见 [作业/代码/README.md](作业/代码/README.md)。

## 注意

- `附件/` 目录下的教师验卷程序与素材一律原样使用，不做任何修改。
- 编译产物（`conv_pthread`、`conv_omp`、`环境验证/test`）不入库，已在 `.gitignore` 排除。
