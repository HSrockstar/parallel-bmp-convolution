# 卷积算法实验

完整说明见 [OPTIMIZATION_RESEARCH.md](OPTIMIZATION_RESEARCH.md)，包含方法、数学条件、资料来源、工程清单、测试和性能数据。

```bash
make -C research
research/conv_lab_pthread --list
research/conv_lab_pthread input.bmp output.bmp --method avx2 --threads 4
research/conv_lab_omp input.bmp output.bmp --method symmetric --threads 4
python3 research/test_methods.py
```

C 程序需要 Linux、C11、Pthread 和 OpenMP。基础测试只需要 Python 标准库。`alternatives.py` 需要 NumPy/SciPy，`benchmark.py` 与 `autotune.py` 需要 NumPy。GPU 需要 CUDA 设备及兼容的 CuPy，目前未在本次环境实测。

| 文件 | 内容 |
|---|---|
| `kernels.c` / `kernels.h` | 17 种 CPU 方法 |
| `conv_lab.c` | BMP I/O、Pthread 池、OpenMP、计时和选项 |
| `alternatives.py` | 17 种库对照与扩展方法 |
| `cuda_kernels.cu` / `gpu.py` | 3 种 CUDA 方法和硬件自测 |
| `test_methods.py` | 独立精确整数参考与文件级回归 |
| `test_buffers.c` | 精确长度缓冲区越界检查 |
| `test_alternatives.py` | 数值对照和递归滤波性质验证 |
| `benchmark.py` | 每轮随机交错、验证输出、保存原始时间 |
| `autotune.py` | 经正确性筛选后选择配置 |
| `results/` | 本次实际测量与验证记录 |

默认 `make` 使用可移植的 x86 构建和运行时 ISA 检测。`make native` 使用当前 CPU 指令集，只在目标机器上编译使用。`--sync` 等待输出文件同步；默认完成页缓存写入。文本导出由历史程序提供，实验程序输出 JSON。
