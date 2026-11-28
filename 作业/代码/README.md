# BMP 图像卷积代码与 O2 测评入口

本目录提供 Pthread 和 OpenMP 两个 BMP 高斯卷积实现。默认编译使用系统 GCC 的朴素 `-O2`，默认运行生成 BMP、完整卷积核文本和完整像素矩阵文本。

老师尚未给出完整编译及运行命令。这里的命令是项目统一复现入口；默认四线程对应项目记录的四核评分要求，不绑核只是普通运行基线，并非老师禁止绑核。

## 默认编译和运行

```bash
cd 作业/代码
make
./conv_pthread <input.bmp> <output.bmp>
./conv_omp     <input.bmp> <output.bmp>
```

`make` 实际执行 `gcc -O2 -pthread conv_pthread.c -o conv_pthread` 和 `gcc -O2 -fopenmp conv_omp.c -o conv_omp`。Makefile 作为产物依赖，切换本轮 Makefile 后会重建此前的 O3 产物。`make official` 还会更新兼容旧命令的 `conv_pthread_o2`、`conv_omp_o2`。

两版默认 4 线程；显式第三个参数可以选择 1、2、4 等线程数。OpenMP 仍支持 `显式参数 > OMP_NUM_THREADS > 默认 4`。测评脚本会清理继承的 OpenMP 绑核和线程限制设置，并确认程序打印的实际线程数。

## 默认输出

每次运行生成 `<output.bmp>`、`<output.bmp>.kernel.txt`、`<output.bmp>.matrix.txt`。矩阵默认覆盖全图，以逻辑上从上到下的顺序保存；每个数据行对应一行图像，共有 `width × 3` 个整数，依次为 B、G、R。以 `#` 开始的两行为格式说明。

`timg.bmp` 的矩阵包含 2304 行、28,311,552 个颜色通道值，文本约 100 MB。大图使用十进制查找表和分块写入；`TXT export` 记录卷积核和矩阵文本导出耗时。程序检查文本写入及关闭错误，导出失败时返回非零，不会把缺失或未完成的文本当作运行成功。

`--dump-full` 保留为兼容选项。`--dump-preview` 仅用于优化分析，输出最多 20×20 的局部矩阵，不属于默认完整输出。

## O2 首次测评

```bash
make bench
# 等价入口
make official-run
bash run_bench.sh
# 可指定版本、输入和一个尚不存在的结果目录
bash run_official.sh both <input.bmp> <new-result-directory>
```

脚本强制以 GCC O2 重新编译，按相同输入与输出路径先执行两版默认四线程，再各执行一次一线程、二线程。没有预跑，也不从多次运行中挑最佳值。结果写入新目录，不覆盖旧轮次；其中保留编译记录、环境、逐线程时间、教师验卷原文、两版四线程完整结果，以及 `summary.txt`、`summary.json` 中的总时间和卷积段加速比、效率。

四线程组调用方式就是 `./conv_pthread input.bmp output.bmp` 或 `./conv_omp input.bmp output.bmp`，不加预览选项。教师验卷器从附件原始源码编译到新结果目录。脚本同时检查退出状态及两项 PASS；出现 FAIL 或缺少 PASS 时立即中止。

## 绑核对照

默认由系统调度。不绑核的大图按 64 行动态分配，绑核或处理小图时静态划分条带。可用 `CONV_CPUS` 在程序内按线程顺序绑定，设置和校验均在 `Total time` 内。本机 0、2、4、6 是四个不同的物理 P 核。线程数小于列表长度时使用列表前缀。程序拒绝重复、受限或不足的 CPU 列表。

```bash
CONV_CPUS=0,2,4,6 ./conv_pthread input.bmp output.bmp
CONV_CPUS=0,2,4,6 ./conv_omp input.bmp output.bmp
```

脚本记录 CPU 允许列表及 `CONV_CPUS`。原有的外层进程限定方式也可使用：

```bash
taskset -c 0,2,4,6 bash run_official.sh both <input.bmp> <new-result-directory>
```

外层 CPU 允许列表不会被脚本清除。这组结果应标为“物理 P 核限定”，与普通运行分别比较。本机 CPU 0/1 是同一物理核的两个逻辑线程，不能把 CPU 0–3 称为四个物理核。

## 计时和数值范围

`Total time` 从 main 开头到输入访问、卷积、输出文件映射写入和 close 后，不含最终文本导出、打印、munmap，也不等待磁盘持久化；`Wall time (full)` 另外包含完整文本导出和 munmap。外部进程墙钟还包括程序启动及退出，因此不能把这三个指标混称为同一种总耗时。

`Conv` 是最慢工作线程的卷积耗时，包含线程内分配及映射缺页，不包含线程创建与回收的全部开销。汇总同时计算 `S_total/E_total` 和 `S_conv/E_conv`。

x86-64 运行时检查 AVX2 与操作系统 YMM 支持。满足条件时以手写 SIMD 对 32 行分块，将垂直对称系数的三个乘积分送到五个输出位置，并保持每个像素原来的 float32 加法次序。较窄的图或不支持 AVX2 的 CPU 沿用原有的标量五行缓冲。两版共用同一卷积和文本编码实现。近似分离与原二维核有极小数值差异，但已通过教师大于 99% 的相似度阈值；`MSE=0.00` 和 `100.00%` 是两位小数显示，不表示逐像素完全相同。没有将“必须逐系数计算原核”视为已确认的新增要求。

## 验证与进一步优化

```bash
make check
python3 -I check_edges.py --pthread ./conv_pthread --omp ./conv_omp --work-dir <new-edge-directory>
```

`make check` 也检查教师验卷文本，避免相似度失败但验卷器退出码为 0 时出现假通过。现有 `check_round.sh`、`check_small.sh` 仍可使用。`bench_round.sh`、`bench_ab.sh` 是多次采样的优化分析工具，数据不得替代 O2 首次测评成绩；当前默认构建已改为 O2，不可把新旧构建结果直接归因于算法收益。

`make optimized` 显式生成 `conv_pthread_opt`、`conv_omp_opt`，采用历史 GCC 11、O3/native 等选项，仅用于性能分析，不覆盖默认 O2 程序。历史优化过程、回退与原始数据保留在 `../优化日志.md` 和 `../优化日志/`。
