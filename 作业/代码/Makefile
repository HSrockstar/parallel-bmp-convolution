# BMP 图像卷积作业 —— Pthread / OpenMP baseline
#
# make            编译两个版本
# make run-small  用 Pure_white.bmp 自测（Pthread 4 线程）
# make check      教师验卷器对比（Pure_white）
# make bench      1/2/4 核 benchmark（timg.bmp，结果存 ../结果/）
# make clean

# 基准整备：钉住 gcc-11（=11.4.0，优化日志协议记录的编译器；系统默认 gcc 已漂移到 10.5，
# 换编译器会破坏 R0-R16 数据可比性，故显式指定）
CC      := gcc-11
# R1: -O2 → -O3 -march=native -funroll-loops（开启自动向量化/FMA/激进调度；
#     -march=native 在编译机上自适应本机指令集，重编译无移植问题）
# R7: -ffp-contract=fast —— ISO C 模式(-std=c11)下 GCC 默认关闭 FMA 收缩，
#     显式打开后 a*b+c 编译为 vfmadd（融合乘加），乘加型卷积循环指令数减半
CFLAGS  := -O3 -march=native -funroll-loops -ffp-contract=fast -Wall -std=c11
LD_PTHREAD := -lpthread

TESTDIR := ../../附件/采用Pthread对BMP图片进行卷积作业附件/测试程序/测试程序
RESULT  := ../结果
CHECKER := ../../环境验证/test

all: conv_pthread conv_omp

conv_pthread: conv_pthread.c bmp_safe.h
	$(CC) $(CFLAGS) $< -o $@ $(LD_PTHREAD)

conv_omp: conv_omp.c bmp_safe.h
	$(CC) $(CFLAGS) -fopenmp $< -o $@

# 自测：全白图 + 4 线程
run-small: all
	mkdir -p $(RESULT)
	./conv_pthread $(TESTDIR)/Pure_white.bmp $(RESULT)/Pure_white_blur_pthread.bmp 4
	./conv_omp     $(TESTDIR)/Pure_white.bmp $(RESULT)/Pure_white_blur_omp.bmp     4

# 教师验卷器（OpenCV）：原图 vs 我们的模糊图
check: run-small
	$(CHECKER) $(TESTDIR)/Pure_white.bmp $(RESULT)/Pure_white_blur_pthread.bmp
	$(CHECKER) $(TESTDIR)/Pure_white.bmp $(RESULT)/Pure_white_blur_omp.bmp

# 正式：timg.bmp + 1/2/4 核加速比（详细用 bash run_bench.sh）
bench: all
	bash run_bench.sh

clean:
	rm -f conv_pthread conv_omp
