# 默认入口使用老师记录的朴素 O2；优化分析构建必须显式选择。
CC := gcc
CFLAGS := -O2
TESTDIR := ../../附件/采用Pthread对BMP图片进行卷积作业附件/测试程序/测试程序
RESULT := ../结果
CHECKER := ../../环境验证/test

.PHONY: all official official-run bench check run-small optimized clean
all: conv_pthread conv_omp

conv_pthread: conv_pthread.c bmp_safe.h conv_core.h conv_fused.h matrix_text.h conv_affinity.h Makefile
	$(CC) $(CFLAGS) -pthread $< -o $@

conv_omp: conv_omp.c bmp_safe.h conv_core.h conv_fused.h matrix_text.h conv_affinity.h Makefile
	$(CC) $(CFLAGS) -fopenmp $< -o $@

# 保留旧的 _o2 文件名，避免历史命令误用过期产物。
official: all conv_pthread_o2 conv_omp_o2

conv_pthread_o2: conv_pthread.c bmp_safe.h conv_core.h conv_fused.h matrix_text.h conv_affinity.h Makefile
	gcc -O2 -pthread $< -o $@

conv_omp_o2: conv_omp.c bmp_safe.h conv_core.h conv_fused.h matrix_text.h conv_affinity.h Makefile
	gcc -O2 -fopenmp $< -o $@

official-run: official
	bash run_official.sh

bench: official-run

run-small: all
	mkdir -p $(RESULT)
	./conv_pthread $(TESTDIR)/Pure_white.bmp $(RESULT)/Pure_white_blur_pthread.bmp 4
	./conv_omp $(TESTDIR)/Pure_white.bmp $(RESULT)/Pure_white_blur_omp.bmp 4

check: run-small
	$(CHECKER) $(TESTDIR)/Pure_white.bmp $(RESULT)/Pure_white_blur_pthread.bmp > $(RESULT)/check_pthread.txt
	cat $(RESULT)/check_pthread.txt
	! grep -q '\[FAIL\]' $(RESULT)/check_pthread.txt
	test "$$(grep -c '^\[PASS\]' $(RESULT)/check_pthread.txt)" -eq 2
	$(CHECKER) $(TESTDIR)/Pure_white.bmp $(RESULT)/Pure_white_blur_omp.bmp > $(RESULT)/check_omp.txt
	cat $(RESULT)/check_omp.txt
	! grep -q '\[FAIL\]' $(RESULT)/check_omp.txt
	test "$$(grep -c '^\[PASS\]' $(RESULT)/check_omp.txt)" -eq 2

# 仅供历史优化分析，不作为 O2 首次测评成绩。
optimized: conv_pthread_opt conv_omp_opt

conv_pthread_opt: conv_pthread.c bmp_safe.h conv_core.h conv_fused.h matrix_text.h conv_affinity.h Makefile
	gcc-11 -O3 -march=native -funroll-loops -ffp-contract=fast -Wall -std=c11 -pthread $< -o $@

conv_omp_opt: conv_omp.c bmp_safe.h conv_core.h conv_fused.h matrix_text.h conv_affinity.h Makefile
	gcc-11 -O3 -march=native -funroll-loops -ffp-contract=fast -Wall -std=c11 -fopenmp $< -o $@

clean:
	rm -f conv_pthread conv_omp conv_pthread_o2 conv_omp_o2 conv_pthread_opt conv_omp_opt
