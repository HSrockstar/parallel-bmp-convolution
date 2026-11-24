CC = gcc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off
LDLIBS = -lm -pthread
SOURCES = conv_lab.c kernels.c
HEADERS = kernels.h ../作业/代码/bmp_safe.h
.PHONY: all check sanitize clean native vector-report
all: conv_lab_pthread conv_lab_omp
conv_lab_pthread: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $@ $(LDLIBS)
conv_lab_omp: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -DUSE_OPENMP -fopenmp $(SOURCES) -o $@ $(LDLIBS)
check: all
	python3 test_methods.py
sanitize:
	$(MAKE) clean
	$(MAKE) CFLAGS='-O1 -g -std=c11 -Wall -Wextra -ffp-contract=off -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie -no-pie'
	ASAN_OPTIONS=$${ASAN_OPTIONS:-detect_leaks=1} python3 test_methods.py --quick
native:
	$(MAKE) clean
	$(MAKE) CFLAGS='-O3 -march=native -std=c11 -Wall -Wextra -ffp-contract=fast'
vector-report:
	$(CC) $(CFLAGS) -fopt-info-vec-all=vectorization.txt -c kernels.c -o kernels.o
clean:
	rm -f conv_lab_pthread conv_lab_omp test_buffers kernels.o vectorization.txt

test_buffers: test_buffers.c kernels.c kernels.h
	$(CC) $(CFLAGS) test_buffers.c kernels.c -o $@ $(LDLIBS)
