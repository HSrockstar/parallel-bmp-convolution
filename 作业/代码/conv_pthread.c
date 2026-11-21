/*
 * conv_pthread.c —— Pthread 并行实现 BMP 图像高斯卷积（优化版）
 *
 * 题目：读入 24 位 BMP 图像，用 5x5 高斯核（sigma=1.5，radius=2，已归一化）、
 *       步长 1、边缘常数 0 填充做卷积，输出模糊后的 BMP；并用 Pthread 按行分条带并行。
 *
 * 用法: ./conv_pthread <input.bmp> <output.bmp> [threads] [--dump-full]
 *   threads      并行线程数，默认 4；做 1/2/4 核对比时分别传 1 / 2 / 4
 *   --dump-full  将卷积后的完整像素矩阵写入 txt（大图会很大，默认只导出局部供验证）
 *
 * 输出文件：
 *   <output>.kernel.txt        5x5 卷积核
 *   <output>.matrix.txt        卷积后的像素矩阵（默认小图全量、大图导出左上角 20x20）
 *
 * 优化历史（详见 ../优化日志.md）：
 *   R1 编译选项 -O3 -march=native -funroll-loops
 *   R2 I/O 重构：BMP 原始行序免翻转读入（核垂直对称 ⇒ 翻转图直接卷积等价）+ 单次整块写
 *   R3 边界分离（后被 R4 结构吸收）
 *   R4 分离核两趟卷积：5x5 核最小二乘秩 1 拟合成 1D 核（残差 5.4e-9），
 *      水平趟 5 行环形缓冲驻 L2 + 垂直趟连续 double 累加，每像素乘加 75→30
 *   R5 I/O 与计算重叠：每线程 pread 自己的源条带（含上下 halo）→ 卷积 →
 *      pwrite 自己的输出条带，四线程 I/O/计算流水线化，消除串行整图读写
 *
 * 计时：clock_gettime(CLOCK_MONOTONIC) 计全程墙钟；逐线程打印
 *       读条带/卷积（并行段）/写条带分段耗时，并行段关键路径 = max(各线程卷积时间)。
 */
#define _POSIX_C_SOURCE 199309L   /* clock_gettime / CLOCK_MONOTONIC */
#define _DEFAULT_SOURCE          /* O_DIRECT 等扩展在本文件未用，open/pread 需 POSIX */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define KSIZE 5
#define KRAD  2                     /* (KSIZE-1)/2 */

/* 与教师验卷器 test.cpp 内置核逐位一致（作业统一核，不得更换）。
 * 卷积实际用其秩 1 分解（见 g_k1），此处保留用于导出核 txt。 */
static const double g_kernel[KSIZE][KSIZE] = {
    {0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881},
    {0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402},
    {0.03507270, 0.06831229, 0.08531173, 0.06831229, 0.03507270},
    {0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402},
    {0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881},
};

/* R4: 分离核。5x5 高斯核是（近似的）秩 1 矩阵：g2d[i][j] ≈ k1[i]*k1[j]。
 * k1 取对称特征分解的主特征向量（最小二乘最优拟合，残差 max 5.4e-9，
 * 最坏像素误差 ~1.3e-5 << 0.5 舍入阈值）。
 * 两趟：水平 H(r)[x] = Σ_kx k1[kx]·src[r][x+kx-2]（5 行环形缓冲驻 L2，
 * 源行只读一次）；垂直 out[y][x] = Σ_ky k1[ky]·H(y+ky-2)[x]（连续 double
 * 累加，可向量化）。每像素乘加 75→30 次。
 * 边界天然处理：源行越界 → H 为零行；x 方向逐 kx 取有效区间。 */
static const float g_k1[KSIZE] = {  /* R6: float32 —— 8 宽向量 */
    0.12007837208047906, 0.2338807408740243, 0.29208172229534735,
    0.2338807408740243, 0.12007837208047906
};

/* ---------------- 计时工具 ---------------- */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------- BMP 头解析 / 头写出 ---------------- */
/* R2/R5: 像素按文件原始行序就地处理（核垂直对称 ⇒ 免翻转），逐条带 pread/pwrite */
typedef struct {
    int w, h;                /* 图像宽、高 */
    int stride;              /* 每行字节数（4 字节对齐，含尾部填充） */
    int topdown;             /* 1: 行序自上而下（文件负高度）；0: 自下而上 */
    off_t data_off;          /* 文件内像素数据起始偏移 */
} BmpInfo;

static int bmp_parse_header(const uint8_t hdr[54], BmpInfo *b)
{
    if (hdr[0] != 'B' || hdr[1] != 'M') return -1;
    uint32_t off      = *(uint32_t *)&hdr[10];
    int32_t  w        = *(int32_t *)&hdr[18];
    int32_t  hraw     = *(int32_t *)&hdr[22];
    uint16_t bpp      = *(uint16_t *)&hdr[28];
    uint32_t comp     = *(uint32_t *)&hdr[30];
    if (bpp != 24 || comp != 0) return -1;
    b->w = w;
    b->h = hraw > 0 ? hraw : -hraw;
    b->stride = (w * 3 + 3) & ~3;
    b->topdown = (hraw < 0);
    b->data_off = (off_t)off;
    return 0;
}

static void bmp_make_header(const BmpInfo *b, uint8_t hdr[54])
{
    uint32_t datasize = (uint32_t)b->stride * b->h;
    memset(hdr, 0, 54);
    hdr[0] = 'B'; hdr[1] = 'M';
    *(uint32_t *)&hdr[2]  = 54 + datasize;         /* bfSize */
    *(uint32_t *)&hdr[10] = 54;                    /* bfOffBits */
    *(uint32_t *)&hdr[14] = 40;                    /* biSize = BITMAPINFOHEADER */
    *(int32_t  *)&hdr[18] = b->w;
    *(int32_t  *)&hdr[22] = b->topdown ? -b->h : b->h;  /* 与输入同向，保证不镜像 */
    *(uint16_t *)&hdr[26] = 1;                     /* biPlanes */
    *(uint16_t *)&hdr[28] = 24;                    /* biBitCount */
    *(uint32_t *)&hdr[34] = datasize;              /* biSizeImage */
}

/* ---------------- Pthread 并行卷积（条带级流水线） ---------------- */
typedef struct {
    int tid;
    int y0, y1;              /* 该线程负责的输出行范围 [y0, y1) */
    double conv_ms;          /* 该线程卷积（并行段，含输入/输出映射按需缺页）时间 */
} ThreadArg;

static BmpInfo g_bmp;                   /* 图像几何，线程只读 */
static int g_in_fd = -1, g_out_fd = -1;
static uint8_t *g_in_map;               /* R8: 输入文件 mmap 基址 */
static uint8_t *g_out_map;              /* R8: 输出文件 mmap 基址 */
static size_t g_out_mapsize;
static uint8_t *g_out;                  /* 输出像素区（= g_out_map + data_off，条带互不重叠） */

/* R9: 通道平面化 —— H 环形缓冲按 [槽][通道] 存放 3 个 w 长 float 平面。
 * 源行先解交织成 B/G/R 平面，水平/垂直趟全部连续 float 访问，
 * 8 宽向量无交织 shuffle 开销；尾趟 pack 回交错 uint8。 */
static void hpass3(float *dst3, const uint8_t *srow, uint8_t *scratch3, int w)
{
    memset(dst3, 0, 3 * (size_t)w * sizeof(float));
    if (!srow) return;                     /* 越界行 → 零平面（常数 0 填充） */
    for (int x = 0; x < w; x++) {          /* 解交织 BGR → 3 个 uint8 平面 */
        scratch3[x]         = srow[(size_t)x * 3 + 0];
        scratch3[w + x]     = srow[(size_t)x * 3 + 1];
        scratch3[2 * w + x] = srow[(size_t)x * 3 + 2];
    }
    for (int kx = 0; kx < KSIZE; kx++) {
        const float kv = g_k1[kx];
        /* 源索引 x+kx-2 ∈ [0,w) ⇒ x ∈ [max(0,2-kx), min(w, w+2-kx)) */
        int x0 = KRAD - kx;
        if (x0 < 0) x0 = 0;
        int x1 = w + KRAD - kx;
        if (x1 > w) x1 = w;
        for (int c = 0; c < 3; c++) {
            const uint8_t *s = scratch3 + (size_t)c * w + (kx - KRAD);
            float *d = dst3 + (size_t)c * w;
            for (int x = x0; x < x1; x++)
                d[x] += kv * (float)s[x];
        }
    }
}

static inline int slot5(int r) { return ((r % 5) + 5) % 5; }   /* 行号 → 环形槽位 */

/* 分离核卷积一个输出行条带 [y0,y1)。
 * src 为本线程条带缓冲首址，base 为其对应的全局行号；全局行 r 的源数据在
 * src + (r-base)*stride，r 超出 [base, base+nrows) 视为越界（零填充）。 */
static void conv_band(const uint8_t *src, int base, int nrows,
                      uint8_t *dst, int w, int stride, int y0, int y1)
{
    const size_t pw = (size_t)w;
    float *hb   = malloc(15 * pw * sizeof(float));   /* 5 槽 × 3 通道平面（~245KB, L2） */
    float *vbuf = malloc(3 * pw * sizeof(float));    /* 垂直累加 3 平面 */
    uint8_t *sp = malloc(3 * pw);                    /* 解交织源行（3 uint8 平面） */
    if (!hb || !vbuf || !sp) { free(hb); free(vbuf); free(sp); return; }

    /* 初始化窗口行 [y0-2, y0+3) */
    for (int i = 0; i < 5; i++) {
        int r = y0 + i - KRAD;
        hpass3(hb + (size_t)slot5(r) * 3 * pw,
               (r >= base && r < base + nrows) ? src + (size_t)(r - base) * stride : NULL,
               sp, w);
    }
    for (int y = y0; y < y1; y++) {
        /* 垂直趟：v[c] = Σ_ky k1[ky]·H(y+ky-2)[c]，每通道连续累加；
         * 首趟直接写，替代 memset */
        for (int c = 0; c < 3; c++) {
            float *v = vbuf + c * pw;
            const float *h0 = hb + ((size_t)slot5(y - KRAD) * 3 + c) * pw;
            const float kv0 = g_k1[0];
            for (size_t x = 0; x < pw; x++)
                v[x] = kv0 * h0[x];
            for (int ky = 1; ky < KSIZE; ky++) {
                const float *hr = hb + ((size_t)slot5(y + ky - KRAD) * 3 + c) * pw;
                const float kv = g_k1[ky];
                for (size_t x = 0; x < pw; x++)
                    v[x] += kv * hr[x];
            }
        }
        /* pack：3 平面 → 舍入饱和 → 交错 uint8 写回 */
        uint8_t *d = dst + (size_t)y * stride;
        for (int x = 0; x < w; x++) {
            int b = (int)(vbuf[x] + 0.5f);
            int g = (int)(vbuf[pw + x] + 0.5f);
            int r = (int)(vbuf[2 * pw + x] + 0.5f);
            d[x * 3 + 0] = b < 0 ? 0 : (b > 255 ? 255 : b);
            d[x * 3 + 1] = g < 0 ? 0 : (g > 255 ? 255 : g);
            d[x * 3 + 2] = r < 0 ? 0 : (r > 255 ? 255 : r);
        }
        /* 下一行滑入窗口：H(y+3) */
        int rr = y + KRAD + 1;
        hpass3(hb + (size_t)slot5(rr) * 3 * pw,
               (rr < base + nrows) ? src + (size_t)(rr - base) * stride : NULL,
               sp, w);
    }
    free(hb); free(vbuf); free(sp);
}

static void *thread_main(void *p)
{
    ThreadArg *arg = (ThreadArg *)p;
    const int w = g_bmp.w, h = g_bmp.h, stride = g_bmp.stride;
    int y0 = arg->y0, y1 = arg->y1;

    /* R8: 源条带直接取输入文件映射（按需缺页，免 pread 拷贝与条带缓冲）；
     * 输出直接写输出文件映射（免 pwrite 拷贝，脏页由内核在退出时回写） */
    int y0c = y0 - KRAD < 0 ? 0 : y0 - KRAD;      /* 含上 halo 的读取范围 */
    int y1c = y1 + KRAD > h ? h : y1 + KRAD;
    const uint8_t *src = g_in_map + g_bmp.data_off + (size_t)y0c * stride;

    double t0 = now_sec();
    conv_band(src, y0c, y1c - y0c, g_out, w, stride, y0, y1);
    arg->conv_ms = (now_sec() - t0) * 1000.0;
    return NULL;
}

/* ---------------- 结果导出（卷积核 + 像素矩阵 txt） ---------------- */
static void dump_kernel(const char *base)
{
    char path[1024];
    snprintf(path, sizeof path, "%s.kernel.txt", base);
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "cannot create %s\n", path); return; }
    fprintf(fp, "# 5x5 Gaussian kernel, sigma=1.5, radius=2, stride=1, normalized\n");
    for (int i = 0; i < KSIZE; i++) {
        for (int j = 0; j < KSIZE; j++)
            fprintf(fp, "%s%.8f", j ? " " : "", g_kernel[i][j]);
        fputc('\n', fp);
    }
    fclose(fp);
    printf("kernel txt      : %s\n", path);
}

static void dump_matrix(const char *base, const uint8_t *data,
                        const BmpInfo *b, int full)
{
    char path[1024];
    snprintf(path, sizeof path, "%s.matrix.txt", base);
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "cannot create %s\n", path); return; }

    int rows = full ? b->h : (b->h < 20 ? b->h : 20);
    int cols = full ? b->w : (b->w < 20 ? b->w : 20);
    fprintf(fp, "# Blurred pixel matrix (B G R per pixel), image %dx%d\n", b->w, b->h);
    if (!full)
        fprintf(fp, "# (large image: showing top-left %dx%d block only, "
                    "run with --dump-full for the whole matrix)\n", rows, cols);
    for (int y = 0; y < rows; y++) {
        int r = b->topdown ? y : b->h - 1 - y;   /* 逻辑行 → 缓冲行号 */
        fprintf(fp, "y=%d\n", y);
        for (int x = 0; x < cols; x++) {
            const uint8_t *p = data + (size_t)r * b->stride + x * 3;
            fprintf(fp, "  x=%-5d B=%-3d G=%-3d R=%-3d\n", x, p[0], p[1], p[2]);
        }
    }
    fclose(fp);
    printf("matrix txt      : %s (%s)\n", path,
           full ? "full matrix" : "top-left 20x20 block");
}

/* ---------------- main ---------------- */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <input.bmp> <output.bmp> [threads] [--dump-full]\n",
                argv[0]);
        return 1;
    }
    const char *in_path  = argv[1];
    const char *out_path = argv[2];
    int nthreads = 4;
    int dump_full = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--dump-full") == 0) dump_full = 1;
        else nthreads = atoi(argv[i]);
    }
    if (nthreads < 1) nthreads = 1;

    double t_all0 = now_sec();

    /* 1. 解析 BMP 头 + mmap 输入文件（像素按需缺页，免整图读拷贝） */
    g_in_fd = open(in_path, O_RDONLY);
    if (g_in_fd < 0) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }
    uint8_t hdr[54];
    if (read(g_in_fd, hdr, 54) != 54 || bmp_parse_header(hdr, &g_bmp) != 0) {
        fprintf(stderr, "%s: not an uncompressed 24-bit BMP (BITMAPINFOHEADER)\n", in_path);
        close(g_in_fd);
        return 1;
    }
    const int w = g_bmp.w, h = g_bmp.h, stride = g_bmp.stride;
    off_t in_size = lseek(g_in_fd, 0, SEEK_END);
    g_in_map = mmap(NULL, (size_t)in_size, PROT_READ, MAP_SHARED, g_in_fd, 0);
    if (g_in_map == MAP_FAILED) { fprintf(stderr, "mmap input failed\n"); return 1; }

    /* 2. 创建输出文件：ftruncate 定长 + 写头 + MAP_SHARED 映射，
     *    卷积结果直接写入映射（免 malloc 输出缓冲与 pwrite 拷贝） */
    uint8_t ohdr[54];
    bmp_make_header(&g_bmp, ohdr);
    g_out_fd = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (g_out_fd < 0) { fprintf(stderr, "cannot create %s\n", out_path); return 1; }
    g_out_mapsize = (size_t)g_bmp.data_off + (size_t)stride * h;
    if (ftruncate(g_out_fd, (off_t)g_out_mapsize) != 0) {
        fprintf(stderr, "ftruncate failed\n"); return 1;
    }
    if (pwrite(g_out_fd, ohdr, 54, 0) != 54) { fprintf(stderr, "write header failed\n"); return 1; }
    g_out_map = mmap(NULL, g_out_mapsize, PROT_READ | PROT_WRITE, MAP_SHARED, g_out_fd, 0);
    if (g_out_map == MAP_FAILED) { fprintf(stderr, "mmap output failed\n"); return 1; }
    g_out = g_out_map + g_bmp.data_off;
    {   /* 头部与像素区之间的间隙及行尾填充字节清零（本实验图片均为 0，通用性保险） */
        int padlen = stride - w * 3;
        if (g_bmp.data_off > 54)
            memset(g_out_map + 54, 0, (size_t)g_bmp.data_off - 54);
        if (padlen)
            for (int y = 0; y < h; y++)
                memset(g_out + (size_t)y * stride + w * 3, 0, (size_t)padlen);
    }

    /* 3. 并行：各线程在自己的条带上卷积（源/目的均为文件映射） */
    if (nthreads > h) nthreads = h;
    ThreadArg *args = calloc((size_t)nthreads, sizeof(ThreadArg));
    pthread_t *tids  = calloc((size_t)nthreads, sizeof(pthread_t));
    int chunk = h / nthreads, rem = h % nthreads;

    for (int i = 0; i < nthreads; i++) {
        args[i].tid = i;
        args[i].y0  = i * chunk + (i < rem ? i : rem);
        args[i].y1  = args[i].y0 + chunk + (i < rem ? 1 : 0);
        pthread_create(&tids[i], NULL, thread_main, &args[i]);
    }
    for (int i = 0; i < nthreads; i++)
        pthread_join(tids[i], NULL);

    close(g_in_fd);
    close(g_out_fd);
    double total_ms = (now_sec() - t_all0) * 1000.0;

    /* 4. 导出卷积核 + 像素矩阵 txt */
    dump_kernel(out_path);
    dump_matrix(out_path, g_out, &g_bmp, dump_full);

    /* 5. 计时结果 */
    printf("================ Pthread BMP Gaussian Convolution ================\n");
    printf("Input           : %s (%d x %d, 24bpp, stride=%d)\n", in_path, w, h, stride);
    printf("Output BMP      : %s\n", out_path);
    printf("Threads         : %d\n", nthreads);
    printf("---- per-thread conv time (R8: I/O 零拷贝 mmap，缺页计入 conv) ----\n");
    double max_conv = 0, sum_conv = 0;
    for (int i = 0; i < nthreads; i++) {
        printf("  thread %-2d rows [%5d,%5d)  conv = %8.3f ms\n",
               args[i].tid, args[i].y0, args[i].y1, args[i].conv_ms);
        if (args[i].conv_ms > max_conv) max_conv = args[i].conv_ms;
        sum_conv += args[i].conv_ms;
    }
    printf("---- timing ----\n");
    printf("I/O    : mmap 零拷贝（读=按需缺页+页缓存命中，写=直接写脏页，无显式读写）\n");
    printf("Conv   : (parallel region, p=%d)  %8.3f ms   (critical path = max thread)\n", nthreads, max_conv);
    printf("Conv   : sum/thread              %8.3f ms   (sum of per-thread times)\n", sum_conv);
    printf("Total time      : %10.3f ms\n", total_ms);

    free(args); free(tids);
    munmap(g_out_map, g_out_mapsize);
    munmap(g_in_map, (size_t)in_size);
    return 0;
}
