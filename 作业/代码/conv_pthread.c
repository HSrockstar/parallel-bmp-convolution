/*
 * conv_pthread.c —— Pthread 并行实现 BMP 图像高斯卷积（baseline 版本）
 *
 * 题目：读入 24 位 BMP 图像，用 5x5 高斯核（sigma=1.5，radius=2，已归一化）、
 *       步长 1、边缘常数 0 填充做卷积，输出模糊后的 BMP；并用 Pthread 按行分块并行。
 *
 * 用法: ./conv_pthread <input.bmp> <output.bmp> [threads] [--dump-full]
 *   threads      并行线程数，默认 4；做 1/2/4 核对比时分别传 1 / 2 / 4
 *   --dump-full  将卷积后的完整像素矩阵写入 txt（大图会很大，默认只导出局部供验证）
 *
 * 输出文件：
 *   <output>.kernel.txt        5x5 卷积核
 *   <output>.matrix.txt        卷积后的像素矩阵（默认小图全量、大图导出左上角 20x20）
 *
 * 计时：gettimeofday / clock_gettime 计读图、卷积（并行段）、写图时间，
 *       并逐线程打印各核运算时间。
 */
#define _POSIX_C_SOURCE 199309L   /* clock_gettime / CLOCK_MONOTONIC */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

#define KSIZE 5
#define KRAD  2                     /* (KSIZE-1)/2 */

/* 与教师验卷器 test.cpp 内置核逐位一致（作业统一核，不得更换） */
static const double g_kernel[KSIZE][KSIZE] = {
    {0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881},
    {0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402},
    {0.03507270, 0.06831229, 0.08531173, 0.06831229, 0.03507270},
    {0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402},
    {0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881},
};

/* ---------------- 计时工具 ---------------- */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------- BMP 读写 ---------------- */
/* R2 优化：像素数据按文件原始行序直接读入最终缓冲（免 28MB 翻转拷贝），写图单次 fwrite。
 * 卷积核垂直方向对称 ⇒ 对"上下翻转的图"直接卷积 = 卷积结果上下翻转，
 * 因此无论缓冲是自下而上还是自上而下行序，用同一个卷积公式原地算即可，结果等价。 */
typedef struct {
    int w, h;                /* 图像宽、高 */
    int stride;              /* 每行字节数（4 字节对齐，含尾部填充） */
    int topdown;             /* 1: data 行序自上而下（文件负高度）；0: 自下而上 */
    uint8_t *data;           /* 像素，BGR 交错，行 r 首地址 = data + r*stride */
} Image;

static int bmp_read(const char *path, Image *img)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return -1; }

    uint8_t hdr[54];
    if (fread(hdr, 1, 54, fp) != 54 || hdr[0] != 'B' || hdr[1] != 'M') {
        fprintf(stderr, "%s: not a BMP24 (BITMAPINFOHEADER) file\n", path);
        fclose(fp);
        return -1;
    }
    uint32_t off      = *(uint32_t *)&hdr[10];
    int32_t  w        = *(int32_t *)&hdr[18];
    int32_t  hraw     = *(int32_t *)&hdr[22];       /* >0 自下而上；<0 自上而下 */
    uint16_t bpp      = *(uint16_t *)&hdr[28];
    uint32_t comp     = *(uint32_t *)&hdr[30];
    if (bpp != 24 || comp != 0) {
        fprintf(stderr, "%s: expect uncompressed 24-bit BMP, got bpp=%u comp=%u\n",
                path, bpp, comp);
        fclose(fp);
        return -1;
    }

    int h = hraw > 0 ? hraw : -hraw;
    int stride = (w * 3 + 3) & ~3;
    img->w = w; img->h = h; img->stride = stride;
    img->topdown = (hraw < 0);
    img->data = malloc((size_t)stride * h);        /* 一步到位，无中间缓冲 */
    if (!img->data) { fclose(fp); return -1; }
    fseek(fp, off, SEEK_SET);
    if (fread(img->data, 1, (size_t)stride * h, fp) != (size_t)stride * h) {
        fprintf(stderr, "%s: pixel data truncated\n", path);
        free(img->data); fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static int bmp_write(const char *path, const Image *img)
{
    int h = img->h;
    uint32_t datasize = (uint32_t)img->stride * h;

    uint8_t hdr[54] = {0};
    hdr[0] = 'B'; hdr[1] = 'M';
    *(uint32_t *)&hdr[2]  = 54 + datasize;   /* bfSize */
    *(uint32_t *)&hdr[10] = 54;              /* bfOffBits */
    *(uint32_t *)&hdr[14] = 40;              /* biSize = BITMAPINFOHEADER */
    *(int32_t  *)&hdr[18] = img->w;
    *(int32_t  *)&hdr[22] = img->topdown ? -h : h;  /* 与输入同向，保证不镜像 */
    *(uint16_t *)&hdr[26] = 1;               /* biPlanes */
    *(uint16_t *)&hdr[28] = 24;              /* biBitCount */
    *(uint32_t *)&hdr[34] = datasize;        /* biSizeImage */

    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "cannot create %s\n", path); return -1; }
    fwrite(hdr, 1, 54, fp);
    fwrite(img->data, 1, datasize, fp);      /* R2: 整块单次写（原为逐行 h 次 fwrite） */
    fclose(fp);
    return 0;
}

/* ---------------- Pthread 并行卷积 ---------------- */
typedef struct {
    int tid;
    int y0, y1;              /* 该线程负责的输出行范围 [y0, y1) */
    double time_ms;          /* 该线程（核）的运算时间 */
} ThreadArg;

static int g_w, g_h, g_stride;        /* 全局尺寸/行距，线程只读 */
static const uint8_t *g_src;
static uint8_t *g_dst;

/* R3: 边界像素慢路径——带 sx/sy 越界检查（常数 0 填充），
 * 仅用于图像四周 2 圈（占总像素 <0.2%），让主循环完全无分支 */
static inline void conv_pixel_guarded(int y, int x)
{
    const int w = g_w, h = g_h, stride = g_stride;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    for (int ky = 0; ky < KSIZE; ky++) {
        int sy = y + ky - KRAD;
        if (sy < 0 || sy >= h) continue;
        const uint8_t *srow = g_src + (size_t)sy * stride;
        for (int kx = 0; kx < KSIZE; kx++) {
            int sx = x + kx - KRAD;
            if (sx < 0 || sx >= w) continue;
            double kv = g_kernel[ky][kx];
            const uint8_t *p = srow + sx * 3;
            a0 += kv * p[0];
            a1 += kv * p[1];
            a2 += kv * p[2];
        }
    }
    uint8_t *d = g_dst + (size_t)y * stride + x * 3;
    int v0 = (int)(a0 + 0.5), v1 = (int)(a1 + 0.5), v2 = (int)(a2 + 0.5);
    d[0] = v0 < 0 ? 0 : (v0 > 255 ? 255 : v0);
    d[1] = v1 < 0 ? 0 : (v1 > 255 ? 255 : v1);
    d[2] = v2 < 0 ? 0 : (v2 > 255 ? 255 : v2);
}

static void conv_rows(int y0, int y1)
{
    const int w = g_w, h = g_h, stride = g_stride;
    for (int y = y0; y < y1; y++) {
        /* 左右各 2 列边界：慢路径（w 很小时整行都走慢路径，通用性保险） */
        if (w > 2 * KRAD) {
            for (int x = 0; x < KRAD; x++)
                conv_pixel_guarded(y, x);
            for (int x = w - KRAD; x < w; x++)
                conv_pixel_guarded(y, x);
        } else {
            for (int x = 0; x < w; x++)
                conv_pixel_guarded(y, x);
            continue;
        }

        if (y >= KRAD && y < h - KRAD) {
            /* R3: 内部行主循环——无任何越界分支，25 次乘加直排，
             * 独立标量累加器驻留寄存器，利于编译器流水线化 */
            for (int x = KRAD; x < w - KRAD; x++) {
                double a0 = 0.0, a1 = 0.0, a2 = 0.0;
                for (int ky = 0; ky < KSIZE; ky++) {
                    const uint8_t *srow = g_src
                        + (size_t)(y + ky - KRAD) * stride + (size_t)(x - KRAD) * 3;
                    for (int kx = 0; kx < KSIZE; kx++) {
                        const double kv = g_kernel[ky][kx];
                        a0 += kv * srow[kx * 3 + 0];
                        a1 += kv * srow[kx * 3 + 1];
                        a2 += kv * srow[kx * 3 + 2];
                    }
                }
                uint8_t *d = g_dst + (size_t)y * stride + x * 3;
                int v0 = (int)(a0 + 0.5), v1 = (int)(a1 + 0.5), v2 = (int)(a2 + 0.5);
                d[0] = v0 < 0 ? 0 : (v0 > 255 ? 255 : v0);
                d[1] = v1 < 0 ? 0 : (v1 > 255 ? 255 : v1);
                d[2] = v2 < 0 ? 0 : (v2 > 255 ? 255 : v2);
            }
        } else {
            /* 上下边界行（共 4 行）：中间列仍需 sy 检查，走慢路径 */
            for (int x = KRAD; x < w - KRAD; x++)
                conv_pixel_guarded(y, x);
        }
    }
}

static void *thread_main(void *p)
{
    ThreadArg *arg = (ThreadArg *)p;
    double t0 = now_sec();
    conv_rows(arg->y0, arg->y1);
    arg->time_ms = (now_sec() - t0) * 1000.0;
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

static void dump_matrix(const char *base, const Image *img, int full)
{
    char path[1024];
    snprintf(path, sizeof path, "%s.matrix.txt", base);
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "cannot create %s\n", path); return; }

    int rows = full ? img->h : (img->h < 20 ? img->h : 20);
    int cols = full ? img->w : (img->w < 20 ? img->w : 20);
    fprintf(fp, "# Blurred pixel matrix (B G R per pixel), image %dx%d\n",
            img->w, img->h);
    if (!full)
        fprintf(fp, "# (large image: showing top-left %dx%d block only, "
                    "run with --dump-full for the whole matrix)\n", rows, cols);
    for (int y = 0; y < rows; y++) {
        int r = img->topdown ? y : img->h - 1 - y;   /* 逻辑行 → 缓冲行号 */
        fprintf(fp, "y=%d\n", y);
        for (int x = 0; x < cols; x++) {
            const uint8_t *p = img->data + (size_t)r * img->stride + x * 3;
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

    /* 1. 读图（串行部分） */
    double t0 = now_sec();
    Image img;
    if (bmp_read(in_path, &img) != 0) return 1;
    double read_ms = (now_sec() - t0) * 1000.0;

    Image out = img;
    out.data = malloc((size_t)img.stride * img.h);
    if (!out.data) { fprintf(stderr, "out of memory\n"); return 1; }
    {   /* 行尾填充字节清零（stride 对齐用；本实验图片 padlen=0，通用性保险） */
        int padlen = img.stride - img.w * 3;
        if (padlen)
            for (int y = 0; y < img.h; y++)
                memset(out.data + (size_t)y * img.stride + img.w * 3, 0, (size_t)padlen);
    }

    /* 2. 并行卷积（并行部分，重点计时） */
    g_w = img.w; g_h = img.h; g_stride = img.stride;
    g_src = img.data; g_dst = out.data;
    if (nthreads > img.h) nthreads = img.h;

    ThreadArg *args = calloc((size_t)nthreads, sizeof(ThreadArg));
    pthread_t *tids  = calloc((size_t)nthreads, sizeof(pthread_t));
    int chunk = img.h / nthreads, rem = img.h % nthreads;

    double t1 = now_sec();
    for (int i = 0; i < nthreads; i++) {
        args[i].tid = i;
        args[i].y0  = i * chunk + (i < rem ? i : rem);
        args[i].y1  = args[i].y0 + chunk + (i < rem ? 1 : 0);
        pthread_create(&tids[i], NULL, thread_main, &args[i]);
    }
    for (int i = 0; i < nthreads; i++)
        pthread_join(tids[i], NULL);
    double conv_ms = (now_sec() - t1) * 1000.0;

    /* 3. 写图（串行部分） */
    t0 = now_sec();
    if (bmp_write(out_path, &out) != 0) return 1;
    double write_ms = (now_sec() - t0) * 1000.0;

    double total_ms = (now_sec() - t_all0) * 1000.0;

    /* 4. 导出卷积核 + 像素矩阵 txt */
    dump_kernel(out_path);
    dump_matrix(out_path, &out, dump_full);

    /* 5. 计时结果 */
    printf("================ Pthread BMP Gaussian Convolution ================\n");
    printf("Input           : %s (%d x %d, 24bpp, stride=%d)\n",
           in_path, img.w, img.h, img.stride);
    printf("Output BMP      : %s\n", out_path);
    printf("Threads         : %d\n", nthreads);
    printf("---- per-thread compute time ----\n");
    double sum_thread = 0.0;
    for (int i = 0; i < nthreads; i++) {
        printf("  thread %-2d rows [%5d,%5d)  time = %8.3f ms\n",
               args[i].tid, args[i].y0, args[i].y1, args[i].time_ms);
        sum_thread += args[i].time_ms;
    }
    printf("---- timing ----\n");
    printf("Read  time      : %10.3f ms   (serial)\n", read_ms);
    printf("Conv  time      : %10.3f ms   (parallel region, p=%d)\n", conv_ms, nthreads);
    printf("Conv  sum/thread: %10.3f ms   (sum of per-thread times)\n", sum_thread);
    printf("Write time      : %10.3f ms   (serial)\n", write_ms);
    printf("Total time      : %10.3f ms\n", total_ms);

    free(args); free(tids);
    free(img.data); free(out.data);
    return 0;
}
