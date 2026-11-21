/*
 * conv_omp.c —— OpenMP 并行实现 BMP 图像高斯卷积（baseline 版本）
 *
 * 题目：读入 24 位 BMP 图像，用 5x5 高斯核（sigma=1.5，radius=2，已归一化）、
 *       步长 1、边缘常数 0 填充做卷积，输出模糊后的 BMP；并用 OpenMP 按行并行。
 *
 * 用法: ./conv_omp <input.bmp> <output.bmp> [threads] [--dump-full]
 *   threads      线程数，默认 4（也可用环境变量 OMP_NUM_THREADS 控制）
 *   --dump-full  将卷积后的完整像素矩阵写入 txt（大图会很大，默认只导出局部供验证）
 *
 * 输出文件：
 *   <output>.kernel.txt        5x5 卷积核
 *   <output>.matrix.txt        卷积后的像素矩阵（默认小图全量、大图导出左上角 20x20）
 *
 * 计时：omp_get_wtime 计读图、卷积（并行段）、写图时间，
 *       并逐线程打印各核运算时间。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

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

/* ---------------- OpenMP 并行卷积 ---------------- */
/* R4: 分离核两趟卷积（与 Pthread 版同构，见 conv_pthread.c 注释）：
 * 水平趟 5 行环形缓冲驻 L2 + 垂直趟连续 double 累加，每像素乘加 75→30 次。
 * k1 为 5x5 核的最小二乘秩 1 拟合（残差 max 5.4e-9）。 */
static const double g_k1[KSIZE] = {
    0.12007837208047906, 0.2338807408740243, 0.29208172229534735,
    0.2338807408740243, 0.12007837208047906
};

static inline int slot5(int r) { return ((r % 5) + 5) % 5; }

static void hpass(double *dst, const uint8_t *srow, int w)
{
    const size_t rowlen = (size_t)w * 3;
    memset(dst, 0, rowlen * sizeof(double));
    if (!srow) return;
    for (int kx = 0; kx < KSIZE; kx++) {
        const double kv = g_k1[kx];
        /* sx = x+kx-2 ∈ [0,w) ⇒ x ∈ [max(0,2-kx), min(w, w+2-kx)) */
        int x0 = KRAD - kx;
        if (x0 < 0) x0 = 0;
        int x1 = w + KRAD - kx;
        if (x1 > w) x1 = w;
        const uint8_t *s = srow + (size_t)(x0 + kx - KRAD) * 3;  /* 对齐到 x=x0 */
        double *d = dst + (size_t)x0 * 3;
        for (int x = x0; x < x1; x++) {
            d[0] += kv * s[0];
            d[1] += kv * s[1];
            d[2] += kv * s[2];
            d += 3; s += 3;
        }
    }
}

/* 一个线程负责输出行条带 [y0, y1) */
static void conv_band(const uint8_t *src, uint8_t *dst,
                      int w, int h, int stride, int y0, int y1)
{
    const size_t rowlen = (size_t)w * 3;
    double *hb   = malloc(5 * rowlen * sizeof(double));
    double *vbuf = malloc(rowlen * sizeof(double));
    if (!hb || !vbuf) { free(hb); free(vbuf); return; }

    for (int i = 0; i < 5; i++) {
        int r = y0 + i - KRAD;
        hpass(hb + (size_t)slot5(r) * rowlen,
              (r >= 0 && r < h) ? src + (size_t)r * stride : NULL, w);
    }
    for (int y = y0; y < y1; y++) {
        memset(vbuf, 0, rowlen * sizeof(double));
        for (int ky = 0; ky < KSIZE; ky++) {
            int r = y + ky - KRAD;
            const double *hr = hb + (size_t)slot5(r) * rowlen;
            const double kv = g_k1[ky];
            for (size_t i = 0; i < rowlen; i++)
                vbuf[i] += kv * hr[i];
        }
        uint8_t *d = dst + (size_t)y * stride;
        for (int x = 0; x < w; x++) {
            int v0 = (int)(vbuf[(size_t)x * 3 + 0] + 0.5);
            int v1 = (int)(vbuf[(size_t)x * 3 + 1] + 0.5);
            int v2 = (int)(vbuf[(size_t)x * 3 + 2] + 0.5);
            d[x * 3 + 0] = v0 < 0 ? 0 : (v0 > 255 ? 255 : v0);
            d[x * 3 + 1] = v1 < 0 ? 0 : (v1 > 255 ? 255 : v1);
            d[x * 3 + 2] = v2 < 0 ? 0 : (v2 > 255 ? 255 : v2);
        }
        int r = y + KRAD + 1;
        hpass(hb + (size_t)slot5(r) * rowlen,
              (r < h) ? src + (size_t)r * stride : NULL, w);
    }
    free(hb); free(vbuf);
}

/* 返回并行区墙钟时间（ms），thread_ms 带回各线程运算时间 */
static double conv_all(const Image *img, Image *out, double *thread_ms)
{
    const int w = img->w, h = img->h, stride = img->stride;
    const uint8_t *src = img->data;
    uint8_t *dst = out->data;

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        double ts = omp_get_wtime();
        /* 手动按行静态等分条带（等价 schedule(static)，各线程缓冲私有、行不重叠） */
        int chunk = (h + nth - 1) / nth;
        int y0 = tid * chunk;
        int y1 = y0 + chunk;
        if (y1 > h) y1 = h;
        if (y0 < h)
            conv_band(src, dst, w, h, stride, y0, y1);
        thread_ms[tid] = (omp_get_wtime() - ts) * 1000.0;
    }
    return (omp_get_wtime() - t0) * 1000.0;
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
    omp_set_num_threads(nthreads);

    double t_all0 = omp_get_wtime();

    /* 1. 读图（串行部分） */
    double t0 = omp_get_wtime();
    Image img;
    if (bmp_read(in_path, &img) != 0) return 1;
    double read_ms = (omp_get_wtime() - t0) * 1000.0;

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
    int maxt = omp_get_max_threads();
    double *thread_ms = calloc((size_t)maxt, sizeof(double));
    double conv_ms = conv_all(&img, &out, thread_ms);

    /* 3. 写图（串行部分） */
    t0 = omp_get_wtime();
    if (bmp_write(out_path, &out) != 0) return 1;
    double write_ms = (omp_get_wtime() - t0) * 1000.0;

    double total_ms = (omp_get_wtime() - t_all0) * 1000.0;

    /* 4. 导出卷积核 + 像素矩阵 txt */
    dump_kernel(out_path);
    dump_matrix(out_path, &out, dump_full);

    /* 5. 计时结果 */
    printf("================ OpenMP BMP Gaussian Convolution ==================\n");
    printf("Input           : %s (%d x %d, 24bpp, stride=%d)\n",
           in_path, img.w, img.h, img.stride);
    printf("Output BMP      : %s\n", out_path);
    printf("Threads         : %d\n", nthreads);
    printf("---- per-thread compute time ----\n");
    double sum_thread = 0.0;
    for (int i = 0; i < nthreads; i++) {
        printf("  thread %-2d                    time = %8.3f ms\n",
               i, thread_ms[i]);
        sum_thread += thread_ms[i];
    }
    printf("---- timing ----\n");
    printf("Read  time      : %10.3f ms   (serial)\n", read_ms);
    printf("Conv  time      : %10.3f ms   (parallel region, p=%d)\n", conv_ms, nthreads);
    printf("Conv  sum/thread: %10.3f ms   (sum of per-thread times)\n", sum_thread);
    printf("Write time      : %10.3f ms   (serial)\n", write_ms);
    printf("Total time      : %10.3f ms\n", total_ms);

    free(thread_ms);
    free(img.data); free(out.data);
    return 0;
}
