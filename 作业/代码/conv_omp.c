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
typedef struct {
    int w, h;                /* 图像宽、高（行序已在内存里转为自上而下） */
    int stride;              /* BMP 文件中每行字节数（4 字节对齐） */
    uint8_t *data;           /* 顶层像素，BGR 交错，data[(y*w + x)*3 + c] */
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
    uint8_t *raw = malloc((size_t)stride * h);
    if (!raw) { fclose(fp); return -1; }
    fseek(fp, off, SEEK_SET);
    if (fread(raw, 1, (size_t)stride * h, fp) != (size_t)stride * h) {
        fprintf(stderr, "%s: pixel data truncated\n", path);
        free(raw); fclose(fp);
        return -1;
    }
    fclose(fp);

    /* 存成自上而下：BMP 第 0 行是图像最底行 */
    img->w = w; img->h = h; img->stride = stride;
    img->data = malloc((size_t)w * h * 3);
    if (!img->data) { free(raw); return -1; }
    for (int y = 0; y < h; y++) {
        const uint8_t *src = raw + (size_t)y * stride;
        uint8_t *dst = img->data + (size_t)(h - 1 - y) * w * 3;
        memcpy(dst, src, (size_t)w * 3);
    }
    free(raw);
    return 0;
}

static int bmp_write(const char *path, const Image *img)
{
    int w = img->w, h = img->h;
    int stride = (w * 3 + 3) & ~3;
    uint32_t datasize = (uint32_t)stride * h;

    uint8_t hdr[54] = {0};
    hdr[0] = 'B'; hdr[1] = 'M';
    *(uint32_t *)&hdr[2]  = 54 + datasize;   /* bfSize */
    *(uint32_t *)&hdr[10] = 54;              /* bfOffBits */
    *(uint32_t *)&hdr[14] = 40;              /* biSize = BITMAPINFOHEADER */
    *(int32_t  *)&hdr[18] = w;
    *(int32_t  *)&hdr[22] = h;               /* 正数：自下而上存储 */
    *(uint16_t *)&hdr[26] = 1;               /* biPlanes */
    *(uint16_t *)&hdr[28] = 24;              /* biBitCount */
    *(uint32_t *)&hdr[34] = datasize;        /* biSizeImage */

    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "cannot create %s\n", path); return -1; }
    fwrite(hdr, 1, 54, fp);

    uint8_t pad[3] = {0, 0, 0};
    int padlen = stride - w * 3;
    for (int y = h - 1; y >= 0; y--) {       /* 自下而上写回 */
        fwrite(img->data + (size_t)y * w * 3, 1, (size_t)w * 3, fp);
        if (padlen) fwrite(pad, 1, (size_t)padlen, fp);
    }
    fclose(fp);
    return 0;
}

/* ---------------- OpenMP 并行卷积 ---------------- */
/* 返回并行区墙钟时间（ms），thread_ms 带回各线程运算时间 */
static double conv_all(const Image *img, Image *out, double *thread_ms)
{
    const int w = img->w, h = img->h;
    const uint8_t *src = img->data;
    uint8_t *dst = out->data;

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double ts = omp_get_wtime();
        #pragma omp for schedule(static)      /* 按行静态划分，各线程互不重叠 */
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double acc[3] = {0.0, 0.0, 0.0};
                for (int ky = 0; ky < KSIZE; ky++) {
                    int sy = y + ky - KRAD;
                    if (sy < 0 || sy >= h) continue;        /* 常数 0 填充 */
                    const uint8_t *srow = src + (size_t)sy * w * 3;
                    for (int kx = 0; kx < KSIZE; kx++) {
                        int sx = x + kx - KRAD;
                        if (sx < 0 || sx >= w) continue;    /* 常数 0 填充 */
                        double kv = g_kernel[ky][kx];
                        const uint8_t *p = srow + sx * 3;
                        acc[0] += kv * p[0];
                        acc[1] += kv * p[1];
                        acc[2] += kv * p[2];
                    }
                }
                uint8_t *d = dst + ((size_t)y * w + x) * 3;
                for (int c = 0; c < 3; c++) {
                    int v = (int)(acc[c] + 0.5);            /* 四舍五入 */
                    d[c] = v < 0 ? 0 : (v > 255 ? 255 : v);
                }
            }
        }
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
        fprintf(fp, "y=%d\n", y);
        for (int x = 0; x < cols; x++) {
            const uint8_t *p = img->data + ((size_t)y * img->w + x) * 3;
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
    out.data = malloc((size_t)img.w * img.h * 3);
    if (!out.data) { fprintf(stderr, "out of memory\n"); return 1; }

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
