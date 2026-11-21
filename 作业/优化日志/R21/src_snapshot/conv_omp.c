/*
 * conv_omp.c —— OpenMP 并行实现 BMP 图像高斯卷积（优化版）
 *
 * 题目：读入 24 位 BMP 图像，用 5x5 高斯核（sigma=1.5，radius=2，已归一化）、
 *       步长 1、边缘常数 0 填充做卷积，输出模糊后的 BMP；并用 OpenMP 按行分条带并行。
 *
 * 用法: ./conv_omp <input.bmp> <output.bmp> [threads] [--dump-full]
 *   threads      线程数；未指定时尊重 OMP_NUM_THREADS 环境变量，
 *                环境变量也未设则用 OpenMP 默认（通常 = 可用核数）。
 *                显式传参 > OMP_NUM_THREADS > 系统默认。
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
 *      pwrite 自己的输出条带，各线程 I/O/计算流水线化，消除串行整图读写
 *   R14 vpass 单趟融合：每 x 一次算完垂直 5 tap（5 load + 4 FMA + 1 store），
 *      免 vbuf 4 次读改写整平面往返（T4 −18%；插桩实测 vpass 占纯计算 49%）
 *   R15 hpass 单趟融合：内部区 5 tap 单趟算完（与 R14 同构），边缘 4 列走
 *      逐 tap 旧路径，窄图（w≤4）保留原实现（T4 再 −2.8%）
 *   R17/R18 探索（回退）：线程绑核与 vpass+pack 融合均无收益（详见
 *      ../优化日志.md R17/R18 小节），代码未保留，数据留档
 *   R19 连续 BGR 单布局：每行视作 3w 字节数组，水平 tap 步长 3 字节（同通道），
 *      H 环形缓冲与源行/输出行同序，垂直趟直接舍入写字节——免解交织/pack/
 *      vbuf（T1 −12%，T4 两版 −6.1/−6.8%，输出逐字节一致）
 *   R20 定点探索（回退）：精度达标（相似度 99.99%）但整数向量化 uop 受限，
 *      T4 无收益；数据留档 优化日志/R20/
 *   R21 posix_fallocate 输出预分配：块分配从并行缺页路径挪到串行预备段
 *      （T4 −6.5%，T1 −1.4%，输出逐字节不变）
 *
 * 计时：omp_get_wtime 计全程墙钟；逐线程打印读条带/卷积（并行段）/写条带
 *       分段耗时，并行段关键路径 = max(各线程卷积时间)。
 *       线程数：显式传参 > OMP_NUM_THREADS > OpenMP 默认（打印实际值）。
 */
#define _POSIX_C_SOURCE 200809L   /* mmap/pread/pwrite */
#define _DEFAULT_SOURCE            /* madvise / MADV_SEQUENTIAL */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <omp.h>

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

/* R4: 分离核（与 Pthread 版同构，见 conv_pthread.c 注释）：
 * 水平趟 5 行环形缓冲驻 L2 + 垂直趟连续 double 累加，每像素乘加 75→30 次。
 * k1 为 5x5 核的最小二乘秩 1 拟合（残差 max 5.4e-9）。 */
static const float g_k1[KSIZE] = {  /* R6: float32 —— 8 宽向量 */
    0.12007837208047906, 0.2338807408740243, 0.29208172229534735,
    0.2338807408740243, 0.12007837208047906
};

/* ---------------- BMP 头解析 / 头写出 ---------------- */
/* R2/R5: 像素按文件原始行序就地处理（核垂直对称 ⇒ 免翻转），逐条带 pread/pwrite */
typedef struct {
    int w, h;
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
    *(uint32_t *)&hdr[2]  = 54 + datasize;
    *(uint32_t *)&hdr[10] = 54;
    *(uint32_t *)&hdr[14] = 40;
    *(int32_t  *)&hdr[18] = b->w;
    *(int32_t  *)&hdr[22] = b->topdown ? -b->h : b->h;
    *(uint16_t *)&hdr[26] = 1;
    *(uint16_t *)&hdr[28] = 24;
    *(uint32_t *)&hdr[34] = datasize;
}

/* ---------------- 分离核卷积 ---------------- */
/* R19: 连续 BGR 单布局 —— 三通道共用同一 1D 核 ⇒ 每行视作长度 3w 的字节
 * 数组，水平 tap 读取 n±6/n±3/n（步长 3 字节，始终落在同一颜色通道）。
 * H 环形缓冲每槽 3w float（与源行字节同序），垂直趟读相同布局后直接
 * 舍入饱和写字节。与 R9 平面化相比：免解交织（scratch3）、免 pack
 * 交织、免 vbuf，数值路径逐元素等价（左结合次序不变）→ 输出逐字节一致。 */
static void hpass_row(float *dst, const uint8_t *srow, int w)
{
    const int n3 = w * 3;
    if (!srow) { memset(dst, 0, (size_t)n3 * sizeof(float)); return; }  /* 零行（越界） */
    if (w > 2 * KRAD) {
        /* 内部区 [6, n3-6)：单趟 5 tap（与 R15 同构），步长 3 字节 */
        const float k0 = g_k1[0], k1 = g_k1[1], k2 = g_k1[2],
                    k3 = g_k1[3], k4 = g_k1[4];
        for (int n = 2 * KRAD * 3; n < n3 - 2 * KRAD * 3; n++)
            dst[n] = k0 * (float)srow[n - 6] + k1 * (float)srow[n - 3]
                   + k2 * (float)srow[n]     + k3 * (float)srow[n + 3]
                   + k4 * (float)srow[n + 6];
        /* 边缘 [0,6) ∪ [n3-6,n3)：tap 偏移 d=(kx-2)*3，逐 tap 有效区间累加
         * （kx 升序；首 tap 先把两窗置零）。左窗 tap d 的有效起点 n ≥ -d，
         * 右窗有效终点 n < n3-d。 */
        for (int kx = 0; kx < KSIZE; kx++) {
            const float kv = g_k1[kx];
            const int d = (kx - KRAD) * 3;
            const int l0 = -d > 0 ? -d : 0;              /* 左窗起点 */
            const int r1 = n3 - d > n3 ? n3 : n3 - d;    /* 右窗终点（开） */
            if (kx == 0) {
                for (int n = 0; n < 2 * KRAD * 3; n++) dst[n] = 0.0f;
                for (int n = n3 - 2 * KRAD * 3; n < n3; n++) dst[n] = 0.0f;
            }
            for (int n = l0; n < 2 * KRAD * 3; n++)
                dst[n] += kv * (float)srow[n + d];
            for (int n = n3 - 2 * KRAD * 3; n < n3 && n < r1; n++)
                dst[n] += kv * (float)srow[n + d];
        }
        return;
    }
    /* 窄图（w ≤ 4，内部区为空；左右窗会重叠，改整行逐 tap 有效区间） */
    for (int kx = 0; kx < KSIZE; kx++) {
        const float kv = g_k1[kx];
        const int d = (kx - KRAD) * 3;
        int n0 = -d > 0 ? -d : 0;
        int n1 = n3 - (d > 0 ? d : 0);
        if (kx == 0)
            for (int n = 0; n < n3; n++) dst[n] = 0.0f;
        for (int n = n0; n < n1; n++)
            dst[n] += kv * (float)srow[n + d];
    }
}

static inline int slot5(int r) { return ((r % 5) + 5) % 5; }

static void conv_band(const uint8_t *src, int base, int nrows,
                      uint8_t *dst, int w, int stride, int y0, int y1)
{
    const size_t sn = (size_t)w * 3;                  /* 每行有效字节数（H 槽步长） */
    float *hb = malloc(5 * sn * sizeof(float));       /* 5 槽 × 3w float（~245KB, L2） */
    if (!hb) return;

    for (int i = 0; i < 5; i++) {
        int r = y0 + i - KRAD;
        hpass_row(hb + (size_t)slot5(r) * sn,
                  (r >= base && r < base + nrows) ? src + (size_t)(r - base) * stride : NULL,
                  w);
    }
    for (int y = y0; y < y1; y++) {
        /* 垂直趟+pack 融合：v[n] = Σ_ky k1[ky]·H(y+ky-2)[n]，直接舍入饱和
         * 写字节（H 与输出行同为 BGR 连续布局，pack 退化为逐字节写出）。 */
        const float k0 = g_k1[0], k1 = g_k1[1], k2 = g_k1[2],
                    k3 = g_k1[3], k4 = g_k1[4];
        const float *h0 = hb + (size_t)slot5(y - KRAD) * sn;
        const float *h1 = hb + (size_t)slot5(y - KRAD + 1) * sn;
        const float *h2 = hb + (size_t)slot5(y - KRAD + 2) * sn;
        const float *h3 = hb + (size_t)slot5(y - KRAD + 3) * sn;
        const float *h4 = hb + (size_t)slot5(y - KRAD + 4) * sn;
        uint8_t *d = dst + (size_t)y * stride;
        const int n3 = w * 3;
        for (int n = 0; n < n3; n++) {
            float v = k0 * h0[n] + k1 * h1[n] + k2 * h2[n] + k3 * h3[n] + k4 * h4[n];
            int o = (int)(v + 0.5f);
            d[n] = o < 0 ? 0 : (o > 255 ? 255 : o);
        }
        int rr = y + KRAD + 1;
        hpass_row(hb + (size_t)slot5(rr) * sn,
                  (rr < base + nrows) ? src + (size_t)(rr - base) * stride : NULL,
                  w);
    }
    free(hb);
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
    int nthreads = 0;      /* 0 = 未显式指定 → 尊重 OMP_NUM_THREADS / 系统默认。
                            * 修复记录：此前默认 4 且无条件 omp_set_num_threads(4)，
                            * 会覆盖 OMP_NUM_THREADS 环境变量，导致基准脚本
                            * "1/2 核"组实际跑的是 4 线程挤 1/2 核。 */
    int dump_full = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--dump-full") == 0) dump_full = 1;
        else if (atoi(argv[i]) > 0) nthreads = atoi(argv[i]);
    }
    if (nthreads > 0) omp_set_num_threads(nthreads);

    double t_all0 = omp_get_wtime();

    /* 1. 解析 BMP 头 + mmap 输入文件（像素按需缺页，免整图读拷贝） */
    int in_fd = open(in_path, O_RDONLY);
    if (in_fd < 0) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }
    uint8_t hdr[54];
    BmpInfo bmp;
    if (read(in_fd, hdr, 54) != 54 || bmp_parse_header(hdr, &bmp) != 0) {
        fprintf(stderr, "%s: not an uncompressed 24-bit BMP (BITMAPINFOHEADER)\n", in_path);
        close(in_fd);
        return 1;
    }
    const int w = bmp.w, h = bmp.h, stride = bmp.stride;
    off_t in_size = lseek(in_fd, 0, SEEK_END);
    uint8_t *in_map = mmap(NULL, (size_t)in_size, PROT_READ, MAP_SHARED, in_fd, 0);
    if (in_map == MAP_FAILED) { fprintf(stderr, "mmap input failed\n"); return 1; }

    /* 2. 创建输出文件：ftruncate 定长 + 写头 + MAP_SHARED 映射，
     *    卷积结果直接写入映射（免 malloc 输出缓冲与 pwrite 拷贝） */
    uint8_t ohdr[54];
    bmp_make_header(&bmp, ohdr);
    int out_fd = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) { fprintf(stderr, "cannot create %s\n", out_path); return 1; }
    size_t out_mapsize = (size_t)bmp.data_off + (size_t)stride * h;
    if (ftruncate(out_fd, (off_t)out_mapsize) != 0) {
        fprintf(stderr, "ftruncate failed\n"); return 1;
    }
    if (posix_fallocate(out_fd, 0, (off_t)out_mapsize) != 0) {  /* R21: 预分配物理块 */
        fprintf(stderr, "fallocate failed\n"); return 1;
    }
    if (pwrite(out_fd, ohdr, 54, 0) != 54) { fprintf(stderr, "write header failed\n"); return 1; }
    uint8_t *out_map = mmap(NULL, out_mapsize, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
    if (out_map == MAP_FAILED) { fprintf(stderr, "mmap output failed\n"); return 1; }
    uint8_t *out = out_map + bmp.data_off;
    {   /* 头部与像素区之间的间隙及行尾填充字节清零（本实验图片均为 0，通用性保险） */
        int padlen = stride - w * 3;
        if (bmp.data_off > 54)
            memset(out_map + 54, 0, (size_t)bmp.data_off - 54);
        if (padlen)
            for (int y = 0; y < h; y++)
                memset(out + (size_t)y * stride + w * 3, 0, (size_t)padlen);
    }

    /* 3. 并行：各线程在自己的条带上卷积（源/目的均为文件映射） */
    int maxt = omp_get_max_threads();
    double *conv_ms = calloc((size_t)maxt, sizeof(double));
    int np_actual = 0;                        /* 并行区实际线程数（用于打印） */

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int np  = omp_get_num_threads();
        if (tid == 0) np_actual = np;
        int chunk = (h + np - 1) / np;
        int y0 = tid * chunk;
        int y1 = y0 + chunk;
        if (y1 > h) y1 = h;
        if (y0 < h) {
            /* R8: 源条带直接取输入映射（免 pread 拷贝），输出直接写输出映射
             * （免 pwrite 拷贝，脏页由内核在退出时回写） */
            int y0c = y0 - KRAD < 0 ? 0 : y0 - KRAD;
            int y1c = y1 + KRAD > h ? h : y1 + KRAD;
            const uint8_t *src = in_map + bmp.data_off + (size_t)y0c * stride;
            double t0 = omp_get_wtime();
            conv_band(src, y0c, y1c - y0c, out, w, stride, y0, y1);
            conv_ms[tid] = (omp_get_wtime() - t0) * 1000.0;
        }
    }

    close(in_fd);
    close(out_fd);
    double total_ms = (omp_get_wtime() - t_all0) * 1000.0;

    /* 4. 导出卷积核 + 像素矩阵 txt */
    dump_kernel(out_path);
    dump_matrix(out_path, out, &bmp, dump_full);

    /* 5. 计时结果 */
    printf("================ OpenMP BMP Gaussian Convolution ==================\n");
    printf("Input           : %s (%d x %d, 24bpp, stride=%d)\n", in_path, w, h, stride);
    printf("Output BMP      : %s\n", out_path);
    printf("Threads         : %d%s\n", np_actual,
           nthreads > 0 ? "" : " (OMP_NUM_THREADS / default)");
    printf("---- per-thread conv time (R8: I/O 零拷贝 mmap，缺页计入 conv) ----\n");
    double max_conv = 0, sum_conv = 0;
    for (int i = 0; i < np_actual; i++) {
        printf("  thread %-2d                    conv = %8.3f ms\n", i, conv_ms[i]);
        if (conv_ms[i] > max_conv) max_conv = conv_ms[i];
        sum_conv += conv_ms[i];
    }
    printf("---- timing ----\n");
    printf("I/O    : mmap 零拷贝（读=按需缺页+页缓存命中，写=直接写脏页，无显式读写）\n");
    printf("Conv   : (parallel region, p=%d)  %8.3f ms   (critical path = max thread)\n", np_actual, max_conv);
    printf("Conv   : sum/thread              %8.3f ms   (sum of per-thread times)\n", sum_conv);
    printf("Total time      : %10.3f ms\n", total_ms);

    free(conv_ms);
    munmap(out_map, out_mapsize);
    munmap(in_map, (size_t)in_size);
    /* 基准整备：Total time 口径保持不变（可比），另报全程墙钟（含文本导出+munmap） */
    printf("Wall time (full) : %10.3f ms   (Total + 文本导出 + munmap)\n",
           (omp_get_wtime() - t_all0) * 1000.0);
    return 0;
}
