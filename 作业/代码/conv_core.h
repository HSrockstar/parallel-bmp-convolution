#ifndef CONV_CORE_H
#define CONV_CORE_H
/* x86-64 AVX2 之外的机器仍使用原来的标量分离卷积。 */
static void hpass_row(float *dst, const uint8_t *srow, int w)
{
    const int n3 = w * 3;
    if (!srow) { memset(dst, 0, (size_t)n3 * sizeof(float)); return; }  /* 零行（越界） */
    if (w > 2 * KRAD) {
        /* 内部区 [6, n3-6)：单趟 5 tap（与 R15 同构），步长 3 字节 */
        const float k0 = g_k1[0], k1 = g_k1[1], k2 = g_k1[2],
                    k3 = g_k1[3], k4 = g_k1[4];
        for (int n = KRAD * 3; n < n3 - KRAD * 3; n++)
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
                for (int n = 0; n < KRAD * 3; n++) dst[n] = 0.0f;
                for (int n = n3 - KRAD * 3; n < n3; n++) dst[n] = 0.0f;
            }
            for (int n = l0; n < KRAD * 3; n++)
                dst[n] += kv * (float)srow[n + d];
            for (int n = n3 - KRAD * 3; n < n3 && n < r1; n++)
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

static inline int slot5(int r) { return ((r % 5) + 5) % 5; }   /* 行号 → 环形槽位 */

/* 分离核卷积一个输出行条带 [y0,y1)。
 * src 为本线程条带缓冲首址，base 为其对应的全局行号；全局行 r 的源数据在
 * src + (r-base)*stride，r 超出 [base, base+nrows) 视为越界（零填充）。 */
static int scalar_band(const uint8_t *src, int base, int nrows,
                      uint8_t *dst, int w, int stride, int y0, int y1)
{
    const size_t sn = (size_t)w * 3;                  /* 每行有效字节数（H 槽步长） */
    float *hb = (float *)malloc(5 * sn * sizeof(float));       /* 5 槽 × 3w float（~245KB, L2） */
    if (!hb) return -1;

    /* 初始化窗口行 [y0-2, y0+3) */
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
        /* 下一行滑入窗口：H(y+3) */
        if (y + 1 == y1) break; /* No unused final horizontal row. */
        int rr = y + KRAD + 1;
        hpass_row(hb + (size_t)slot5(rr) * sn,
                  (rr < base + nrows) ? src + (size_t)(rr - base) * stride : NULL,
                  w);
    }
    free(hb);
    return 0;
}

#if defined(__x86_64__) && defined(__GNUC__)
static uint8_t baseline_round(const uint8_t *src, int rows, int stride,
                              int width_bytes, int y, int n)
{
    float v = 0;
    for (int ky = 0; ky < 5; ++ky) {
        int r = y+ky-2;
        if (r < 0 || r >= rows) continue;
        const uint8_t *p = src + (size_t)r * stride;
        float h = 0;
        for (int kx = 0; kx < 5; ++kx) {
            int col = n+(kx-2)*3;
            if (col >= 0 && col < width_bytes) h += g_k1[kx] * p[col];
        }
        v += g_k1[ky] * h;
    }
    return (uint8_t)(int)(v+0.5f);
}

#include "conv_fused.h"
#endif

static int conv_band(const uint8_t *src, int base, int rows, uint8_t *dst,
                     int w, int stride, int y0, int y1)
{
#if defined(__x86_64__) && defined(__GNUC__)
    if (w >= 10 && have_avx2())
        return conv_fused(src,base,rows,dst,w,stride,y0,y1);
#endif
    return scalar_band(src,base,rows,dst,w,stride,y0,y1);
}
/* 用 GCC 内建原子操作，gcc 与 g++ 均可编译（C11 atomic_uint 在 C++ 下不可用）。 */
enum { STRIPE_ROWS = 64 };

/* 不绑核且每个线程有足够多行时按需取条带，较快的核心可多处理几条。
 * 固定核或小图保留连续静态分区。每个输出行只由一个工作线程写。 */
static int process_rows(const uint8_t *src, uint8_t *dst, const BmpInfo *b,
                        int y0, int y1, unsigned *next)
{
    if (!next)
        return conv_band(src,0,b->h,dst,b->w,b->stride,y0,y1) ? -1 : y1-y0;
    int processed = 0;
    for (;;) {
        unsigned stripe = __atomic_fetch_add(next,1,__ATOMIC_RELAXED);
        if (stripe >= ((unsigned)b->h+STRIPE_ROWS-1)/STRIPE_ROWS) return processed;
        int first = (int)(stripe * STRIPE_ROWS);
        int end = b->h-first < STRIPE_ROWS ? b->h : first+STRIPE_ROWS;
        if (conv_band(src,0,b->h,dst,b->w,b->stride,first,end)) return -1;
        processed += end-first;
    }
}
#endif
