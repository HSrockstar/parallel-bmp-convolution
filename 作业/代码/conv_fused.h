#ifndef CONV_FUSED_H
#define CONV_FUSED_H

#include <cpuid.h>
/* 在线程内确认 CPU 和操作系统都支持 YMM，探测也计入 Total。 */
static int have_avx2(void)
{
    unsigned a,b,c,d;
    if (!__get_cpuid(1,&a,&b,&c,&d) || !(c & bit_OSXSAVE) || !(c & bit_AVX))
        return 0;
    unsigned xlo,xhi;
    __asm__ volatile ("xgetbv" : "=a"(xlo),"=d"(xhi) : "c"(0));
    return (xlo & 6) == 6 && __get_cpuid_count(7,0,&a,&b,&c,&d) && (b & bit_AVX2);
}
/* 将垂直卷积写成转置 FIR。一个水平结果分别乘以外侧、内侧和中心
 * 三个系数。这三个乘积可分送给接下来五行；各行仍按原来的 tap 顺序相加。
 * 四个未完成的和保留在寄存器中，无需在内存写回水平中间行。 */
#define FIR_STEP(A,B,C,D,OFFSET) \
"vpmovzxbd -6(%%rax), %%ymm3\n\t" \
"vpmovzxbd -3(%%rax), %%ymm4\n\t" \
"vpmovzxbd 0(%%rax), %%ymm5\n\t" \
"vpmovzxbd 3(%%rax), %%ymm6\n\t" \
"vpmovzxbd 6(%%rax), %%ymm7\n\t" \
"vcvtdq2ps %%ymm3, %%ymm3\n\t" \
"vcvtdq2ps %%ymm4, %%ymm4\n\t" \
"vcvtdq2ps %%ymm5, %%ymm5\n\t" \
"vcvtdq2ps %%ymm6, %%ymm6\n\t" \
"vcvtdq2ps %%ymm7, %%ymm7\n\t" \
"vmulps %%ymm0, %%ymm3, %%ymm3\n\t" \
"vmulps %%ymm1, %%ymm4, %%ymm4\n\t" \
"vmulps %%ymm2, %%ymm5, %%ymm5\n\t" \
"vmulps %%ymm1, %%ymm6, %%ymm6\n\t" \
"vmulps %%ymm0, %%ymm7, %%ymm7\n\t" \
"vaddps %%ymm4, %%ymm3, %%ymm3\n\t" \
"vaddps %%ymm5, %%ymm3, %%ymm3\n\t" \
"vaddps %%ymm6, %%ymm3, %%ymm3\n\t" \
"vaddps %%ymm7, %%ymm3, %%ymm3\n\t" \
"vmulps %%ymm0, %%ymm3, %%ymm4\n\t" \
"vmulps %%ymm1, %%ymm3, %%ymm5\n\t" \
"vmulps %%ymm2, %%ymm3, %%ymm6\n\t" \
"vaddps %%ymm4, " A ", %%ymm7\n\t" \
"vaddps %%ymm5, " B ", " A "\n\t" \
"vaddps %%ymm6, " C ", " B "\n\t" \
"vaddps %%ymm5, " D ", " C "\n\t" \
"vmovaps %%ymm4, " D "\n\t" \
"cmp $4, %[j]\n\t" \
"jb 2f\n\t" \
"vaddps %[half], %%ymm7, %%ymm7\n\t" \
"vcvttps2dq %%ymm7, %%ymm7\n\t" \
"vextracti128 $1, %%ymm7, %%xmm6\n\t" \
"vpackssdw %%xmm6, %%xmm7, %%xmm7\n\t" \
"vpackuswb %%xmm7, %%xmm7, %%xmm7\n\t" \
"vmovq %%xmm7, " OFFSET "(%[dst])\n\t" \
"2:\n\t"

static inline void fused_column16(const uint8_t *const *rows, int nr, int x,
                                 uint8_t *dst, int stride)
{
    static const float half[8] = {0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f};
    size_t j = 0;
    __asm__ volatile (
        "vbroadcastss 0(%[k]), %%ymm0\n\t"
        "vbroadcastss 4(%[k]), %%ymm1\n\t"
        "vbroadcastss 8(%[k]), %%ymm2\n\t"
        "vxorps %%ymm8, %%ymm8, %%ymm8\n\t"
        "vxorps %%ymm9, %%ymm9, %%ymm9\n\t"
        "vxorps %%ymm10, %%ymm10, %%ymm10\n\t"
        "vxorps %%ymm11, %%ymm11, %%ymm11\n\t"
        "vxorps %%ymm12, %%ymm12, %%ymm12\n\t"
        "vxorps %%ymm13, %%ymm13, %%ymm13\n\t"
        "vxorps %%ymm14, %%ymm14, %%ymm14\n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15\n\t"
        "1:\n\t"
        "mov (%[rows],%[j],8), %%rax\n\t"
        "add %[x], %%rax\n\t"
        FIR_STEP("%%ymm8","%%ymm9","%%ymm10","%%ymm11","")
        "add $8, %%rax\n\t"
        FIR_STEP("%%ymm12","%%ymm13","%%ymm14","%%ymm15","8")
        "cmp $4, %[j]\n\t"
        "jb 3f\n\t"
        "add %[stride], %[dst]\n\t"
        "3: inc %[j]\n\t"
        "cmp %[nr], %[j]\n\t"
        "jb 1b\n\t"
        "vzeroupper"
        : [j] "+&r"(j), [dst] "+&r"(dst)
        : [k] "r"(g_k1), [half] "m"(half), [rows] "r"(rows), [x] "r"((size_t)x),
          [stride] "r"((size_t)stride), [nr] "r"((size_t)nr)
        : "rax","ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",
          "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","ymm15","cc","memory");
}

#undef FIR_STEP

#define BLOCK_ROWS 32
static int conv_fused(const uint8_t *src, int base, int nrows, uint8_t *dst,
                      int w, int stride, int y0, int y1)
{
    const int bytes = 3*w;
    uint8_t *zero = NULL;
    if (y0-base < 2 || y1-base+2 > nrows) {
        zero = (uint8_t *)calloc((size_t)bytes,1);
        if (!zero) return -1;
    }
    for (int y = y0; y < y1; ) {
        int height = y1-y < BLOCK_ROWS ? y1-y : BLOCK_ROWS;
        const uint8_t *rows[BLOCK_ROWS+4];
        for (int j = 0; j < height+4; ++j) {
            int r = y-base+j-2;
            rows[j] = (r < 0 || r >= nrows) ? zero : src+(size_t)r*stride;
        }
        uint8_t *d = dst+(size_t)y*stride;
        int start = bytes < 6 ? bytes : 6, end = start;
        for (; end+16 <= bytes-6; end+=16)
            fused_column16(rows,height+4,end,d+end,stride);
        for (int j=0; j<height; ++j) {
            for (int n=0; n<start; ++n) d[n] = baseline_round(src,nrows,stride,bytes,y-base+j,n);
            for (int n=end; n<bytes; ++n) d[n] = baseline_round(src,nrows,stride,bytes,y-base+j,n);
            if (j+1 < height) d += stride;
        }
        y += height;
    }
    free(zero);
    return 0;
}
#endif
