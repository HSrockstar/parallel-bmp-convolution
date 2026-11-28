/* 编译时指定 CONV_SOURCE。ASan 检查标量水平卷积；保护页检查直接
 * 用汇编读写的 AVX2 边界（编译器无法给这些内存访问插入 ASan 检查）。 */
#define main conv_program_main
#include CONV_SOURCE
#undef main

#if defined(__x86_64__) && defined(__GNUC__)
static int check_vector_bounds(void)
{
    if (!have_avx2()) return 0;
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    enum { HEIGHT = 32 };
    size_t span = 3*page;
    uint8_t *m = mmap(NULL,span*(HEIGHT+1),PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (m == MAP_FAILED) return 1;
    for (int j = 0; j <= HEIGHT; ++j)
        if (mprotect(m+j*span+page,page,PROT_READ|PROT_WRITE)) return 1;
    for (int width = 10; width <= 375; ++width) {
        int n = 3*width;
        for (int at_end = 0; at_end <= 1; ++at_end) {
            size_t offset = page + (at_end ? page-(size_t)n : 0);
            uint8_t *input = m+offset;
            uint8_t *dst = input+span;
            const uint8_t *rows[HEIGHT+4];
            for (int j = 0; j < HEIGHT+4; ++j) rows[j] = input;
            for (int j = 0; j < n; ++j) input[j] = (uint8_t)(31+j*29);
            for (int x = 6; x+16 <= n-6; x+=16)
                fused_column16(rows,HEIGHT+4,x,dst+x,(int)span);
        }
    }
    return munmap(m,span*(HEIGHT+1)) != 0;
}
#endif

int main(void)
{
    for (int w = 1; w <= 375; ++w) {
        size_t n = (size_t)w*3;
        uint8_t *row = (uint8_t *)malloc(n);
        float *h = (float *)malloc(n*sizeof(*h));
        if (!row || !h) return 2;
        for (size_t i = 0; i < n; ++i) row[i] = (uint8_t)(31+i*29);
        hpass_row(h,row,w);
        free(h);
        free(row);
    }
#if defined(__x86_64__) && defined(__GNUC__)
    return check_vector_bounds();
#else
    return 0;
#endif
}
