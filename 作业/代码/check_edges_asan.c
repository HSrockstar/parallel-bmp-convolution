/* hpass_row 单行地址检查：编译时指定 -DCONV_SOURCE='"conv_pthread.c"' 或 omp。
 * 只分配当前行的精确大小，避免 BMP 映射中相邻行/文件头掩盖越界访问。
 * 必须配合 AddressSanitizer 使用；不参与正式卷积构建与计时。 */
#define main conv_program_main
#include CONV_SOURCE
#undef main

int main(void)
{
    for (int w = 1; w <= 12; w++) {
        size_t n = (size_t)w * 3;
        uint8_t *row = malloc(n);
        float *h = malloc(n * sizeof(*h));
        if (!row || !h) return 2;
        for (size_t i = 0; i < n; i++) row[i] = (uint8_t)(31 + i * 29);
        hpass_row(h, row, w);
        free(h);
        free(row);
    }
    return 0;
}
