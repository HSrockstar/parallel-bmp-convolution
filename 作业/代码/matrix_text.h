#ifndef MATRIX_TEXT_H
#define MATRIX_TEXT_H
/* 每项含十进制数字和一个空格。固定四字节拷贝，实际前进 2 至 4 字节，
 * 下一项覆盖多写的尾字节。行末把最后一个空格换成换行。 */
typedef struct { char chars[4]; unsigned char length; } DecimalByte;

typedef struct { char chars[8]; unsigned char length; } DecimalPair;

/* 两个通道合并为一次拷贝。小图和预览只使用单字节查找表。 */
static DecimalPair *make_pairs(const DecimalByte *lut)
{
    DecimalPair *pairs = (DecimalPair *)malloc(65536 * sizeof(*pairs));
    if (!pairs) return NULL;
    for (int b = 0; b < 256; ++b) {
        for (int a = 0; a < 256; ++a) {
            DecimalPair *pair = &pairs[(b << 8) | a];
            memset(pair->chars,0,8);
            memcpy(pair->chars,lut[a].chars,4);
            memcpy(pair->chars+lut[a].length,lut[b].chars,4);
            pair->length = lut[a].length + lut[b].length;
        }
    }
    return pairs;
}

static int dump_matrix(const char *base, const uint8_t *data, const BmpInfo *b, int full)
{
    char path[1024];
    if (snprintf(path,sizeof(path),"%s.matrix.txt",base) >= (int)sizeof(path)) {
        fprintf(stderr,"matrix output path too long\n");
        return -1;
    }
    FILE *fp = fopen(path,"w");
    if (!fp) { perror(path); return -1; }
    int rows = full || b->h < 20 ? b->h : 20;
    int cols = full || b->w < 20 ? b->w : 20;
    int large = (size_t)rows * cols >= 65536;
    char *io = large ? (char *)malloc(256*1024) : NULL;
    if (io) setvbuf(fp,io,_IOFBF,256*1024);
    fprintf(fp,"# image=%dx%d channels=3 order=BGR rows=top-down\n",b->w,b->h);
    fprintf(fp,"# exported=%dx%d scope=%s; each data line is one image row\n",
            cols,rows,full ? "full" : "preview");
    DecimalByte lut[256] = {0};
    for (int i = 0; i < 256; ++i) {
        char repr[5];
        int len = snprintf(repr,sizeof(repr),"%d ",i);
        memcpy(lut[i].chars,repr,(size_t)len);
        lut[i].length = (unsigned char)len;
    }
    enum { CHUNK = 8192 };
    char block[4*CHUNK];
    DecimalPair *pairs = large ? make_pairs(lut) : NULL;
    int failed = ferror(fp);
    for (int y = 0; y < rows && !failed; ++y) {
        int r = b->topdown ? y : b->h-1-y;
        const uint8_t *src = data+(size_t)r*b->stride;
        int left = cols*3;
        while (left > 0) {
            int count = left < CHUNK ? left : CHUNK;
            char *end = block;
            int i = 0;
            if (pairs) {
                for (; i+2 <= count; i+=2) {
                    unsigned key = src[i] | ((unsigned)src[i+1] << 8);
                    memcpy(end,pairs[key].chars,8);
                    end += pairs[key].length;
                }
            }
            for (; i < count; ++i) {
                const DecimalByte *d = &lut[src[i]];
                memcpy(end,d->chars,4);
                end += d->length;
            }
            src += count;
            left -= count;
            if (!left) end[-1] = '\n';
            size_t bytes = (size_t)(end-block);
            if (fwrite(block,1,bytes,fp) != bytes) { failed = 1; break; }
        }
    }
    if (fclose(fp) != 0) failed = 1;
    free(io);
    free(pairs);
    if (failed) { fprintf(stderr,"cannot finish %s\n",path); return -1; }
    printf("matrix txt      : %s (%s, %d x %d)\n",path,full ? "full matrix" : "preview",cols,rows);
    return 0;
}
#endif
