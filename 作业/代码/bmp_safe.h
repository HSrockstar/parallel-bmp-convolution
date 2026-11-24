#ifndef BMP_SAFE_H
#define BMP_SAFE_H
/* Shared little-endian BMP validation. No unaligned or aliasing-unsafe casts. */
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
typedef struct { int w,h,stride,topdown; off_t data_off; } BmpInfo;
static uint16_t bmp_u16(const uint8_t *p) { return (uint16_t)(p[0] | (uint16_t)p[1]<<8); }
static uint32_t bmp_u32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24; }
static void bmp_put16(uint8_t *p,uint16_t n) { p[0]=(uint8_t)n; p[1]=(uint8_t)(n>>8); }
static void bmp_put32(uint8_t *p,uint32_t n) { for(int i=0;i<4;i++) p[i]=(uint8_t)(n>>(8*i)); }
static int bmp_parse_header(const uint8_t *p,BmpInfo *b) {
    uint32_t dib=bmp_u32(p+14),off=bmp_u32(p+10),wu=bmp_u32(p+18),hu=bmp_u32(p+22);
    int64_t h=hu<=INT32_MAX?(int64_t)hu:(int64_t)hu-4294967296LL;
    if(p[0]!='B'||p[1]!='M'||dib<40||dib>UINT32_MAX-14||off<14+dib||
       bmp_u16(p+26)!=1||bmp_u16(p+28)!=24||bmp_u32(p+30)!=0||
       wu==0||wu>(INT_MAX-32)/3||h==0||h==INT32_MIN) return -1;
    uint64_t stride=((uint64_t)wu*3+3)&~UINT64_C(3), ah=(uint64_t)(h<0?-h:h);
    /* Conventional BMP has a 32-bit file-size field; reject >4 GiB. */
    if(stride*ah>UINT32_MAX-54||stride*ah>SIZE_MAX-off) return -1;
    b->w=(int)wu; b->h=(int)ah; b->stride=(int)stride;
    b->topdown=h<0; b->data_off=(off_t)off;
    return 0;
}
static int bmp_file_size(int fd,const BmpInfo *b,off_t *size) {
    struct stat st;
    if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<0||
       (uint64_t)st.st_size>SIZE_MAX||
       (uint64_t)st.st_size<(uint64_t)b->data_off+(uint64_t)b->stride*b->h) return -1;
    *size=st.st_size; return 0;
}
static void bmp_make_header(const BmpInfo *b,uint8_t *p) {
    uint32_t n=(uint32_t)b->stride*(uint32_t)b->h;
    memset(p,0,54); p[0]='B';p[1]='M';
    bmp_put32(p+2,54+n);bmp_put32(p+10,54);bmp_put32(p+14,40);
    bmp_put32(p+18,(uint32_t)b->w);bmp_put32(p+22,b->topdown?0U-(uint32_t)b->h:(uint32_t)b->h);
    bmp_put16(p+26,1);bmp_put16(p+28,24);bmp_put32(p+34,n);
}
/* Check descriptors BEFORE truncation, including symlinks and hard links. */
static int bmp_open_output(const char *path,int input_fd) {
    int fd=open(path,O_RDWR|O_CREAT|O_NONBLOCK,0644);struct stat a,b;
    if(fd<0) return -1;
    if(fstat(input_fd,&a)||fstat(fd,&b)||!S_ISREG(b.st_mode)||(a.st_dev==b.st_dev&&a.st_ino==b.st_ino)) {
        close(fd);errno=EINVAL;return -1;
    }
    if(ftruncate(fd,0)) { int e=errno;close(fd);errno=e;return -1; }
    return fd;
}
static int bmp_preallocate(int fd,off_t n) {
    int e=posix_fallocate(fd,0,n);
    if(e==0||e==EOPNOTSUPP||e==ENOSYS||e==EINVAL) return 0;
    errno=e;return -1;
}
static int bmp_positive_int(const char *s,int *out) {
    unsigned n=0;if(!s||!*s) return -1;
    for(;*s;s++) { if(*s<'0'||*s>'9'||n>((unsigned)INT_MAX-(unsigned)(*s-'0'))/10) return -1;n=n*10+(unsigned)(*s-'0'); }
    if(!n) return -1;
    *out=(int)n;return 0;
}
#endif
