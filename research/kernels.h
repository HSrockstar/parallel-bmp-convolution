#ifndef CONV_KERNELS_H
#define CONV_KERNELS_H
#include <stdint.h>
#include <stddef.h>
typedef enum {
 M_RING,M_SYMMETRIC,M_TILED,M_VFIRST,M_BOX,M_PREFIX,M_Q15,M_WINOGRAD,
 M_AVX2,M_AVX512,M_DIRECT,M_SYM2D,M_EXACT,M_LUT,M_RANK3,M_FFT,M_OVERLAP,M_COUNT
} Method;
extern const char *const method_names[M_COUNT];
int method_find(const char *name);
int method_available(Method m);
/* Input/output must not overlap. Stride includes BMP padding. Each worker owns
   [y0,y1), including its output padding; halo is read-only. 0 success, -1 failure. */
int conv_band_run(Method m,const uint8_t *src,uint8_t *dst,int w,int h,int stride,
                  int y0,int y1,int tile);
#endif
