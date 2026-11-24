#include "kernels.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <limits.h>
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <immintrin.h>
#define X86_TARGET 1
#else
#define X86_TARGET 0
#endif
const char *const method_names[M_COUNT]={"ring","symmetric","tiled","vfirst","box","prefix","q15","winograd","avx2","avx512","direct","sym2d","exact-int","lut","rank3","fft","overlap-save"};
static const float a=.12007837208047906f,b=.2338807408740243f,c=.29208172229534735f;
static const double K[5][5]={
 {.01441881,.02808402,.03507270,.02808402,.01441881},
 {.02808402,.05470020,.06831229,.05470020,.02808402},
 {.03507270,.06831229,.08531173,.06831229,.03507270},
 {.02808402,.05470020,.06831229,.05470020,.02808402},
 {.01441881,.02808402,.03507270,.02808402,.01441881}};
static const uint32_t KI[5][5]={
 {1441881,2808402,3507270,2808402,1441881},
 {2808402,5470020,6831229,5470020,2808402},
 {3507270,6831229,8531173,6831229,3507270},
 {2808402,5470020,6831229,5470020,2808402},
 {1441881,2808402,3507270,2808402,1441881}};
int method_find(const char *s) {for(int i=0;i<M_COUNT;i++)if(!strcmp(s,method_names[i]))return i;return -1;}
int method_available(Method m) {
#if X86_TARGET
 if(m==M_AVX2)return __builtin_cpu_supports("avx2")!=0;
 if(m==M_AVX512)return __builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw");
#else
 if(m==M_AVX2||m==M_AVX512)return 0;
#endif
 return m>=0&&m<M_COUNT;
}
static uint8_t pack(double v) {return v<=0?0:v>=255?255:(uint8_t)(v+.5);}
static uint8_t packf(float v) { int o=(int)(v+.5f);return o<0?0:o>255?255:(uint8_t)o; }
static int slot(int y){return (y%5+5)%5;}
static float sample(const uint8_t *s,int n,int len){return n>=0&&n<len?s[n]:0;}
static float symmetric_value(const uint8_t *s,int n,int len) {
 return a*(sample(s,n-6,len)+sample(s,n+6,len))+b*(sample(s,n-3,len)+sample(s,n+3,len))+c*s[n];
}
/* Read exactly 8/16 bytes, never a full wider vector beyond a row. */
#if X86_TARGET
__attribute__((target("avx2"))) static __m256 load8(const uint8_t *s) {
 return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)s)));
}
__attribute__((target("avx2"))) static int horizontal8(float *d,const uint8_t *s,int lo,int hi) {
 int n=lo;__m256 aa=_mm256_set1_ps(a),bb=_mm256_set1_ps(b),cc=_mm256_set1_ps(c);
 for(;n+8<=hi;n+=8) {
  __m256 v=_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(aa,_mm256_add_ps(load8(s+n-6),load8(s+n+6))),_mm256_mul_ps(bb,_mm256_add_ps(load8(s+n-3),load8(s+n+3)))),_mm256_mul_ps(cc,load8(s+n)));
  _mm256_storeu_ps(d+n-lo,v);
 }return n;
}
__attribute__((target("avx512f,avx512bw"))) static __m512 load16(const uint8_t *s) {
 return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)s)));
}
__attribute__((target("avx512f,avx512bw"))) static int horizontal16(float *d,const uint8_t *s,int lo,int hi) {
 int n=lo;__m512 aa=_mm512_set1_ps(a),bb=_mm512_set1_ps(b),cc=_mm512_set1_ps(c);
 for(;n+16<=hi;n+=16) {
  __m512 v=_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(aa,_mm512_add_ps(load16(s+n-6),load16(s+n+6))),_mm512_mul_ps(bb,_mm512_add_ps(load16(s+n-3),load16(s+n+3)))),_mm512_mul_ps(cc,load16(s+n)));
  _mm512_storeu_ps(d+n-lo,v);
 }return n;
}
__attribute__((target("avx2"))) static int vertical8(uint8_t *d,const float *h0,const float *h1,const float *h2,const float *h3,const float *h4,int len) {
 int n=0;__m256 aa=_mm256_set1_ps(a),bb=_mm256_set1_ps(b),cc=_mm256_set1_ps(c);
 for(;n+8<=len;n+=8){
  __m256 v=_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(aa,_mm256_add_ps(_mm256_loadu_ps(h0+n),_mm256_loadu_ps(h4+n))),_mm256_mul_ps(bb,_mm256_add_ps(_mm256_loadu_ps(h1+n),_mm256_loadu_ps(h3+n)))),_mm256_mul_ps(cc,_mm256_loadu_ps(h2+n)));
  __m256i q=_mm256_cvttps_epi32(_mm256_add_ps(v,_mm256_set1_ps(.5f)));
  __m128i p=_mm_packus_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
  p=_mm_packus_epi16(p,p);_mm_storel_epi64((__m128i*)(d+n),p);
 }return n;
}
__attribute__((target("avx512f,avx512bw"))) static int vertical16(uint8_t *d,const float *h0,const float *h1,const float *h2,const float *h3,const float *h4,int len) {
 int n=0;__m512 aa=_mm512_set1_ps(a),bb=_mm512_set1_ps(b),cc=_mm512_set1_ps(c);
 for(;n+16<=len;n+=16){
  __m512 v=_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(aa,_mm512_add_ps(_mm512_loadu_ps(h0+n),_mm512_loadu_ps(h4+n))),_mm512_mul_ps(bb,_mm512_add_ps(_mm512_loadu_ps(h1+n),_mm512_loadu_ps(h3+n)))),_mm512_mul_ps(cc,_mm512_loadu_ps(h2+n)));
  __m512i q=_mm512_cvttps_epi32(_mm512_add_ps(v,_mm512_set1_ps(.5f)));
  _mm_storeu_si128((__m128i*)(d+n),_mm512_cvtusepi32_epi8(q));
 }return n;
}
#endif
/* F(2,3) applied to the first three taps, then two direct taps. The two
   outputs share four transformed products. Only horizontal pass transformed. */
static void horizontal_winograd(float *d,const uint8_t *s,int lo,int hi,int len) {
 int n=lo;
 for(;n+6<=hi;n+=6)for(int ch=0;ch<3;ch++){
  int p=n+ch;float d0=sample(s,p-6,len),d1=sample(s,p-3,len),d2=sample(s,p,len),d3=sample(s,p+3,len);
  float m0=(d0-d2)*a,m1=(d1+d2)*((a+b+c)*.5f),m2=(d2-d1)*((a-b+c)*.5f),m3=(d1-d3)*c;
  d[p-lo]=m0+m1+m2+b*d3+a*sample(s,p+6,len);
  d[p+3-lo]=m1-m2-m3+b*sample(s,p+6,len)+a*sample(s,p+9,len);
 }
 for(;n<hi;n++)d[n-lo]=symmetric_value(s,n,len);
}
static void horizontal(float *d,const uint8_t *s,int lo,int hi,int len,Method m,uint64_t *prefix) {
 if(!s){memset(d,0,(size_t)(hi-lo)*sizeof(*d));return;}
 if(m==M_WINOGRAD){horizontal_winograd(d,s,lo,hi,len);return;}
 if(m==M_BOX){
  for(int ch=0;ch<3;ch++){
   int p=lo+ch;float sum3=0,sum5=0;
   for(int k=-2;k<=2;k++){float v=sample(s,p+3*k,len);sum5+=v;if(k>=-1&&k<=1)sum3+=v;}
   for(;p<hi;p+=3){
    d[p-lo]=a*sum5+(b-a)*sum3+(c-b)*s[p];
    sum5+=sample(s,p+9,len)-sample(s,p-6,len);
    sum3+=sample(s,p+6,len)-sample(s,p-3,len);
   }
  }return;
 }
 if(m==M_PREFIX){
  int w=len/3;
  for(int ch=0;ch<3;ch++){
   prefix[0]=0;for(int x=0;x<w;x++)prefix[x+1]=prefix[x]+s[x*3+ch];
   for(int x=lo/3;x<hi/3;x++){
    int l5=x>1?x-2:0,r5=x+3<w?x+3:w,l3=x>0?x-1:0,r3=x+2<w?x+2:w;
    d[x*3+ch-lo]=a*(float)(prefix[r5]-prefix[l5])+(b-a)*(float)(prefix[r3]-prefix[l3])+(c-b)*s[x*3+ch];
   }
  }return;
 }
 int n=lo,begin=lo>6?lo:6,end=hi<len-6?hi:len-6;
 for(;n<hi&&n<begin;n++)d[n-lo]=symmetric_value(s,n,len);
#if X86_TARGET
 if(n<end&&m==M_AVX2)n=horizontal8(d+n-lo,s,n,end);
 if(n<end&&m==M_AVX512)n=horizontal16(d+n-lo,s,n,end);
#endif
 if(m==M_RING){
  for(;n<end;n++)d[n-lo]=a*s[n-6]+b*s[n-3]+c*s[n]+b*s[n+3]+a*s[n+6];
 }else for(;n<end;n++)d[n-lo]=a*(s[n-6]+s[n+6])+b*(s[n-3]+s[n+3])+c*s[n];
 for(;n<hi;n++)d[n-lo]=symmetric_value(s,n,len);
}
static int ring(Method m,const uint8_t *s,uint8_t *d,int w,int h,int stride,int y0,int y1,int tile) {
 int len=w*3,tw=m==M_TILED?(tile<w?tile:w):w;
 size_t sn=((size_t)tw*3+15)&~(size_t)15; /* each row starts on a cache line */
 float *buf=aligned_alloc(64,5*sn*sizeof(float));
 uint64_t *prefix=m==M_PREFIX?malloc(((size_t)w+1)*sizeof(uint64_t)):NULL;
 if(!buf||(m==M_PREFIX&&!prefix)){free(buf);free(prefix);return -1;}
 for(int x0=0;x0<w;){
  int width=w-x0<tw?w-x0:tw,lo=x0*3,hi=lo+width*3;
  for(int r=y0-2;r<y0+3;r++)horizontal(buf+(size_t)slot(r)*sn,r>=0&&r<h?s+(size_t)r*stride:NULL,lo,hi,len,m,prefix);
  for(int y=y0;y<y1;y++){
   float *h0=buf+(size_t)slot(y-2)*sn,*h1=buf+(size_t)slot(y-1)*sn,*h2=buf+(size_t)slot(y)*sn,*h3=buf+(size_t)slot(y+1)*sn,*h4=buf+(size_t)slot(y+2)*sn;
   uint8_t *out=d+(size_t)y*stride+lo;int n=0,count=hi-lo;
#if X86_TARGET
   if(m==M_AVX2)n=vertical8(out,h0,h1,h2,h3,h4,count);
   if(m==M_AVX512)n=vertical16(out,h0,h1,h2,h3,h4,count);
#endif
   if(m==M_RING)for(;n<count;n++)out[n]=packf(a*h0[n]+b*h1[n]+c*h2[n]+b*h3[n]+a*h4[n]);
   else for(;n<count;n++)out[n]=packf(a*(h0[n]+h4[n])+b*(h1[n]+h3[n])+c*h2[n]);
   int r=y+3;
   if(y+1<y1)horizontal(buf+(size_t)slot(r)*sn,r<h?s+(size_t)r*stride:NULL,lo,hi,len,m,prefix);
  }x0+=width;
 }
 free(prefix);free(buf);return 0;
}
static int vfirst(const uint8_t *s,uint8_t *d,int w,int h,int st,int y0,int y1) {
 int len=w*3;float *base=calloc((size_t)len+12,sizeof(float));if(!base)return -1;
 float *v=base+6;uint8_t *zero=calloc((size_t)len,1);if(!zero){free(base);return -1;}
 for(int y=y0;y<y1;y++){
  const uint8_t *r[5];for(int k=0;k<5;k++){int yy=y+k-2;r[k]=yy>=0&&yy<h?s+(size_t)yy*st:zero;}
  for(int n=0;n<len;n++)v[n]=a*(r[0][n]+r[4][n])+b*(r[1][n]+r[3][n])+c*r[2][n];
  for(int n=0;n<len;n++)d[(size_t)y*st+n]=packf(a*(v[n-6]+v[n+6])+b*(v[n-3]+v[n+3])+c*v[n]);
 }free(zero);free(base);return 0;
}
static int q15(const uint8_t *s,uint8_t *d,int w,int h,int st,int y0,int y1) {
 /* Full Q15 horizontal value in int32, NO intermediate rounding. Q30 vertical
    sum uses uint64. q sum=32768 and max final numerator=255*2^30. */
 const uint32_t q[5]={3935,7664,9570,7664,3935};int len=w*3;
 uint32_t *buf=malloc((size_t)5*len*sizeof(*buf));if(!buf)return -1;
 for(int y=y0-2;y<y1+2;y++){
  uint32_t *row=buf+(size_t)slot(y)*len;
  if(y<0||y>=h)memset(row,0,(size_t)len*sizeof(*row));
  else for(int n=0;n<len;n++){
   uint32_t v=0;for(int k=0;k<5;k++){int p=n+3*(k-2);if(p>=0&&p<len)v+=q[k]*s[(size_t)y*st+p];}row[n]=v;
  }
  if(y>=y0+2){int yy=y-2;uint8_t *out=d+(size_t)yy*st;
   for(int n=0;n<len;n++){uint64_t v=UINT64_C(1)<<29;
    for(int k=0;k<5;k++)v+=(uint64_t)q[k]*buf[(size_t)slot(yy+k-2)*len+n];
    out[n]=(uint8_t)(v>>30);
   }
  }
 }free(buf);return 0;
}
static int direct(Method m,const uint8_t *s,uint8_t *d,int w,int h,int st,int y0,int y1) {
 uint64_t (*lut)[256]=NULL;
 if(m==M_LUT){lut=malloc(25*sizeof(*lut));if(!lut)return -1;
  for(int k=0;k<25;k++)for(int v=0;v<256;v++)lut[k][v]=(uint64_t)KI[k/5][k%5]*v;
 }
 for(int y=y0;y<y1;y++)for(int x=0;x<w;x++)for(int ch=0;ch<3;ch++){
  double sum=0;uint64_t total=0;
  if(m==M_SYM2D){
   for(int dy=0;dy<3;dy++)for(int dx=0;dx<3;dx++){
    unsigned group=0;
    for(int iy=0;iy<(dy?2:1);iy++)for(int ix=0;ix<(dx?2:1);ix++){
     int yy=y+(iy?-dy:dy),xx=x+(ix?-dx:dx);
     if(yy>=0&&yy<h&&xx>=0&&xx<w)group+=s[(size_t)yy*st+xx*3+ch];
    }sum+=K[2+dy][2+dx]*group;
   }
  }else for(int ky=0;ky<5;ky++){int yy=y+ky-2;if(yy<0||yy>=h)continue;
   for(int kx=0;kx<5;kx++){int xx=x+kx-2;if(xx<0||xx>=w)continue;
    uint8_t v=s[(size_t)yy*st+xx*3+ch];
    if(m==M_EXACT)total+=(uint64_t)KI[ky][kx]*v;
    else if(m==M_LUT)total+=lut[ky*5+kx][v];
    else sum+=K[ky][kx]*v;
   }
  }
  d[(size_t)y*st+x*3+ch]=(m==M_EXACT||m==M_LUT)?(uint8_t)((total+50000000)/100000000):pack(sum);
 }free(lut);return 0;
}
static int rank3(const uint8_t *s,uint8_t *d,int w,int h,int st,int y0,int y1) {
 /* Rounded 2D kernel has three distinct rows. Exact factorization into three
    horizontal filters and vertical selectors [1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0]. */
 int len=3*w;double *buf=malloc((size_t)15*len*sizeof(double));if(!buf)return -1;
 for(int r=y0-2;r<y1+2;r++){
  for(int k=0;k<3;k++){double *row=buf+((size_t)slot(r)*3+k)*len;
   for(int n=0;n<len;n++){
    double v=0;if(r>=0&&r<h)for(int dx=-2;dx<=2;dx++){int p=n+3*dx;if(p>=0&&p<len)v+=K[k][dx+2]*s[(size_t)r*st+p];}row[n]=v;
   }
  }
  if(r>=y0+2){int y=r-2;
   for(int n=0;n<len;n++){double v=0;
    for(int k=0;k<5;k++){int row=k<3?k:4-k;v+=buf[((size_t)slot(y+k-2)*3+row)*len+n];}
    d[(size_t)y*st+n]=pack(v);
   }
  }
 }free(buf);return 0;
}
/* Iterative radix-2 DFT. Inverse is normalized once per dimension. */
static void fft1(double complex *a0,size_t n,int inverse) {
 for(size_t i=1,j=0;i<n;i++){size_t bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){double complex t=a0[i];a0[i]=a0[j];a0[j]=t;}}
 for(size_t len=2;len<=n;len*=2){double angle=(inverse?2:-2)*3.14159265358979323846/(double)len;double complex root=cos(angle)+I*sin(angle);
  for(size_t i=0;i<n;i+=len){double complex z=1;for(size_t j=0;j<len/2;j++){double complex u=a0[i+j],v=z*a0[i+j+len/2];a0[i+j]=u+v;a0[i+j+len/2]=u-v;z*=root;}}
  if(len==n)break;
 }if(inverse)for(size_t i=0;i<n;i++)a0[i]/=(double)n;
}
static void fft2(double complex *a0,size_t nx,size_t ny,int inv,double complex *col) {
 for(size_t y=0;y<ny;y++)fft1(a0+y*nx,nx,inv);
 for(size_t x=0;x<nx;x++){for(size_t y=0;y<ny;y++)col[y]=a0[y*nx+x];fft1(col,ny,inv);for(size_t y=0;y<ny;y++)a0[y*nx+x]=col[y];}
}
static size_t pow2(size_t n){size_t p=1;while(p<n){if(p>SIZE_MAX/2)return 0;p*=2;}return p;}
static int frequency(Method m,const uint8_t *s,uint8_t *d,int w,int h,int st,int y0,int y1,int tile) {
 size_t nx=pow2(m==M_FFT?(size_t)w+4:(size_t)tile+4),ny=pow2(m==M_FFT?(size_t)(y1-y0)+4:64);
 /* Explicit workspace cap per worker; failures are surfaced, never silently
    substituted with another method. Full-frame FFT is intentionally expensive. */
 if(!nx||!ny||nx>SIZE_MAX/ny||nx*ny>256*1024*1024/(2*sizeof(double complex)))return -1;
 size_t n=nx*ny;double complex *v=calloc(n,sizeof(*v)),*k=calloc(n,sizeof(*k)),*col=malloc(ny*sizeof(*col));
 if(!v||!k||!col){free(v);free(k);free(col);return -1;}
 for(int y=0;y<5;y++)for(int x=0;x<5;x++)k[(size_t)y*nx+x]=K[y][x];
 fft2(k,nx,ny,0,col);
 for(int yb=y0;yb<y1;){int bh=(size_t)(y1-yb)<ny-4?y1-yb:(int)(ny-4);
  for(int xb=0;xb<w;){int bw=(size_t)(w-xb)<nx-4?w-xb:(int)(nx-4);
   for(int ch=0;ch<3;ch++){
    memset(v,0,n*sizeof(*v));
    for(size_t y=0;y<ny;y++){int64_t yy=(int64_t)yb+(int64_t)y-2;if(yy<0||yy>=h)continue;
     for(size_t x=0;x<nx;x++){int64_t xx=(int64_t)xb+(int64_t)x-2;if(xx>=0&&xx<w)v[y*nx+x]=s[(size_t)yy*st+(size_t)xx*3+ch];}}
    fft2(v,nx,ny,0,col);for(size_t i=0;i<n;i++)v[i]*=k[i];fft2(v,nx,ny,1,col);
    for(int y=0;y<bh;y++)for(int x=0;x<bw;x++)d[(size_t)(yb+y)*st+(xb+x)*3+ch]=pack(creal(v[(size_t)(y+4)*nx+x+4]));
   }xb+=bw;
  }yb+=bh;
 }free(v);free(k);free(col);return 0;
}
int conv_band_run(Method m,const uint8_t *s,uint8_t *d,int w,int h,int st,int y0,int y1,int tile) {
 if(!s||!d||s==d||w<1||w>(INT_MAX-32)/3||h<1||st<w*3||y0<0||y1>h||y0>y1||tile<1||!method_available(m))return -1;
 if((size_t)w>SIZE_MAX/(15*sizeof(double))||(size_t)h>SIZE_MAX/(size_t)st)return -1;
 if(y0==y1)return 0;
 int rc;
 if(m==M_DIRECT||m==M_SYM2D||m==M_EXACT||m==M_LUT)rc=direct(m,s,d,w,h,st,y0,y1);
 else if(m==M_RANK3)rc=rank3(s,d,w,h,st,y0,y1);
 else if(m==M_VFIRST)rc=vfirst(s,d,w,h,st,y0,y1);
 else if(m==M_Q15)rc=q15(s,d,w,h,st,y0,y1);
 else if(m==M_FFT||m==M_OVERLAP)rc=frequency(m,s,d,w,h,st,y0,y1,tile);
 else rc=ring(m,s,d,w,h,st,y0,y1,tile);
 if(!rc)for(int y=y0;y<y1;y++)memset(d+(size_t)y*st+w*3,0,(size_t)(st-w*3));
 return rc;
}
