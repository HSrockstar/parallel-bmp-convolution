// Original CUDA kernels loaded by gpu.py through CuPy/NVRTC.
// All buffers are tightly packed BGR. No boundary renormalization.
extern "C" {
__device__ __constant__ float k1[5]={.12007837208047906f,.2338807408740243f,.29208172229534735f,.2338807408740243f,.12007837208047906f};
__device__ __constant__ float k2[25]={
 .01441881f,.02808402f,.03507270f,.02808402f,.01441881f,
 .02808402f,.05470020f,.06831229f,.05470020f,.02808402f,
 .03507270f,.06831229f,.08531173f,.06831229f,.03507270f,
 .02808402f,.05470020f,.06831229f,.05470020f,.02808402f,
 .01441881f,.02808402f,.03507270f,.02808402f,.01441881f};
__device__ unsigned char quantize(float v){return (unsigned char)fminf(255.f,fmaxf(0.f,floorf(v+.5f)));}
__global__ void direct(const unsigned char *s,unsigned char *d,int w,int h){
 int n=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 int len=w*3;if(n>=len||y>=h)return;
 float v=0;
 for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){
  int yy=y+dy,nn=n+dx*3;
  if(yy>=0&&yy<h&&nn>=0&&nn<len)v+=k2[(dy+2)*5+dx+2]*s[(unsigned long long)yy*len+nn];
 }d[(unsigned long long)y*len+n]=quantize(v);
}
__global__ void horizontal(const unsigned char *s,float *t,int w,int h){
 int n=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 int len=w*3;if(n>=len||y>=h)return;float v=0;
 for(int k=-2;k<=2;k++){int nn=n+3*k;if(nn>=0&&nn<len)v+=k1[k+2]*s[(unsigned long long)y*len+nn];}
 t[(unsigned long long)y*len+n]=v;
}
__global__ void vertical(const float *t,unsigned char *d,int w,int h){
 int n=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 int len=w*3;if(n>=len||y>=h)return;float v=0;
 for(int k=-2;k<=2;k++){int yy=y+k;if(yy>=0&&yy<h)v+=k1[k+2]*t[(unsigned long long)yy*len+n];}
 d[(unsigned long long)y*len+n]=quantize(v);
}
__global__ void fused(const unsigned char *s,unsigned char *d,int w,int h){
 // 32 byte positions by 8 rows, with six byte / two row halos.
 // All 256 threads reach both barriers, including partial edge blocks.
 __shared__ unsigned char input[12][44];
 __shared__ float intermediate[12][32];
 int tid=threadIdx.y*32+threadIdx.x,len=w*3;
 int bx=blockIdx.x*32,by=blockIdx.y*8;
 for(int i=tid;i<12*44;i+=256){int ly=i/44,lx=i%44,y=by+ly-2,n=bx+lx-6;
  input[ly][lx]=(y>=0&&y<h&&n>=0&&n<len)?s[(unsigned long long)y*len+n]:0;
 }
 __syncthreads();
 for(int i=tid;i<12*32;i+=256){int ly=i/32,lx=i%32;float v=0;
  for(int k=0;k<5;k++)v+=k1[k]*input[ly][lx+3*k];intermediate[ly][lx]=v;
 }
 __syncthreads();
 int n=bx+threadIdx.x,y=by+threadIdx.y;
 if(n<len&&y<h){float v=0;for(int k=0;k<5;k++)v+=k1[k]*intermediate[threadIdx.y+k][threadIdx.x];d[(unsigned long long)y*len+n]=quantize(v);}
}
}
