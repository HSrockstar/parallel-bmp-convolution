/* ASan sees exact heap ends; BMP page rounding must not hide overreads. */
#include "kernels.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
int main(void){
 for(int w=1;w<266;w++){
  int h=1+w%7,st=w*3;size_t size=(size_t)st*h;
  unsigned char *s=malloc(size),*d=malloc(size),*ref=malloc(size);
  if(!s||!d||!ref)return 2;
  for(size_t i=0;i<size;i++)s[i]=(unsigned char)(i*31+w);
  if(conv_band_run(M_EXACT,s,ref,w,h,st,0,h,7))return 3;
  for(int m=0;m<M_COUNT;m++){
   if(!method_available((Method)m))continue;
   memset(d,197,size);
   for(int y=0;y<h;y++)if(conv_band_run((Method)m,s,d,w,h,st,y,y+1,7))return 4;
   for(size_t i=0;i<size;i++)if(abs((int)d[i]-(int)ref[i])>1){fprintf(stderr,"m=%d w=%d byte=%zu\n",m,w,i);return 5;}
  }
  free(s);free(d);free(ref);
 }puts("265 exact-allocation shapes passed");return 0;
}
