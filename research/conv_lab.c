#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sched.h>
#include <pthread.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "kernels.h"
#include "../作业/代码/bmp_safe.h"
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int transfer(int fd,void *buf,size_t n,off_t off,int writing){
 size_t p=0;while(p<n){size_t chunk=n-p>1048576?1048576:n-p;
  ssize_t r=writing?pwrite(fd,(char*)buf+p,chunk,off+(off_t)p):pread(fd,(char*)buf+p,chunk,off+(off_t)p);
  if(r<0&&errno==EINTR)continue;
  if(r<=0)return -1;
  p+=(size_t)r;
 }return 0;
}
typedef struct Pool Pool;
typedef struct {Pool *p;int id;unsigned generation;pthread_t thread;int status;} Worker;
struct Pool {
 const uint8_t *src;uint8_t *dst;int w,h,st,n,tile,pin;Method method;
 pthread_mutex_t lock;pthread_cond_t start,done;
 unsigned generation;int stop,pending;Worker *workers;
 cpu_set_t allowed;
};
static int pin_worker(Pool *p,int tid){
 if(!p->pin)return 0;
 int count=CPU_COUNT(&p->allowed);if(!count)return -1;
 int target=tid%count;
 for(int cpu=0;cpu<CPU_SETSIZE;cpu++)if(CPU_ISSET(cpu,&p->allowed)&&target--==0){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);return pthread_setaffinity_np(pthread_self(),sizeof(set),&set);}
 return -1;
}
static int work(Pool *p,int tid,int nt){
 int y0=(int)((int64_t)p->h*tid/nt),y1=(int)((int64_t)p->h*(tid+1)/nt);
 return conv_band_run(p->method,p->src,p->dst,p->w,p->h,p->st,y0,y1,p->tile);
}
#ifndef USE_OPENMP
static void *worker(void *arg){
 Worker *w=arg;Pool *p=w->p;int affinity_error=pin_worker(p,w->id);
 pthread_mutex_lock(&p->lock);
 for(;;){
  while(!p->stop&&w->generation==p->generation)pthread_cond_wait(&p->start,&p->lock);
  if(p->stop)break;
  w->generation=p->generation;pthread_mutex_unlock(&p->lock);
  int status=affinity_error?affinity_error:work(p,w->id,p->n);
  pthread_mutex_lock(&p->lock);w->status=status;if(--p->pending==0)pthread_cond_signal(&p->done);
 }
 pthread_mutex_unlock(&p->lock);return NULL;
}
#endif
static int pool_init(Pool *p){
 if(p->pin&&sched_getaffinity(0,sizeof(p->allowed),&p->allowed))return -1;
#ifdef USE_OPENMP
 omp_set_dynamic(0);omp_set_num_threads(p->n);return 0;
#else
 if(p->n==1)return pin_worker(p,0);
 p->workers=calloc((size_t)p->n,sizeof(Worker));if(!p->workers)return -1;
 if(pthread_mutex_init(&p->lock,NULL)){free(p->workers);p->workers=NULL;return -1;}
 if(pthread_cond_init(&p->start,NULL)){pthread_mutex_destroy(&p->lock);free(p->workers);p->workers=NULL;return -1;}
 if(pthread_cond_init(&p->done,NULL)){pthread_cond_destroy(&p->start);pthread_mutex_destroy(&p->lock);free(p->workers);p->workers=NULL;return -1;}
 for(int i=0;i<p->n;i++){
  p->workers[i].p=p;p->workers[i].id=i;
  if(pthread_create(&p->workers[i].thread,NULL,worker,&p->workers[i])){
   pthread_mutex_lock(&p->lock);p->stop=1;pthread_cond_broadcast(&p->start);pthread_mutex_unlock(&p->lock);
   for(int j=0;j<i;j++)pthread_join(p->workers[j].thread,NULL);
   pthread_cond_destroy(&p->done);pthread_cond_destroy(&p->start);pthread_mutex_destroy(&p->lock);free(p->workers);p->workers=NULL;return -1;
  }
 }return 0;
#endif
}
static int pool_run(Pool *p,int *actual){
#ifdef USE_OPENMP
 int error=0;
 #pragma omp parallel reduction(|:error)
 {
  int tid=omp_get_thread_num(),nt=omp_get_num_threads();
  #pragma omp single
  *actual=nt;
  error|=pin_worker(p,tid);error|=work(p,tid,nt);
 }return error;
#else
 *actual=p->n;if(p->n==1)return work(p,0,1);
 pthread_mutex_lock(&p->lock);p->pending=p->n;p->generation++;pthread_cond_broadcast(&p->start);
 while(p->pending)pthread_cond_wait(&p->done,&p->lock);
 int error=0;for(int i=0;i<p->n;i++)error|=p->workers[i].status;
 pthread_mutex_unlock(&p->lock);return error;
#endif
}
static void pool_close(Pool *p){
#ifndef USE_OPENMP
 if(!p->workers)return;
 pthread_mutex_lock(&p->lock);p->stop=1;pthread_cond_broadcast(&p->start);pthread_mutex_unlock(&p->lock);
 for(int i=0;i<p->n;i++)pthread_join(p->workers[i].thread,NULL);
 pthread_cond_destroy(&p->done);pthread_cond_destroy(&p->start);pthread_mutex_destroy(&p->lock);free(p->workers);
#else
 (void)p;
#endif
}
static int compare_double(const void *a0,const void *b0){double x=*(const double*)a0,y=*(const double*)b0;return(x>y)-(x<y);}
static void usage(const char *name){fprintf(stderr,"Usage: %s input.bmp output.bmp [--method NAME] [--threads N] [--tile N]\n  [--repeat N] [--warmup N] [--io mmap|buffered] [--sync] [--pin]\n  [--prefault] [--advice sequential|huge] [--no-preallocate]\n  %s --list\n",name,name);}
int main(int argc,char **argv){
 if(argc==2&&!strcmp(argv[1],"--list")){for(int i=0;i<M_COUNT;i++)printf("%s %s\n",method_names[i],method_available((Method)i)?"available":"unavailable");return 0;}
 if(argc<3){usage(argv[0]);return 1;}
 int threads=4,tile=128,repeat=1,warmup=0,buffered=0,sync_output=0,pin=0,prefault=0,advice=0,preallocate=1;Method method=M_RING;
 for(int i=3;i<argc;i++){
  if(!strcmp(argv[i],"--sync")){sync_output=1;continue;}
  if(!strcmp(argv[i],"--pin")){pin=1;continue;}
  if(!strcmp(argv[i],"--prefault")){prefault=1;continue;}
  if(!strcmp(argv[i],"--no-preallocate")){preallocate=0;continue;}
  if(i+1==argc){usage(argv[0]);return 1;}
  const char *key=argv[i],*value=argv[++i];
  if(!strcmp(key,"--method")){int m=method_find(value);if(m<0){fprintf(stderr,"unknown method\n");return 1;}method=(Method)m;}
  else if(!strcmp(key,"--io")){if(!strcmp(value,"buffered"))buffered=1;else if(strcmp(value,"mmap")){usage(argv[0]);return 1;}}
  else if(!strcmp(key,"--advice")){if(!strcmp(value,"sequential"))advice=MADV_SEQUENTIAL;else if(!strcmp(value,"huge"))advice=MADV_HUGEPAGE;else{usage(argv[0]);return 1;}}
  else {
   int n;if(!strcmp(key,"--warmup")&&!strcmp(value,"0")){warmup=0;continue;}
   if(bmp_positive_int(value,&n)){usage(argv[0]);return 1;}
   if(!strcmp(key,"--threads"))threads=n;else if(!strcmp(key,"--tile"))tile=n;else if(!strcmp(key,"--repeat"))repeat=n;else if(!strcmp(key,"--warmup"))warmup=n;else{usage(argv[0]);return 1;}
  }
 }
 if(threads>4096||tile>65536||repeat>10000||warmup>10000){fprintf(stderr,"parameter resource limit exceeded\n");return 1;}
 if(!method_available(method)){fprintf(stderr,"requested ISA unavailable\n");return 2;}
 int rc=1,in=-1,out=-1,started=0,actual=0;size_t insize=0,outsize=0,pixels=0;
 uint8_t *im=NULL,*om=NULL,*src=NULL,*dst=NULL;double *samples=NULL;Pool p={0};
 double begin=now(),setup_end=0,compute_end=0,finish=0;
 uint8_t header[54];BmpInfo info;off_t filesize;
 in=open(argv[1],O_RDONLY|O_NONBLOCK);if(in<0){perror("input");goto done;}
 if(transfer(in,header,54,0,0)||bmp_parse_header(header,&info)||bmp_file_size(in,&info,&filesize)){fprintf(stderr,"invalid/truncated BMP\n");goto done;}
 insize=(size_t)filesize;pixels=(size_t)info.stride*info.h;outsize=54+pixels;
 if(threads>info.h)threads=info.h;
 if(buffered){src=malloc(pixels);dst=malloc(pixels);if(!src||!dst||transfer(in,src,pixels,info.data_off,0)){fprintf(stderr,"buffered input allocation/read failed\n");goto done;}}
 else {im=mmap(NULL,insize,PROT_READ,MAP_PRIVATE,in,0);if(im==MAP_FAILED){im=NULL;perror("input mmap");goto done;}src=im+info.data_off;}
 out=bmp_open_output(argv[2],in);if(out<0){perror("output (must differ from input)");goto done;}
 if(ftruncate(out,(off_t)outsize)||(preallocate&&bmp_preallocate(out,(off_t)outsize))){perror("output allocation");goto done;}
 bmp_make_header(&info,header);if(transfer(out,header,54,0,1)){perror("header write");goto done;}
 if(!buffered){om=mmap(NULL,outsize,PROT_READ|PROT_WRITE,MAP_SHARED,out,0);if(om==MAP_FAILED){om=NULL;perror("output mmap");goto done;}dst=om+54;}
 if(advice){if(!buffered){if(madvise(im,insize,advice)||madvise(om,outsize,advice)){perror("madvise");goto done;}}else{fprintf(stderr,"--advice requires --io mmap\n");goto done;}}
 if(prefault){volatile unsigned sum=0;long page=sysconf(_SC_PAGESIZE);if(page<=0)page=4096;for(size_t i=0;i<pixels;i+=(size_t)page)sum+=src[i];(void)sum;memset(dst,0,pixels);}
 samples=malloc((size_t)repeat*sizeof(*samples));if(!samples){perror("samples");goto done;}
 p.src=src;p.dst=dst;p.w=info.w;p.h=info.h;p.st=info.stride;p.n=threads;p.tile=tile;p.method=method;p.pin=pin;
 if(pool_init(&p)){fprintf(stderr,"thread initialization failed\n");goto done;}started=1;setup_end=now();
 for(int i=0;i<warmup+repeat;i++){double t=now();if(pool_run(&p,&actual)){fprintf(stderr,"kernel/workspace/affinity failure\n");goto done;}if(i>=warmup)samples[i-warmup]=(now()-t)*1000;}
 compute_end=now();pool_close(&p);started=0;
 if(buffered&&transfer(out,dst,pixels,54,1)){perror("output write");goto done;}
 if(sync_output&&((!buffered&&msync(om,outsize,MS_SYNC))||fsync(out))){perror("sync");goto done;}
 if(im){if(munmap(im,insize)){perror("unmap input");goto done;}im=NULL;}
 if(om){if(munmap(om,outsize)){perror("unmap output");goto done;}om=NULL;}
 if(close(out)){out=-1;perror("close output");goto done;}out=-1;finish=now();
 printf("{\"backend\":\"%s\",\"method\":\"%s\",\"width\":%d,\"height\":%d,\"threads\":%d,\"tile\":%d,\"io\":\"%s\",\"sync\":%s,\"warmup\":%d,\"samples_ms\":[",
#ifdef USE_OPENMP
 "openmp",
#else
 "pthread",
#endif
 method_names[method],info.w,info.h,actual,tile,buffered?"buffered":"mmap",sync_output?"true":"false",warmup);
 for(int i=0;i<repeat;i++)printf("%s%.6f",i?",":"",samples[i]);
 qsort(samples,(size_t)repeat,sizeof(*samples),compare_double);
 double median=repeat%2?samples[repeat/2]:(samples[repeat/2-1]+samples[repeat/2])/2;
 printf("],\"median_ms\":%.6f,\"setup_ms\":%.6f,\"compute_window_ms\":%.6f,\"finish_ms\":%.6f,\"wall_ms\":%.6f}\n",median,(setup_end-begin)*1000,(compute_end-setup_end)*1000,(finish-compute_end)*1000,(finish-begin)*1000);rc=0;
 done:
 if(started)pool_close(&p);
 if(im)munmap(im,insize);
 if(om)munmap(om,outsize);
 if(buffered){free(src);free(dst);}
 free(samples);if(in>=0)close(in);if(out>=0)close(out);return rc;
}
