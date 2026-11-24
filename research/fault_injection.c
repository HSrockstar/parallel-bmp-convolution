/* Link-time wrappers used only by test_failures.py, never in normal binaries. */
#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>
#include <errno.h>
static _Atomic int allocation_count,thread_count;
static int fail(const char *name,_Atomic int *counter){
 const char *s=getenv(name);int n=atomic_fetch_add(counter,1)+1;return s&&atoi(s)==n;
}
void *__real_malloc(size_t);void *__real_calloc(size_t,size_t);void *__real_aligned_alloc(size_t,size_t);
int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
void *__wrap_malloc(size_t n){if(fail("CONV_FAIL_ALLOC",&allocation_count)){errno=ENOMEM;return NULL;}return __real_malloc(n);}
void *__wrap_calloc(size_t a,size_t b){if(fail("CONV_FAIL_ALLOC",&allocation_count)){errno=ENOMEM;return NULL;}return __real_calloc(a,b);}
void *__wrap_aligned_alloc(size_t a,size_t n){if(fail("CONV_FAIL_ALLOC",&allocation_count)){errno=ENOMEM;return NULL;}return __real_aligned_alloc(a,n);}
int __wrap_pthread_create(pthread_t *t,const pthread_attr_t *a,void*(*f)(void*),void *arg){if(fail("CONV_FAIL_THREAD",&thread_count))return EAGAIN;return __real_pthread_create(t,a,f,arg);}
