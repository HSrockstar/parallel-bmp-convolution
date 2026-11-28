#ifndef CONV_AFFINITY_H
#define CONV_AFFINITY_H
#include <sched.h>

/* 可选环境变量按工作线程顺序给出 Linux 逻辑 CPU 编号。任务集受限时显式
 * 报错，保证标为绑核的运行确实执行了请求的绑定。默认沿用系统调度。 */
typedef struct {
    int cpus[CPU_SETSIZE];
    int count;
} ConvAffinity;

static int affinity_init(ConvAffinity *a, int workers)
{
    a->count = 0;
    const char *p = getenv("CONV_CPUS");
    if (!p || !*p) return 0;
    cpu_set_t allowed, seen;
    if (sched_getaffinity(0,sizeof(allowed),&allowed)) { perror("sched_getaffinity"); return -1; }
    CPU_ZERO(&seen);
    for (;;) {
        char *end;
        if (*p < '0' || *p > '9') break;
        long cpu = strtol(p,&end,10);
        if (cpu < 0 || cpu >= CPU_SETSIZE || !CPU_ISSET(cpu,&allowed) || CPU_ISSET(cpu,&seen)) break;
        CPU_SET(cpu,&seen);
        a->cpus[a->count++] = (int)cpu;
        if (!*end && a->count >= workers) return 0;
        if (*end != ',') break;
        p = end+1;
    }
    fprintf(stderr,"invalid CONV_CPUS: require distinct allowed CPUs for all workers\n");
    return -1;
}

static int affinity_bind(const ConvAffinity *a, int tid)
{
    if (!a->count) return 0;
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(a->cpus[tid],&one);
    if (sched_setaffinity(0,sizeof(one),&one)) { perror("sched_setaffinity"); return -1; }
    if (sched_getcpu() != a->cpus[tid]) { fprintf(stderr,"worker CPU verification failed\n"); return -1; }
    return 0;
}

static void affinity_print(const ConvAffinity *a, int workers)
{
    if (!a->count) return;
    printf("Worker CPUs     : ");
    for (int i = 0; i < workers; ++i) printf("%s%d",i ? "," : "",a->cpus[i]);
    putchar('\n');
}
#endif
