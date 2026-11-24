#!/usr/bin/env python3
"""Optional CUDA direct/separable/shared-memory fusion benchmark, no CPU fallback.
Requires a CUDA device and a CuPy build compatible with its driver.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from alternatives import read_bmp, write_bmp, apply, rounded

def prepare(x,method):
    import cupy as cp
    h,w,_=x.shape
    if w*3>2147483647 or h>2147483647:raise ValueError('GPU index limit exceeded')
    start=time.perf_counter()
    module=cp.RawModule(code=(Path(__file__).with_name('cuda_kernels.cu')).read_text(),options=('--std=c++11','--fmad=false'))
    # Resolve functions before reporting resident kernel time, so NVRTC JIT is separate.
    names=('horizontal','vertical') if method=='separable' else (method,)
    funcs=[module.get_function(n) for n in names]
    device=cp.empty(x.shape,dtype=cp.uint8);out=cp.empty_like(device)
    tmp=cp.empty(x.shape,dtype=cp.float32) if method=='separable' else None
    # Pinned host source permits a measurable asynchronous DMA transfer.
    pinned=cp.cuda.alloc_pinned_memory(x.nbytes);host=np.frombuffer(pinned,dtype=np.uint8).reshape(x.shape);host[:]=x
    cp.cuda.get_current_stream().synchronize();setup_ms=(time.perf_counter()-start)*1000
    t=time.perf_counter();device.set(host);cp.cuda.get_current_stream().synchronize();h2d_ms=(time.perf_counter()-t)*1000
    grid=((w*3+31)//32,(h+7)//8);block=(32,8);dims=(np.int32(w),np.int32(h))
    def execute(_module=module):
        if method=='separable':
            funcs[0](grid,block,(device,tmp,*dims));funcs[1](grid,block,(tmp,out,*dims))
        else:funcs[0](grid,block,(device,out,*dims))
    return cp,out,execute,{'setup_jit_allocation_ms':setup_ms,'h2d_ms':h2d_ms}

def compute(x,method,repeat=5,warmup=1):
    cp,out,execute,report=prepare(np.ascontiguousarray(x),method)
    for _ in range(warmup):execute()
    cp.cuda.get_current_stream().synchronize();samples=[]
    for _ in range(repeat):
        a,b=cp.cuda.Event(),cp.cuda.Event();a.record();execute();b.record();b.synchronize();samples.append(cp.cuda.get_elapsed_time(a,b))
    t=time.perf_counter();result=cp.asnumpy(out);report['d2h_ms']=(time.perf_counter()-t)*1000
    report.update(method=method,samples_ms=samples,median_ms=float(np.median(samples)))
    return result,report

def selftest():
    rng=np.random.default_rng(20261124);cases=0
    for h,w in ((1,1),(2,5),(5,2),(7,7),(8,11),(9,12),(17,33),(65,129)):
        x=rng.integers(0,256,(h,w,3),dtype=np.uint8);ref=rounded(apply(x,'scipy-direct')).astype(int)
        for method in ('direct','separable','fused'):
            y,_=compute(x,method,repeat=2);assert np.max(np.abs(y.astype(int)-ref))<=1;cases+=1
    print(json.dumps({'gpu_selftest_cases':cases}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',nargs='?',type=Path);p.add_argument('output',nargs='?',type=Path)
    p.add_argument('--method',choices=('direct','separable','fused'),default='fused');p.add_argument('--repeat',type=int,default=5);p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.repeat<1 or args.repeat>10000:p.error('invalid repeat')
    if args.self_test:selftest();return
    if args.input is None or args.output is None:p.error('input and output required')
    if args.output.exists() and args.input.samefile(args.output):p.error('input/output must differ')
    t=time.perf_counter();x,topdown=read_bmp(args.input);y,report=compute(x,args.method,args.repeat)
    reference=rounded(apply(x,'scipy-direct'));diff=y.astype(float)-reference
    report.update(max_abs_error=float(abs(diff).max()),mse=float(np.mean(diff**2)))
    if report['max_abs_error']>1:raise RuntimeError('GPU output failed independent reference check')
    write_bmp(args.output,y,topdown);report['wall_with_reference_ms']=(time.perf_counter()-t)*1000
    print(json.dumps(report))
if __name__=='__main__':main()
