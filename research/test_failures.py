#!/usr/bin/env python3
"""Allocation/thread failure injection verifies nonzero exit without deadlock."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from test_methods import make_bmp
ROOT=Path(__file__).resolve().parent

def main():
    count=0
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);inp=td/'input.bmp';out=td/'output.bmp';inp.write_bytes(make_bmp(33,17))
        for backend in ('pthread','omp'):
            exe=td/backend
            compile=['gcc','-O1','-std=c11',str(ROOT/'conv_lab.c'),str(ROOT/'kernels.c'),str(ROOT/'fault_injection.c'),'-lm','-pthread','-Wl,--wrap=malloc,--wrap=calloc,--wrap=aligned_alloc,--wrap=pthread_create','-o',str(exe)]
            if backend=='omp':compile+=['-DUSE_OPENMP','-fopenmp']
            subprocess.run(compile,check=True,capture_output=True)
            for method in ('ring','prefix','vfirst','q15','rank3','lut','fft','overlap-save'):
                # First allocation is timing samples, subsequent allocations are
                # thread bookkeeping/workspaces. Test the first three reliably reached.
                for n in (1,2,3):
                    env={**os.environ,'CONV_FAIL_ALLOC':str(n)}
                    p=subprocess.run([exe,inp,out,'--method',method,'--threads','4'],env=env,capture_output=True,timeout=20)
                    assert p.returncode!=0,(backend,method,n);count+=1
            if backend=='pthread':
                for n in (1,2,4):
                    p=subprocess.run([exe,inp,out,'--threads','4'],env={**os.environ,'CONV_FAIL_THREAD':str(n)},capture_output=True,timeout=20)
                    assert p.returncode!=0;count+=1
    print(json.dumps({'injected_failure_cases':count,'status':'passed'},indent=2))
if __name__=='__main__':main()
