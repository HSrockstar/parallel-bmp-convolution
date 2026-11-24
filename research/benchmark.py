#!/usr/bin/env python3
"""Randomized, paired benchmark; validates every output before ranking it.
One-shot wall_ms includes mapping/preallocation, computation, unmap/close. It
excludes process startup. process_ms includes startup and captured stdout.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parent

def command(cmd):
    t=time.perf_counter();p=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,timeout=180)
    return p,(time.perf_counter()-t)*1000

def pixels(blob):
    import struct
    off=struct.unpack_from('<I',blob,10)[0];w,h=struct.unpack_from('<ii',blob,18);st=(w*3+3)&~3
    return memoryview(blob)[off:off+st*abs(h)],(w,h,st)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',default='ring,symmetric,tiled,vfirst,box,prefix,q15,winograd,avx2,avx512,direct,sym2d,exact-int,lut,rank3,fft,overlap-save')
    p.add_argument('--threads',default='1,2,4');p.add_argument('--rounds',type=int,default=5);p.add_argument('--repeat',type=int,default=1);p.add_argument('--warmup',type=int,default=0)
    p.add_argument('--tile',type=int,default=128);p.add_argument('--legacy',action='store_true');p.add_argument('--seed',type=int,default=20261124)
    p.add_argument('--legacy-pthread',type=Path,default=ROOT.parent/'作业/代码/conv_pthread');p.add_argument('--legacy-omp',type=Path,default=ROOT.parent/'作业/代码/conv_omp')
    args=p.parse_args();args.input=args.input.resolve()
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('kernels.c','conv_lab.c','Makefile','../作业/代码/bmp_safe.h')}
    if args.rounds<1 or args.repeat<1 or args.warmup<0:p.error('invalid repetitions')
    threads=[int(x) for x in args.threads.split(',')]
    if any(n<1 for n in threads):p.error('invalid threads')
    methods=args.methods.split(',');rows=[];rng=random.Random(args.seed)
    binaries=[ROOT/'conv_lab_pthread',ROOT/'conv_lab_omp']
    available={line.split()[0] for line in command([binaries[0],'--list'])[0].stdout.splitlines() if line.endswith(' available')}
    unsupported=set(methods)-available
    if unsupported:p.error(f'unavailable or unknown methods: {unsupported}')
    tasks=[(exe,m,n) for exe in binaries for m in methods for n in threads]
    if args.legacy:tasks += [(exe,'legacy',n) for exe in (args.legacy_pthread.resolve(),args.legacy_omp.resolve()) for n in threads]
    with tempfile.TemporaryDirectory(prefix='conv-benchmark-') as td:
        td=Path(td);ref=td/'reference.bmp';out=td/'output.bmp'
        proc,_=command([binaries[0],args.input,ref,'--method','exact-int','--threads','1'])
        if proc.returncode:raise RuntimeError(proc.stderr)
        reference=ref.read_bytes();rp,geometry=pixels(reference)
        import numpy as np
        expected=np.frombuffer(rp,dtype=np.uint8).astype(np.int16)
        for round_no in range(args.rounds):
            rng.shuffle(tasks)
            for exe,m,n in tasks:
                if m=='legacy':cmd=[exe,args.input,out,str(n)]
                else:cmd=[exe,args.input,out,'--method',m,'--threads',str(n),'--tile',str(args.tile),'--repeat',str(args.repeat),'--warmup',str(args.warmup)]
                proc,process_ms=command(cmd)
                row={'round':round_no,'binary':exe.name,'method':m,'threads':n,'process_ms':process_ms,'status':'ok'}
                if proc.returncode:row.update(status='failed',error=proc.stderr.strip());rows.append(row);continue
                data=out.read_bytes();actual,shape=pixels(data)
                if shape!=geometry:raise AssertionError('output geometry mismatch')
                difference=np.frombuffer(actual,dtype=np.uint8).astype(np.int16)-expected
                maximum=int(np.abs(difference).max());mse=float(np.mean(difference.astype(np.float64)**2))
                if maximum>1 or (m in ('exact-int','lut') and maximum!=0):raise AssertionError((m,maximum))
                row.update(max_abs_error=maximum,mse=mse,output_sha256=hashlib.sha256(data).hexdigest())
                if m=='legacy':
                    row['legacy_total_ms']=float(re.search(r'Total time\s*:\s*([0-9.]+)',proc.stdout)[1])
                    row['legacy_full_ms']=float(re.search(r'Wall time \(full\)\s*:\s*([0-9.]+)',proc.stdout)[1])
                else:row.update(json.loads(proc.stdout))
                rows.append(row)
            print(f'finished round {round_no+1}/{args.rounds}',flush=True)
    meta={'source_sha256':source_hashes,'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'geometry':geometry,'platform':platform.platform(),'compiler':command(['gcc','--version'])[0].stdout.splitlines()[0],
          'cpuinfo':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown'),
          'affinity':sorted(os.sched_getaffinity(0)),'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'timing_contract':'lab wall_ms excludes process startup; legacy_full_ms includes text dump; process_ms includes subprocess launch for both'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'environment':meta,'runs':rows},indent=2)+'\n')
    csv_path=args.output.with_suffix('.csv')
    fields=['binary','method','threads','successes','metric','median_ms','min_ms','max_ms','max_abs_error']
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for exe,m,n in sorted(set((r['binary'],r['method'],r['threads']) for r in rows)):
            group=[r for r in rows if (r['binary'],r['method'],r['threads'])==(exe,m,n) and r['status']=='ok']
            metrics=['process_ms','legacy_total_ms','legacy_full_ms'] if m=='legacy' else ['process_ms','wall_ms','median_ms']
            for metric in metrics:
                values=[r[metric] for r in group]
                if values:writer.writerow(dict(binary=exe,method=m,threads=n,successes=len(values),metric=metric,median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),max_abs_error=max(r['max_abs_error'] for r in group)))
    print(args.output)
if __name__=='__main__':main()
