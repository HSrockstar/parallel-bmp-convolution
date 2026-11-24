#!/usr/bin/env python3
"""Measure candidates in randomized rounds and print a command; never edit defaults."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import numpy as np
from benchmark import pixels
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--backend',choices=('pthread','omp'),default='pthread');p.add_argument('--threads',default='1,2,4');p.add_argument('--rounds',type=int,default=5);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.rounds<3:p.error('at least three randomized rounds required')
    exe=ROOT/f'conv_lab_{args.backend}';available={line.split()[0] for line in subprocess.check_output([exe,'--list'],text=True).splitlines() if line.endswith(' available')}
    candidates=[(m,t,n) for m in ('ring','symmetric','vfirst','tiled','avx2','avx512') if m in available for t in ((32,64,128,256,512) if m=='tiled' else (128,)) for n in map(int,args.threads.split(','))]
    if any(n<1 or n>4096 for _,_,n in candidates):p.error('invalid threads')
    rng=random.Random(20261124);runs={c:[] for c in candidates};bad=set()
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/'out.bmp';ref=Path(td)/'ref.bmp'
        subprocess.run([exe,args.input,ref,'--method','exact-int','--threads','1'],check=True,capture_output=True)
        r,shape=pixels(ref.read_bytes());r=np.frombuffer(r,np.uint8).astype(np.int16)
        for _ in range(args.rounds):
            rng.shuffle(candidates)
            for m,t,n in candidates:
                c=(m,t,n);q=subprocess.run([str(v) for v in (exe,args.input,out,'--method',m,'--tile',t,'--threads',n)],text=True,capture_output=True,check=True)
                data,geom=pixels(out.read_bytes());d=np.frombuffer(data,np.uint8).astype(np.int16)
                if geom!=shape or np.max(np.abs(d-r))>1:bad.add(c);continue
                runs[c].append(json.loads(q.stdout)['wall_ms'])
    ranked=sorted((statistics.median(v),c,v) for c,v in runs.items() if len(v)==args.rounds and c not in bad)
    if not ranked:raise RuntimeError('no validated candidates')
    time_ms,(method,tile,threads),_=ranked[0]
    report={'input':str(args.input.resolve()),'affinity':sorted(os.sched_getaffinity(0)),'winner':{'method':method,'tile':tile,'threads':threads,'median_wall_ms':time_ms},'ranked':[{'median_ms':v,'candidate':c,'samples_ms':s} for v,c,s in ranked],'note':'winner applies to this image, build, host, cache and I/O conditions only'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['winner']))
if __name__=='__main__':main()
