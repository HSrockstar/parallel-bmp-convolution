#!/usr/bin/env python3
"""Independent decimal-integer oracle, malformed BMP and parallel regressions."""
import argparse
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
KI = ((1441881,2808402,3507270,2808402,1441881),
      (2808402,5470020,6831229,5470020,2808402),
      (3507270,6831229,8531173,6831229,3507270),
      (2808402,5470020,6831229,5470020,2808402),
      (1441881,2808402,3507270,2808402,1441881))

def make_bmp(w,h,seed=1,offset=54,topdown=False,kind='random'):
    rng=random.Random(seed);st=(w*3+3)&~3;data=bytearray(st*h)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                v=rng.randrange(256)
                if kind=='white':v=255
                if kind=='black':v=0
                if kind=='impulse':v=255 if (x,y,c)==(w//2,h//2,1) else 0
                if kind=='checker':v=255*((x+y+c)%2)
                data[y*st+x*3+c]=v
        data[y*st+w*3:(y+1)*st]=bytes([197])*(st-w*3)
    hdr=bytearray(offset);hdr[:2]=b'BM'
    struct.pack_into('<I',hdr,2,offset+len(data));struct.pack_into('<I',hdr,10,offset)
    struct.pack_into('<IiiHHII',hdr,14,40,w,-h if topdown else h,1,24,0,len(data))
    hdr[54:]=bytes([113])*(offset-54)
    return bytes(hdr+data)

def oracle(blob):
    w,hr=struct.unpack_from('<ii',blob,18);h=abs(hr);off=struct.unpack_from('<I',blob,10)[0];st=(w*3+3)&~3
    out=bytearray(st*h)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                v=0
                for ky in range(5):
                    yy=y+ky-2
                    if not 0<=yy<h:continue
                    for kx in range(5):
                        xx=x+kx-2
                        if 0<=xx<w:v+=KI[ky][kx]*blob[off+yy*st+xx*3+c]
                out[y*st+x*3+c]=(v+50000000)//100000000
    return bytes(out)

def run(cmd,ok=True):
    p=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,timeout=45,
                     env={**os.environ,'OMP_DYNAMIC':'TRUE','OMP_NUM_THREADS':'2'})
    if ok and p.returncode:raise AssertionError(f'{cmd}: rc={p.returncode}\n{p.stderr}')
    if not ok and p.returncode==0:raise AssertionError(f'accepted invalid input: {cmd}')
    return p

def verify(blob,result,expected,exact=False):
    w,hr=struct.unpack_from('<ii',blob,18);h=abs(hr);st=(w*3+3)&~3
    assert result[:2]==b'BM' and len(result)==54+st*h
    assert struct.unpack_from('<I',result,2)[0]==len(result)
    assert struct.unpack_from('<I',result,10)[0]==54
    assert struct.unpack_from('<ii',result,18)==(w,hr)
    actual=result[54:]
    ds=[abs(x-y) for x,y in zip(actual,expected)]
    assert max(ds,default=0)<=(0 if exact else 1), max(ds)
    mse=sum(d*d for d in ds)/(w*h*3)
    assert 100-(mse**.5)/2.55>99
    for y in range(h):assert actual[y*st+w*3:(y+1)*st]==bytes(st-w*3)
    return sum(d!=0 for d in ds)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    bins=[ROOT/'conv_lab_pthread',ROOT/'conv_lab_omp']
    old=[ROOT.parent/'作业/代码/conv_pthread',ROOT.parent/'作业/代码/conv_omp']
    available=[line.split()[0] for line in run([bins[0],'--list']).stdout.splitlines() if line.endswith(' available')]
    dims=[(w,1+w%7) for w in range(1,13)]+[(17,13),(31,19),(33,5),(63,9),(65,17),(129,7)]
    if not args.quick:dims += [(w,h) for w in (4,5,7,8,9,15,16,32,127,128,257) for h in (1,2,3,4,8,17)]
    counts={m:0 for m in available};cases=0
    with tempfile.TemporaryDirectory(prefix='conv-test-') as td:
        td=Path(td);inp=td/'in.bmp';out=td/'out.bmp'
        for index,(w,h) in enumerate(dims):
            kind=('random','checker','white','black','impulse')[index%5]
            blob=make_bmp(w,h,index,offset=54 if index%2 else 122,topdown=index%3==0,kind=kind)
            inp.write_bytes(blob);expected=oracle(blob)
            for m in available:
                first=None
                for exe in bins:
                    cmd=[exe,inp,out,'--method',m,'--threads','7','--tile','7','--repeat','2','--warmup','1']
                    if index%2:cmd+=['--io','buffered']
                    report=json.loads(run(cmd).stdout)
                    assert report['threads']==min(7,h) and len(report['samples_ms'])==2
                    result=out.read_bytes();counts[m]+=verify(blob,result,expected,m in ('exact-int','lut'))
                    if first is not None:assert result==first,(m,'backend mismatch')
                    first=result;cases+=1
                if index%7==0:
                    run([bins[0],inp,out,'--method',m,'--threads','1','--tile','128'])
                    assert out.read_bytes()==first,(m,'thread/tile invariance')
                    cases+=1
            for exe in old:
                if exe.exists():run([exe,inp,out,'7']);verify(blob,out.read_bytes(),expected);cases+=1
        # All implementation variants exercise constant/impulse/random at a vector-tail size.
        for kind in ('random','white','black','impulse','checker'):
            blob=make_bmp(35,11,99,kind=kind);inp.write_bytes(blob);expected=oracle(blob)
            for m in available:
                run([bins[0],inp,out,'--method',m,'--threads','3'])
                verify(blob,out.read_bytes(),expected,m in ('exact-int','lut'));cases+=1
        good=make_bmp(7,5,offset=122);inp.write_bytes(good)
        # Short file, impossible geometry, headers and pixel offset before DIB end.
        malformed=[b'',good[:53],good[:-1]]
        for offset,fmt,value in ((18,'i',0),(18,'i',-1),(18,'i',2147483647),(22,'i',0),(22,'i',-2147483648),(14,'I',12),(14,'I',0xffffffff),(10,'I',20),(10,'I',4000000000),(26,'H',2),(28,'H',32),(30,'I',1)):
            buf=bytearray(good);struct.pack_into('<'+fmt,buf,offset,value);malformed.append(buf)
        for exe in bins+[x for x in old if x.exists()]:
            for blob in malformed:
                inp.write_bytes(blob);run([exe,inp,out],ok=False);cases+=1
            inp.write_bytes(good)
            for alias in (inp,td/'hard.bmp',td/'link.bmp'):
                if alias!=inp:
                    if alias.exists():alias.unlink()
                    if alias.name=='hard.bmp':os.link(inp,alias)
                    else:alias.symlink_to(inp)
                run([exe,inp,alias],ok=False);assert inp.read_bytes()==good;cases+=1
        for exe in bins:
            for opts in (['--threads','0'],['--threads','abc'],['--tile','0'],['--method','bad'],['--repeat','10001'],['--io','bad'],['--missing'],['--warmup','-1']):
                run([exe,inp,out,*opts],ok=False);cases+=1
            for opts in (['--sync'],['--prefault'],['--advice','sequential'],['--pin'],['--no-preallocate']):
                run([exe,inp,out,*opts]);verify(good,out.read_bytes(),oracle(good));cases+=1
    print(json.dumps({'passed_cases':cases,'methods':available,'differing_bytes_across_main_cases':counts},indent=2))
if __name__=='__main__':main()
