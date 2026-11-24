#!/usr/bin/env python3
"""Runnable NumPy/SciPy comparison methods; approximation modes are explicit.
These timings include Python/array allocation and cannot rank C instruction schedules.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import time
import numpy as np
from scipy import fft, ndimage, signal
K = np.array([[.01441881,.02808402,.03507270,.02808402,.01441881],
 [.02808402,.05470020,.06831229,.05470020,.02808402],
 [.03507270,.06831229,.08531173,.06831229,.03507270],
 [.02808402,.05470020,.06831229,.05470020,.02808402],
 [.01441881,.02808402,.03507270,.02808402,.01441881]])
K1=np.array([.12007837208047906,.2338807408740243,.29208172229534735,.2338807408740243,.12007837208047906])
EXACT=('scipy-direct','scipy-fft','overlap-add','dct','integral','im2col','svd3','transpose','cascade','sliding-dft')
APPROX=('boxes','extended-boxes','binomial','am-recursive','deriche','vyv','downsample')

def read_bmp(path):
    blob=Path(path).read_bytes()
    if len(blob)<54 or blob[:2]!=b'BM':raise ValueError('invalid BMP')
    off=struct.unpack_from('<I',blob,10)[0]
    dib,w,hr,planes,bpp,comp=struct.unpack_from('<IiiHHI',blob,14)
    if dib<40 or off<14+dib or w<1 or hr in (0,-2147483648) or planes!=1 or bpp!=24 or comp!=0:raise ValueError('unsupported BMP')
    h=abs(hr);st=(w*3+3)&~3
    if off+st*h>len(blob):raise ValueError('truncated BMP')
    x=np.frombuffer(blob,dtype=np.uint8,count=st*h,offset=off).reshape(h,st)[:,:w*3].reshape(h,w,3).copy()
    return x,hr<0

def rounded(x):return np.floor(np.clip(x,0,255)+.5).astype(np.uint8)

def write_bmp(path,x,topdown):
    h,w,c=x.shape
    if c!=3:raise ValueError('three channels required')
    st=(3*w+3)&~3;size=54+st*h
    if size>0xffffffff:raise ValueError('BMP too large')
    hdr=bytearray(54);hdr[:2]=b'BM'
    struct.pack_into('<I',hdr,2,size);struct.pack_into('<I',hdr,10,54)
    struct.pack_into('<IiiHHII',hdr,14,40,w,-h if topdown else h,1,24,0,st*h)
    rows=np.zeros((h,st),np.uint8);rows[:,:w*3]=x.reshape(h,w*3)
    Path(path).write_bytes(hdr+rows.tobytes())

def separable(x,k):
    y=ndimage.convolve1d(x,k,axis=1,mode='constant',cval=0.)
    return ndimage.convolve1d(y,k,axis=0,mode='constant',cval=0.)

def integral(x):
    # Any symmetric 5x5 kernel is a sum of nine nested box kernels.
    # Prefix sums use float64 (exact for these 8-bit image sums at practical sizes).
    h,w,_=x.shape;y=np.zeros_like(x)
    p=np.pad(x,((2,2),(2,2),(0,0)))
    sat=np.pad(p.cumsum(0).cumsum(1),((1,0),(1,0),(0,0)))
    f=np.zeros((4,4));f[:3,:3]=K[2:,2:]
    for ry in range(3):
        for rx in range(3):
            weight=f[ry,rx]-f[ry+1,rx]-f[ry,rx+1]+f[ry+1,rx+1]
            l,t=2-rx,2-ry;r,b=3+rx,3+ry
            rect=sat[b:b+h,r:r+w]-sat[t:t+h,r:r+w]-sat[b:b+h,l:l+w]+sat[t:t+h,l:l+w]
            y+=weight*rect
    return y

def dct_exact(x):
    p=np.pad(x,((2,2),(2,2),(0,0)));h,w,_=p.shape
    fy=np.pi*np.arange(h)/h;fx=np.pi*np.arange(w)/w
    response=np.zeros((h,w))
    for dy in range(3):
        for dx in range(3):
            response+=K[2+dy,2+dx]*np.outer(np.ones(h) if dy==0 else 2*np.cos(dy*fy),np.ones(w) if dx==0 else 2*np.cos(dx*fx))
    y=fft.idctn(fft.dctn(p,type=2,norm='ortho',axes=(0,1))*response[:,:,None],type=2,norm='ortho',axes=(0,1))
    return y[2:-2,2:-2]

def im2col(x):
    h,w,_=x.shape;p=np.pad(x,((2,2),(2,2),(0,0)));out=np.empty_like(x)
    # Row tiles cap materialized patch memory at roughly 64 MiB.
    rows=max(1,min(h,(64*1024*1024)//max(1,w*3*25*8)))
    for y in range(0,h,rows):
        bh=min(rows,h-y)
        patches=np.lib.stride_tricks.sliding_window_view(p[y:y+bh+4],(5,5),axis=(0,1))
        out[y:y+bh]=(np.ascontiguousarray(patches).reshape(-1,25)@K.reshape(25)).reshape(bh,w,3)
    return out

def sliding_axis(x,axis):
    # Exact five-sample DFT window, three bins suffice for real input.
    # Reset every 256 positions to bound recurrence drift.
    z=np.moveaxis(x,axis,0);p=np.pad(z,[(2,2)]+[(0,0)]*(z.ndim-1));out=np.empty_like(z)
    omega=2*np.pi*np.arange(3)/5
    phase=np.exp(1j*omega);basis=np.exp(-1j*np.outer(omega,np.arange(5)))
    g=np.fft.fft(K1)[:3].conj();shape=(3,)+(1,)*(z.ndim-1)
    for n in range(len(z)):
        if n%256==0:state=np.tensordot(basis,p[n:n+5],axes=(1,0))
        else:state=phase.reshape(shape)*(state-p[n-1]+p[n+4])
        weighted=state*g.reshape(shape)
        out[n]=(weighted[0].real+2*weighted[1:].real.sum(axis=0))/5
    return np.moveaxis(out,0,axis)

def recursive_axis(x,axis,mode,sigma=1.5):
    # Pad far beyond the IIR decay length, then crop. Starting/ending recurrence
    # state is zero in the padded domain. Padding sensitivity is tested below.
    pad=64
    pads=[(0,0)]*x.ndim;pads[axis]=(pad,pad);p=np.pad(x,pads)
    if mode=='am-recursive':
        passes=4;lam=sigma*sigma/(2*passes);nu=(1+2*lam-math.sqrt(1+4*lam))/(2*lam)
        y=p
        for _ in range(passes):
            y=signal.lfilter([1-nu],[1,-nu],y,axis=axis)
            y=np.flip(signal.lfilter([1-nu],[1,-nu],np.flip(y,axis),axis=axis),axis)
    elif mode=='deriche':
        alpha=1.695/sigma;e=math.exp(-alpha);e2=e*e
        scale=(1-e)**2/(1+2*alpha*e-e2)
        denominator=[1,-2*e,e2]
        causal=signal.lfilter([scale,scale*(alpha-1)*e],denominator,p,axis=axis)
        anti=np.flip(signal.lfilter([0,scale*(alpha+1)*e,-scale*e2],denominator,np.flip(p,axis),axis=axis),axis)
        y=causal+anti
    else:
        # Young & van Vliet's standard third-order coefficient approximation.
        q=.98711*sigma-.96330 if sigma>=2.5 else 3.97156-4.14554*math.sqrt(1-.26891*sigma)
        b0=1.57825+2.44413*q+1.4281*q*q+.422205*q**3
        b1=2.44413*q+2.85619*q*q+1.26661*q**3;b2=-(1.4281*q*q+1.26661*q**3);b3=.422205*q**3
        gain=1-(b1+b2+b3)/b0;den=[1,-b1/b0,-b2/b0,-b3/b0]
        y=signal.lfilter([gain],den,p,axis=axis)
        y=np.flip(signal.lfilter([gain],den,np.flip(y,axis),axis=axis),axis)
    take=[slice(None)]*x.ndim;take[axis]=slice(pad,-pad)
    return y[tuple(take)]

def apply(x,method):
    x=np.asarray(x,dtype=np.float64)
    if method=='scipy-direct':return ndimage.convolve(x,K[:,:,None],mode='constant',cval=0.)
    if method=='scipy-fft':return signal.fftconvolve(x,K[:,:,None],mode='same',axes=(0,1))
    if method=='overlap-add':return signal.oaconvolve(x,K[:,:,None],mode='same',axes=(0,1))
    if method=='dct':return dct_exact(x)
    if method=='integral':return integral(x)
    if method=='im2col':return im2col(x)
    if method=='svd3':
        u,s,vh=np.linalg.svd(K);y=np.zeros_like(x)
        for r in range(3):y+=ndimage.convolve1d(ndimage.convolve1d(x,vh[r],axis=1,mode='constant'),u[:,r]*s[r],axis=0,mode='constant')
        return y
    if method=='sliding-dft':return sliding_axis(sliding_axis(x,1),0)
    if method=='cascade':
        a,b,c=K1[:3];total=b/a;product=c/a-2
        t=(total+math.sqrt(total*total-4*product))/2;z=total-t
        # One intermediate halo per side preserves the five-tap zero boundary.
        y=np.pad(x,((2,2),(2,2),(0,0)))
        y=separable(y,np.array([a,a*t,a]))
        y=separable(y,np.array([1,z,1]))
        return y[2:-2,2:-2]
    if method=='transpose':
        y=ndimage.convolve1d(x,K1,axis=1,mode='constant')
        y=np.ascontiguousarray(y.transpose(1,0,2))
        return ndimage.convolve1d(y,K1,axis=1,mode='constant').transpose(1,0,2)
    if method=='binomial':return separable(x,np.array([1,4,6,4,1])/16)
    if method in ('boxes','extended-boxes'):
        # Three passes on a domain padded by the total support. Cropping between
        # box passes would incorrectly discard values that return across edges.
        k=np.ones(3)/3 if method=='boxes' else np.array([.0125,.325,.325,.325,.0125])
        radius=len(k)//2;pad=3*radius;y=np.pad(x,((pad,pad),(pad,pad),(0,0)))
        for _ in range(3):y=separable(y,k)
        return y[pad:-pad,pad:-pad]
    if method in ('am-recursive','deriche','vyv'):
        return recursive_axis(recursive_axis(x,1,method),0,method)
    if method=='downsample':
        # Explicit quality-changing downsample/filter/bilinear-upsample experiment.
        h,w,_=x.shape
        low=ndimage.zoom(x,(.5,.5,1),order=1,mode='constant',grid_mode=False) if h>1 and w>1 else x
        y=separable(low,K1);return ndimage.zoom(y,(h/y.shape[0],w/y.shape[1],1),order=1,mode='constant')[:h,:w]
    raise ValueError(method)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--method',choices=EXACT+APPROX,required=True);p.add_argument('--allow-approx',action='store_true');p.add_argument('--repeat',type=int,default=3);p.add_argument('--warmup',type=int,default=1)
    args=p.parse_args()
    if args.repeat<1 or args.repeat>10000 or args.warmup<0 or args.warmup>10000:p.error('invalid repetition count')
    if args.output.exists() and args.input.samefile(args.output):p.error('input and output must differ')
    if args.method in APPROX and not args.allow_approx:p.error('changed filter requires --allow-approx')
    start=time.perf_counter();x,topdown=read_bmp(args.input);times=[]
    for i in range(args.warmup+args.repeat):
        t=time.perf_counter();y=apply(x,args.method);q=rounded(y)
        if i>=args.warmup:times.append((time.perf_counter()-t)*1000)
    reference=rounded(apply(x,'scipy-direct'));diff=q.astype(float)-reference;write_bmp(args.output,q,topdown)
    print(json.dumps({'method':args.method,'changed_filter':args.method in APPROX,'samples_ms':times,'median_ms':float(np.median(times)),'max_abs_error':float(abs(diff).max()),'mse':float(np.mean(diff**2)),'similarity':100-float(np.sqrt(np.mean(diff**2)))/2.55,'wall_with_reference_ms':(time.perf_counter()-start)*1000}))
if __name__=='__main__':main()
