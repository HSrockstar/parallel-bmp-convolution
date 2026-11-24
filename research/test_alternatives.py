#!/usr/bin/env python3
import json
import numpy as np
from alternatives import apply, rounded, EXACT, APPROX

def main():
    rng=np.random.default_rng(20261124);count=0;errors={}
    for h,w in ((1,1),(2,7),(7,2),(11,13),(64,65),(129,135)):
        for kind in ('random','impulse','white'):
            x=rng.integers(0,256,(h,w,3),dtype=np.uint8)
            if kind=='impulse':x[:]=0;x[h//2,w//2,1]=255
            if kind=='white':x[:]=255
            ref=apply(x,'scipy-direct')
            for method in EXACT:
                result=apply(x,method)
                assert result.shape==x.shape
                # Rank-1 transpose uses the previous implementation's fitted kernel.
                assert np.max(np.abs(result-ref)) < (2e-5 if method in ('transpose','cascade','sliding-dft') else 1e-8),(method,np.max(np.abs(result-ref)))
                count+=1
            for method in APPROX:
                result=apply(x,method)
                assert result.shape==x.shape and np.isfinite(result).all()
                err=np.abs(rounded(result).astype(float)-rounded(ref))
                errors[method]=max(errors.get(method,0),float(err.max()));count+=1
    # IIRs must preserve mass, symmetry and channel isolation for an interior impulse.
    impulse=np.zeros((257,257,3));impulse[128,128,1]=255
    for method in ('am-recursive','deriche','vyv'):
        result=apply(impulse,method)
        assert abs(result.sum()-255)<1e-7,(method,result.sum())
        assert np.max(np.abs(result-result[::-1]))<1e-10
        assert np.max(np.abs(result-result[:,::-1]))<1e-10
        assert np.count_nonzero(result[:,:,[0,2]])==0;count+=1
    print(json.dumps({'passed_cases':count,'approximation_max_errors':errors},indent=2))
if __name__=='__main__':main()
