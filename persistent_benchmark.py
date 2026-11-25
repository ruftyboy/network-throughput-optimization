"""Persistent bottleneck screening and full Germany50 chronological replay.

All plan selection is confined to the first half. Historical capacities may be
hypothetical. Prior exploratory snapshots overlap the replay.
"""
import argparse
import hashlib
import io
import json
import math
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix

import robust_benchmark as rb
import balanced_benchmark as bb
from sndlib_benchmark import Problem, parse, URL

class SourceProblem(Problem):
    """Equivalent source-aggregated concurrent-flow LP.

    For each source, one commodity delivers the required amount to each sink.
    A nonnegative flow can be decomposed into source-to-sink paths plus cycles.
    """
    def __init__(self, edges, capacities, demands, budget):
        self.edges,self.budget=edges,budget
        self.m=m=len(edges)
        nodes=sorted({v for e in edges for v in e})
        idx={v:i for i,v in enumerate(nodes)}
        self.n=n=len(nodes)
        groups={}
        for s,t,w in demands:
            if s==t or w<=0: continue
            sinks=groups.setdefault(s,{})
            sinks[t]=sinks.get(t,0.)+w
        self.k=k=len(groups)
        if not k: raise ValueError("No positive demand")
        scale=max(max(capacities),max(sum(v.values()) for v in groups.values()))
        self.cap=np.asarray(capacities,dtype=float)/scale
        self.q=q=n*k
        self.f=f=2*m*k
        rows=[];cols=[];vals=[]
        def put(r,c,v):
            rows.append(r);cols.append(c);vals.append(v)
        for j,(s,sinks) in enumerate(sorted(groups.items())):
            put(j*n+idx[s],f,-sum(sinks.values())/scale)
            for t,w in sorted(sinks.items()):
                put(j*n+idx[t],f,w/scale)
            for i,(u,v) in enumerate(edges):
                a=j*2*m+2*i
                for col,x,y in ((a,u,v),(a+1,v,u)):
                    put(j*n+idx[x],col,1);put(j*n+idx[y],col,-1);put(q+i,col,1)
        mat=coo_matrix((vals,(rows,cols)),shape=(q+m,f+1)).tocsr()
        self.eq,self.ub=mat[:q],mat[q:]
        self.obj=np.zeros(f+1);self.obj[-1]=-1
        mr=[];mc=[]
        for i,(u,v) in enumerate(edges):
            mr.extend([idx[u],idx[v],n]);mc.extend([i,i,i])
        self.match=coo_matrix((np.ones(len(mr)),(mr,mc)),shape=(n+1,m)).tocsr()
        self.limits=np.r_[np.ones(n),budget]

def scan(edges,cap,matrices,budget,seconds):
    models=[];base=[];slopes=[]
    for d in matrices:
        p=SourceProblem(edges,cap,d,budget)
        v,g=p.lp(np.zeros(p.m),seconds)
        if v<=0: raise RuntimeError("Zero baseline")
        g=np.asarray(g)/v
        if np.min(g)<-1e-7 or abs(g.sum()-1)>1e-5:
            raise RuntimeError("Unexpected baseline dual normalization")
        models.append(p);base.append(v);slopes.append(g)
    return models,np.asarray(base),np.asarray(slopes)

def matching(p,weights,seconds):
    r=milp(-np.asarray(weights),integrality=np.ones(p.m),
           bounds=Bounds(np.zeros(p.m),np.ones(p.m)),
           constraints=LinearConstraint(p.match,-np.inf,p.limits),
           options={"time_limit":seconds,"mip_rel_gap":0.})
    if r.x is None: raise RuntimeError("Matching solver found no plan")
    return p.plan(r.x)

def envelope(p,G,threshold,seconds):
    """Maximize count of scenarios whose dual bound permits threshold gain.

    Among equal counts prefer largest mean dual-envelope gain. The count
    upper bound also bounds the number of actual threshold successes.
    """
    k,m=G.shape
    from scipy.sparse import hstack,vstack,eye
    # variables x (upgrades), z (threshold eligibility), h (gain envelope)
    match=hstack([p.match,csr_matrix((p.n+1,2*k))])
    A=vstack([match,hstack([-csr_matrix(G),threshold*eye(k),csr_matrix((k,k))]),
              hstack([-csr_matrix(G),csr_matrix((k,k)),eye(k)])]).tocsr()
    cost=np.r_[np.zeros(m),-np.ones(k),np.full(k,-.5/k)]
    r=milp(cost,integrality=np.r_[np.ones(m+k),np.zeros(k)],
           bounds=Bounds(np.zeros(m+2*k),np.ones(m+2*k)),
           constraints=LinearConstraint(A,-np.inf,np.r_[p.limits,np.zeros(2*k)]),
           options={"time_limit":seconds,"mip_rel_gap":0.})
    if r.x is None:
        x=np.zeros(m)
    else:
        x=p.plan(r.x[:m])
    dual=getattr(r,"mip_dual_bound",None)
    # Objective = integer count + bonus in [0, 0.5].
    # Numerical tolerance prevents rounding just below an integer.
    ub=k if dual is None or not math.isfinite(dual) else min(k,math.floor(-dual+1e-6))
    eligible=int(np.sum(G@x>=threshold-1e-7))
    if eligible>ub: raise RuntimeError("Inconsistent coverage bound")
    return x,{"count_upper_bound":ub,"eligible_for_plan":eligible,
              "status":int(r.status),"scenarios":k}

def assess(models,bases,G,x,seconds,threshold,cache):
    key=tuple(np.flatnonzero(x))
    if key not in cache:
        vals=[]
        for p,b,g in zip(models,bases,G):
            p.plan(x)
            rate,_=p.lp(x,seconds)
            gain=rate/b-1
            if gain>float(g@x)+1e-6 or gain < -1e-6:
                raise RuntimeError("Dual upper bound or monotonicity violated")
            vals.append(gain)
        cache[key]=np.asarray(vals)
    a=cache[key]
    return {"gain_pct":(100*a).tolist(),"mean_gain_pct":float(100*a.mean()),
            "min_gain_pct":float(100*a.min()),"median_gain_pct":float(100*np.median(a)),
            "fraction_threshold":float(np.mean(a>=threshold-1e-7)),
            "fraction_positive":float(np.mean(a>1e-7)),
            "mean_dual_upper_gain_pct":float(100*np.mean(G@x))}

def study(edges,cap,train,test,seconds,threshold):
    start=time.monotonic()
    budget=len(edges)//20
    tm,tb,G=scan(edges,cap,train,budget,seconds)
    print("Training dual scan complete",len(train),flush=True)
    p=tm[0];zero=np.zeros(p.m)
    candidates={"no_upgrade":zero,"betweenness":bb.centrality_plan(edges,budget)}
    single=zero.copy()
    if budget:
        single[p.cuts(seconds)["chosen"]]=1
    candidates["single_snapshot"]=single
    # Use a single optimal dual per baseline, so frequency is a proxy:
    # degeneracy can hide alternatives. All reported bounds remain valid.
    freq=np.mean(G>1e-7,axis=0)
    candidates["dual_frequency"]=matching(p,freq,seconds)
    candidates["dual_mean"]=matching(p,G.mean(axis=0),seconds)
    coverage,certificate=envelope(p,G,threshold,seconds)
    candidates["dual_coverage"]=coverage
    # Reproduce the balanced approach on six deterministic training samples.
    ix=np.linspace(0,len(train)-1,min(6,len(train)),dtype=int)
    seeds=[(x,np.asarray([tm[i].lp(x,seconds)[0]/tb[i] for i in ix]))
           for x in [zero,candidates["betweenness"],single]]
    extra=list(bb.singleton_rows(edges,cap,[train[i] for i in ix],tb[ix]))
    result=bb.optimize([tm[i] for i in ix],tb[ix],.5,seconds,seeds,extra)
    balanced=zero.copy();balanced[result["chosen"]]=1
    candidates["balanced_sample"]=balanced
    train_cache={};training={}
    for name,x in candidates.items():
        training[name]=assess(tm,tb,G,x,seconds,threshold,train_cache)
        print("train",name,training[name]["mean_gain_pct"],training[name]["fraction_threshold"],flush=True)
    # Predeclared selection rule, independent of all second-half matrices.
    # Include persistent candidates only to compare this family with baselines.
    persistent=["dual_frequency","dual_mean","dual_coverage"]
    winner=max(persistent,key=lambda n:(training[n]["fraction_threshold"],training[n]["mean_gain_pct"],n))
    candidates["persistent_selected"]=candidates[winner].copy()
    training["persistent_selected"]=training[winner]
    selection_seconds=time.monotonic()-start
    del tm
    held,hb,H=scan(edges,cap,test,budget,seconds)
    print("Holdout dual scan complete",len(test),flush=True)
    cache={};plans={}
    for name,x in candidates.items():
        scores=assess(held,hb,H,x,seconds,threshold,cache)
        plans[name]={"edges":[list(edges[i]) for i in np.flatnonzero(x)],
                     "train":training[name],"test":scores}
        print("test",name,scores["mean_gain_pct"],scores["fraction_threshold"],flush=True)
    # Retrospective upper bound on the best possible threshold coverage.
    # The optimizing plan is deliberately discarded: no holdout tuning.
    _,retrospective=envelope(held[0],H,threshold,seconds)
    ranking=sorted(range(len(edges)),key=lambda i:(-freq[i],-G[:,i].mean(),i))
    return {"nodes":p.n,"links":p.m,"budget":budget,"threshold_gain_pct":100*threshold,
            "train_count":len(train),"test_count":len(test),"training_coverage_certificate":certificate,
            "test_retrospective_coverage_upper_bound":retrospective["count_upper_bound"],
            "persistent_variant_selected":winner,"selection_seconds":selection_seconds,
            "plans":plans,"dual_ranking":[{"edge":list(edges[i]),"positive_shadow_fraction":float(freq[i]),
                                           "mean_normalized_shadow":float(G[:,i].mean())} for i in ranking],
            "elapsed_seconds":time.monotonic()-start}

def self_test():
    edges=[("a","b"),("b","c"),("c","d"),("a","d"),("a","c")]
    cap=[1.,2.,1.,3.,1.]
    demand=[("a","b",1.),("a","c",2.),("d","b",1.),("a","b",.5)]
    for mask in range(1<<len(edges)):
        x=np.array([(mask>>i)&1 for i in range(len(edges))],dtype=float)
        p=SourceProblem(edges,cap,demand,2)
        try: p.plan(x)
        except RuntimeError: continue
        a,_=p.lp(x,10);b,_=Problem(edges,cap,demand,2).lp(x,10)
        assert abs(a-b)<1e-7
    edges=[("a","b"),("b","c"),("c","d")]
    matrices=[[("a","b",1.)],[("c","d",1.)]]
    models,b,G=scan(edges,[1.,1.,1.],matrices,1,10)
    x,cert=envelope(models[0],G,.05,10)
    assert cert["count_upper_bound"]==1
    assert assess(models,b,G,x,10,.05,{})["fraction_threshold"]==.5

def historical(args):
    static={Path(k).stem.lower():v for k,v in rb.members(rb.fetch(URL)).items()}
    name="germany50"
    edges,cap,_,meta=parse(static[name],name,"module")
    nodes={v for e in edges for v in e}
    data=rb.fetch(rb.TRACES[name])
    files=rb.members(data)
    names=sorted((k for k in files if Path(k).name.lower().startswith("demandmatrix") and k.endswith(".xml")),
                 key=lambda k:Path(k).name)
    if len(names)!=288:
        raise RuntimeError("Expected 288 Germany50 matrices, received "+str(len(names)))
    midpoint=len(names)//2
    train=[];test=[];excluded=[]
    train_names=[];test_names=[]
    for i,key in enumerate(names):
        try: d=rb.demands(files[key],nodes)
        except ValueError as e:
            if str(e)!="Empty traffic matrix":raise
            excluded.append(key);continue
        if i<midpoint:train.append(d);train_names.append(key)
        else:test.append(d);test_names.append(key)
    return {**meta,"network":name,"trace_sha256":hashlib.sha256(data).hexdigest(),
            "available_matrices":len(names),"excluded_empty":excluded,"train_files":train_names,
            "test_files":test_names,"traffic":"historical measurement-derived; exploratory replay",
            **study(edges,cap,train,test,args.seconds,args.threshold)}

def synthetic(args):
    # Fresh predeclared topology traffic seeds; separate from historical replay.
    data=rb.fetch(rb.ZOO);eligible=[]
    for name,raw in sorted(rb.members(data).items()):
        g=nx.Graph(nx.read_graphml(io.BytesIO(raw)))
        g.remove_edges_from(list(nx.selfloop_edges(g)))
        g=g.subgraph(max(nx.connected_components(g),key=lambda c:(len(c),sorted(c)))).copy()
        if 20<=g.number_of_edges()<=250:eligible.append((name,g,raw))
    random.Random(2026).shuffle(eligible)
    if len(eligible)<args.limit:raise RuntimeError("Insufficient topology instances")
    for name,g,raw in eligible[:args.limit]:
        nodes=sorted(g.nodes);edges=sorted(tuple(sorted(e)) for e in g.edges)
        rng=random.Random(20261126)
        matrices=[[(*rng.sample(nodes,2),rng.uniform(.5,2)) for _ in range(12)] for _ in range(48)]
        yield {"network":name,"traffic":"synthetic","capacity":"unit","seed":20261126,
               "archive_sha256":hashlib.sha256(data).hexdigest(),"graph_sha256":hashlib.sha256(raw).hexdigest(),
               **study(edges,[1.]*len(edges),matrices[:24],matrices[24:],args.seconds,args.threshold)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["self-test","historical","synthetic"],required=True)
    ap.add_argument("--seconds",type=float,default=60)
    ap.add_argument("--threshold",type=float,default=.05)
    ap.add_argument("--limit",type=int,default=30)
    ap.add_argument("--output",default="results/persistent.json")
    args=ap.parse_args()
    if not 0<args.threshold<=1: raise ValueError("threshold must be in (0,1]")
    self_test()
    if args.mode=="self-test":
        print("Source aggregation and coverage checks passed");return
    rows=[];errors=[]
    try:
        for row in ([historical(args)] if args.mode=="historical" else synthetic(args)):
            rows.append(row)
            rb.save(args,rows,errors,[])
    except Exception as e:
        errors.append({"error":str(e)})
        print(type(e).__name__,str(e),flush=True)
    rb.save(args,rows,errors,[])
    if errors or not rows:raise SystemExit("Incomplete persistent benchmark")

if __name__=="__main__":
    main()
