"""Balanced fixed-link augmentation: multi-cut mean/min objective.

Training selects plans; holdout is used only for evaluation.
No claim of novelty or operational capacity is made.
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
from scipy.sparse import csr_matrix, hstack, vstack

import robust_benchmark as rb
from sndlib_benchmark import URL

WEIGHTS = {"mean_only":0.0, "balanced":0.5, "min_only":1.0}

def utility(values, weight):
    return float((1-weight)*np.mean(values)+weight*np.min(values))

def metrics(values):
    a=100*(np.asarray(values)-1)
    return {"gain_pct":a.tolist(),"mean_gain_pct":float(np.mean(a)),
            "min_gain_pct":float(np.min(a)),"balanced_gain_pct":float((np.mean(a)+np.min(a))/2),
            "fraction_positive":float(np.mean(a>1e-6))}

def fixed(models,bases,x,seconds):
    models[0].plan(x)
    return np.array([p.lp(x,seconds)[0]/b for p,b in zip(models,bases)])

def centrality_plan(edges,budget):
    g=nx.Graph()
    g.add_edges_from(edges)
    central={frozenset(e):v for e,v in nx.edge_betweenness_centrality(g).items()}
    chosen,used=[],set()
    for i in sorted(range(len(edges)),key=lambda i:(-central[frozenset(edges[i])],i)):
        if len(chosen)>=budget:
            break
        u,v=edges[i]
        if u not in used and v not in used:
            chosen.append(i);used.update((u,v))
    x=np.zeros(len(edges));x[chosen]=1
    return x

def singleton_rows(edges,cap,matrices,bases):
    for i,(d,b) in enumerate(zip(matrices,bases)):
        for g,a in rb.endpoint_cuts(edges,cap,[d],[b]):
            yield i,g,a

def optimize(models,bases,weight,seconds,seeds,extra=(),iterations=150):
    """Master: one throughput upper-envelope per scenario, plus minimum z.

    LP dual cuts upper-bound individual normalized scenario throughput.
    Monotone objective gives a global numerical upper bound.
    Feasible plans are evaluated by routing LPs, including all shared seeds.
    """
    p=models[0];m=p.m;k=len(models);n=m+k+1
    start=time.monotonic()
    incumbent,values=max(seeds,key=lambda pair:utility(pair[1],weight))
    incumbent=incumbent.copy();values=values.copy()
    best=utility(values,weight);upper=2.;count=0;solves=0
    if p.budget==0:
        return {"chosen":[],"objective":1.,"upper_bound":1.,"closed_gap":True,
                "iterations":0,"master_solves":0,"seconds":time.monotonic()-start}
    rows=[];rhs=[]
    def add(i,g,a):
        row=np.zeros(n);row[:m]=-g;row[m+i]=1.
        rows.append(row);rhs.append(float(a))
    for i,g,a in extra:
        add(i,g,a)
    for i in range(k):
        row=np.zeros(n);row[m+i]=-1.;row[-1]=1.
        rows.append(row);rhs.append(0.)
    cost=np.r_[np.zeros(m),np.full(k,-(1-weight)/k),-weight]
    integral=np.r_[np.ones(m),np.zeros(k+1)]
    bounds=Bounds(np.r_[np.zeros(m),np.ones(k+1)],np.r_[np.ones(m),np.full(k+1,2.)])
    match=hstack([p.match,csr_matrix((p.n+1,k+1))])
    # Begin at no upgrades; after that, query the current master plan.
    x=np.zeros(m)
    for _ in range(iterations):
        vals=[]
        for i,(model,base) in enumerate(zip(models,bases)):
            remaining=seconds-(time.monotonic()-start)
            if remaining<=.05:
                break
            try:
                rate,g=model.lp(x,remaining)
            except RuntimeError:
                break
            rate,g=rate/base,g/base
            vals.append(rate);add(i,g,rate-float(g@x))
        if len(vals)!=k:
            break
        count+=1
        score=utility(vals,weight)
        if score>best+1e-9:
            best=score;incumbent=x.copy();values=np.asarray(vals)
        if upper-best<=1e-7:
            break
        remaining=seconds-(time.monotonic()-start)
        if remaining<=.05:
            break
        A=vstack([match,csr_matrix(np.asarray(rows))]).tocsr()
        r=milp(cost,integrality=integral,bounds=bounds,
               constraints=LinearConstraint(A,-np.inf,np.r_[p.limits,rhs]),
               options={"time_limit":remaining,"mip_rel_gap":0.})
        solves+=1
        dual=getattr(r,"mip_dual_bound",None)
        if dual is not None and math.isfinite(dual):
            upper=min(upper,float(-dual))
        if upper-best<=1e-7 or r.x is None:
            break
        x=p.plan(r.x[:m])
    if upper<best-1e-6:
        raise RuntimeError("Inconsistent numerical objective bounds")
    return {"chosen":np.flatnonzero(incumbent).tolist(),"objective":best,
            "upper_bound":max(best,upper),"closed_gap":bool(upper-best<=1e-7),
            "iterations":count,"master_solves":solves,"seconds":time.monotonic()-start}

def evaluate(edges,cap,train,test,seconds,compare=True):
    budget=len(edges)//20
    start=time.monotonic()
    models,bases=rb.make_models(edges,cap,train,budget,seconds)
    held,hbases=rb.make_models(edges,cap,test,budget,seconds)
    setup=time.monotonic()-start
    start=time.monotonic()
    zero=np.zeros(len(edges))
    central=centrality_plan(edges,budget)
    single=zero.copy()
    if budget:
        single[models[0].cuts(seconds)["chosen"]]=1
    baseline={"no_upgrade":zero,"betweenness":central,"single_snapshot":single}
    cache={}
    def train_values(x):
        key=tuple(np.flatnonzero(x))
        if key not in cache:
            cache[key]=fixed(models,bases,x,seconds)
        return cache[key]
    seeds=[(x,train_values(x)) for x in baseline.values()]
    extra=list(singleton_rows(edges,cap,train,bases))
    shared=time.monotonic()-start
    results={};plans=dict(baseline)
    for name,w in WEIGHTS.items():
        r=optimize(models,bases,w,seconds,seeds,extra)
        x=zero.copy();x[r["chosen"]]=1
        v=train_values(x)
        if abs(utility(v,w)-r["objective"])>1e-6:
            raise RuntimeError("Recomputed incumbent objective disagrees")
        results[name]=r;plans[name]=x
    # Cross-check: all evaluated feasible plans must respect each solver bound.
    for name,w in WEIGHTS.items():
        if max(utility(train_values(x),w) for x in plans.values())>results[name]["upper_bound"]+1e-6:
            raise RuntimeError("Cross-objective feasible value exceeds solver bound")
    # Only now evaluate holdout. No plan selection or parameter tuning on test.
    test_cache={}
    evaluated={}
    for name,x in plans.items():
        key=tuple(np.flatnonzero(x))
        if key not in test_cache:
            test_cache[key]=fixed(held,hbases,x,seconds)
        evaluated[name]={"links":[list(edges[i]) for i in key],
                         "train":metrics(train_values(x)),"test":metrics(test_cache[key])}
    return {"budget":budget,"nodes":models[0].n,"links":len(edges),
            "training_count":len(train),"test_count":len(test),
            "train_baselines":bases.tolist(),"test_baselines":hbases.tolist(),
            "setup_seconds":setup,"shared_seed_seconds":shared,
            "optimizers":results,"plans":evaluated}

def self_test():
    # Two disjoint single-edge demands. With one upgrade, worst gain
    # is zero but mean gain is 50%; the balanced objective is 1.25.
    edges=[("a","b"),("b","c"),("c","d")]
    cap=[1.,1.,1.];train=[[("a","b",1.)],[("c","d",1.)]]
    for budget in (0,1,2):
        models,bases=rb.make_models(edges,cap,train,budget,10)
        zero=np.zeros(3);seeds=[(zero,np.ones(2))]
        extra=list(singleton_rows(edges,cap,train,bases))
        # Enumerate every binary matching to independently test the master.
        feasible=[]
        for mask in range(8):
            x=np.array([(mask>>i)&1 for i in range(3)],dtype=float)
            try:
                models[0].plan(x)
            except RuntimeError:
                continue
            feasible.append(fixed(models,bases,x,10))
        for w in WEIGHTS.values():
            r=optimize(models,bases,w,10,seeds,extra)
            exact=max(utility(v,w) for v in feasible)
            assert abs(r["objective"]-exact)<1e-7 and r["closed_gap"],(budget,w,r,exact)
    # Independent numerical enumeration on a cyclic graph.
    edges=[("a","b"),("b","c"),("c","d"),("d","a"),("a","c")]
    cap=[1.,2.,1.,3.,1.];train=[[("a","c",2.),("b","d",1.)],[("a","d",3.),("b","c",2.)]]
    models,bases=rb.make_models(edges,cap,train,2,10)
    vals=[]
    for mask in range(32):
        x=np.array([(mask>>i)&1 for i in range(5)],dtype=float)
        try: models[0].plan(x)
        except RuntimeError: continue
        vals.append(fixed(models,bases,x,10))
    for w in WEIGHTS.values():
        r=optimize(models,bases,w,10,[(np.zeros(5),np.ones(2))],
                   list(singleton_rows(edges,cap,train,bases)))
        assert abs(r["objective"]-max(utility(v,w) for v in vals))<1e-7 and r["closed_gap"]

def zoo(args,rows,errors,skips):
    data=rb.fetch(rb.ZOO);eligible=[]
    for name,raw in sorted(rb.members(data).items()):
        try:
            graph=nx.Graph(nx.read_graphml(io.BytesIO(raw)))
            graph.remove_edges_from(list(nx.selfloop_edges(graph)))
            graph=graph.subgraph(max(nx.connected_components(graph),key=lambda c:(len(c),sorted(c)))).copy()
            if 20<=graph.number_of_edges()<=250:
                eligible.append((name,raw,graph))
            else:
                skips.append({"network":name,"reason":"outside edge screening range"})
        except Exception as e:
            errors.append({"network":name,"stage":"parse","error":str(e)})
    random.Random(2026).shuffle(eligible)
    if args.offset+args.limit>len(eligible):
        errors.append({"stage":"sampling","eligible":len(eligible),"requested":args.offset+args.limit})
    for name,raw,g in eligible[args.offset:args.offset+args.limit]:
        try:
            nodes=sorted(g.nodes);edges=sorted(tuple(sorted(e)) for e in g.edges)
            rng=random.Random(args.seed)
            matrices=[ [(*rng.sample(nodes,2),rng.uniform(.5,2)) for _ in range(12)]
                       for _ in range(args.train+args.test)]
            row={"network":name,"traffic":"synthetic","capacity":"unit",
                 "seed":args.seed,"archive_sha256":hashlib.sha256(data).hexdigest(),
                 "graph_sha256":hashlib.sha256(raw).hexdigest(),
                 **evaluate(edges,[1.]*len(edges),matrices[:args.train],matrices[args.train:],args.seconds)}
            rows.append(row)
            print(name,row["plans"]["balanced"]["test"],flush=True)
        except Exception as e:
            errors.append({"network":name,"error":str(e)})
            print(name,type(e).__name__,str(e),flush=True)
        rb.save(args,rows,errors,skips)

def traces(args,rows,errors,skips):
    # Reuse the existing historical chronological sampling implementation
    # with this evaluator substituted for the original max-min evaluator.
    # Historical dates have been inspected previously; exploratory only.
    old=rb.evaluate
    rb.evaluate=evaluate
    try:
        static={Path(k).stem.lower():v for k,v in rb.members(rb.fetch(URL)).items()}
        for name in rb.TRACES:
            try:
                row=rb.trace_run(name,static[name],args)
                rows.append(row)
                print(name,row["plans"]["balanced"]["test"],flush=True)
            except Exception as e:
                errors.append({"network":name,"error":str(e)})
                print(name,type(e).__name__,str(e),flush=True)
            rb.save(args,rows,errors,skips)
    finally:
        rb.evaluate=old

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["zoo","traces","self-test"],required=True)
    ap.add_argument("--train",type=int,default=6)
    ap.add_argument("--test",type=int,default=12)
    ap.add_argument("--seconds",type=float,default=15)
    ap.add_argument("--limit",type=int,default=30)
    ap.add_argument("--offset",type=int,default=0)
    ap.add_argument("--seed",type=int,default=20261125)
    ap.add_argument("--output",default="results/balanced.json")
    args=ap.parse_args()
    self_test()
    if args.mode=="self-test":
        print("Independent enumeration checks passed")
        return
    rows,errors,skips=[],[],[]
    try:
        (zoo if args.mode=="zoo" else traces)(args,rows,errors,skips)
    except Exception as e:
        errors.append({"stage":"setup","error":str(e)})
        print(type(e).__name__,str(e),flush=True)
    rb.save(args,rows,errors,skips)
    if errors or not rows:
        raise SystemExit("Incomplete benchmark: inspect artifact")

if __name__=="__main__":
    main()
