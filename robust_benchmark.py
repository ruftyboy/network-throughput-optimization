"""Robust fixed-upgrade benchmark: chronological holdout and topology sweep.

Optimize worst per-scenario relative concurrent throughput; routes may adapt.
All capacities, traffic provenance, and time limits are reported separately.
"""
import argparse
import hashlib
import io
import json
import math
import random
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, vstack
from sndlib_benchmark import Problem, parse, URL

ROOT = "https://sndlib.put.poznan.pl/download/"
TRACES = {
    "germany50": ROOT+"directed-germany50-DFN-aggregated-5min-over-1day.tgz",
    "geant": ROOT+"directed-geant-uhlig-15min-over-4months-ALL.tgz",
    "abilene": ROOT+"directed-abilene-zhang-5min-over-6months-ALL.tgz",
}
ZOO = "https://raw.githubusercontent.com/mroughan/InternetTopologyZoo/main/graphml.tar.gz"

def fetch(url):
    with urlopen(url, timeout=120) as f:
        return f.read()

def members(data):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
        return {m.name:t.extractfile(m).read() for m in t.getmembers()
                if m.isfile() and m.name.lower().endswith((".xml",".graphml"))}

def demands(data, nodes):
    root=ET.fromstring(data)
    for el in root.iter():
        el.tag=el.tag.split("}")[-1]
    result=[]
    for d in root.findall(".//demands/demand"):
        u,v=d.findtext("source"),d.findtext("target")
        w=float(d.findtext("demandValue"))*float(d.findtext("routingUnit","1"))
        if u not in nodes or v not in nodes or not math.isfinite(w) or w<0:
            raise ValueError("Invalid traffic matrix endpoint or value")
        if u!=v and w>0:
            result.append((u,v,w))
    if not result:
        raise ValueError("Empty traffic matrix")
    return result

def spread(names, count):
    if count>len(names):
        raise ValueError("Insufficient matrices")
    return [names[i] for i in np.linspace(0,len(names)-1,count,dtype=int)]

def make_models(edges, cap, matrices, budget, seconds):
    models=[Problem(edges,cap,d,budget) for d in matrices]
    bases=[p.lp(np.zeros(p.m),seconds)[0] for p in models]
    if min(bases)<=0:
        raise ValueError("Zero baseline")
    return models,np.asarray(bases)

def score(models,bases,x,seconds):
    return [p.lp(x,seconds)[0]/b for p,b in zip(models,bases)]

def endpoint_cuts(edges,cap,matrices,bases):
    """Every cut separating a single router must carry its incident demands."""
    cap=np.asarray(cap)
    nodes=sorted(set(v for e in edges for v in e))
    result=[]
    for d,b in zip(matrices,bases):
        weights=dict.fromkeys(nodes,0.)
        for u,v,w in d:
            weights[u]+=w
            weights[v]+=w
        for v in nodes:
            den=weights[v]*b
            if den>0:
                g=cap*np.asarray([v in e for e in edges])/den
                result.append((g,float(g.sum())))
    return result

def robust(models,bases,seconds,extra=(),max_iterations=100):
    """Multi-scenario cutting plane, optional singleton-cut initialization."""
    start=time.monotonic()
    p=models[0]; m=p.m
    rows=[np.r_[-g,1.] for g,_ in extra]
    rhs=[a for _,a in extra]
    x=np.zeros(m)
    best=1.; incumbent=x.copy(); upper=2.; count=0; complete=0
    for _ in range(max_iterations):
        vals=[]
        for model,base in zip(models,bases):
            remaining=seconds-(time.monotonic()-start)
            if remaining<=.05:
                break
            try:
                rate,g=model.lp(x,remaining)
            except RuntimeError:
                break
            rate,g=rate/base,g/base
            vals.append(rate)
            rows.append(np.r_[-g,1.])
            rhs.append(rate-float(g@x))
        if len(vals)!=len(models):
            break
        count+=1
        if min(vals)>best:
            best=min(vals); incumbent=x.copy()
        if upper-best<=1e-7:
            break
        remaining=seconds-(time.monotonic()-start)
        if remaining<=.05:
            break
        A=vstack([hstack([p.match,csr_matrix((p.n+1,1))]),csr_matrix(np.asarray(rows))])
        r=milp(np.r_[np.zeros(m),-1.],integrality=np.r_[np.ones(m),0],
               bounds=Bounds(np.r_[np.zeros(m),1.],np.r_[np.ones(m),2.]),
               constraints=LinearConstraint(A,-np.inf,np.r_[p.limits,rhs]),
               options={"time_limit":remaining,"mip_rel_gap":0.})
        complete+=1
        dual=getattr(r,"mip_dual_bound",None)
        if dual is not None and math.isfinite(dual):
            upper=min(upper,float(-dual))
        if upper-best<=1e-7 or r.x is None:
            break
        x=p.plan(r.x[:m])
    if upper<best-1e-6:
        raise RuntimeError("Inconsistent numeric bounds")
    return {"chosen":np.flatnonzero(incumbent).tolist(),"worst_train_ratio":best,
            "upper_bound":max(best,upper),"iterations":count,"master_solves":complete,
            "seconds":time.monotonic()-start,"closed_gap":upper-best<=1e-7}

def evaluate(edges,cap,train,test,seconds,compare):
    budget=len(edges)//20
    models,bases=make_models(edges,cap,train,budget,seconds)
    held,hbases=make_models(edges,cap,test,budget,seconds)
    plain=robust(models,bases,seconds) if compare else None
    strong=robust(models,bases,seconds,endpoint_cuts(edges,cap,train,bases))
    if plain and (strong["worst_train_ratio"]>plain["upper_bound"]+1e-6 or
                  plain["worst_train_ratio"]>strong["upper_bound"]+1e-6):
        raise RuntimeError("Methods disagree on valid bounds")
    single=models[0].cuts(seconds)
    plans={"endpoint_robust":strong["chosen"],"single_snapshot":single["chosen"],"no_upgrade":[]}
    if plain:
        plans["plain_robust"]=plain["chosen"]
    # A deterministic topology-only baseline, same upgrade count budget.
    g=nx.Graph();g.add_edges_from(edges)
    central={frozenset(e):v for e,v in nx.edge_betweenness_centrality(g).items()}
    used=set(); chosen=[]
    for i in sorted(range(len(edges)),key=lambda i:(-central[frozenset(edges[i])],i)):
        if len(chosen)>=budget:
            break
        u,v=edges[i]
        if u not in used and v not in used:
            chosen.append(i);used.update((u,v))
    plans["betweenness"]=chosen
    results={}
    for name,idx in plans.items():
        x=np.zeros(len(edges));x[idx]=1;models[0].plan(x)
        a=score(models,bases,x,seconds)
        b=score(held,hbases,x,seconds)
        results[name]={"links":[list(edges[i]) for i in idx],
                       "train_gain_pct":[100*(v-1) for v in a],
                       "test_gain_pct":[100*(v-1) for v in b],
                       "worst_train_gain_pct":100*(min(a)-1),
                       "worst_test_gain_pct":100*(min(b)-1),
                       "mean_test_gain_pct":100*(np.mean(b)-1)}
    return {"budget":budget,"nodes":models[0].n,"links":len(edges),
            "train_demands":[len(d) for d in train],"test_demands":[len(d) for d in test],
            "plain":plain,"strengthened":strong,"plans":results}

def trace_run(name,raw,args):
    edges,cap,_,meta=parse(raw,name,"module")
    nodes=set(v for e in edges for v in e)
    data=fetch(TRACES[name])
    files=members(data)
    # SNDlib uses timestamped matrix filenames. Sort by basename, not archive order.
    names=sorted(files,key=lambda k:Path(k).name)
    split=len(names)//2
    a=spread(names[:split],args.train);b=spread(names[split:],args.test)
    train=[demands(files[k],nodes) for k in a]
    test=[demands(files[k],nodes) for k in b]
    return {**meta,"network":name,"traffic":"historical measurement-derived",
            "trace_url":TRACES[name],"trace_sha256":hashlib.sha256(data).hexdigest(),
            "available_matrices":len(names),"train_files":a,"test_files":b,
            **evaluate(edges,cap,train,test,args.seconds,True)}

def zoo_run(args,rows,errors,skips):
    data=fetch(ZOO)
    files=members(data)
    eligible=[]
    for name,raw in sorted(files.items()):
        try:
            graph=nx.Graph(nx.read_graphml(io.BytesIO(raw)))
            graph.remove_edges_from(list(nx.selfloop_edges(graph)))
            graph=graph.subgraph(max(nx.connected_components(graph),key=lambda c:(len(c),sorted(c)))).copy()
            if not 20<=graph.number_of_edges()<=250:
                skips.append({"network":name,"reason":"outside 20–250 edge screening range"});continue
            eligible.append((name,raw,graph))
        except Exception as e:
            errors.append({"network":name,"stage":"graph parse","error":str(e)})
    # Deterministic sample of eligible files, fixed before any optimization.
    random.Random(2026).shuffle(eligible)
    for name,raw,g in eligible[:args.limit]:
        try:
            nodes=sorted(g.nodes);edges=sorted(tuple(sorted(e)) for e in g.edges)
            rng=random.Random(42)
            matrices=[]
            for _ in range(args.train+args.test):
                matrices.append([(*rng.sample(nodes,2),rng.uniform(.5,2)) for _ in range(12)])
            row={"network":name,"traffic":"synthetic","capacity":"unit",
                 "archive_sha256":hashlib.sha256(data).hexdigest(),"graph_sha256":hashlib.sha256(raw).hexdigest(),
                 **evaluate(edges,[1.]*len(edges),matrices[:args.train],matrices[args.train:],args.seconds,False)}
            rows.append(row)
            print(name,row["plans"]["endpoint_robust"]["worst_test_gain_pct"],flush=True)
        except Exception as e:
            errors.append({"network":name,"error":str(e)})
        save(args,rows,errors,skips)

def save(args,rows,errors,skips):
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"mode":args.mode,"results":rows,"errors":errors,"skipped":skips},
                               indent=2,allow_nan=False))

def self_test():
    # Disjoint bottlenecks: no single edge upgrade helps both scenarios.
    edges=[("a","b"),("b","c"),("c","d")]
    cap=[1.,1.,1.]; train=[[("a","b",1.)],[("c","d",1.)]]
    p,b=make_models(edges,cap,train,1,10)
    for extra in ([],endpoint_cuts(edges,cap,train,b)):
        r=robust(p,b,10,extra)
        assert abs(r["worst_train_ratio"]-1)<1e-7 and r["closed_gap"]
    p,b=make_models(edges,cap,train,2,10)
    r=robust(p,b,10,endpoint_cuts(edges,cap,train,b))
    assert abs(r["worst_train_ratio"]-2)<1e-7 and r["closed_gap"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["traces","zoo"],required=True)
    ap.add_argument("--train",type=int,default=3)
    ap.add_argument("--test",type=int,default=6)
    ap.add_argument("--seconds",type=float,default=30)
    ap.add_argument("--limit",type=int,default=120)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    self_test()
    rows,errors,skips=[],[],[]
    if args.mode=="traces":
        try:
            static={Path(k).stem.lower():v for k,v in members(fetch(URL)).items()}
            for name in TRACES:
                try:
                    row=trace_run(name,static[name],args)
                    rows.append(row)
                    print(name,row["plans"]["endpoint_robust"]["worst_test_gain_pct"],flush=True)
                except Exception as e:
                    errors.append({"network":name,"error":str(e)})
                save(args,rows,errors,skips)
        except Exception as e:
            errors.append({"stage":"trace setup","error":str(e)})
    else:
        try:
            zoo_run(args,rows,errors,skips)
        except Exception as e:
            errors.append({"stage":"zoo setup","error":str(e)})
    save(args,rows,errors,skips)
    if errors or not rows:
        raise SystemExit("Incomplete benchmark: inspect artifact")

if __name__=="__main__":
    main()
