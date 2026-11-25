"""Generalization of concentrated augmentation across topology instances.

Compare nested router constraints under identical count and modeled cost
budgets. Fresh traffic; topology files may overlap historical experiments.
"""
import argparse
import hashlib
import io
import math
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import robust_benchmark as rb
from persistent_benchmark import scan
from cost_relaxation import Policy, evaluate, metrics

def bridge_requirements(edges, cap, matrices, bases, threshold=.05):
    """Necessary threshold conditions from bridges; sufficient on a tree."""
    g=nx.Graph();g.add_edges_from(edges)
    lookup={frozenset(e):i for i,e in enumerate(edges)}
    bridges=[]
    for u,v in list(nx.bridges(g)):
        g.remove_edge(u,v)
        side=set(nx.node_connected_component(g,u))
        g.add_edge(u,v)
        bridges.append((lookup[frozenset((u,v))],side))
    result=[]
    for demands,b in zip(matrices,bases):
        required=[];impossible=False
        for i,side in bridges:
            load=sum(w for s,t,w in demands if (s in side)!=(t in side))
            target=(1+threshold)*b*load
            tol=1e-7*max(1.,abs(target),cap[i])
            if target>2*cap[i]+tol:impossible=True
            if target>cap[i]+tol:required.append(i)
        result.append((required,impossible))
    return result

def coverage_bound(policy,G,requirements,threshold,seconds):
    """Upper bound on threshold successes using dual and mandatory bridge cuts."""
    n,m=policy.n,policy.m;k=len(G)
    rows=[np.c_[policy.A[:,:m+n],np.zeros((len(policy.b),k))]]
    rhs=list(policy.b)
    dual_rows=np.zeros((k,m+n+k))
    dual_rows[:,:m]=-G
    dual_rows[:,m+n:]=threshold*np.eye(k)
    rows.append(dual_rows);rhs.extend([0.]*k)
    upper_z=np.ones(k)
    for s,(mandatory,impossible) in enumerate(requirements):
        if impossible:upper_z[s]=0.
        for e in mandatory:
            row=np.zeros(m+n+k);row[e]=-1;row[m+n+s]=1
            rows.append(row[None,:]);rhs.append(0.)
    c=np.r_[np.zeros(m+n),-np.ones(k)]
    from scipy.sparse import csr_matrix
    r=milp(c,integrality=np.ones(m+n+k),
           bounds=Bounds(np.zeros(m+n+k),np.r_[np.ones(m+n),upper_z]),
           constraints=LinearConstraint(csr_matrix(np.vstack(rows)),-np.inf,np.asarray(rhs)),
           options={"time_limit":seconds,"mip_rel_gap":0.})
    d=getattr(r,"mip_dual_bound",None)
    upper=k if d is None or not math.isfinite(d) else min(k,math.floor(-d+1e-6))
    return {"upper_count":int(upper),"scenarios":k,"status":int(r.status)}

def optimize(policy,models,bases,G,seconds,iterations,cache,warm=()):
    # Carry feasible incumbents across nested policies. Add their cuts too.
    start=time.monotonic()
    best=np.zeros(policy.m);value=0.;upper=1.
    cuts=[(G.mean(axis=0),0.)]
    history=[]
    for x in warm:
        try:policy.validate(x)
        except RuntimeError:continue
        v,grad=evaluate(models,bases,x,seconds,cache)
        avg=float(v.mean());a=grad.mean(axis=0)
        cuts.append((a,avg-float(a@x)))
        if avg>value+1e-9:best=x.copy();value=avg
    for i in range(iterations):
        x,ub,status=policy.master(cuts,seconds)
        upper=min(upper,ub)
        if upper-value<=1e-7 or x is None:break
        v,grad=evaluate(models,bases,x,seconds,cache)
        if np.any(v>G@x+1e-6):raise RuntimeError("Baseline dual upper bound violation")
        avg=float(v.mean());a=grad.mean(axis=0)
        cuts.append((a,avg-float(a@x)))
        history.append({"train_gain_pct":100*avg,"upper_pct":100*upper,"status":status})
        if avg>value+1e-9:best=x.copy();value=avg
        if upper-value<=1e-7:break
    if value>upper+1e-6:raise RuntimeError("Invalid upper bound")
    policy.validate(best)
    return best,{"lower_pct":100*value,"upper_pct":100*max(value,upper),
                 "closed_gap":bool(upper-value<=1e-7),"seconds":time.monotonic()-start,
                 "history":history}

def study(edges,cap,train,test,seconds,iterations):
    k=len(edges)//20
    m=len(edges)
    models,bases,G=scan(edges,cap,train,k,seconds)
    costs=np.asarray(cap)/np.median(cap)
    budget=1.5*k
    # Identical count and normalized cost budgets across all cost-aware policies.
    specs={"count_matching":Policy(edges,cap,k,1),
           "count_unrestricted":Policy(edges,cap,k,k),
           "cost_matching":Policy(edges,cap,k,1,costs,budget,.25),
           "cost_degree2":Policy(edges,cap,k,min(2,k),costs,budget,.25),
           "cost_unrestricted":Policy(edges,cap,k,k,costs,budget,.25),
           "cost_headroom":Policy(edges,cap,k,k,costs,budget,.25,.5)}
    cache={};plans={};records={}
    req=bridge_requirements(edges,cap,train,bases)
    for name,p in specs.items():
        # Baseline dual bound isolates opportunity, even if iterations time out.
        _,initial_ub,_=p.master([(G.mean(axis=0),0.)],seconds)
        x,out=optimize(p,models,bases,G,seconds,iterations,cache,list(plans.values()))
        v,_=evaluate(models,bases,x,seconds,cache)
        plans[name]=x
        records[name]={**out,"dual_screen_upper_pct":100*initial_ub,
                       "plan_edges":[list(edges[i]) for i in np.flatnonzero(x)],
                       "accounting":p.accounting(x),"train":metrics(v),
                       "train_threshold_bound":coverage_bound(p,G,req,.05,seconds)}
    # Validate every upper bound against every known feasible placement.
    for name,p in specs.items():
        for x in plans.values():
            try:p.validate(x)
            except RuntimeError:continue
            v,_=evaluate(models,bases,x,seconds,cache)
            if 100*v.mean()>records[name]["upper_pct"]+1e-4:
                raise RuntimeError("Cross-policy upper bound violation")
    del models,cache
    held,hb,H=scan(edges,cap,test,k,seconds)
    hreq=bridge_requirements(edges,cap,test,hb)
    cache={}
    for name,x in plans.items():
        p=specs[name];v,_=evaluate(held,hb,x,seconds,cache)
        if np.any(v>H@x+1e-6):raise RuntimeError("Holdout dual bound violation")
        b=coverage_bound(p,H,hreq,.05,seconds)
        if int(np.sum(v>=.05-1e-7))>b["upper_count"]:
            raise RuntimeError("Coverage bound violation")
        records[name]["test"]=metrics(v)
        records[name]["test_retrospective_threshold_bound"]=b
    comparisons={}
    for label,limited,relaxed in [("count","count_matching","count_unrestricted"),
                                  ("cost","cost_matching","cost_unrestricted")]:
        a,b=records[limited],records[relaxed]
        comparisons[label]={"training_advantage_lower_pct_points":max(0.,b["lower_pct"]-a["upper_pct"]),
                            "training_advantage_upper_pct_points":max(0.,b["upper_pct"]-a["lower_pct"]),
                            "certified_strict_training_advantage":bool(b["lower_pct"]>a["upper_pct"]+1e-5),
                            "test_mean_difference_pct_points":b["test"]["mean_gain_pct"]-a["test"]["mean_gain_pct"],
                            "test_minimum_difference_pct_points":b["test"]["minimum_gain_pct"]-a["test"]["minimum_gain_pct"]}
    return {"nodes":len({v for e in edges for v in e}),"links":m,"budget":k,
            "train_count":len(train),"test_count":len(test),"records":records,
            "comparisons":comparisons,
            "train_scenarios_adjacent_required_bridges":sum(
                len({v for e in mandatory for v in edges[e]})<2*len(mandatory)
                for mandatory,_ in req),
            "max_mean_shadow_at_router":float(np.max(specs["count_matching"].incidence@G.mean(axis=0)))}

def self_test():
    # A serial two-edge bottleneck: one upgrade cannot improve throughput,
    # two adjacent upgrades double it. Thus no finite universal ratio of gains.
    edges=[("a","b"),("b","c")];cap=[1.,1.]
    matrices=[[("a","c",1.)]]
    models,b,G=scan(edges,cap,matrices,2,10)
    req=bridge_requirements(edges,cap,matrices,b,.05)
    assert req==[([0,1],False)] or req==[([1,0],False)]
    cache={}
    a=Policy(edges,cap,2,1);z=Policy(edges,cap,2,2)
    x,r=optimize(a,models,b,G,10,10,cache)
    y,s=optimize(z,models,b,G,10,10,cache)
    assert abs(r["upper_pct"])<1e-6 and abs(s["lower_pct"]-100)<1e-6
    assert coverage_bound(a,G,req,.05,10)["upper_count"]==0
    assert coverage_bound(z,G,req,.05,10)["upper_count"]==1
    # Verify threshold characterization on a heterogeneous tree.
    edges=[("a","b"),("b","c"),("b","d")];cap=[2.,1.,3.]
    matrices=[[("a","c",2.),("a","d",1.)],[("c","d",1.),("a","d",2.)]]
    models,b,G=scan(edges,cap,matrices,3,10)
    req=bridge_requirements(edges,cap,matrices,b,.2)
    for mask in range(8):
        x=np.array([(mask>>i)&1 for i in range(3)],dtype=float)
        v,_=evaluate(models,b,x,10,{})
        for i,(mandatory,impossible) in enumerate(req):
            expected=not impossible and all(x[e]>0.5 for e in mandatory)
            assert expected==bool(v[i]>=.2-1e-7)

def run(args):
    data=rb.fetch(rb.ZOO);eligible=[];errors=[];skips=[]
    for name,raw in sorted(rb.members(data).items()):
        try:
            graph=nx.Graph(nx.read_graphml(io.BytesIO(raw)))
            graph.remove_edges_from(list(nx.selfloop_edges(graph)))
            graph=graph.subgraph(max(nx.connected_components(graph),key=lambda c:(len(c),sorted(c)))).copy()
            if 20<=graph.number_of_edges()<=250:eligible.append((name,raw,graph))
            else:skips.append({"network":name,"reason":"outside edge screening range"})
        except Exception as e:errors.append({"network":name,"error":str(e)})
    random.Random(2026).shuffle(eligible)
    if len(eligible)<args.offset+args.limit:
        errors.append({"stage":"sampling","eligible":len(eligible)})
    rows=[]
    for name,raw,g in eligible[args.offset:args.offset+args.limit]:
        nodes=sorted(g.nodes);edges=sorted(tuple(sorted(e)) for e in g.edges)
        rng=random.Random(20261127)
        matrices=[[(*rng.sample(nodes,2),rng.uniform(.5,2)) for _ in range(12)]
                  for _ in range(args.train+args.test)]
        # Separate fixed seed for heterogeneous modeled capacities.
        crng=random.Random(1701)
        cap=[float(crng.choice([1,2,4])) for _ in edges] if args.capacity=="heterogeneous" else [1.]*len(edges)
        try:
            row={"network":name,"traffic":"synthetic","capacity":args.capacity,
                 "archive_sha256":hashlib.sha256(data).hexdigest(),"graph_sha256":hashlib.sha256(raw).hexdigest(),
                 **study(edges,cap,matrices[:args.train],matrices[args.train:],args.seconds,args.iterations)}
            rows.append(row)
            print(name,args.capacity,row["comparisons"],flush=True)
        except Exception as e:
            errors.append({"network":name,"error":str(e)})
            print(name,type(e).__name__,str(e),flush=True)
        rb.save(args,rows,errors,skips)
    rb.save(args,rows,errors,skips)
    if errors or len(rows)!=args.limit:raise SystemExit("Incomplete topology experiment")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["self-test","zoo"],required=True)
    ap.add_argument("--capacity",choices=["unit","heterogeneous"],default="unit")
    ap.add_argument("--offset",type=int,default=0)
    ap.add_argument("--limit",type=int,default=30)
    ap.add_argument("--train",type=int,default=12)
    ap.add_argument("--test",type=int,default=24)
    ap.add_argument("--seconds",type=float,default=5)
    ap.add_argument("--iterations",type=int,default=6)
    ap.add_argument("--output",default="results/concentration.json")
    args=ap.parse_args()
    self_test()
    if args.mode=="self-test":
        print("Bridge characterization and strict concentration example passed");return
    run(args)

if __name__=="__main__":
    main()
