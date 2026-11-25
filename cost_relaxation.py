"""Cost-aware relaxation of matching-constrained upgrades.

Compare identical budgets under different per-router upgrade limits.
All actual throughput values are evaluated by source-aggregated routing LPs.
"""
import argparse
import hashlib
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

import robust_benchmark as rb
from persistent_benchmark import scan, SourceProblem
from sndlib_benchmark import URL, parse

def module_prices(raw,cap):
    root=ET.fromstring(raw)
    for el in root.iter():
        el.tag=el.tag.split("}")[-1]
    out=[]
    for link,c in zip(root.findall("./networkStructure/links/link"),cap):
        offers=[]
        for module in link.findall("additionalModules/*"):
            try:
                size=float(module.findtext("capacity"))
                price=float(module.findtext("cost"))
            except (TypeError,ValueError):
                continue
            if math.isfinite(size) and math.isfinite(price) and size>=c and price>0:
                offers.append(price)
        out.append(min(offers) if offers else None)
    return out

class Policy:
    def __init__(self,edges,cap,k,degree,cost=None,budget=None,setup=0.,headroom=None):
        self.edges=edges
        self.cap=np.asarray(cap,dtype=float)
        self.m=m=len(edges)
        self.nodes=sorted({v for e in edges for v in e})
        self.n=n=len(self.nodes)
        node={v:i for i,v in enumerate(self.nodes)}
        incidence=np.zeros((n,m))
        for i,(u,v) in enumerate(edges):
            incidence[node[u],i]=incidence[node[v],i]=1.
        self.incidence=incidence
        self.cost=None if cost is None else np.asarray(cost,dtype=float)
        self.budget=budget
        self.setup=setup
        self.degree=degree
        self.k=k
        self.headroom=headroom
        # Variables: binary link choices x, binary router activations z,
        # scalar throughput gain upper envelope t.
        rows=[];rhs=[]
        def add(x,z,b):
            rows.append(np.r_[x,z,0.]);rhs.append(b)
        add(np.ones(m),np.zeros(n),k)
        for i in range(n):
            z=np.zeros(n);z[i]=1.
            add(incidence[i],-degree*z,0.)
            add(-incidence[i],z,0.)
            if headroom is not None:
                w=incidence[i]*self.cap
                add(w,np.zeros(n),headroom*w.sum())
        if self.cost is not None:
            add(self.cost,np.full(n,setup),budget)
        self.A=np.asarray(rows);self.b=np.asarray(rhs)
    def accounting(self,x):
        z=(self.incidence@x>0).astype(float)
        return {"upgrades":int(np.sum(x)),"routers_touched":int(z.sum()),
                "max_upgrades_at_router":int(np.max(self.incidence@x)),
                "added_capacity_fraction":float(self.cap@x/self.cap.sum()),
                "modeled_cost":None if self.cost is None else float(self.cost@x+self.setup*z.sum())}
    def validate(self,x):
        z=(self.incidence@x>0).astype(float)
        if np.max(np.abs(x-np.rint(x)))>1e-6 or np.min(x)<-1e-6 or np.max(x)>1+1e-6:
            raise RuntimeError("Invalid upgrade vector")
        if np.max(self.A@np.r_[x,z,0.]-self.b)>1e-6:
            raise RuntimeError("Infeasible upgrade policy")
    def master(self,cuts,seconds):
        A=np.vstack([self.A]+[np.r_[-g,np.zeros(self.n),1.][None,:] for g,_ in cuts])
        rhs=np.r_[self.b,[a for _,a in cuts]]
        objective=np.r_[np.zeros(self.m+self.n),-1.]
        r=milp(objective,integrality=np.r_[np.ones(self.m+self.n),0.],
               bounds=Bounds(np.zeros(len(objective)),np.ones(len(objective))),
               constraints=LinearConstraint(csr_matrix(A),-np.inf,rhs),
               options={"time_limit":seconds,"mip_rel_gap":0.})
        bound=getattr(r,"mip_dual_bound",None)
        upper=1. if bound is None or not math.isfinite(bound) else min(1.,float(-bound))
        x=None if r.x is None else np.rint(r.x[:self.m])
        if x is not None:self.validate(x)
        return x,upper,int(r.status)

def evaluate(models,bases,x,seconds,cache):
    key=tuple(np.flatnonzero(x))
    if key not in cache:
        vals=[];grads=[]
        for p,b in zip(models,bases):
            rate,g=p.lp(x,seconds)
            vals.append(rate/b-1.)
            grads.append(g/b)
        vals=np.asarray(vals);grads=np.asarray(grads)
        if np.min(vals)<-1e-6 or np.max(vals)>1+1e-6:
            raise RuntimeError("Routing gain outside physical range")
        cache[key]=(vals,grads)
    return cache[key]

def metrics(v):
    return {"gain_pct":(100*v).tolist(),"mean_gain_pct":float(100*v.mean()),
            "minimum_gain_pct":float(100*v.min()),"median_gain_pct":float(100*np.median(v)),
            "fraction_at_least_5pct":float(np.mean(v>=.05-1e-7))}

def optimize(policy,models,bases,G,seconds,iterations,cache):
    # Each evaluated plan supplies a global supporting hyperplane for the
    # mean normalized gain. The master upper bound is valid for all plans.
    cuts=[(G.mean(axis=0),0.)]
    x=np.zeros(policy.m);best=x.copy();value=0.;upper=1.
    history=[];start=time.monotonic()
    for i in range(iterations):
        x,ub,status=policy.master(cuts,seconds)
        upper=min(upper,ub)
        if x is None:break
        vals,grads=evaluate(models,bases,x,seconds,cache)
        if np.any(vals>G@x+1e-6):raise RuntimeError("Baseline dual bound violated")
        avg=float(vals.mean())
        g=grads.mean(axis=0)
        cuts.append((g,avg-float(g@x)))
        history.append({"chosen":np.flatnonzero(x).tolist(),"mean_gain_pct":100*avg,
                        "upper_gain_pct":100*upper,"master_status":status})
        if avg>value+1e-9:best=x.copy();value=avg
        if value>upper+1e-6:raise RuntimeError("Mean gain exceeds master bound")
        if upper-value<=1e-7:break
    policy.validate(best)
    return best,{"mean_train_gain_pct":100*value,"train_upper_gain_pct":100*max(value,upper),
                 "numerically_closed_gap":bool(upper-value<=1e-7),"history":history,
                 "selection_seconds":time.monotonic()-start}

def historical():
    static={Path(k).stem.lower():v for k,v in rb.members(rb.fetch(URL)).items()}
    raw=static["germany50"]
    edges,cap,_,meta=parse(raw,"germany50","module")
    prices=module_prices(raw,cap)
    nodes={v for e in edges for v in e}
    data=rb.fetch(rb.TRACES["germany50"]);files=rb.members(data)
    names=sorted((k for k in files if Path(k).name.lower().startswith("demandmatrix") and k.endswith(".xml")),
                 key=lambda k:Path(k).name)
    if len(names)!=288:raise RuntimeError("Expected 288 snapshots")
    train=[];test=[];excluded=[];train_names=[];test_names=[]
    for i,key in enumerate(names):
        try:d=rb.demands(files[key],nodes)
        except ValueError as e:
            if str(e)!="Empty traffic matrix":raise
            excluded.append(key);continue
        if i<144:train.append(d);train_names.append(key)
        else:test.append(d);test_names.append(key)
    return edges,cap,prices,train,test,{**meta,"network":"germany50",
           "trace_sha256":hashlib.sha256(data).hexdigest(),"available_matrices":len(names),
           "excluded_empty":excluded,"train_files":train_names,"test_files":test_names}

def study(args):
    edges,cap,prices,train,test,meta=historical()
    k=len(edges)//20
    if k<1:raise RuntimeError("No upgrades allowed")
    models,bases,G=scan(edges,cap,train,k,args.seconds)
    print("Training snapshots scanned",len(train),flush=True)
    policies={}
    for degree in range(1,k+1):
        policies["count_degree_"+str(degree)]=Policy(edges,cap,k,degree)
    # Fixed count cap, matched cost budgets across degree limits.
    # Capacity cost is a normalized engineering proxy, not a monetary cost.
    cost_sets={"capacity_proxy":np.asarray(cap,dtype=float)}
    if all(p is not None for p in prices):
        cost_sets["published_module_cost_proxy"]=np.asarray(prices,dtype=float)
    for name,raw_cost in cost_sets.items():
        cost=raw_cost/np.median(raw_cost)
        setup=.25
        budget=k+2*k*setup
        for degree in (1,k):
            policies[name+"_degree_"+str(degree)]=Policy(edges,cap,k,degree,cost,budget,setup)
        policies[name+"_headroom_50pct"]=Policy(edges,cap,k,k,cost,budget,setup,.5)
    # Additional pure count headroom sensitivity, not a measured router limit.
    policies["count_headroom_50pct"]=Policy(edges,cap,k,k,headroom=.5)
    cache={};chosen={};records={}
    for name,policy in policies.items():
        x,r=optimize(policy,models,bases,G,args.seconds,args.iterations,cache)
        chosen[name]=x
        vals,_=evaluate(models,bases,x,args.seconds,cache)
        records[name]={**r,"edges":[list(edges[i]) for i in np.flatnonzero(x)],
                       "accounting":policy.accounting(x),"train":metrics(vals),
                       "policy":{"degree":policy.degree,"count":k,"cost_budget":policy.budget,
                                 "router_setup_cost":policy.setup,"headroom":policy.headroom}}
        print("train",name,r["mean_train_gain_pct"],r["train_upper_gain_pct"],records[name]["edges"],flush=True)
    # Cross-check valid upper bounds against all known feasible plans.
    for name,policy in policies.items():
        for x in chosen.values():
            try:policy.validate(x)
            except RuntimeError:continue
            v,_=evaluate(models,bases,x,args.seconds,cache)
            if 100*v.mean()>records[name]["train_upper_gain_pct"]+1e-4:
                raise RuntimeError("Cross-policy feasible plan violates upper bound")
    del models,cache
    held,hbases,H=scan(edges,cap,test,k,args.seconds)
    print("Holdout snapshots scanned",len(test),flush=True)
    cache={}
    for name,x in chosen.items():
        v,_=evaluate(held,hbases,x,args.seconds,cache)
        if np.any(v>H@x+1e-6):raise RuntimeError("Holdout dual bound violated")
        records[name]["test"]=metrics(v)
        print("test",name,records[name]["test"]["mean_gain_pct"],records[name]["test"]["minimum_gain_pct"],flush=True)
    return {**meta,"train_count":len(train),"test_count":len(test),"budget_count":k,
            "published_prices_available":sum(p is not None for p in prices),
            "cost_note":"Published module prices are benchmark proxies; capacity proxy and setup costs are assumptions.",
            "records":records}

def self_test():
    # Star bottleneck: four equal branches, all demand has one source.
    # Upgrading d branches raises rate by d/4 only if demand distribution
    # permits; choose one separate scenario per branch to get exact mean d/4.
    edges=[("a",v) for v in ("b","c","d","e")]
    cap=[1.]*4
    matrices=[[("a",v,1.)] for v in ("b","c","d","e")]
    models,bases,G=scan(edges,cap,matrices,4,10)
    cache={}
    for d in range(1,5):
        p=Policy(edges,cap,4,d)
        x,r=optimize(p,models,bases,G,10,4,cache)
        assert abs(r["mean_train_gain_pct"]-25*d)<1e-6 and r["numerically_closed_gap"]
    p=Policy(edges,cap,4,4,cost=np.ones(4),budget=3.,setup=.25,headroom=.5)
    x,r=optimize(p,models,bases,G,10,4,cache)
    # leaf headroom 50% makes any doubled leaf edge infeasible.
    assert r["mean_train_gain_pct"]<1e-7
    p=Policy(edges,cap,4,4,cost=np.ones(4),budget=3.,setup=.25)
    x,r=optimize(p,models,bases,G,10,4,cache)
    assert abs(r["mean_train_gain_pct"]-50)<1e-6
    # Independent enumeration over all subsets to verify policy/cost master.
    feasible=[]
    for mask in range(16):
        x=np.array([(mask>>i)&1 for i in range(4)],dtype=float)
        try:p.validate(x)
        except RuntimeError:continue
        vals,_=evaluate(models,bases,x,10,cache)
        feasible.append(float(vals.mean()))
    assert abs(max(feasible)*100-r["mean_train_gain_pct"])<1e-7

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["self-test","historical"],required=True)
    ap.add_argument("--seconds",type=float,default=60)
    ap.add_argument("--iterations",type=int,default=6)
    ap.add_argument("--output",default="results/cost-relaxation.json")
    args=ap.parse_args()
    self_test()
    if args.mode=="self-test":
        print("Degree, cost, headroom and enumeration checks passed");return
    try:
        row=study(args)
        rb.save(args,[row],[],[])
    except Exception as e:
        rb.save(args,[],[{"error":str(e)}],[])
        raise

if __name__=="__main__":
    main()
