"""Routing-solve elimination with reusable primal/dual certificates.

All methods use identical multicut masters and budgets. Reuse only within
the same traffic scenario; no holdout information enters plan selection.
"""
import argparse, hashlib, io, math, random, time
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
import robust_benchmark as rb
from persistent_benchmark import SourceProblem
from cost_relaxation import Policy, metrics, historical
from screening_benchmark import build
from adaptive_benchmark import EPS, CUT_PAD, NUM

PAD=5e-6
LOCAL=EPS/4

def solve(p,x,seconds):
    b=p.cap*(1+x)
    r=linprog(p.obj,A_eq=p.eq,b_eq=np.zeros(p.q),A_ub=p.ub,b_ub=b,
              bounds=(0,None),method="highs",
              options={"time_limit":seconds})
    if not r.success:raise RuntimeError("routing LP: "+r.message)
    load=np.asarray(p.ub@r.x)
    if max(np.max(np.abs(p.eq@r.x)),np.max(load-b),-np.min(r.x))>1e-6:
        raise RuntimeError("routing residual too large")
    rate=float(r.x[-1])
    grad=-np.asarray(r.ineqlin.marginals)*p.cap
    if rate<=0 or np.min(grad)<-1e-7:raise RuntimeError("invalid routing certificate")
    # Store aggregate load normalized by original capacities. Scaling a
    # feasible multicommodity flow scales every demand and every link load.
    return rate,grad,np.maximum(load,0.)/p.cap

def scan(edges,cap,matrices,K,seconds):
    models=[];bases=[];slopes=[];loads=[]
    for d in matrices:
        p=SourceProblem(edges,cap,d,K)
        v,g,l=solve(p,np.zeros(p.m),seconds)
        if abs(g.sum()/v-1)>1e-5:raise RuntimeError("baseline dual normalization")
        models.append(p);bases.append(v);slopes.append(g/v);loads.append(l)
    return models,np.asarray(bases),np.asarray(slopes),np.asarray(loads)

def lower_from_flows(x,ratios,loads):
    # A prior flow with normalized throughput r and normalized loads l
    # can be scaled by min_e (1+x_e)/l_e. Zero loads impose no restriction.
    factors=np.min(np.divide(1.+x,loads,out=np.full_like(loads,np.inf),where=loads>1e-14),axis=1)
    return max(0.,min(1.,float(np.max(ratios*factors))-1.-PAD))

class ReuseSolver:
    def __init__(self,p,models,bases,G,loads,seconds,mode):
        self.p=p;self.models=models;self.bases=bases;self.seconds=seconds;self.mode=mode
        self.k=len(models);self.m=p.m
        self.cuts=[(G.copy(),np.zeros(self.k))]
        self.flows=[[(1.,l.copy())] for l in loads]
        self.active=np.arange(self.m)
        self.best=0.;self.plan=np.zeros(self.m);self.upper=1.
        self.queries=0;self.calls=0;self.skipped=0;self.rejected=0
        self.master_calls=0;self.routing_time=0.;self.bound_time=0.;self.master_time=0.
        self.incidents=[];self.invalid=False
        self.cache={(): (np.zeros(self.k),np.zeros(self.k))}
        self.history=[]

    def bounds(self,x,use_primal=True):
        start=time.monotonic()
        vals=np.stack([np.maximum(g,0.)@x+np.maximum(b,0.)+CUT_PAD for g,b in self.cuts])
        idx=np.argmin(vals,axis=0)
        hi=np.minimum(1.,np.min(vals,axis=0)+PAD)
        lo=[]
        if use_primal:
            for flows in self.flows:
                rates=np.asarray([r for r,_ in flows]);loads=np.asarray([l for _,l in flows])
                lo.append(lower_from_flows(x,rates,loads))
        else:
            lo=np.zeros(self.k)
        lo=np.asarray(lo)
        if np.any(lo>hi+NUM):raise RuntimeError("reused lower exceeds upper")
        self.bound_time+=time.monotonic()-start
        return lo,hi,idx

    def query(self,x):
        self.p.validate(x);key=tuple(np.flatnonzero(x))
        if key in self.cache:return False
        self.queries+=1
        lo,hi,idx=self.bounds(x,use_primal=self.mode!='full')
        # Fill skipped cut rows with a previously valid supporting cut.
        # For solved rows, replace with the new supporting cut.
        gs=np.stack([self.cuts[j][0][s] for s,j in enumerate(idx)])
        bs=np.asarray([self.cuts[j][1][s] for s,j in enumerate(idx)])
        done=np.zeros(self.k,dtype=bool);solved=0
        while not done.all():
            if self.mode=="lazy" and hi.mean()<=self.best+LOCAL:
                self.rejected+=1;break
            if self.mode!="full" and (hi-lo).mean()<=LOCAL:break
            remaining=np.flatnonzero(~done)
            if self.mode=="full":s=int(remaining[0])
            else:
                s=int(remaining[np.argmax((hi-lo)[remaining])])
                # Reuse a sufficiently tight individual certificate even if
                # other scenarios need solving.
                if hi[s]-lo[s]<=LOCAL:break
            start=time.monotonic()
            rate,g,load=solve(self.models[s],x,self.seconds)
            self.routing_time+=time.monotonic()-start
            self.calls+=1;solved+=1;done[s]=True
            val=rate/self.bases[s]-1.;g=g/self.bases[s]
            if val<lo[s]-2*PAD or val>hi[s]+2*PAD or val< -NUM or val>1+NUM:
                raise RuntimeError("routing value contradicts reused certificate")
            val=max(0.,min(1.,val))
            lo[s]=max(0.,val-PAD);hi[s]=min(1.,val+PAD)
            intercept=val-float(g@x)
            for oldkey,(oldlo,_) in self.cache.items():
                ox=np.zeros(self.m);ox[list(oldkey)]=1
                if oldlo[s]>np.maximum(g,0.)@ox+max(intercept,0.)+CUT_PAD+2*PAD:
                    raise RuntimeError("new cut contradicts known feasible value")
            gs[s]=g;bs[s]=intercept
            self.flows[s].append((rate/self.bases[s],load))
        self.skipped+=self.k-solved
        self.cache[key]=(lo.copy(),hi.copy())
        if solved:self.cuts.append((gs,bs))
        value=float(lo.mean())
        if value>self.best:
            self.best=value;self.plan=x.copy()
        self.history.append({"queries":self.queries,"solved":solved,
                             "lower_pct":100*value,"upper_pct":100*float(hi.mean())})
        return bool(solved)

    def master(self):
        start=time.monotonic()
        A,b,c,integer,routers=build(self.p,self.cuts,self.active,self.k)
        candidate=None
        for presolve in (True,False):
            r=milp(-c,integrality=integer,bounds=Bounds(np.zeros(len(c)),np.ones(len(c))),
                   constraints=LinearConstraint(A,-np.inf,b),
                   options={"time_limit":self.seconds,"mip_rel_gap":0.,"presolve":presolve})
            self.master_calls+=1
            d=getattr(r,"mip_dual_bound",None)
            ub=1. if d is None or not math.isfinite(d) else min(1.,max(0.,float(-d)+PAD))
            error=None
            if r.status==2:error="master infeasible despite zero plan"
            if ub<self.best-NUM:error="master upper contradicted feasible lower"
            if r.x is not None:
                raw=r.x[:self.m];x=np.rint(raw)
                try:
                    if np.max(abs(raw-x))>1e-5:raise RuntimeError("nonintegral plan")
                    self.p.validate(x);candidate=x
                except RuntimeError as e:error=str(e)
            if error is None:
                self.upper=min(self.upper,ub);break
            self.incidents.append(error+"; presolve="+str(presolve))
        else:self.invalid=True
        self.master_time+=time.monotonic()-start
        return candidate

    def run(self,iterations):
        start=time.monotonic()
        for _ in range(iterations+1):
            if self.upper-self.best<=EPS:break
            x=self.master()
            if self.invalid or self.upper-self.best<=EPS or x is None:break
            if self.queries>=iterations:break
            key=tuple(np.flatnonzero(x))
            if key in self.cache:break
            self.query(x)
            if self.best>self.upper+NUM:
                self.invalid=True;self.incidents.append("incumbent exceeds global upper");break
        selection=time.monotonic()-start
        # Full final training replay for EVERY method, counted separately.
        # This verifies the lower certificate and permits like-for-like
        # metrics, even when a chosen plan was evaluated only approximately.
        start=time.monotonic()
        verified=[]
        for p,b in zip(self.models,self.bases):
            v,_=p.lp(self.plan,self.seconds);verified.append(v/b-1.)
        verified=np.asarray(verified)
        replay=time.monotonic()-start
        key=tuple(np.flatnonzero(self.plan));lo,hi=self.cache[key]
        if np.any(verified<lo-2*PAD) or np.any(verified>hi+2*PAD):
            self.invalid=True;self.incidents.append("final replay contradicts candidate bounds")
        if verified.mean()>self.upper+NUM:
            self.invalid=True;self.incidents.append("final replay contradicts global upper")
        U=1. if self.invalid else max(float(verified.mean()),self.upper)
        return self.plan,{"train":metrics(verified),"train_lower_pct":100*self.best,
                          "train_upper_pct":100*U,
                          "train_gap_pct_points":100*max(0.,U-float(verified.mean())),
                          "within_tolerance":bool(not self.invalid and U-verified.mean()<=EPS+NUM),
                          "certificate_withheld":self.invalid,"numerical_incidents":self.incidents,
                          "plan_queries":self.queries,"routing_lp_calls":self.calls,
                          "routing_calls_skipped":self.skipped,
                          "routing_skip_fraction":self.skipped/max(1,self.skipped+self.calls),
                          "early_rejected_queries":self.rejected,"master_calls":self.master_calls,
                          "routing_seconds":self.routing_time,"bound_seconds":self.bound_time,
                          "master_seconds":self.master_time,"selection_seconds":selection,
                          "final_replay_seconds":replay,"final_replay_calls":self.k,
                          "history":self.history}

def study(edges,cap,train,test,seconds,iterations):
    K=len(edges)//20
    p=Policy(edges,cap,K,max(1,K),np.asarray(cap)/np.median(cap),1.5*K,.25)
    start=time.monotonic()
    models,bases,G,loads=scan(edges,cap,train,K,seconds)
    setup=time.monotonic()-start
    records={};plans={}
    for mode in ("full","reuse","lazy"):
        solver=ReuseSolver(p,models,bases,G,loads,seconds,mode)
        x,r=solver.run(iterations)
        records[mode]={**r,"accounting":p.accounting(x),
                       "plan_edges":[list(edges[i]) for i in np.flatnonzero(x)],
                       "total_selection_seconds":setup+r["selection_seconds"],
                       "total_verified_seconds":setup+r["selection_seconds"]+r["final_replay_seconds"],
                       "total_routing_calls_including_scan_replay":len(train)+r["routing_lp_calls"]+r["final_replay_calls"]}
        plans[mode]=x
    best=max(r["train"]["mean_gain_pct"] for r in records.values())
    for r in records.values():
        if best>r["train_upper_pct"]+1e-4:
            r["certificate_withheld"]=True;r["within_tolerance"]=False
            r["numerical_incidents"].append("cross-method upper bound contradicted")
            r["train_upper_pct"]=100.
            r["train_gap_pct_points"]=max(0.,100.-r["train"]["mean_gain_pct"])
    if test:
        held,hb,_,_=scan(edges,cap,test,K,seconds)
        cache={}
        for name,x in plans.items():
            key=tuple(np.flatnonzero(x))
            if key not in cache:cache[key]=np.asarray([q.lp(x,seconds)[0]/b-1 for q,b in zip(held,hb)])
            records[name]["test"]=metrics(cache[key])
    return {"nodes":p.n,"links":p.m,"budget":K,"train_count":len(train),"test_count":len(test),
            "shared_scan_seconds":setup,"methods":records}

def self_test():
    # Independently enumerate plans, check both reused bounds for every
    # scenario, then verify the three complete solver modes.
    edges=[("a","b"),("b","c"),("c","d"),("d","a"),("a","c")]
    cap=[1.,2.,1.,3.,1.]
    train=[[("a","c",2.),("b","d",1.)],[("a","d",3.),("b","c",2.)]]
    for K in (1,2,3):
        p=Policy(edges,cap,K,K,np.asarray(cap),K+1.,.25)
        models,bases,G,loads=scan(edges,cap,train,K,10)
        oracle=ReuseSolver(p,models,bases,G,loads,10,"full")
        feasible=[]
        for mask in range(32):
            x=np.asarray([(mask>>i)&1 for i in range(5)],dtype=float)
            try:p.validate(x)
            except RuntimeError:continue
            v=np.asarray([q.lp(x,10)[0]/b-1 for q,b in zip(models,bases)])
            lo,hi,_=oracle.bounds(x)
            assert np.all(v>=lo-2*PAD) and np.all(v<=hi+2*PAD)
            oracle.query(x);feasible.append(float(v.mean()))
        opt=max(feasible)
        for mode in ("full","reuse","lazy"):
            _,r=ReuseSolver(p,models,bases,G,loads,10,mode).run(40)
            assert not r["certificate_withheld"] and r["within_tolerance"],r
            assert r["train_upper_pct"]>=100*opt-1e-5,r
            assert r["train"]["mean_gain_pct"]>=100*(opt-EPS)-1e-5,r
    # A one-link bottleneck and unrelated traffic: upgraded baseline flow
    # can be scaled to the dual upper bound without any new candidate solve.
    edges=[("a","b"),("b","c")];cap=[1.,1.]
    train=[[("a","b",1.)],[("b","c",1.)]]
    p=Policy(edges,cap,1,1);models,bases,G,loads=scan(edges,cap,train,1,10)
    _,r=ReuseSolver(p,models,bases,G,loads,10,"reuse").run(10)
    assert r["routing_calls_skipped"]>0 and r["within_tolerance"],r
    assert abs(r["train"]["mean_gain_pct"]-50)<1e-5,r

def zoo(args,rows,errors,skips):
    data=rb.fetch(rb.ZOO);eligible=[]
    for name,raw in sorted(rb.members(data).items()):
        try:
            g=nx.Graph(nx.read_graphml(io.BytesIO(raw)))
            g.remove_edges_from(list(nx.selfloop_edges(g)))
            g=g.subgraph(max(nx.connected_components(g),key=lambda c:(len(c),sorted(c)))).copy()
            if 20<=g.number_of_edges()<=250:eligible.append((name,raw,g))
            else:skips.append({"network":name,"reason":"outside edge screening range"})
        except Exception as e:errors.append({"network":name,"error":str(e)})
    random.Random(2026).shuffle(eligible)
    if len(eligible)<args.offset+args.limit:errors.append({"stage":"sampling","eligible":len(eligible)})
    for name,raw,g in eligible[args.offset:args.offset+args.limit]:
        try:
            nodes=sorted(g.nodes);edges=sorted(tuple(sorted(e)) for e in g.edges)
            rng=random.Random(20261130);crng=random.Random(1701)
            matrices=[[(*rng.sample(nodes,2),rng.uniform(.5,2)) for _ in range(12)] for _ in range(48)]
            cap=[float(crng.choice([1,2,4])) for _ in edges] if args.capacity=="heterogeneous" else [1.]*len(edges)
            row={"network":name,"capacity":args.capacity,"traffic":"synthetic","seed":20261130,
                 "archive_sha256":hashlib.sha256(data).hexdigest(),"graph_sha256":hashlib.sha256(raw).hexdigest(),
                 **study(edges,cap,matrices[:24],matrices[24:],args.seconds,args.iterations)}
            rows.append(row)
            print(name,{k:(v["routing_skip_fraction"],v["train_gap_pct_points"],v["total_verified_seconds"])
                        for k,v in row["methods"].items()},flush=True)
        except Exception as e:
            errors.append({"network":name,"error":str(e)});print(name,type(e).__name__,str(e),flush=True)
        rb.save(args,rows,errors,skips)
    if len(rows)!=args.limit:errors.append({"stage":"count","actual":len(rows),"expected":args.limit})

def trace(args,rows):
    edges,cap,prices,train,test,meta=historical()
    rows.append({**meta,"capacity":"historical_modeled","traffic":"historical exploratory",
                 **study(edges,cap,train,test,args.seconds,args.iterations)})
    print("Germany50",{k:(v["routing_skip_fraction"],v["train"]["mean_gain_pct"],v["test"]["mean_gain_pct"])
                       for k,v in rows[-1]["methods"].items()},flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["self-test","zoo","historical"],required=True)
    ap.add_argument("--capacity",choices=["unit","heterogeneous"],default="unit")
    ap.add_argument("--offset",type=int,default=0);ap.add_argument("--limit",type=int,default=30)
    ap.add_argument("--seconds",type=float,default=5);ap.add_argument("--iterations",type=int,default=12)
    ap.add_argument("--output",default="results/reuse.json")
    args=ap.parse_args()
    self_test()
    if args.mode=="self-test":print("Routing reuse exhaustive and skip checks passed");return
    rows=[];errors=[];skips=[]
    try:
        if args.mode=="zoo":zoo(args,rows,errors,skips)
        else:trace(args,rows)
    except Exception as e:errors.append({"error":str(e)});print(type(e).__name__,str(e),flush=True)
    rb.save(args,rows,errors,skips)
    if errors or not rows:raise SystemExit("Incomplete routing-reuse benchmark")

if __name__=="__main__":main()
