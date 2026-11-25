"""Dual-guided safe screening for multi-scenario capacity augmentation.

Compare identical multicut drivers with screening off, cardinality screening,
and cardinality plus LP-dual screening. All decisions use training only.
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
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, hstack, vstack, eye

import robust_benchmark as rb
from adaptive_benchmark import Solver, EPS, NUM, CUT_PAD
from persistent_benchmark import scan
from cost_relaxation import Policy, evaluate, metrics
from sndlib_benchmark import URL, parse

MARGIN=5e-6

def cardinality_bounds(cuts,active,K):
    """Upper bound on mean gain for each forced active edge and any plan
    containing it. Relax costs and use the K largest cut coefficients.
    """
    active=np.asarray(active,dtype=int)
    m=len(active)
    if not m or K<1:return np.zeros(m)
    out=None
    for g,b in cuts:
        g=np.maximum(g[:,active],0.)
        b=np.maximum(b,0.)+CUT_PAD
        order=np.argsort(-g,axis=1,kind="stable")
        h=min(K,m)-1
        top=order[:,:h]
        total=np.take_along_axis(g,top,axis=1).sum(axis=1)
        forced=g+total[:,None]
        if h:
            kth=np.take_along_axis(g,order[:,h:h+1],axis=1)
            ii=np.arange(g.shape[0])[:,None]
            forced[ii,top]=total[:,None]+kth
        forced=np.minimum(1.,forced+b[:,None]+MARGIN)
        out=forced if out is None else np.minimum(out,forced)
    return out.mean(axis=0)

def build(policy,cuts,active,scenario_count):
    """Physically remove inactive link columns and unused router columns."""
    active=np.asarray(active,dtype=int)
    routers=np.flatnonzero(np.any(policy.incidence[:,active]>0,axis=1))
    a=len(active);r=len(routers);k=scenario_count
    cols=np.r_[active,policy.m+routers]
    P=csr_matrix(policy.A[:,cols])
    A=[hstack([P,csr_matrix((len(policy.b),k))])]
    rhs=[policy.b]
    for g,b in cuts:
        g=np.maximum(g[:,active],0.)
        b=np.maximum(b,0.)+CUT_PAD
        A.append(hstack([-csr_matrix(g),csr_matrix((k,r)),eye(k)]))
        rhs.append(b)
    c=np.r_[np.zeros(a+r),np.full(k,1./k)]
    integer=np.r_[np.ones(a+r),np.zeros(k)]
    return vstack(A).tocsr(),np.concatenate(rhs),c,integer,routers

def dual_box_bounds(A,b,c,seconds):
    """Lagrangian box bounds using any nonnegative multiplier vector.
    For Ax<=b and 0<=x<=1, U=lambda*b+sum positive residual.
    For forcing x_j=1, subtract the negative residual magnitude.
    """
    r=linprog(-c,A_ub=A,b_ub=b,bounds=(0,1),method="highs",
              options={"time_limit":seconds})
    if not r.success:return None
    lam=np.maximum(0.,-np.asarray(r.ineqlin.marginals))
    residual=c-A.T@lam
    U=float(lam@b+np.maximum(residual,0.).sum())
    # Outward numerical padding. No exact rational certificate is claimed.
    pad=MARGIN*(1+float(np.abs(lam*b).sum())+float(np.abs(residual).sum()))
    forced=U+np.minimum(residual,0.)+pad
    return min(1.,U+pad),np.minimum(1.,forced)

class ScreenSolver:
    def __init__(self,policy,models,bases,G,seconds,mode):
        self.p=policy;self.models=models;self.bases=bases
        self.seconds=seconds;self.mode=mode;self.G=G
        # Reuse identical primal evaluation and routing-cut validation
        # from the corrected multicut benchmark; never call its master.
        self.ev=Solver(policy,models,bases,G,seconds,"multicut")
        self.active=np.arange(policy.m)
        self.removed_upper=0.
        self.active_upper=1.
        self.screen_time=0.;self.master_time=0.;self.master_calls=0
        self.lp_screen_calls=0;self.removed=[]
        self.incidents=[];self.history=[]
        self.invalid=False

    def global_upper(self):
        return max(self.active_upper,self.removed_upper)

    def screen(self):
        if self.mode=="none" or not len(self.active):return
        start=time.monotonic()
        L=self.ev.best["all"][0]
        forced=cardinality_bounds(self.ev.cuts,self.active,self.p.k)
        if self.mode=="hybrid":
            A,b,c,_,_=build(self.p,self.ev.cuts,self.active,len(self.models))
            dual=dual_box_bounds(A,b,c,self.seconds)
            self.lp_screen_calls+=1
            if dual is not None:
                u,q=dual
                forced=np.minimum(forced,q[:len(self.active)])
                self.active_upper=min(self.active_upper,u)
        # A known incumbent lying in the active region must satisfy every
        # conditional bound for edges it uses. Check before deleting.
        x=self.ev.best["all"][1]
        if not np.any(x[np.setdiff1d(np.arange(self.p.m),self.active)]>0.5):
            selected=x[self.active]>.5
            if np.any(forced[selected]<L-NUM) or self.active_upper<L-NUM:
                self.invalid=True
                self.incidents.append("screening bound contradicted active incumbent")
                self.screen_time+=time.monotonic()-start
                return
        # Every discarded plan contains at least one discarded edge.
        # Record its bound so global certificates include discarded regions.
        remove=(forced<=L+EPS-MARGIN)
        if np.any(remove):
            self.removed_upper=max(self.removed_upper,float(np.max(forced[remove])))
            self.removed.extend({"edge":int(e),"conditional_upper_pct":float(100*u),
                                 "incumbent_pct":float(100*L)}
                                for e,u in zip(self.active[remove],forced[remove]))
            self.active=self.active[~remove]
        if not len(self.active):self.active_upper=0.
        self.screen_time+=time.monotonic()-start

    def master(self):
        if not len(self.active):return None
        start=time.monotonic()
        A,b,c,integer,routers=build(self.p,self.ev.cuts,self.active,len(self.models))
        a=len(self.active)
        x=None
        for presolve in (True,False):
            r=milp(-c,integrality=integer,bounds=Bounds(np.zeros(len(c)),np.ones(len(c))),
                   constraints=LinearConstraint(A,-np.inf,b),
                   options={"time_limit":self.seconds,"mip_rel_gap":0.,"presolve":presolve})
            self.master_calls+=1
            d=getattr(r,"mip_dual_bound",None)
            ub=1. if d is None or not math.isfinite(d) else min(1.,max(0.,float(-d)+MARGIN))
            problem=None
            if r.status==2:problem="infeasible master despite feasible zero plan"
            candidate=None
            if r.x is not None:
                raw=r.x[:a]
                candidate=np.zeros(self.p.m);candidate[self.active]=np.rint(raw)
                try:
                    if np.max(abs(raw-np.rint(raw)))>1e-5:raise RuntimeError("nonintegral plan")
                    self.p.validate(candidate)
                except RuntimeError as e:
                    problem=str(e);candidate=None
            # Check known evaluated plans still in reduced feasible space.
            for key,(values,_) in self.ev.cache.items():
                if set(key).issubset(set(self.active)) and values.mean()>ub+NUM:
                    problem="master upper bound contradicted known active plan"
                    break
            if problem is None:
                self.active_upper=min(self.active_upper,ub);x=candidate;break
            self.incidents.append(problem+"; presolve="+str(presolve))
        else:
            self.invalid=True
        self.master_time+=time.monotonic()-start
        return x

    def run(self,iterations):
        start=time.monotonic()
        for _ in range(iterations+1):
            self.screen()
            L=self.ev.best["all"][0]
            if self.invalid or self.global_upper()-L<=EPS:break
            if len(self.ev.cache)-1>=iterations:break
            x=self.master()
            if self.invalid or self.global_upper()-L<=EPS or x is None:break
            if tuple(np.flatnonzero(x)) in self.ev.cache:break
            self.ev.evaluate(x)
            value=self.ev.best["all"][0]
            self.history.append({"active_edges":len(self.active),
                                 "incumbent_pct":100*value,"global_upper_pct":100*self.global_upper()})
            if value>self.global_upper()+NUM:
                self.invalid=True;self.incidents.append("global bound contradicted by evaluated plan");break
        L,x=self.ev.best["all"]
        if L>self.global_upper()+NUM:
            self.invalid=True;self.incidents.append("final global bound contradicted by incumbent")
        U=1. if self.invalid or self.ev.untrusted else max(L,self.global_upper())
        invalid=bool(self.invalid or self.ev.untrusted)
        return x,{"train_lower_pct":100*L,"train_upper_pct":100*U,
                  "train_gap_pct_points":100*max(0.,U-L),
                  "within_tolerance":bool(not invalid and U-L<=EPS+NUM),
                  "certificate_withheld":invalid,"numerical_incidents":self.incidents+self.ev.incidents,
                  "initial_edges":self.p.m,"remaining_edges":len(self.active),
                  "fraction_edges_removed":1-len(self.active)/self.p.m,
                  "discarded_region_upper_pct":100*self.removed_upper,
                  "removed_edges":self.removed,"history":self.history,
                  "routing_lp_calls":self.ev.calls,"master_calls":self.master_calls,
                  "screening_lp_calls":self.lp_screen_calls,
                  "screening_seconds":self.screen_time,"master_seconds":self.master_time,
                  "selection_seconds":time.monotonic()-start}

def study(edges,cap,train,test,seconds,iterations):
    K=len(edges)//20
    p=Policy(edges,cap,K,max(1,K),np.asarray(cap)/np.median(cap),1.5*K,.25)
    start=time.monotonic()
    models,bases,G=scan(edges,cap,train,K,seconds)
    setup=time.monotonic()-start
    records={};plans={}
    for name,mode in (("multicut","none"),("cardinality_screen","cardinality"),("hybrid_screen","hybrid")):
        solver=ScreenSolver(p,models,bases,G,seconds,mode)
        x,r=solver.run(iterations)
        v,_=solver.ev.cache[tuple(np.flatnonzero(x))]
        records[name]={**r,"train":metrics(v),"accounting":p.accounting(x),
                       "plan_edges":[list(edges[i]) for i in np.flatnonzero(x)],
                       "total_selection_seconds":setup+r["selection_seconds"]}
        plans[name]=x
    best=max(r["train_lower_pct"] for r in records.values())
    for r in records.values():
        if best>r["train_upper_pct"]+1e-4:
            r["certificate_withheld"]=True;r["within_tolerance"]=False
            r["numerical_incidents"].append("cross-method global bound contradicted")
            r["train_upper_pct"]=100.
            r["train_gap_pct_points"]=max(0.,100-r["train_lower_pct"])
    if test:
        held,hb,H=scan(edges,cap,test,K,seconds);cache={}
        for name,x in plans.items():
            v,_=evaluate(held,hb,x,seconds,cache)
            if np.any(v>H@x+NUM):raise RuntimeError("Holdout bound violated")
            records[name]["test"]=metrics(v)
    return {"nodes":p.n,"links":p.m,"budget":K,"train_count":len(train),"test_count":len(test),
            "shared_scan_seconds":setup,"methods":records}

def self_test():
    # Verify forced-edge bounds independently against all feasible subsets.
    edges=[("a","b"),("b","c"),("c","d"),("d","a"),("a","c")]
    cap=[1.,2.,1.,3.,1.]
    train=[[("a","c",2.),("b","d",1.)],[("a","d",3.),("b","c",2.)]]
    for K in (1,2,3):
        p=Policy(edges,cap,K,K,np.asarray(cap),float(K)+1.,.25)
        models,bases,G=scan(edges,cap,train,K,10)
        oracle=Solver(p,models,bases,G,10,"multicut")
        feasible=[]
        for mask in range(32):
            x=np.array([(mask>>i)&1 for i in range(5)],dtype=float)
            try:p.validate(x)
            except RuntimeError:continue
            oracle.evaluate(x)
            v,_=oracle.cache[tuple(np.flatnonzero(x))]
            feasible.append((x,float(v.mean())))
        optimum=max(v for _,v in feasible)
        for active in (np.arange(5),np.array([0,2,3]),np.array([1])):
            f=cardinality_bounds(oracle.cuts,active,K)
            A,b,c,_,_=build(p,oracle.cuts,active,len(train))
            d=dual_box_bounds(A,b,c,10);assert d is not None
            U,q=d
            for x,v in feasible:
                if not set(np.flatnonzero(x)).issubset(set(active)):continue
                assert v<=U+1e-6
                for j,e in enumerate(active):
                    if x[e]>.5:assert v<=min(f[j],q[j])+1e-6,(K,e,v,f[j],q[j])
        for mode in ("none","cardinality","hybrid"):
            x,r=ScreenSolver(p,models,bases,G,10,mode).run(40)
            assert not r["certificate_withheld"],r
            assert r["train_lower_pct"]<=100*optimum+1e-5
            assert r["train_upper_pct"]>=100*optimum-1e-5
            assert r["within_tolerance"],r
    # One active bottleneck plus irrelevant leaf edges. Screening must
    # actually remove at least one edge and still certify the optimum.
    edges=[("a","b"),("b","c"),("b","d"),("b","e")]
    cap=[1.,10.,10.,10.];train=[[("a","b",1.)]]
    p=Policy(edges,cap,1,1);models,bases,G=scan(edges,cap,train,1,10)
    _,r=ScreenSolver(p,models,bases,G,10,"hybrid").run(10)
    assert r["fraction_edges_removed"]>0 and r["within_tolerance"] and abs(r["train_lower_pct"]-100)<1e-6,r

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
            rng=random.Random(20261129);crng=random.Random(1701)
            matrices=[[(*rng.sample(nodes,2),rng.uniform(.5,2)) for _ in range(12)] for _ in range(36)]
            cap=[float(crng.choice([1,2,4])) for _ in edges] if args.capacity=="heterogeneous" else [1.]*len(edges)
            row={"network":name,"capacity":args.capacity,"traffic":"synthetic","seed":20261129,
                 "archive_sha256":hashlib.sha256(data).hexdigest(),"graph_sha256":hashlib.sha256(raw).hexdigest(),
                 **study(edges,cap,matrices[:12],matrices[12:],args.seconds,args.iterations)}
            rows.append(row)
            print(name,{k:(v["fraction_edges_removed"],v["train_gap_pct_points"],v["selection_seconds"])
                        for k,v in row["methods"].items()},flush=True)
        except Exception as e:
            errors.append({"network":name,"error":str(e)})
            print(name,type(e).__name__,str(e),flush=True)
        rb.save(args,rows,errors,skips)
    if len(rows)!=args.limit:errors.append({"stage":"count","actual":len(rows),"expected":args.limit})

def published(args,rows,errors,skips):
    data=rb.fetch(URL)
    for name,raw in sorted(rb.members(data).items()):
        if not name.endswith(".xml"):continue
        label=Path(name).stem
        try:edges,cap,dem,meta=parse(raw,label,"module")
        except ValueError as e:
            skips.append({"network":label,"reason":str(e)});continue
        if not 20<=len(edges)<=120 or len(dem)>1000:
            skips.append({"network":label,"reason":"outside size range"});continue
        try:
            regime="published_preinstalled" if meta["assumed_module_links"]==0 else "hypothetical_module"
            rows.append({**meta,"network":label,"capacity":regime,"traffic":"published static",
                         "archive_sha256":hashlib.sha256(data).hexdigest(),
                         **study(edges,cap,[dem],[],args.seconds,args.iterations)})
            print(label,rows[-1]["methods"]["hybrid_screen"]["fraction_edges_removed"],flush=True)
        except Exception as e:
            errors.append({"network":label,"error":str(e)})
            print(label,type(e).__name__,str(e),flush=True)
        rb.save(args,rows,errors,skips)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["self-test","zoo","published"],required=True)
    ap.add_argument("--capacity",choices=["unit","heterogeneous"],default="unit")
    ap.add_argument("--offset",type=int,default=0);ap.add_argument("--limit",type=int,default=30)
    ap.add_argument("--seconds",type=float,default=5);ap.add_argument("--iterations",type=int,default=12)
    ap.add_argument("--output",default="results/screening.json")
    args=ap.parse_args()
    self_test()
    if args.mode=="self-test":
        print("Conditional bounds and exhaustive enumeration checks passed");return
    rows=[];errors=[];skips=[]
    try:
        (zoo if args.mode=="zoo" else published)(args,rows,errors,skips)
    except Exception as e:
        errors.append({"stage":"setup","error":str(e)});print(type(e).__name__,str(e),flush=True)
    rb.save(args,rows,errors,skips)
    if errors or not rows:raise SystemExit("Incomplete screening benchmark")

if __name__=="__main__":main()
