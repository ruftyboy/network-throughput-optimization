"""Certificate-guided partition of network augmentation plans.

All methods use the same source-aggregated routing model, budget and baseline
duals. Compare standard aggregate/multicut masters and adaptive partition.
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
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, hstack, vstack, lil_matrix

import robust_benchmark as rb
from persistent_benchmark import scan
from cost_relaxation import Policy, metrics
from sndlib_benchmark import parse, URL

EPS=0.005  # 0.5 percentage point training objective tolerance
NUM=1e-6
CUT_PAD=2e-6

class Solver:
    def __init__(self,policy,models,bases,G,seconds,kind):
        self.p=policy;self.models=models;self.bases=bases;self.G=G
        self.seconds=seconds;self.kind=kind
        self.m=policy.m;self.k=len(models)
        self.cuts=[(np.asarray(G),np.zeros(self.k))]
        self.zero=np.zeros(self.m)
        self.cache={(): (np.zeros(self.k),np.asarray(G))}
        self.best={"all":(0.,self.zero.copy()),"matching":(0.,self.zero.copy()),
                   "concentrated":(None,None)}
        self.calls=0;self.master_calls=0;self.master_seconds=0.
        self.upper={"all":1.,"matching":1.,"concentrated":1.}
        self.empty=set()
        self.incidents=[]
        self.untrusted=set()
        # All implemented policy restrictions are downward closed. A
        # concentrated feasible plan exists iff some adjacent pair is feasible.
        self.witness={"all":self.zero.copy(),"matching":self.zero.copy()}
        if policy.k>=2:
            for row in policy.incidence:
                ids=np.flatnonzero(row)
                for i,e in enumerate(ids):
                    for f in ids[i+1:]:
                        x=np.zeros(self.m);x[[e,f]]=1.
                        try: policy.validate(x)
                        except RuntimeError: continue
                        self.witness["concentrated"]=x
                        break
                    if "concentrated" in self.witness:break
                if "concentrated" in self.witness:break
        if "concentrated" not in self.witness:
            self.upper["concentrated"]=0.;self.empty.add("concentrated")

    def invalidate(self,label,reason):
        # Never silently replace a contradictory numerical bound by the
        # incumbent. Fall back to the physical doubling bound, and withhold
        # optimality/classification certificates for this solver.
        self.incidents.append({"class":label,"reason":reason})
        self.untrusted.add(label)
        self.upper[label]=1.

    def classify(self,x):
        return "concentrated" if np.max(self.p.incidence@x)>1.5 else "matching"

    def evaluate(self,x):
        self.p.validate(x)
        key=tuple(np.flatnonzero(x))
        if key in self.cache:return
        values=[];grad=[]
        for p,b in zip(self.models,self.bases):
            rate,g=p.lp(x,self.seconds)
            values.append(rate/b-1.)
            grad.append(g/b)
            self.calls+=1
        v=np.asarray(values);g=np.asarray(grad)
        if np.any(v>self.G@x+NUM) or min(v)<-NUM or max(v)>1+NUM:
            raise RuntimeError("Invalid routing gain or baseline bound")
        for a,b in self.cuts:
            if np.any(v>np.maximum(a,0.)@x+np.maximum(b,0.)+CUT_PAD+NUM):
                # A violated supporting cut calls the routing certificate into
                # question. Fail explicitly rather than conceal that problem.
                raise RuntimeError("Dual cut violation after numerical padding")
        intercept=v-g@x
        for oldkey,(oldv,_) in self.cache.items():
            ox=np.zeros(self.m);ox[list(oldkey)]=1.
            if np.any(oldv>np.maximum(g,0.)@ox+np.maximum(intercept,0.)+CUT_PAD+NUM):
                raise RuntimeError("New dual cut contradicts prior evaluated plan")
        self.cache[key]=(v,g)
        self.cuts.append((g,intercept))
        val=float(v.mean())
        for label in ("all",self.classify(x)):
            if self.best[label][0] is None or val>self.best[label][0]+1e-9:
                self.best[label]=(val,x.copy())
            if val>self.upper[label]+NUM:
                self.invalidate(label,"new feasible value exceeds previously reported master bound")

    def master(self,label):
        if label in self.empty:return None,0.
        start=time.monotonic();p=self.p;m=p.m;n=p.n
        k=1 if self.kind=="aggregate" else self.k
        w=n if label=="concentrated" else 0
        N=m+n+k+w
        # x = upgrades, z = router activation, y = gain envelope,
        # w = witness for a vertex with at least two upgraded incident links.
        A=[hstack([csr_matrix(p.A[:,:m+n]),csr_matrix((len(p.b),k+w))])]
        rhs=list(p.b)
        if label=="matching":
            A.append(hstack([csr_matrix(p.incidence),csr_matrix((n,n+k+w))]))
            rhs.extend([1.]*n)
        if label=="concentrated":
            M=lil_matrix((n+1,N))
            M[:n,:m]=-p.incidence
            for i in range(n):M[i,m+n+k+i]=2.
            M[n,m+n+k:]=-1.
            A.append(M.tocsr());rhs.extend([0.]*n+[-1.])
        for g,b in self.cuts:
            # Gains are nonnegative and x>=0. Raising tiny negative slopes
            # and intercepts to zero can only relax (not strengthen) a cut.
            # Add explicit outward rounding margin for numerical LP duals.
            g=np.maximum(g,0.);b=np.maximum(b,0.)+CUT_PAD
            if self.kind=="aggregate":
                g=g.mean(axis=0)[None,:];b=np.array([b.mean()])
            from scipy.sparse import eye
            A.append(hstack([-csr_matrix(g),csr_matrix((k,n)),eye(k),csr_matrix((k,w))]))
            rhs.extend(b.tolist())
        c=np.r_[np.zeros(m+n),np.full(k,-1./k),np.zeros(w)]
        constraints=LinearConstraint(vstack(A).tocsr(),-np.inf,np.asarray(rhs))
        known=self.best[label][0]
        x=None
        for presolve in (True,False):
            r=milp(c,integrality=np.r_[np.ones(m+n),np.zeros(k),np.ones(w)],
                   bounds=Bounds(np.zeros(N),np.ones(N)),constraints=constraints,
                   options={"time_limit":self.seconds,"mip_rel_gap":0.,"presolve":presolve})
            self.master_calls+=1
            d=getattr(r,"mip_dual_bound",None)
            ub=1. if d is None or not math.isfinite(d) else min(1.,max(0.,float(-d)+CUT_PAD))
            problem=None
            if r.status==2:
                problem="master infeasible despite independently feasible class witness"
            elif known is not None and known>ub+NUM:
                problem="known lower bound exceeds reported master bound"
            candidate=None
            if r.x is not None:
                raw=r.x[:m];candidate=np.rint(raw)
                try:
                    if np.max(abs(raw-candidate))>1e-5:raise RuntimeError("nonintegral incumbent")
                    p.validate(candidate)
                    if label!="all" and self.classify(candidate)!=label:raise RuntimeError("wrong partition")
                except RuntimeError as e:
                    problem=str(e);candidate=None
            if candidate is not None:x=candidate
            if problem is None:
                if label not in self.untrusted:
                    self.upper[label]=min(self.upper[label],ub)
                break
            self.incidents.append({"class":label,"reason":problem,"presolve":presolve})
        else:
            self.invalidate(label,"inconsistent master after presolve-disabled retry")
        self.master_seconds+=time.monotonic()-start
        if x is None and label in self.untrusted:
            x=self.witness[label].copy()
        return x,self.upper[label]

    def run(self,adaptive,iterations):
        start=time.monotonic()
        if adaptive:
            # Try one matching master first: it supplies a feasible incumbent,
            # not a proof of matching optimality.
            x,_=self.master("matching")
            if x is not None:self.evaluate(x)
        for _ in range(iterations):
            if len(self.cache)-1>=iterations:break
            if adaptive:
                candidates={}
                for label in ("matching","concentrated"):
                    if self.upper[label]<=self.best["all"][0]+EPS:continue
                    x,u=self.master(label)
                    if x is not None and u>self.best["all"][0]+EPS:
                        candidates[label]=(x,u)
                upper=max(self.upper["matching"],self.upper["concentrated"])
                if upper-self.best["all"][0]<=EPS or not candidates:break
                label=max(candidates,key=lambda q:(candidates[q][1],q=="matching"))
                x=candidates[label][0]
            else:
                x,upper=self.master("all")
                if upper-self.best["all"][0]<=EPS or x is None:break
            if tuple(np.flatnonzero(x)) in self.cache:
                # Bound is still valid if a time-limited master repeats a plan.
                break
            self.evaluate(x)
        value,x=self.best["all"]
        upper=max(self.upper["matching"],self.upper["concentrated"]) if adaptive else self.upper["all"]
        if value>upper+NUM:
            self.invalidate("all","global certificate contradicted by feasible incumbent")
            upper=1.
        if self.untrusted:upper=1.
        match=self.best["matching"][0];conc=self.best["concentrated"][0]
        classification="not_assessed"
        if adaptive and self.untrusted:
            classification="unresolved_numerical"
        elif adaptive:
            if upper-match<=EPS:
                classification="matching_sufficient_within_tolerance"
            elif conc is not None and conc>self.upper["matching"]+EPS:
                classification="concentration_advantage_certified"
            else:classification="unresolved"
        return x,{"train_lower_pct":100*value,"train_upper_pct":100*max(value,upper),
                 "train_gap_pct_points":100*max(0.,upper-value),
                 "within_tolerance":bool(not self.untrusted and upper-value<=EPS+NUM),
                 "numerical_incidents":self.incidents,"certificate_withheld":bool(self.untrusted),
                 "classification":classification,"matching_lower_pct":100*match,
                 "matching_upper_pct":100*self.upper["matching"] if adaptive else None,
                 "concentrated_lower_pct":None if conc is None else 100*conc,
                 "concentrated_upper_pct":100*self.upper["concentrated"] if adaptive else None,
                 "concentrated_empty":bool("concentrated" in self.empty),
                 "distinct_nonzero_plan_evaluations":len(self.cache)-1,
                 "routing_lp_calls":self.calls,"master_calls":self.master_calls,
                 "master_seconds":self.master_seconds,"selection_seconds":time.monotonic()-start}

def study(edges,cap,train,test,seconds,iterations):
    K=len(edges)//20
    costs=np.asarray(cap)/np.median(cap)
    policy=Policy(edges,cap,K,max(1,K),costs,1.5*K,.25)
    start=time.monotonic()
    models,bases,G=scan(edges,cap,train,K,seconds)
    setup=time.monotonic()-start
    records={};plans={};values={}
    # Randomize nothing: identical fixed baseline and budgets for all methods.
    # Each has a private plan cache; none gets the other's evaluated cuts.
    for name,kind,adapt in (("aggregate_direct","aggregate",False),
                            ("multicut_direct","multicut",False),
                            ("adaptive_partition","multicut",True)):
        solver=Solver(policy,models,bases,G,seconds,kind)
        x,r=solver.run(adapt,iterations)
        plans[name]=x
        v,_=solver.cache[tuple(np.flatnonzero(x))]
        values[name]=float(100*v.mean())
        records[name]={**r,"plan_edges":[list(edges[i]) for i in np.flatnonzero(x)],
                       "accounting":policy.accounting(x),"train":metrics(v),
                       "total_selection_seconds_including_scan":setup+r["selection_seconds"]}
        del solver
    for r in records.values():
        if max(values.values())>r["train_upper_pct"]+1e-4:
            # A different independently evaluated placement disproves the
            # reported upper bound. Retain feasible gains, but no certificate.
            r["numerical_incidents"].append({"class":"all","reason":"cross-method upper bound contradicted"})
            r["certificate_withheld"]=True
            r["train_upper_pct"]=100.
            r["train_gap_pct_points"]=max(0.,100.-r["train_lower_pct"])
            r["within_tolerance"]=False
            if r["classification"]!="not_assessed":
                r["classification"]="unresolved_numerical"
                r["matching_upper_pct"]=100.
                r["concentrated_upper_pct"]=100.
    # Test only after all plans have been frozen. No tuning on test.
    if test:
        held,hb,H=scan(edges,cap,test,K,seconds);cache={}
        from cost_relaxation import evaluate
        for name,x in plans.items():
            v,_=evaluate(held,hb,x,seconds,cache)
            if np.any(v>H@x+NUM):raise RuntimeError("Test bound violation")
            records[name]["test"]=metrics(v)
    return {"nodes":policy.n,"links":policy.m,"budget":K,"train_count":len(train),
            "test_count":len(test),"shared_training_scan_seconds":setup,"methods":records}

def self_test():
    # Two adjacent serial bottlenecks: matching cannot improve, unrestricted doubles.
    edges=[("a","b"),("b","c")];cap=[1.,1.]
    train=[[("a","c",1.)]]
    p=Policy(edges,cap,2,2)
    models,b,G=scan(edges,cap,train,2,10)
    for kind,adaptive in (("aggregate",False),("multicut",False),("multicut",True)):
        x,r=Solver(p,models,b,G,10,kind).run(adaptive,12)
        assert abs(r["train_lower_pct"]-100)<1e-6 and r["within_tolerance"],r
        if adaptive:assert r["classification"]=="concentration_advantage_certified",r
    # With one upgrade the concentrated class is empty.
    p=Policy(edges,cap,1,1)
    x,r=Solver(p,models,b,G,10,"multicut").run(True,12)
    assert r["concentrated_empty"] and r["classification"]=="matching_sufficient_within_tolerance",r
    # Enumerate every feasible subset in a cyclic graph, independently verify
    # partition certificates and multicut bounds.
    edges=[("a","b"),("b","c"),("c","d"),("d","a"),("a","c")]
    cap=[1.,2.,1.,3.,1.];train=[[("a","c",2.),("b","d",1.)],[("a","d",3.),("b","c",2.)]]
    p=Policy(edges,cap,2,2,np.asarray(cap),4.,.25)
    models,b,G=scan(edges,cap,train,2,10)
    exact={"matching":0.,"concentrated":None}
    for mask in range(32):
        x=np.array([(mask>>i)&1 for i in range(5)],dtype=float)
        try:p.validate(x)
        except RuntimeError:continue
        label="matching" if np.max(p.incidence@x)<1.5 else "concentrated"
        v=float(np.mean([q.lp(x,10)[0]/z-1 for q,z in zip(models,b)]))
        exact[label]=v if exact[label] is None else max(exact[label],v)
    opt=max(v for v in exact.values() if v is not None)
    for kind,adaptive in (("aggregate",False),("multicut",False),("multicut",True)):
        _,r=Solver(p,models,b,G,10,kind).run(adaptive,30)
        assert r["train_lower_pct"]<=100*opt+1e-5<=r["train_upper_pct"]+2e-5,r
        assert r["train_gap_pct_points"]<=100*EPS+1e-5,r
        if adaptive:
            assert r["matching_upper_pct"]>=100*exact["matching"]-1e-5
            if exact["concentrated"] is not None:
                assert r["concentrated_upper_pct"]>=100*exact["concentrated"]-1e-5

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
    if len(eligible)<args.offset+args.limit:
        errors.append({"stage":"sample","eligible":len(eligible)})
    for name,raw,g in eligible[args.offset:args.offset+args.limit]:
        try:
            nodes=sorted(g.nodes);edges=sorted(tuple(sorted(e)) for e in g.edges)
            rng=random.Random(20261128);crng=random.Random(1701)
            matrices=[[(*rng.sample(nodes,2),rng.uniform(.5,2)) for _ in range(12)] for _ in range(36)]
            cap=[float(crng.choice([1,2,4])) for _ in edges] if args.capacity=="heterogeneous" else [1.]*len(edges)
            row={"network":name,"capacity":args.capacity,"traffic":"synthetic","seed":20261128,
                 "archive_sha256":hashlib.sha256(data).hexdigest(),"graph_sha256":hashlib.sha256(raw).hexdigest(),
                 **study(edges,cap,matrices[:12],matrices[12:],args.seconds,args.iterations)}
            rows.append(row)
            print(name,{k:(v["train_gap_pct_points"],v["selection_seconds"],v["classification"])
                        for k,v in row["methods"].items()},flush=True)
        except Exception as e:
            errors.append({"network":name,"error":str(e)})
            print(name,type(e).__name__,str(e),flush=True)
        rb.save(args,rows,errors,skips)
    if len(rows)!=args.limit:errors.append({"stage":"count","actual":len(rows),"expected":args.limit})

def published(args,rows,errors,skips):
    # No fabricated test matrices: these are static published benchmark demands.
    # Screen all files; report unsupported or oversized cases separately.
    data=rb.fetch(URL)
    for name,raw in sorted(rb.members(data).items()):
        if not name.endswith(".xml"):continue
        label=Path(name).stem
        try:edges,cap,dem,meta=parse(raw,label,"module")
        except ValueError as e:
            skips.append({"network":label,"reason":str(e)});continue
        if not 20<=len(edges)<=120 or len(dem)>1000:
            skips.append({"network":label,"reason":"outside prespecified size range"});continue
        regime="published_preinstalled" if meta["assumed_module_links"]==0 else "hypothetical_module"
        try:
            row={**meta,"network":label,"capacity":regime,"traffic":"published static demands",
                 "archive_sha256":hashlib.sha256(data).hexdigest(),
                 **study(edges,cap,[dem],[],args.seconds,args.iterations)}
            rows.append(row)
            print(label,regime,{k:v["train_lower_pct"] for k,v in row["methods"].items()},flush=True)
        except Exception as e:
            errors.append({"network":label,"error":str(e)})
            print(label,type(e).__name__,str(e),flush=True)
        rb.save(args,rows,errors,skips)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["self-test","zoo","published"],required=True)
    ap.add_argument("--capacity",choices=["unit","heterogeneous"],default="unit")
    ap.add_argument("--offset",type=int,default=0)
    ap.add_argument("--limit",type=int,default=30)
    ap.add_argument("--seconds",type=float,default=5)
    ap.add_argument("--iterations",type=int,default=12)
    ap.add_argument("--output",default="results/adaptive.json")
    args=ap.parse_args()
    self_test()
    if args.mode=="self-test":
        print("Partition enumeration and bound checks passed");return
    rows=[];errors=[];skips=[]
    try:
        if args.mode=="zoo":zoo(args,rows,errors,skips)
        else:published(args,rows,errors,skips)
    except Exception as e:
        errors.append({"stage":"setup","error":str(e)});print(type(e).__name__,str(e),flush=True)
    rb.save(args,rows,errors,skips)
    if errors or not rows:raise SystemExit("Incomplete adaptive experiment")

if __name__=="__main__":main()
