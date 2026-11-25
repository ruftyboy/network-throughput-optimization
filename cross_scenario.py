"""Cross-scenario primal path templates and shortest-path dual certificates.

Numerical bounds, not rational certificates. All selection uses training only.
"""
import argparse, hashlib, io, math, random, time
from pathlib import Path
from collections import deque
import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import robust_benchmark as rb
import routing_reuse as old
from persistent_benchmark import SourceProblem
from cost_relaxation import Policy, metrics, historical
from adaptive_benchmark import EPS, NUM

PAD=old.PAD
BASE_TOL=EPS/20
LOCAL=old.LOCAL

class Transfer:
    def __init__(self,edges,cap,matrices,K,seconds):
        self.edges=edges;self.cap=np.asarray(cap,dtype=float)
        self.seconds=seconds;self.m=len(edges);self.k=len(matrices)
        self.nodes=sorted({v for e in edges for v in e})
        self.n=len(self.nodes);idx={v:i for i,v in enumerate(self.nodes)}
        self.ends=np.asarray([(idx[u],idx[v]) for u,v in edges])
        self.adj=[[] for _ in self.nodes]
        for e,(u,v) in enumerate(self.ends):
            self.adj[u].append((v,2*e));self.adj[v].append((u,2*e+1))
        ds=[]
        for matrix in matrices:
            d={}
            for s,t,w in matrix:
                if w>0 and s!=t:
                    key=(idx[s],idx[t]);d[key]=d.get(key,0.)+w
            ds.append(d)
        self.pairs=sorted(set().union(*(set(d) for d in ds)))
        self.pi={pair:i for i,pair in enumerate(self.pairs)}
        self.D=np.asarray([[d.get(pair,0.) for pair in self.pairs] for d in ds])
        self.groups=[]
        for d in ds:
            by={}
            for (s,t),w in d.items():by.setdefault(s,{})[t]=w
            self.groups.append(by)
        self.models=[SourceProblem(edges,cap,d,K) for d in matrices]
        # A shortest-hop route exists for every union demand pair. Use it
        # whenever a solved flow does not cover a target pair or residual.
        self.default=[]
        for s,t in self.pairs:
            path=self.path(s,t,None)
            if path is None:raise RuntimeError("disconnected demand")
            self.default.append([a//2 for a in path])
        self.primal=[];self.dual=[]
        self.calls=0;self.lp_seconds=0.;self.transfer_seconds=0.
        self.fallback_pairs=0;self.template_pairs=0
        self.add_template([{e:1. for e in path} for path in self.default])

    def path(self,s,t,res):
        parent={s:None};q=deque([s])
        while q and t not in parent:
            u=q.popleft()
            for v,a in self.adj[u]:
                if v not in parent and (res is None or res[a]>1e-14):
                    parent[v]=(u,a);q.append(v)
        if t not in parent:return None
        out=[];u=t
        while u!=s:
            u,a=parent[u];out.append(a)
        return out[::-1]

    def add_template(self,rows):
        rr=[];cc=[];vv=[]
        for i,weights in enumerate(rows):
            for e,w in weights.items():
                rr.append(i);cc.append(e);vv.append(w)
        T=csr_matrix((vv,(rr,cc)),shape=(len(rows),self.m))
        # Per-unit-throughput link loads for every target demand matrix.
        loads=np.asarray((T.T@self.D.T).T)/self.cap
        self.primal.append(loads)

    def template(self,s,rate,flows):
        groups=self.groups[s]
        scale=max(float(self.cap.max()),max(sum(v.values()) for v in groups.values()))
        rows=[{e:1. for e in path} for path in self.default]
        for j,(u,sinks) in enumerate(sorted(groups.items())):
            residual=np.maximum(flows[j*2*self.m:(j+1)*2*self.m],0.).copy()
            for v,w in sorted(sinks.items()):
                amount=rate*w/scale
                if amount<=0:continue
                remaining=amount;weights={};i=self.pi[(u,v)]
                # Every extracted path is a genuine graph path. Any missing
                # flow is replaced by a default path; normalize by the full
                # requested amount. Thus the template is feasible even when
                # the solver's primal has small conservation residuals.
                for _ in range(2*self.m+1):
                    if remaining<=0:break
                    path=self.path(u,v,residual)
                    if path is None:break
                    take=min(remaining,min(residual[a] for a in path))
                    if take<=0:break
                    for a in path:
                        residual[a]-=take;e=a//2
                        weights[e]=weights.get(e,0.)+take/amount
                    remaining=max(0.,remaining-take)
                if remaining>0:
                    if remaining/amount>1e-6:self.fallback_pairs+=1
                    for e in self.default[i]:
                        weights[e]=weights.get(e,0.)+remaining/amount
                self.template_pairs+=1
                rows[i]=weights
        self.add_template(rows)

    def prices(self,g):
        # g_e = capacity price times capacity increment. Divide by original
        # physical capacity to obtain nonnegative edge lengths up to scale.
        length=np.maximum(g,0.)/self.cap
        mx=float(length.max())
        if mx<=0:raise RuntimeError("zero dual length")
        length/=mx
        arcs={}
        for (u,v),w in zip(self.ends,length):
            for pair in ((u,v),(v,u)):
                arcs[pair]=min(arcs.get(pair,math.inf),float(w))
        ij=list(arcs)
        A=csr_matrix(([arcs[a] for a in ij],([a[0] for a in ij],[a[1] for a in ij])),shape=(self.n,self.n))
        # Explicit zero-weight sparse entries are valid edges.
        dist=dijkstra(A,directed=True)
        q=np.asarray([dist[u,v] for u,v in self.pairs])
        if not np.all(np.isfinite(q)):raise RuntimeError("disconnected dual graph")
        den=self.D@q
        num=self.cap*length
        slope=np.full((self.k,self.m),np.inf)
        good=den>0
        slope[good]=num[None,:]/den[good,None]*(1+PAD)
        self.dual.append(slope)

    def add_solve(self,s,x):
        p=self.models[s];b=p.cap*(1+x)
        start=time.monotonic()
        r=linprog(p.obj,A_eq=p.eq,b_eq=np.zeros(p.q),A_ub=p.ub,b_ub=b,
                  bounds=(0,None),method="highs",options={"time_limit":self.seconds})
        self.calls+=1;self.lp_seconds+=time.monotonic()-start
        if not r.success:raise RuntimeError("routing LP: "+r.message)
        if max(np.max(abs(p.eq@r.x)),np.max(p.ub@r.x-b),-np.min(r.x))>1e-6:
            raise RuntimeError("routing residual")
        rate=float(r.x[-1]);g=-np.asarray(r.ineqlin.marginals)*p.cap
        if rate<=0 or min(g)<-1e-7:raise RuntimeError("invalid LP certificate")
        start=time.monotonic()
        self.template(s,rate,r.x[:-1])
        self.prices(g)
        lo,hi,_=self.latest(x)
        if rate<lo[s]*(1-2*PAD) or rate>hi[s]*(1+2*PAD):
            raise RuntimeError("transferred anchor certificate inconsistent")
        self.transfer_seconds+=time.monotonic()-start
        return rate

    def latest(self,x):
        loads=self.primal[-1]
        factor=np.max(loads/(1+x),axis=1)
        if np.any(factor<=0):raise RuntimeError("nonpositive route load")
        lo=(1-PAD)/factor
        if not self.dual:return lo,np.full(self.k,np.inf),None
        slope=self.dual[-1]
        good=np.all(np.isfinite(slope),axis=1)
        hi=np.full(self.k,np.inf);hi[good]=slope[good]@(1+x)
        return lo,hi,slope

    def bounds(self,x):
        lo=np.zeros(self.k);hi=np.full(self.k,np.inf)
        best=np.full((self.k,self.m),np.inf)
        for loads in self.primal:
            lo=np.maximum(lo,(1-PAD)/np.max(loads/(1+x),axis=1))
        for slope in self.dual:
            good=np.all(np.isfinite(slope),axis=1)
            v=np.full(self.k,np.inf);v[good]=slope[good]@(1+x)
            take=v<hi;hi[take]=v[take];best[take]=slope[take]
        return lo,hi,best

    def baseline(self):
        x=np.zeros(self.m);lo,hi,_=self.bounds(x)
        known=np.full(self.k,np.nan)
        while True:
            ratio=hi/lo-1.
            need=np.flatnonzero(np.isnan(known)&(ratio>BASE_TOL))
            if not len(need):break
            s=int(need[np.argmax(ratio[need])])
            rate=self.add_solve(s,x)
            if rate<lo[s]*(1-2*PAD) or rate>hi[s]*(1+2*PAD):
                raise RuntimeError("baseline interval contradicted")
            known[s]=rate
            a,b,_=self.latest(x)
            lo=np.maximum(lo,a);hi=np.minimum(hi,b)
            mask=np.isfinite(known);lo[mask]=known[mask];hi[mask]=known[mask]
            if np.any(lo>hi*(1+2*PAD)):raise RuntimeError("baseline transfer interval inconsistent")
        return lo,hi,known

class Cross(old.ReuseSolver):
    def __init__(self,p,T,bl,bh,known):
        self.p=p;self.T=T;self.bl=bl;self.bh=bh;self.known=known
        self.models=T.models;self.seconds=T.seconds
        self.k=T.k;self.m=T.m;self.active=np.arange(self.m)
        self.best=0.;self.plan=np.zeros(self.m);self.upper=1.
        self.master_calls=0;self.master_time=0.;self.invalid=False;self.incidents=[]
        self.queries=0;self.skipped=0;self.history=[];self.cache={}
        self.cuts=[];self.add_cut(np.zeros(self.m))
        self.cache[()]=(np.zeros(self.k),np.zeros(self.k))

    def add_cut(self,x):
        _,_,slope=self.T.bounds(x)
        # For targets with no informative dual, the physical doubling
        # ceiling gives a constant upper bound of 1 on normalized gain.
        valid=np.all(np.isfinite(slope),axis=1)
        g=np.zeros((self.k,self.m));b=np.ones(self.k)
        g[valid]=slope[valid]/self.bl[valid,None]
        b[valid]=g[valid].sum(axis=1)-1.
        self.cuts.append((g,b))

    def normalized(self,lo,hi):
        l=np.clip(lo/self.bh-1.,0.,1.)
        u=np.clip(hi/self.bl-1.,0.,1.)
        if np.any(l>u+2*PAD):raise RuntimeError("normalized interval inconsistent")
        return l,np.maximum(l,u)

    def query(self,x):
        self.p.validate(x);key=tuple(np.flatnonzero(x))
        if key in self.cache:return
        self.queries+=1
        lo,hi,_=self.T.bounds(x)
        exact=np.full(self.k,np.nan);solved=0
        while True:
            l,u=self.normalized(lo,hi)
            if (u-l).mean()<=LOCAL:break
            remaining=np.flatnonzero(np.isnan(exact))
            if not len(remaining):break
            s=int(remaining[np.argmax((u-l)[remaining])])
            rate=self.T.add_solve(s,x);solved+=1
            if rate<lo[s]*(1-2*PAD) or rate>hi[s]*(1+2*PAD):
                raise RuntimeError("candidate interval contradicted")
            exact[s]=rate
            a,b,_=self.T.latest(x)
            lo=np.maximum(lo,a);hi=np.minimum(hi,b)
            mask=np.isfinite(exact)
            lo[mask]=exact[mask]*(1-PAD);hi[mask]=exact[mask]*(1+PAD)
            if np.any(lo>hi*(1+2*PAD)):raise RuntimeError("candidate transfer interval inconsistent")
        l,u=self.normalized(lo,hi)
        self.cache[key]=(l.copy(),u.copy())
        self.add_cut(x)
        self.skipped+=self.k-solved
        if l.mean()>self.best:self.best=float(l.mean());self.plan=x.copy()
        self.history.append({"solved":solved,"lower_pct":100*float(l.mean()),"upper_pct":100*float(u.mean())})

    def run_cross(self,iterations):
        start=time.monotonic()
        baseline_calls=self.T.calls
        for _ in range(iterations+1):
            if self.upper-self.best<=EPS:break
            x=self.master()
            if self.invalid or self.upper-self.best<=EPS or x is None or self.queries>=iterations:break
            if tuple(np.flatnonzero(x)) in self.cache:break
            self.query(x)
            if self.best>self.upper+NUM:
                self.invalid=True;self.incidents.append("global upper contradicted");break
        seconds=time.monotonic()-start
        candidate_calls=self.T.calls-baseline_calls
        # Full independent replay of the selected plan. Also verify every
        # previously skipped baseline. Charge these additional solves.
        start=time.monotonic();verified=[];base_extra=0
        for i,p in enumerate(self.models):
            b=self.known[i]
            if not np.isfinite(b):
                b=p.lp(np.zeros(self.m),self.seconds)[0];base_extra+=1
                if b<self.bl[i]*(1-2*PAD) or b>self.bh[i]*(1+2*PAD):
                    self.invalid=True;self.incidents.append("baseline replay contradicted")
            rate=p.lp(self.plan,self.seconds)[0];verified.append(rate/b-1.)
        verified=np.asarray(verified);replay=time.monotonic()-start
        lo,hi=self.cache[tuple(np.flatnonzero(self.plan))]
        if np.any(verified<lo-4*PAD) or np.any(verified>hi+4*PAD) or verified.mean()>self.upper+NUM:
            self.invalid=True;self.incidents.append("final replay contradicted certificate")
        U=1. if self.invalid else max(float(verified.mean()),self.upper)
        return self.plan,{"train":metrics(verified),"train_upper_pct":100*U,
            "train_gap_pct_points":100*max(0.,U-float(verified.mean())),
            "within_tolerance":bool(not self.invalid and U-verified.mean()<=EPS+NUM),
            "certificate_withheld":self.invalid,"numerical_incidents":self.incidents,
            "baseline_lp_calls":baseline_calls,"baseline_skips":self.k-baseline_calls,
            "candidate_lp_calls":candidate_calls,"candidate_skips":self.skipped,
            "replay_lp_calls":self.k+base_extra,"total_lp_calls":self.T.calls+self.k+base_extra,
            "selection_seconds":seconds,"replay_seconds":replay,
            "transfer_seconds":self.T.transfer_seconds,"master_calls":self.master_calls,
            "template_count":len(self.T.primal),"template_pairs":self.T.template_pairs,
            "template_fallback_pairs":self.T.fallback_pairs,"history":self.history}

def study(edges,cap,train,test,seconds,iterations):
    K=len(edges)//20;p=Policy(edges,cap,K,max(1,K),np.asarray(cap)/np.median(cap),1.5*K,.25)
    records={};plans={}
    # Separate setup and caches for a fair end-to-end comparison.
    start=time.monotonic()
    models,bases,G,loads=old.scan(edges,cap,train,K,seconds)
    setup=time.monotonic()-start
    x,r=old.ReuseSolver(p,models,bases,G,loads,seconds,"reuse").run(iterations)
    records["within"]={"train":r["train"],"train_upper_pct":r["train_upper_pct"],
        "train_gap_pct_points":r["train_gap_pct_points"],"within_tolerance":r["within_tolerance"],
        "certificate_withheld":r["certificate_withheld"],"numerical_incidents":r["numerical_incidents"],
        "baseline_lp_calls":len(train),"baseline_skips":0,"candidate_lp_calls":r["routing_lp_calls"],
        "candidate_skips":r["routing_calls_skipped"],"replay_lp_calls":r["final_replay_calls"],
        "total_lp_calls":len(train)+r["routing_lp_calls"]+r["final_replay_calls"],
        "total_selection_seconds":setup+r["selection_seconds"],
        "total_verified_seconds":setup+r["selection_seconds"]+r["final_replay_seconds"]}
    plans["within"]=x;del models
    start=time.monotonic()
    T=Transfer(edges,cap,train,K,seconds);bl,bh,known=T.baseline()
    solver=Cross(p,T,bl,bh,known)
    setup=time.monotonic()-start
    x,r=solver.run_cross(iterations)
    records["cross"]={**r,"total_selection_seconds":setup+r["selection_seconds"],
                      "total_verified_seconds":setup+r["selection_seconds"]+r["replay_seconds"]}
    plans["cross"]=x
    best=max(v["train"]["mean_gain_pct"] for v in records.values())
    for name,r in records.items():
        if best>r["train_upper_pct"]+1e-4:
            r["certificate_withheld"]=True;r["within_tolerance"]=False
            r["train_upper_pct"]=100.;r["train_gap_pct_points"]=max(0.,100-r["train"]["mean_gain_pct"])
            r["numerical_incidents"].append("cross-method upper contradicted")
        r["plan_edges"]=[list(edges[i]) for i in np.flatnonzero(plans[name])]
    if test:
        held,hb,_,_=old.scan(edges,cap,test,K,seconds);cache={}
        for name,x in plans.items():
            key=tuple(np.flatnonzero(x))
            if key not in cache:cache[key]=np.asarray([q.lp(x,seconds)[0]/b-1 for q,b in zip(held,hb)])
            records[name]["test"]=metrics(cache[key])
    return {"nodes":p.n,"links":p.m,"budget":K,"train_count":len(train),"test_count":len(test),"methods":records}

def self_test():
    edges=[("a","b"),("b","c"),("c","d"),("d","a"),("a","c")]
    cap=[1.,2.,1.,3.,1.]
    train=[[("a","c",2.),("b","d",1.)],[("a","c",4.),("b","d",2.)],
           [("a","d",3.),("b","c",2.)],[("a","c",1.),("a","d",2.),("d","b",3.)]]
    T=Transfer(edges,cap,train,2,10)
    T.add_solve(0,np.zeros(5))
    # Check all plans, including differing demand support and opposite directions.
    for mask in range(32):
        x=np.array([(mask>>i)&1 for i in range(5)],dtype=float)
        lo,hi,_=T.bounds(x)
        for s,p in enumerate(T.models):
            r=p.lp(x,10)[0]
            assert lo[s]<=r*(1+2*PAD) and hi[s]>=r*(1-2*PAD),(mask,s,lo[s],r,hi[s])
        if mask in (3,12):T.add_solve(2,x)
    p=Policy(edges,cap,2,2)
    T=Transfer(edges,cap,train,2,10);bl,bh,known=T.baseline()
    assert T.calls<len(train),(T.calls,known)
    opt=0.
    bases=[q.lp(np.zeros(5),10)[0] for q in T.models]
    for mask in range(32):
        x=np.array([(mask>>i)&1 for i in range(5)],dtype=float)
        try:p.validate(x)
        except RuntimeError:continue
        opt=max(opt,float(np.mean([q.lp(x,10)[0]/b-1 for q,b in zip(T.models,bases)])))
    _,r=Cross(p,T,bl,bh,known).run_cross(40)
    assert not r["certificate_withheld"] and r["within_tolerance"],r
    assert r["train_upper_pct"]>=100*opt-1e-5 and r["train"]["mean_gain_pct"]>=100*(opt-EPS)-1e-5,r
    # Single edge plus proportional and disjoint traffic: transfer is exact
    # (within padding) without another baseline solve for scaled demands.
    T=Transfer([("a","b"),("b","c")],[1.,1.],[[("a","b",1.)],[("a","b",5.)],[("b","c",2.)]],1,10)
    bl,bh,known=T.baseline()
    assert T.calls<3 and np.all(bh/bl-1<=BASE_TOL+1e-7),(T.calls,bl,bh)

def zoo(args,rows,errors,skips):
    data=rb.fetch(rb.ZOO);eligible=[]
    for name,raw in sorted(rb.members(data).items()):
        try:
            g=nx.Graph(nx.read_graphml(io.BytesIO(raw)));g.remove_edges_from(list(nx.selfloop_edges(g)))
            g=g.subgraph(max(nx.connected_components(g),key=lambda c:(len(c),sorted(c)))).copy()
            if 20<=g.number_of_edges()<=250:eligible.append((name,raw,g))
            else:skips.append({"network":name,"reason":"outside edge screening range"})
        except Exception as e:errors.append({"network":name,"error":str(e)})
    random.Random(2026).shuffle(eligible)
    for name,raw,g in eligible[args.offset:args.offset+args.limit]:
        try:
            nodes=sorted(g.nodes);edges=sorted(tuple(sorted(e)) for e in g.edges)
            rng=random.Random(20261201);crng=random.Random(1701)
            matrices=[]
            for t in range(48):
                matrices.append([(*rng.sample(nodes,2),rng.uniform(.5,2)) for _ in range(12)])
            cap=[float(crng.choice([1,2,4])) for _ in edges] if args.capacity=="heterogeneous" else [1.]*len(edges)
            # Paired persistent-demand workload, independently seeded; demand
            # endpoints are fixed within a topology, weights fluctuate.
            rng=random.Random(20261202)
            pairs=[tuple(rng.sample(nodes,2)) for _ in range(12)]
            weights=[rng.uniform(.5,2) for _ in pairs]
            correlated=[[(*pair,w*rng.uniform(.8,1.2)) for pair,w in zip(pairs,weights)] for _ in range(48)]
            for workload,mats in (("independent",matrices),("correlated",correlated)):
                row={"network":name,"capacity":args.capacity,"workload":workload,
                    "archive_sha256":hashlib.sha256(data).hexdigest(),"graph_sha256":hashlib.sha256(raw).hexdigest(),
                    **study(edges,cap,mats[:24],mats[24:],args.seconds,args.iterations)}
                rows.append(row)
                print(name,workload,{k:(v["baseline_skips"],v["candidate_skips"],v["total_verified_seconds"]) for k,v in row["methods"].items()},flush=True)
                rb.save(args,rows,errors,skips)
        except Exception as e:
            errors.append({"network":name,"error":str(e)});print(name,type(e).__name__,str(e),flush=True)
        rb.save(args,rows,errors,skips)
    if len(rows)!=2*args.limit:errors.append({"stage":"count","actual":len(rows),"expected":2*args.limit})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["self-test","zoo","historical"],required=True)
    ap.add_argument("--capacity",choices=["unit","heterogeneous"],default="unit")
    ap.add_argument("--offset",type=int,default=0);ap.add_argument("--limit",type=int,default=30)
    ap.add_argument("--seconds",type=float,default=5);ap.add_argument("--iterations",type=int,default=12)
    ap.add_argument("--output",default="results/cross.json");args=ap.parse_args()
    self_test()
    if args.mode=="self-test":print("Cross-scenario transfer and enumeration checks passed");return
    rows=[];errors=[];skips=[]
    try:
        if args.mode=="zoo":zoo(args,rows,errors,skips)
        else:
            edges,cap,prices,train,test,meta=historical()
            rows.append({**meta,"capacity":"historical_modeled","workload":"historical",
                         **study(edges,cap,train,test,args.seconds,args.iterations)})
    except Exception as e:errors.append({"error":str(e)});print(type(e).__name__,str(e),flush=True)
    rb.save(args,rows,errors,skips)
    if errors or not rows:raise SystemExit("Incomplete cross-scenario benchmark")

if __name__=="__main__":main()
