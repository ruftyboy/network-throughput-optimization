"""Persistent native HiGHS LPs and simplex basis reuse.

Same routing formulation and certificate-screened master as phase 10.
Native cold and reset controls isolate model persistence and basis reuse.
"""
import argparse, hashlib, io, random, time
import numpy as np
import networkx as nx
import highspy as hs
from scipy.sparse import vstack
import robust_benchmark as rb
import routing_reuse as old
from routing_reuse import PAD, LOCAL, NUM
from persistent_benchmark import SourceProblem
from cost_relaxation import Policy, metrics, historical

def check(status):
    if status==hs.HighsStatus.kError:
        raise RuntimeError("HiGHS API call failed")

class Engine:
    def __init__(self,mode,seconds):
        self.mode=mode;self.seconds=seconds
        self.pool={};self.bases={}
        self.calls=0;self.builds=0;self.updates=0;self.coeff_changes=0
        self.basis_attempts=0;self.basis_rejections=0;self.fallbacks=0;self.scipy_fallbacks=0
        self.iterations=0;self.native_runs=0
        self.update_seconds=0.;self.solve_seconds=0.

    def new(self,p,column,b):
        h=hs.Highs()
        for name,value in (("output_flag",False),("solver","simplex"),
                           ("threads",1),("parallel","off"),
                           ("time_limit",float(self.seconds)),
                           ("primal_feasibility_tolerance",1e-8),
                           ("dual_feasibility_tolerance",1e-8)):
            check(h.setOptionValue(name,value))
        A=vstack([p.eq,p.ub],format="csc")
        A.sort_indices()
        lp=hs.HighsLp()
        lp.num_col_=p.f+1;lp.num_row_=p.q+p.m
        lp.col_cost_=p.obj
        lp.col_lower_=np.zeros(p.f+1)
        lp.col_upper_=np.full(p.f+1,hs.kHighsInf)
        lp.row_lower_=np.r_[np.zeros(p.q),np.full(p.m,-hs.kHighsInf)]
        lp.row_upper_=np.r_[np.zeros(p.q),b]
        lp.a_matrix_.format_=hs.MatrixFormat.kColwise
        lp.a_matrix_.start_=A.indptr.astype(np.int32)
        lp.a_matrix_.index_=A.indices.astype(np.int32)
        lp.a_matrix_.value_=A.data
        check(h.passModel(lp));self.builds+=1
        return h,column.copy(),b.copy()

    def extract(self,h,p,b):
        if h.getModelStatus()!=hs.HighsModelStatus.kOptimal:
            raise RuntimeError("native LP: "+str(h.getModelStatus()))
        sol=h.getSolution();v=np.asarray(sol.col_value)
        y=np.asarray(sol.row_dual)
        if len(v)!=p.f+1 or len(y)!=p.q+p.m or not np.isfinite(v).all() or not np.isfinite(y).all():
            raise RuntimeError("nonfinite native certificate")
        load=np.asarray(p.ub@v)
        reduced=p.obj-p.eq.T@y[:p.q]-p.ub.T@y[p.q:]
        dual=float(y[p.q:]@b)
        rate=float(v[-1])
        # Validate the returned primal/dual against the requested model,
        # not merely the persistent object's internal status.
        if max(np.max(abs(p.eq@v)),np.max(load-b),-np.min(v))>1e-6:
            raise RuntimeError("native primal residual")
        if max(np.max(y[p.q:]),-np.min(reduced))>1e-6:
            raise RuntimeError("native dual residual")
        if rate<=0 or abs(dual+rate)>1e-6*max(1.,rate):
            raise RuntimeError("native primal-dual gap")
        return rate,-y[p.q:]*p.cap,np.maximum(load,0.)/p.cap

    def solve(self,p,x):
        self.calls+=1
        if self.mode=="scipy":
            start=time.monotonic()
            out=old.solve(p,x,self.seconds)
            self.solve_seconds+=time.monotonic()-start
            return out
        key=(p.k,p.n,p.m)
        # For one fixed graph and a fixed number of source blocks the
        # incidence part is identical even if the source identities change.
        # Only the multiplier column and capacity RHS depend on traffic.
        column=np.asarray(p.eq[:,-1].toarray()).ravel()
        b=p.cap*(1+x)
        start=time.monotonic()
        try:
            entry=self.pool.get(key) if self.mode!="cold" else None
            if entry is None:
                h,prev,prev_b=self.new(p,column,b)
            else:
                h,prev,prev_b=entry
                previous_basis=h.getBasis() if self.mode=="warm" else None
                for i in np.flatnonzero(column!=prev):
                    check(h.changeCoeff(int(i),p.f,float(column[i])))
                    self.coeff_changes+=1
                if np.any(b!=prev_b):
                    ids=np.arange(p.q,p.q+p.m,dtype=np.int32)
                    check(h.changeRowsBounds(p.m,ids,np.full(p.m,-hs.kHighsInf),b))
                self.updates+=1
                check(h.clearSolver())
                if self.mode=="warm":
                    # Prefer the last basis for the same scenario, otherwise
                    # reuse the immediately preceding compatible LP basis.
                    basis=self.bases.get(id(p),previous_basis)
                    if basis is not None and basis.valid:
                        self.basis_attempts+=1
                        status=h.setBasis(basis)
                        if status!=hs.HighsStatus.kOk:
                            self.basis_rejections+=1
                            check(h.clearSolver())
                prev=column.copy();prev_b=b.copy()
            if self.mode!="cold":self.pool[key]=(h,prev,prev_b)
            self.update_seconds+=time.monotonic()-start
            start=time.monotonic()
            check(h.run());self.native_runs+=1
            self.iterations+=max(0,int(h.getInfo().simplex_iteration_count))
            out=self.extract(h,p,b)
            if self.mode=="warm":self.bases[id(p)]=h.getBasis()
            self.solve_seconds+=time.monotonic()-start
            return out
        except RuntimeError:
            # Never use a failed incremental certificate. Rebuild a native
            # cold model, then use the independent scipy solve if necessary.
            self.fallbacks+=1
            self.pool.pop(key,None);self.bases.pop(id(p),None)
            self.update_seconds+=time.monotonic()-start
            start=time.monotonic()
            try:
                h,prev,prev_b=self.new(p,column,b)
                check(h.run());self.native_runs+=1
                self.iterations+=max(0,int(h.getInfo().simplex_iteration_count))
                out=self.extract(h,p,b)
                if self.mode!="cold":self.pool[key]=(h,prev,prev_b)
                if self.mode=="warm":self.bases[id(p)]=h.getBasis()
            except RuntimeError:
                self.scipy_fallbacks+=1
                out=old.solve(p,x,self.seconds)
            self.solve_seconds+=time.monotonic()-start
            return out

    def stats(self):
        return {"model_builds":self.builds,"model_updates":self.updates,
                "coefficient_changes":self.coeff_changes,"basis_attempts":self.basis_attempts,
                "basis_rejections":self.basis_rejections,"fallbacks":self.fallbacks,"scipy_fallbacks":self.scipy_fallbacks,
                "native_runs":self.native_runs,"simplex_iterations":self.iterations,
                "engine_update_seconds":self.update_seconds,"engine_solve_seconds":self.solve_seconds}

class Incremental(old.ReuseSolver):
    def __init__(self,p,models,bases,G,loads,seconds,engine,screen=True):
        super().__init__(p,models,bases,G,loads,seconds,"reuse" if screen else "full")
        self.engine=engine

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
            rate,g,load=self.engine.solve(self.models[s],x)
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


def scan(edges,cap,train,K,seconds,engine):
    models=[];bases=[];slopes=[];loads=[]
    for d in train:
        p=SourceProblem(edges,cap,d,K)
        v,g,l=engine.solve(p,np.zeros(p.m))
        if abs(g.sum()/v-1)>1e-5 or min(g)<-1e-7:
            raise RuntimeError("baseline dual normalization")
        models.append(p);bases.append(v);slopes.append(g/v);loads.append(l)
    return models,np.asarray(bases),np.asarray(slopes),np.asarray(loads)

MODES=(("scipy_reuse","scipy",True),("native_cold","cold",True),
       ("persistent_reset","reset",True),("persistent_warm","warm",True),
       ("warm_no_screen","warm",False))

def study(edges,cap,train,test,seconds,iterations):
    K=len(edges)//20
    p=Policy(edges,cap,K,max(1,K),np.asarray(cap)/np.median(cap),1.5*K,.25)
    records={};plans={};reference=None
    # Deterministic order rotation reduces systematic first/last effects.
    shift=len(edges)%len(MODES)
    for name,mode,screen in MODES[shift:]+MODES[:shift]:
        start=time.monotonic()
        engine=Engine(mode,seconds)
        models,bases,G,loads=scan(edges,cap,train,K,seconds,engine)
        setup=time.monotonic()-start
        baseline_stats=engine.stats()
        if reference is None:reference=bases.copy()
        elif not np.allclose(reference,bases,rtol=1e-6,atol=1e-8):
            raise RuntimeError("baseline values differ across engines")
        solver=Incremental(p,models,bases,G,loads,seconds,engine,screen)
        x,r=solver.run(iterations)
        records[name]={**r,**engine.stats(),"baseline_lp_calls":len(train),
            "baseline_skips":0,"candidate_skips":r["routing_calls_skipped"],
            "baseline_seconds":setup,"baseline_engine_stats":baseline_stats,"total_selection_seconds":setup+r["selection_seconds"],
            "total_verified_seconds":setup+r["selection_seconds"]+r["final_replay_seconds"],
            "total_lp_calls":len(train)+r["routing_lp_calls"]+r["final_replay_calls"],
            "plan_edges":[list(edges[i]) for i in np.flatnonzero(x)]}
        plans[name]=x
        del solver,engine,models
    best=max(r["train"]["mean_gain_pct"] for r in records.values())
    for r in records.values():
        if best>r["train_upper_pct"]+1e-4:
            r["certificate_withheld"]=True;r["within_tolerance"]=False
            r["numerical_incidents"].append("cross-method upper contradicted")
            r["train_upper_pct"]=100.
            r["train_gap_pct_points"]=max(0.,100-r["train"]["mean_gain_pct"])
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
           [("d","a",3.),("c","b",2.)],[("a","d",1.)],
           [("a","b",.5),("a","d",2.),("b","c",1.)]]
    models=[SourceProblem(edges,cap,d,2) for d in train]
    for mode in ("cold","reset","warm"):
        engine=Engine(mode,10)
        for mask in (0,3,12,1,16,0):
            x=np.array([(mask>>i)&1 for i in range(5)],dtype=float)
            for p in models:
                rate,g,load=engine.solve(p,x)
                exact,_=p.lp(x,10)
                assert abs(rate-exact)<1e-6*max(1.,exact),(mode,rate,exact)
                assert np.max(load-1-x)<1e-6
                assert abs(g@(1+x)-rate)<1e-6*max(1.,rate)
        assert engine.fallbacks==0,engine.stats()
        if mode=="warm":assert engine.basis_attempts>0 and engine.updates>0
        if mode=="cold":assert engine.builds==engine.calls
    # Complete optimization compared against independently enumerated optimum.
    p=Policy(edges,cap,2,2)
    models,b,G,loads=old.scan(edges,cap,train,2,10)
    opt=0.
    for mask in range(32):
        x=np.array([(mask>>i)&1 for i in range(5)],dtype=float)
        try:p.validate(x)
        except RuntimeError:continue
        opt=max(opt,float(np.mean([q.lp(x,10)[0]/v-1 for q,v in zip(models,b)])))
    for name,mode,screen in MODES:
        engine=Engine(mode,10)
        ms,bs,G,loads=scan(edges,cap,train,2,10,engine)
        _,r=Incremental(p,ms,bs,G,loads,10,engine,screen).run(40)
        assert not r["certificate_withheld"] and r["within_tolerance"],(name,r)
        assert r["train_upper_pct"]>=100*opt-1e-5 and r["train"]["mean_gain_pct"]>=100*(opt-old.EPS)-1e-5,(name,r)

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
            rng=random.Random(20261203);crng=random.Random(1701)
            independent=[[(*rng.sample(nodes,2),rng.uniform(.5,2)) for _ in range(12)] for _ in range(48)]
            cap=[float(crng.choice([1,2,4])) for _ in edges] if args.capacity=="heterogeneous" else [1.]*len(edges)
            rng=random.Random(20261204)
            pairs=[tuple(rng.sample(nodes,2)) for _ in range(12)]
            weights=[rng.uniform(.5,2) for _ in pairs]
            correlated=[[(*pair,w*rng.uniform(.8,1.2)) for pair,w in zip(pairs,weights)] for _ in range(48)]
            for workload,mats in (("independent",independent),("correlated",correlated)):
                row={"network":name,"capacity":args.capacity,"workload":workload,
                     "archive_sha256":hashlib.sha256(data).hexdigest(),"graph_sha256":hashlib.sha256(raw).hexdigest(),
                     **study(edges,cap,mats[:24],mats[24:],args.seconds,args.iterations)}
                rows.append(row)
                print(name,workload,{k:(v["total_verified_seconds"],v["simplex_iterations"],v["fallbacks"]) for k,v in row["methods"].items()},flush=True)
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
    ap.add_argument("--output",default="results/incremental.json");args=ap.parse_args()
    self_test()
    if args.mode=="self-test":print("Native incremental basis and enumeration checks passed");return
    rows=[];errors=[];skips=[]
    try:
        if args.mode=="zoo":zoo(args,rows,errors,skips)
        else:
            edges,cap,prices,train,test,meta=historical()
            rows.append({**meta,"capacity":"historical_modeled","workload":"historical",
                         **study(edges,cap,train,test,args.seconds,args.iterations)})
    except Exception as e:errors.append({"error":str(e)});print(type(e).__name__,str(e),flush=True)
    rb.save(args,rows,errors,skips)
    if errors or not rows:raise SystemExit("Incomplete incremental benchmark")

if __name__=="__main__":main()
