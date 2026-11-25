"""SNDlib pilot: weighted traffic, heterogeneous capacity, dual-cut decomposition.

The decomposition is classical Benders/LP duality; novelty is not claimed.
"""
import argparse
import hashlib
import io
import json
import math
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

URL = "https://sndlib.put.poznan.pl/download/sndlib-networks-xml.tgz"

def parse(data, name, policy):
    root = ET.fromstring(data)
    for el in root.iter():
        el.tag = el.tag.split("}")[-1]
    edges, caps, kinds = [], [], []
    for link in root.findall("./networkStructure/links/link"):
        u, v = link.findtext("source"), link.findtext("target")
        installed = float(link.findtext("preInstalledModule/capacity", "0"))
        modules = [float(x.text) for x in link.findall("additionalModules/addModule/capacity")]
        if not modules:
            modules = [float(x.text) for x in link.findall("additionalModules/*/capacity")]
        if installed > 0:
            c, kind = installed, "preinstalled"
        elif policy == "module" and any(x > 0 for x in modules):
            c, kind = min(x for x in modules if x > 0), "one_candidate_module_assumed"
        else:
            raise ValueError("No positive preinstalled capacity for " + str(link.get("id")))
        if not math.isfinite(c) or c <= 0 or u == v:
            raise ValueError("Invalid link")
        edges.append((u, v))
        caps.append(c)
        kinds.append(kind)
    pairs = []
    for d in root.findall("./demands/demand"):
        amount = float(d.findtext("demandValue")) * float(d.findtext("routingUnit", "1"))
        u, v = d.findtext("source"), d.findtext("target")
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Invalid demand")
        if amount > 0 and u != v:
            pairs.append((u, v, amount))
    if not edges or not pairs:
        raise ValueError("No links or positive demands")
    return edges, caps, pairs, {"name":name, "xml_sha256":hashlib.sha256(data).hexdigest(),
                                "capacity_policy":policy, "preinstalled_links":kinds.count("preinstalled"),
                                "assumed_module_links":kinds.count("one_candidate_module_assumed")}

class Problem:
    def __init__(self, edges, capacities, demands, budget):
        self.edges, self.budget = edges, budget
        self.m = m = len(edges)
        nodes = sorted(set(v for e in edges for v in e))
        idx = {v:i for i,v in enumerate(nodes)}
        n, k = len(nodes), len(demands)
        self.n, self.k = n, k
        # Scaling keeps coefficients reasonable; t is the scale of the
        # original weighted demand matrix, independent of this normalization.
        scale = max(max(capacities), max(d for _,_,d in demands))
        self.cap = np.asarray(capacities, dtype=float)/scale
        q = self.q = n*k
        f = self.f = 2*m*k
        rows, cols, vals = [], [], []
        def put(r,c,v):
            rows.append(r); cols.append(c); vals.append(v)
        for j,(s,t,d) in enumerate(demands):
            put(j*n+idx[s],f,-d/scale)
            put(j*n+idx[t],f,d/scale)
            for i,(u,v) in enumerate(edges):
                a = j*2*m+2*i
                for col,x,y in ((a,u,v),(a+1,v,u)):
                    put(j*n+idx[x],col,1)
                    put(j*n+idx[y],col,-1)
                    put(q+i,col,1)
        mat = coo_matrix((vals,(rows,cols)),shape=(q+m,f+1)).tocsr()
        self.eq, self.ub = mat[:q],mat[q:]
        self.obj = np.zeros(f+1); self.obj[-1] = -1
        mr, mc = [], []
        for i,(u,v) in enumerate(edges):
            mr.extend([idx[u],idx[v],n]); mc.extend([i,i,i])
        self.match = coo_matrix((np.ones(len(mr)),(mr,mc)),shape=(n+1,m)).tocsr()
        self.limits = np.r_[np.ones(n),budget]

    def lp(self, x, seconds=None):
        b = self.cap*(1+x)
        r = linprog(self.obj,A_eq=self.eq,b_eq=np.zeros(self.q),
                    A_ub=self.ub,b_ub=b,bounds=(0,None),method="highs",
                    options={} if seconds is None else {"time_limit":max(.01,seconds)})
        if not r.success:
            raise RuntimeError("routing LP: "+r.message)
        if max(np.max(np.abs(self.eq@r.x)), np.max(self.ub@r.x-b), -np.min(r.x)) > 1e-6:
            raise RuntimeError("routing residual too large")
        # The value of max-flow is concave in capacities. The LP dual
        # yields a global upper supporting hyperplane for any upgrade plan.
        slope = -r.ineqlin.marginals*self.cap
        return float(r.x[-1]), slope

    def plan(self, x):
        y = np.rint(x)
        if np.max(np.abs(x-y)) > 1e-5 or np.min(y)<0 or np.max(y)>1 or np.max(self.match@y-self.limits)>1e-6:
            raise RuntimeError("invalid matching")
        return y

    def direct(self, seconds):
        m, f = self.m,self.f
        a = hstack([self.eq,csr_matrix((self.q,m))])
        from scipy.sparse import diags
        b = hstack([self.ub,-diags(self.cap)])
        c = hstack([csr_matrix((self.n+1,f+1)),self.match])
        A = vstack([a,b,c]).tocsr()
        hi = np.r_[np.zeros(self.q), self.cap,self.limits]
        lo = np.r_[np.zeros(self.q),np.full(m+self.n+1,-np.inf)]
        start = time.monotonic()
        r = milp(np.r_[self.obj,np.zeros(m)],integrality=np.r_[np.zeros(f+1),np.ones(m)],
                 bounds=Bounds(np.zeros(f+1+m),np.r_[np.full(f+1,np.inf),np.ones(m)]),
                 constraints=LinearConstraint(A,lo,hi),
                 options={"time_limit":seconds,"mip_rel_gap":0.0})
        bound = getattr(r,"mip_dual_bound",None)
        out = {"status":int(r.status),"upper_bound":float(-bound) if bound is not None and math.isfinite(bound) else None,
               "rate":None,"chosen":None,"seconds":time.monotonic()-start}
        if r.x is not None:
            x = self.plan(r.x[-m:]); out["rate"],_ = self.lp(x,seconds)
            out["chosen"] = np.flatnonzero(x).tolist()
        out["seconds"] = time.monotonic()-start
        return out

    def cuts(self, seconds, iterations=200):
        start = time.monotonic()
        m = self.m
        x = np.zeros(m)
        base, slope = self.lp(x,seconds)
        best, incumbent = base, x.copy()
        # Doubling every capacity doubles the original optimum, even
        # without matching/budget restrictions. This is a global bound.
        upper = 2*base
        cuts, rhs = [], []
        count = 0
        for _ in range(iterations):
            remaining = seconds-(time.monotonic()-start)
            if remaining <= .05:
                break
            if count:
                try:
                    rate,slope = self.lp(x,remaining)
                except RuntimeError:
                    break
                if rate > best:
                    best,incumbent = rate,x.copy()
            else:
                rate = base
            cuts.append(np.r_[-slope,1.])
            rhs.append(rate-float(slope@x))
            count += 1
            A = vstack([hstack([self.match,csr_matrix((self.n+1,1))]),csr_matrix(np.asarray(cuts))])
            remaining = seconds-(time.monotonic()-start)
            if remaining <= .05:
                break
            r = milp(np.r_[np.zeros(m),-1.],integrality=np.r_[np.ones(m),0],
                     bounds=Bounds(np.zeros(m+1),np.r_[np.ones(m),2*base+1e-9]),
                     constraints=LinearConstraint(A,-np.inf,np.r_[self.limits,rhs]),
                     options={"time_limit":remaining,"mip_rel_gap":0.0})
            dual = getattr(r,"mip_dual_bound",None)
            if dual is not None and math.isfinite(dual):
                upper = min(upper,float(-dual))
            if upper-best <= 1e-7*max(1,abs(best)) or r.x is None:
                break
            x = self.plan(r.x[:m])
        return {"rate":best,"upper_bound":max(best,upper),"chosen":np.flatnonzero(incumbent).tolist(),
                "iterations":count,"seconds":time.monotonic()-start,
                "numerically_closed_gap":upper-best<=1e-7*max(1,abs(best))}

def self_test():
    p = Problem([("a","b"),("b","c")],[2,1],[("a","c",1)],1)
    assert abs(p.lp(np.zeros(2))[0]-1)<1e-7
    for out in (p.direct(10),p.cuts(10)):
        assert abs(out["rate"]-2)<1e-7 and abs(out["upper_bound"]-2)<1e-6
    p = Problem([("a","b"),("b","c")],[1,1],[("a","c",1)],1)
    assert abs(p.cuts(10)["rate"]-1)<1e-7
    xml = b'<network><networkStructure><links><link id="x"><source>a</source><target>b</target><preInstalledModule><capacity>3</capacity></preInstalledModule></link></links></networkStructure><demands><demand><source>a</source><target>b</target><routingUnit>2</routingUnit><demandValue>4</demandValue></demand></demands></network>'
    e,c,d,_ = parse(xml,"fixture","installed")
    assert c==[3] and d==[("a","b",8)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seconds",type=int,default=25)
    ap.add_argument("--output",default="results/sndlib.json")
    ap.add_argument("--archive",help="Local official SNDlib XML archive instead of download")
    ap.add_argument("--names",nargs="+",default=["abilene","geant","germany50","nobel-us","polska","pdh"])
    args=ap.parse_args()
    self_test()
    data=Path(args.archive).read_bytes() if args.archive else urlopen(URL,timeout=90).read()
    results,errors,skipped=[],[],[]
    with tarfile.open(fileobj=io.BytesIO(data),mode="r:*") as tar:
        files={Path(m.name).stem.lower():m for m in tar.getmembers() if m.isfile() and m.name.endswith(".xml")}
        for name in args.names:
            try:
                raw=tar.extractfile(files[name]).read()
                for policy in ("installed","module"):
                    try:
                        edges,cap,dem,meta=parse(raw,name,policy)
                    except ValueError as exc:
                        skipped.append({"network":name,"policy":policy,"reason":str(exc)})
                        continue
                    if policy=="module" and meta["assumed_module_links"]==0:
                        continue
                    p=Problem(edges,cap,dem,len(edges)//20)
                    base,_=p.lp(np.zeros(p.m),args.seconds)
                    if base<=0:
                        raise RuntimeError("Nonpositive baseline")
                    a=p.cuts(args.seconds)
                    b=p.direct(args.seconds)
                    if b["upper_bound"] is not None and a["rate"]>b["upper_bound"]+1e-6:
                        raise RuntimeError("Dual cut incumbent exceeds direct upper bound")
                    if b["rate"] is not None and b["rate"]>a["upper_bound"]+1e-6:
                        raise RuntimeError("Direct incumbent exceeds dual cut upper bound")
                    row={**meta,"nodes":p.n,"links":p.m,"demands":p.k,"budget":p.budget,"base":base,
                         "dual_cuts":a,"direct_milp":b,"dual_gain_pct":100*(a["rate"]/base-1),
                         "direct_gain_pct":None if b["rate"] is None else 100*(b["rate"]/base-1)}
                    results.append(row)
                    print(json.dumps(row),flush=True)
            except Exception as exc:
                errors.append({"network":name,"error":str(exc)})
                print(name,str(exc),flush=True)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"source":URL,"archive_sha256":hashlib.sha256(data).hexdigest(),
        "results":results,"skipped":skipped,"errors":errors},indent=2,allow_nan=False))
    if errors or not results:
        raise SystemExit("Incomplete benchmark; inspect artifact")

if __name__=="__main__":
    main()
