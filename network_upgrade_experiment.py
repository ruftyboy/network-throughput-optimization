import numpy as np, networkx as nx
from scipy.optimize import milp, linprog, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
import time

def solve(g, pairs, factor=1.068):
    nodes=list(g); ix={v:i for i,v in enumerate(nodes)}; es=list(g.edges()); n=len(nodes); m=len(es); k=len(pairs); nv=2*m*k+m+1; t=nv-1
    A=lil_matrix((k*n+m+n,nv)); lo=np.full(A.shape[0],-np.inf); hi=np.zeros(A.shape[0]); lo[:k*n]=0; hi[k*n:k*n+m]=1; hi[k*n+m:]=1
    for j,(s,d) in enumerate(pairs):
        A[j*n+ix[s],t]=-1; A[j*n+ix[d],t]=1
        for e,(u,v) in enumerate(es):
            for a,b,q in [(u,v,2*e),(v,u,2*e+1)]:
                p=j*2*m+q; A[j*n+ix[a],p]=1; A[j*n+ix[b],p]=-1; A[k*n+e,p]=1
    for e,(u,v) in enumerate(es):
        p=2*m*k+e; A[k*n+e,p]=-1; A[k*n+m+ix[u],p]=1; A[k*n+m+ix[v],p]=1
    A=A.tocsr(); lb=np.zeros(nv); ub=np.full(nv,np.inf); ub[2*m*k:t]=0
    c=np.zeros(nv); c[t]=-1
    res=linprog(c,A_ub=A[k*n:],b_ub=hi[k*n:],A_eq=A[:k*n],b_eq=hi[:k*n],bounds=list(zip(lb,ub)),method='highs')
    base=res.x[t]; target=base*factor; lb[t]=ub[t]=target; ub[2*m*k:t]=1; c[t]=0; c[2*m*k:t]=1
    integ=np.zeros(nv); integ[2*m*k:t]=1
    r=milp(c,integrality=integ,bounds=Bounds(lb,ub),constraints=LinearConstraint(A,lo,hi),options={'time_limit':12,'mip_rel_gap':0})
    print('n,m,k',n,m,k,'base,target',base,target,'status',r.status,'upgrades',r.fun,'bound',getattr(r,'mip_dual_bound',None),flush=True)
    if r.x is not None:
        y=r.x[2*m*k:t]; a=A@r.x; print('verify',np.max(a-hi),np.max(lo-a),np.max(abs(y-np.round(y))), 'edges', [es[i] for i in np.where(y>.5)[0]],flush=True)
    if r.x is not None:
        B=int(round(r.fun)); lb[t]=0; ub[t]=np.inf; c[2*m*k:t]=0; c[t]=-1
        from scipy.sparse import vstack,csr_matrix
        row=np.zeros(nv); row[2*m*k:t]=1
        A2=vstack([A,csr_matrix(row.reshape(1,-1))]); lo2=np.r_[lo,-np.inf]; hi2=np.r_[hi,B]
        z=milp(c,integrality=integ,bounds=Bounds(lb,ub),constraints=LinearConstraint(A2,lo2,hi2),options={'time_limit':12,'mip_rel_gap':0})
        print('max at B',B,'status',z.status,'rate',z.x[t] if z.x is not None else None,'gain',100*(z.x[t]/base-1) if z.x is not None else None,'upper',-z.mip_dual_bound,flush=True)
    return r
rng=np.random.default_rng(42)
for name,g in [('grid',nx.convert_node_labels_to_integers(nx.grid_2d_graph(5,5))),('regular',nx.random_regular_graph(4,24,seed=8)),('smallworld',nx.watts_strogatz_graph(24,4,.2,seed=13))]:
    print(name,flush=True); pairs=[tuple(rng.choice(list(g.nodes),2,replace=False)) for _ in range(12)]; print('pairs',pairs,flush=True); solve(g,pairs)
