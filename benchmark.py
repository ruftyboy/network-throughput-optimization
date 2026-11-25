"""Reproducible matching-constrained capacity augmentation benchmark."""
import argparse
import hashlib
import io
import json
import math
import random
import time
from pathlib import Path
from urllib.request import urlopen

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix

SOURCES = {
    "Abilene": "https://raw.githubusercontent.com/sk2/topologyzoo/master/sources/Abilene.graphml",
    "Esnet": "https://raw.githubusercontent.com/mroughan/InternetTopologyZoo/main/graphml/Esnet.graphml",
    "Geant2012": "https://raw.githubusercontent.com/PacktPublishing/Network-Science-with-Python-and-NetworkX-Quick-Start-Guide/master/data/UAITZ/Geant2012.graphml",
}

class Model:
    def __init__(self, graph, pairs, budget):
        self.nodes = sorted(graph.nodes)
        self.edges = sorted(tuple(sorted(e)) for e in graph.edges)
        self.m = m = len(self.edges)
        n, k = len(self.nodes), len(pairs)
        idx = {v: i for i, v in enumerate(self.nodes)}
        self.p = p = 2*m*k
        self.t = t = p+m
        self.q = q = n*k
        A = lil_matrix((q+m+n+1, t+1))
        for j, (s, d) in enumerate(pairs):
            A[j*n+idx[s], t] = -1
            A[j*n+idx[d], t] = 1
            for i, (u, v) in enumerate(self.edges):
                a, b = j*2*m+2*i, j*2*m+2*i+1
                A[j*n+idx[u], a] = 1
                A[j*n+idx[v], a] = -1
                A[j*n+idx[v], b] = 1
                A[j*n+idx[u], b] = -1
                A[q+i, a] = A[q+i, b] = 1
        for i, (u, v) in enumerate(self.edges):
            A[q+i, p+i] = -1
            A[q+m+idx[u], p+i] = 1
            A[q+m+idx[v], p+i] = 1
            A[-1, p+i] = 1
        self.A = A.tocsr()
        self.hi = np.r_[np.zeros(q), np.ones(m+n), budget]
        self.lo = np.r_[np.zeros(q), np.full(m+n+1, -np.inf)]
        self.lb = np.zeros(t+1)
        self.ub = np.full(t+1, np.inf)
        self.ub[p:t] = 1
        self.c = np.zeros(t+1)
        self.c[t] = -1
        self.integrality = np.zeros(t+1)
        self.integrality[p:t] = 1

    def validate(self, x):
        if x is None or not np.all(np.isfinite(x)):
            raise RuntimeError("No finite solution")
        a = self.A @ x
        if max(np.max(a-self.hi), np.max(self.lo-a),
               np.max(self.lb-x), np.max(x-self.ub)) > 1e-6:
            raise RuntimeError("Flow violates constraints")
        if np.max(np.abs(x[self.p:self.t]-np.rint(x[self.p:self.t]))) > 1e-6:
            raise RuntimeError("Nonintegral upgrade plan")

    def fixed(self, chosen):
        x = np.zeros(self.m)
        x[list(chosen)] = 1
        bounds = list(zip(self.lb, self.ub))
        bounds[self.p:self.t] = [(v, v) for v in x]
        r = linprog(self.c, A_eq=self.A[:self.q], b_eq=self.hi[:self.q],
                    A_ub=self.A[self.q:], b_ub=self.hi[self.q:],
                    bounds=bounds, method="highs")
        if not r.success:
            raise RuntimeError(r.message)
        self.validate(r.x)
        return float(r.x[self.t])

    def optimize(self, seconds):
        start = time.monotonic()
        r = milp(self.c, integrality=self.integrality,
                 bounds=Bounds(self.lb, self.ub),
                 constraints=LinearConstraint(self.A, self.lo, self.hi),
                 options={"time_limit": seconds, "mip_rel_gap": 0.0})
        info = {"status": int(r.status), "message": str(r.message),
                "seconds": time.monotonic()-start,
                "upper_bound": None, "rate": None, "upgrades": None}
        bound = getattr(r, "mip_dual_bound", None)
        if bound is not None and math.isfinite(bound):
            info["upper_bound"] = float(-bound)
        if r.x is not None:
            self.validate(r.x)
            chosen = np.flatnonzero(r.x[self.p:self.t] > .5).tolist()
            info["rate"] = self.fixed(chosen)
            info["upgrades"] = [list(self.edges[i]) for i in chosen]
        return info

def greedy_matching(edges, order, budget):
    chosen, used = [], set()
    for i in order:
        if len(chosen) >= budget:
            break
        u, v = edges[i]
        if u not in used and v not in used:
            chosen.append(i)
            used.update((u, v))
    return chosen

def evaluate(name, graph, seed, seconds, source=None):
    rng = random.Random(seed)
    nodes = sorted(graph.nodes)
    pairs = [tuple(rng.sample(nodes, 2)) for _ in range(12)]
    # Strict maximum of 5% of links, with zero upgrades allowed.
    budget = graph.number_of_edges() // 20
    model = Model(graph, pairs, budget)
    base = model.fixed([])
    random_rates = []
    for _ in range(20):
        order = list(range(model.m))
        rng.shuffle(order)
        random_rates.append(model.fixed(greedy_matching(model.edges, order, budget)))
    scores = {frozenset(e): v for e, v in nx.edge_betweenness_centrality(graph).items()}
    order = sorted(range(model.m), key=lambda i: (-scores[frozenset(model.edges[i])], i))
    central = model.fixed(greedy_matching(model.edges, order, budget))
    # Congestion-aware greedy: choose the single feasible addition maximizing
    # throughput at each step; bounded to small cases for runtime.
    chosen, used = [], set()
    greedy_rate = base
    if model.m <= 100:
        for _ in range(budget):
            candidates = [i for i, (u, v) in enumerate(model.edges)
                          if u not in used and v not in used]
            if not candidates:
                break
            greedy_rate, i = max((model.fixed(chosen+[i]), i) for i in candidates)
            chosen.append(i)
            used.update(model.edges[i])
    else:
        greedy_rate = None
    opt = model.optimize(seconds)
    rates = {"random_mean": float(np.mean(random_rates)),
             "random_best": max(random_rates), "betweenness": central,
             "throughput_greedy": greedy_rate, "milp": opt["rate"]}
    return {"network": name, "source": source, "nodes": len(nodes),
            "links": model.m, "seed": seed, "pairs": pairs, "budget": budget,
            "base_rate": base, "rates": rates, "optimizer": opt,
            "gain_pct": {k: None if v is None else 100*(v/base-1)
                         for k, v in rates.items()}}

def load_graph(name, url):
    with urlopen(url, timeout=40) as response:
        data = response.read()
    raw = nx.read_graphml(io.BytesIO(data))
    graph = nx.Graph(raw)
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    graph = graph.subgraph(max(nx.connected_components(graph), key=lambda c: (len(c), sorted(c)))).copy()
    return graph, {"url": url, "sha256": hashlib.sha256(data).hexdigest(),
                   "original_nodes": raw.number_of_nodes(),
                   "original_edges": raw.number_of_edges()}

def self_test():
    g = nx.path_graph(3)
    model = Model(g, [(0, 1)], 1)
    assert abs(model.fixed([])-1) < 1e-8
    r = model.optimize(10)
    assert r["status"] == 0 and abs(r["rate"]-2) < 1e-8
    model = Model(g, [(0, 2)], 1)
    assert abs(model.optimize(10)["rate"]-1) < 1e-8

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    self_test()
    networks = [
        ("grid", nx.convert_node_labels_to_integers(nx.grid_2d_graph(5,5)), None),
        ("regular", nx.random_regular_graph(4,24,seed=8), None),
        ("smallworld", nx.watts_strogatz_graph(24,4,.2,seed=13), None),
    ]
    errors = []
    if args.real:
        for name, url in SOURCES.items():
            try:
                graph, meta = load_graph(name, url)
                networks.append((name, graph, meta))
            except Exception as exc:
                errors.append({"network": name, "stage": "download", "error": str(exc)})
    rows = []
    for name, graph, source in networks:
        for seed in range(args.seeds):
            try:
                row = evaluate(name, graph, seed, args.seconds, source)
                rows.append(row)
                print(name, seed, row["gain_pct"], flush=True)
            except Exception as exc:
                errors.append({"network": name, "seed": seed, "error": str(exc)})
                print(name, seed, str(exc), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": "unit shared capacities; equal synthetic demands; matching upgrades",
                                  "results": rows, "errors": errors}, indent=2, allow_nan=False))
    if errors:
        raise SystemExit("Some experiments failed; inspect JSON errors.")

if __name__ == "__main__":
    main()
