"""Aggregate adaptive benchmark with paired runtime and quality metrics."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np

def describe(rows):
    if not rows:return {"instances":0}
    out={"instances":len(rows),"test_scenarios":sum(r["test_count"] for r in rows),"methods":{},"paired":{}}
    names=("aggregate_direct","multicut_direct","adaptive_partition")
    for name in names:
        a=[r["methods"][name] for r in rows]
        d={"certified_within_0_5pp":sum(v["within_tolerance"] for v in a),
           "mean_training_lower_pct":float(np.mean([v["train_lower_pct"] for v in a])),
           "mean_training_gap_pp":float(np.mean([v["train_gap_pct_points"] for v in a])),
           "median_selection_seconds":float(np.median([v["selection_seconds"] for v in a])),
           "median_total_seconds_including_scan":float(np.median([v["total_selection_seconds_including_scan"] for v in a])),
           "total_selection_seconds_including_scan":float(np.sum([v["total_selection_seconds_including_scan"] for v in a])),
           "mean_lp_calls":float(np.mean([v["routing_lp_calls"] for v in a])),
           "mean_master_calls":float(np.mean([v["master_calls"] for v in a])),
           "classification":dict(Counter(v["classification"] for v in a))}
        if all("test" in v for v in a):
            d.update({"mean_test_gain_pct":float(np.mean([v["test"]["mean_gain_pct"] for v in a])),
                      "mean_minimum_test_gain_pct":float(np.mean([v["test"]["minimum_gain_pct"] for v in a])),
                      "mean_fraction_test_at_least_5pct":float(np.mean([v["test"]["fraction_at_least_5pct"] for v in a]))})
        out["methods"][name]=d
    for name in names[:2]:
        a=[r["methods"]["adaptive_partition"] for r in rows]
        b=[r["methods"][name] for r in rows]
        times=np.array([q["total_selection_seconds_including_scan"]/max(p["total_selection_seconds_including_scan"],1e-9) for p,q in zip(a,b)])
        diffs=np.array([p["train_lower_pct"]-q["train_lower_pct"] for p,q in zip(a,b)])
        d={"median_paired_speedup":float(np.median(times)),
           "total_time_speedup":float(sum(q["total_selection_seconds_including_scan"] for q in b)/max(sum(p["total_selection_seconds_including_scan"] for p in a),1e-9)),
           "faster_instances":int((times>1).sum()),
           "mean_training_incumbent_difference_pp":float(diffs.mean()),
           "training_incumbent_wins":int((diffs>1e-5).sum()),"training_incumbent_losses":int((diffs<-1e-5).sum()),
           "both_certified_instances":sum(p["within_tolerance"] and q["within_tolerance"] for p,q in zip(a,b))}
        if all("test" in v for v in a+b):
            t=np.array([p["test"]["mean_gain_pct"]-q["test"]["mean_gain_pct"] for p,q in zip(a,b)])
            d.update({"mean_test_difference_pp":float(t.mean()),"test_wins":int((t>1e-5).sum()),
                      "test_ties":int((abs(t)<=1e-5).sum()),"test_losses":int((t<-1e-5).sum())})
        out["paired"][name]=d
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("directory");p.add_argument("--output",default="results/adaptive-summary.json");a=p.parse_args()
    groups={k:[] for k in ("unit","heterogeneous","published_preinstalled","hypothetical_module")}
    errors=[];skips=[];files=[];seen=set()
    for f in sorted(Path(a.directory).rglob("*.json")):
        j=json.loads(f.read_text());files.append(f.name);errors.extend(j["errors"])
        if j["mode"]=="published":skips.extend(j["skipped"])
        for r in j["results"]:
            key=(r["capacity"],r["network"])
            if key in seen:raise ValueError("Duplicate "+str(key))
            seen.add(key);groups[key[0]].append(r)
    report={"files":files,"errors":errors,"published_skipped":skips,
            "regimes":{k:describe(v) for k,v in groups.items()},
            "same_topology_set":{r["network"] for r in groups["unit"]}=={r["network"] for r in groups["heterogeneous"]}}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))
    if errors or any(len(groups[k])!=120 for k in ("unit","heterogeneous")) or not report["same_topology_set"] or not (groups["published_preinstalled"] or groups["hypothetical_module"]):
        raise SystemExit("Incomplete benchmark")

if __name__=="__main__":main()
