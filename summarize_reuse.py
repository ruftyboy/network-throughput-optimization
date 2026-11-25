"""Paired routing reuse benchmark summary."""
import argparse,json
from pathlib import Path
import numpy as np

def describe(rows):
    if not rows:return {"instances":0}
    out={"instances":len(rows),"heldout_scenarios":sum(r["test_count"] for r in rows),"methods":{},"paired":{}}
    for name in ("full","reuse","lazy"):
        a=[r["methods"][name] for r in rows]
        calls=sum(v["routing_lp_calls"] for v in a)
        skips=sum(v["routing_calls_skipped"] for v in a)
        d={"candidate_routing_calls":calls,"candidate_calls_skipped":skips,
           "candidate_skip_fraction":skips/max(1,calls+skips),
           "total_calls_including_scan_replay":sum(v["total_routing_calls_including_scan_replay"] for v in a),
           "total_plan_queries":sum(v["plan_queries"] for v in a),
           "total_early_rejections":sum(v["early_rejected_queries"] for v in a),
           "total_integer_master_calls":sum(v["master_calls"] for v in a),
           "total_selection_seconds":sum(v["total_selection_seconds"] for v in a),
           "total_verified_seconds":sum(v["total_verified_seconds"] for v in a),
           "total_routing_seconds":sum(v["routing_seconds"] for v in a),
           "total_bound_seconds":sum(v["bound_seconds"] for v in a),
           "total_master_seconds":sum(v["master_seconds"] for v in a),
           "certified_within_0_5pp":sum(v["within_tolerance"] for v in a),
           "certificates_withheld":sum(v["certificate_withheld"] for v in a),
           "instances_with_numerical_incidents":sum(bool(v["numerical_incidents"]) for v in a),
           "mean_training_gap_pp":float(np.mean([v["train_gap_pct_points"] for v in a])),
           "mean_verified_training_gain_pct":float(np.mean([v["train"]["mean_gain_pct"] for v in a])),
           "mean_test_gain_pct":float(np.mean([v["test"]["mean_gain_pct"] for v in a])),
           "mean_minimum_test_gain_pct":float(np.mean([v["test"]["minimum_gain_pct"] for v in a]))}
        out["methods"][name]=d
    for name in ("reuse","lazy"):
        a=[r["methods"][name] for r in rows];b=[r["methods"]["full"] for r in rows]
        ratios=np.array([q["total_verified_seconds"]/max(v["total_verified_seconds"],1e-9) for v,q in zip(a,b)])
        tr=np.array([v["train"]["mean_gain_pct"]-q["train"]["mean_gain_pct"] for v,q in zip(a,b)])
        te=np.array([v["test"]["mean_gain_pct"]-q["test"]["mean_gain_pct"] for v,q in zip(a,b)])
        d={"selection_speedup":out["methods"]["full"]["total_selection_seconds"]/max(out["methods"][name]["total_selection_seconds"],1e-9),
           "verified_speedup":out["methods"]["full"]["total_verified_seconds"]/max(out["methods"][name]["total_verified_seconds"],1e-9),
           "median_verified_paired_speedup":float(np.median(ratios)),
           "verified_faster_instances":int((ratios>1).sum()),"verified_10x_faster_instances":int((ratios>=10).sum()),
           "total_routing_call_reduction_fraction":1-out["methods"][name]["total_calls_including_scan_replay"]/max(1,out["methods"]["full"]["total_calls_including_scan_replay"]),
           "training_losses_over_0_5pp":int((tr<-.5).sum()),
           "mean_training_gain_difference_pp":float(tr.mean()),"mean_test_gain_difference_pp":float(te.mean()),
           "test_wins":int((te>1e-6).sum()),"test_ties":int((abs(te)<=1e-6).sum()),"test_losses":int((te<-1e-6).sum())}
        out["paired"][name]=d
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("directory");p.add_argument("--output",default="results/reuse-summary.json");a=p.parse_args()
    groups={k:[] for k in ("unit","heterogeneous","historical_modeled")}
    errors=[];files=[];seen=set()
    for f in sorted(Path(a.directory).rglob("*.json")):
        j=json.loads(f.read_text());files.append(f.name);errors.extend(j["errors"])
        for r in j["results"]:
            key=(r["capacity"],r["network"])
            if key in seen:raise ValueError("duplicate "+str(key))
            seen.add(key);groups[key[0]].append(r)
    report={"files":files,"errors":errors,"regimes":{k:describe(v) for k,v in groups.items()},
            "same_topology_set":{r["network"] for r in groups["unit"]}=={r["network"] for r in groups["heterogeneous"]}}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))
    if errors or any(len(groups[k])!=120 for k in ("unit","heterogeneous")) or len(groups["historical_modeled"])!=1 or not report["same_topology_set"]:
        raise SystemExit("Incomplete routing reuse benchmark")

if __name__=="__main__":main()
