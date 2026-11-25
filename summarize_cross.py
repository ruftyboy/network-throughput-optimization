"""Summarize paired cross-scenario performance; keep workloads separate."""
import argparse,json
from pathlib import Path
import numpy as np

def describe(rows):
    if not rows:return {"instances":0}
    out={"instances":len(rows),"heldout_scenarios":sum(r["test_count"] for r in rows),"methods":{}}
    for name in ("within","cross"):
        a=[r["methods"][name] for r in rows]
        d={"baseline_calls":sum(v["baseline_lp_calls"] for v in a),
           "baseline_skips":sum(v["baseline_skips"] for v in a),
           "candidate_calls":sum(v["candidate_lp_calls"] for v in a),
           "candidate_skips":sum(v["candidate_skips"] for v in a),
           "replay_calls":sum(v["replay_lp_calls"] for v in a),
           "total_routing_calls":sum(v["total_lp_calls"] for v in a),
           "total_selection_seconds":sum(v["total_selection_seconds"] for v in a),
           "total_verified_seconds":sum(v["total_verified_seconds"] for v in a),
           "certified_within_0_5pp":sum(v["within_tolerance"] for v in a),
           "certificates_withheld":sum(v["certificate_withheld"] for v in a),
           "numerical_incident_instances":sum(bool(v["numerical_incidents"]) for v in a),
           "mean_training_gap_pp":float(np.mean([v["train_gap_pct_points"] for v in a])),
           "mean_training_gain_pct":float(np.mean([v["train"]["mean_gain_pct"] for v in a])),
           "mean_test_gain_pct":float(np.mean([v["test"]["mean_gain_pct"] for v in a])),
           "mean_minimum_test_gain_pct":float(np.mean([v["test"]["minimum_gain_pct"] for v in a]))}
        if name=="cross":
            d.update({"certificate_transfer_seconds":sum(v["transfer_seconds"] for v in a),
                      "template_fallback_pairs":sum(v["template_fallback_pairs"] for v in a),
                      "template_pairs":sum(v["template_pairs"] for v in a)})
        out["methods"][name]=d
    a=[r["methods"]["cross"] for r in rows];b=[r["methods"]["within"] for r in rows]
    speed=np.array([q["total_verified_seconds"]/max(v["total_verified_seconds"],1e-9) for v,q in zip(a,b)])
    train=np.array([v["train"]["mean_gain_pct"]-q["train"]["mean_gain_pct"] for v,q in zip(a,b)])
    test=np.array([v["test"]["mean_gain_pct"]-q["test"]["mean_gain_pct"] for v,q in zip(a,b)])
    x=out["methods"]["cross"];y=out["methods"]["within"]
    out["paired"]={"selection_speedup":y["total_selection_seconds"]/max(x["total_selection_seconds"],1e-9),
                   "verified_speedup":y["total_verified_seconds"]/max(x["total_verified_seconds"],1e-9),
                   "median_paired_verified_speedup":float(np.median(speed)),
                   "faster_verified_instances":int((speed>1).sum()),"verified_10x_instances":int((speed>=10).sum()),
                   "routing_call_reduction_including_replay":1-x["total_routing_calls"]/max(y["total_routing_calls"],1),
                   "baseline_skip_fraction":x["baseline_skips"]/max(x["baseline_calls"]+x["baseline_skips"],1),
                   "training_losses_over_0_5pp":int((train<-.5).sum()),
                   "mean_training_difference_pp":float(train.mean()),"mean_test_difference_pp":float(test.mean()),
                   "test_wins":int((test>1e-6).sum()),"test_ties":int((abs(test)<=1e-6).sum()),"test_losses":int((test<-1e-6).sum())}
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("directory");p.add_argument("--output",default="results/cross-summary.json");a=p.parse_args()
    groups={k:[] for k in ("unit_independent","unit_correlated","heterogeneous_independent","heterogeneous_correlated","historical_modeled_historical")}
    errors=[];files=[];seen=set()
    for f in sorted(Path(a.directory).rglob("*.json")):
        j=json.loads(f.read_text());files.append(f.name);errors.extend(j["errors"])
        for r in j["results"]:
            group=r["capacity"]+"_"+r["workload"];key=(group,r["network"])
            if key in seen:raise ValueError("duplicate "+str(key))
            seen.add(key);groups[group].append(r)
    synthetic=[{r["network"] for r in v} for k,v in groups.items() if k!="historical_modeled_historical"]
    report={"files":files,"errors":errors,"regimes":{k:describe(v) for k,v in groups.items()},
            "same_topology_sets":all(v==synthetic[0] for v in synthetic)}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))
    if errors or any(len(v)!=(1 if k=="historical_modeled_historical" else 120) for k,v in groups.items()) or not report["same_topology_sets"]:
        raise SystemExit("Incomplete cross-scenario benchmark")

if __name__=="__main__":main()
