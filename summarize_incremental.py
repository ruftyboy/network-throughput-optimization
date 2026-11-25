"""Paired warm-start, reset, cold and screening comparisons."""
import argparse,json
from pathlib import Path
import numpy as np
NAMES=("scipy_reuse","native_cold","persistent_reset","persistent_warm","warm_no_screen")

def describe(rows):
    if not rows:return {"instances":0}
    out={"instances":len(rows),"heldout_scenarios":sum(r["test_count"] for r in rows),"methods":{},"paired":{}}
    for name in NAMES:
        a=[r["methods"][name] for r in rows]
        sums=("model_builds","model_updates","coefficient_changes","basis_attempts","basis_rejections",
              "fallbacks","scipy_fallbacks","native_runs","simplex_iterations","engine_update_seconds",
              "engine_solve_seconds","routing_lp_calls","routing_calls_skipped","total_lp_calls",
              "baseline_seconds","total_selection_seconds","total_verified_seconds")
        d={k:sum(v[k] for v in a) for k in sums}
        d.update({"baseline_simplex_iterations":sum(v["baseline_engine_stats"]["simplex_iterations"] for v in a),
                  "baseline_native_runs":sum(v["baseline_engine_stats"]["native_runs"] for v in a),
                  "baseline_fallbacks":sum(v["baseline_engine_stats"]["fallbacks"] for v in a),
                  "certified_within_0_5pp":sum(v["within_tolerance"] for v in a),
                  "certificates_withheld":sum(v["certificate_withheld"] for v in a),
                  "numerical_incident_instances":sum(bool(v["numerical_incidents"]) for v in a),
                  "mean_training_gap_pp":float(np.mean([v["train_gap_pct_points"] for v in a])),
                  "mean_training_gain_pct":float(np.mean([v["train"]["mean_gain_pct"] for v in a])),
                  "mean_test_gain_pct":float(np.mean([v["test"]["mean_gain_pct"] for v in a])),
                  "mean_minimum_test_gain_pct":float(np.mean([v["test"]["minimum_gain_pct"] for v in a]))})
        out["methods"][name]=d
    for name in ("scipy_reuse","native_cold","persistent_reset","warm_no_screen"):
        a=[r["methods"]["persistent_warm"] for r in rows];b=[r["methods"][name] for r in rows]
        x=out["methods"]["persistent_warm"];y=out["methods"][name]
        ratio=np.array([q["total_verified_seconds"]/max(v["total_verified_seconds"],1e-9) for v,q in zip(a,b)])
        train=np.array([v["train"]["mean_gain_pct"]-q["train"]["mean_gain_pct"] for v,q in zip(a,b)])
        test=np.array([v["test"]["mean_gain_pct"]-q["test"]["mean_gain_pct"] for v,q in zip(a,b)])
        out["paired"]["warm_vs_"+name]={"verified_speedup":y["total_verified_seconds"]/max(x["total_verified_seconds"],1e-9),
            "selection_speedup":y["total_selection_seconds"]/max(x["total_selection_seconds"],1e-9),
            "baseline_scan_speedup":y["baseline_seconds"]/max(x["baseline_seconds"],1e-9),
            "median_verified_paired_speedup":float(np.median(ratio)),
            "verified_faster_instances":int((ratio>1).sum()),"verified_10x_instances":int((ratio>=10).sum()),
            "baseline_iteration_reduction":None if not y["baseline_simplex_iterations"] else 1-x["baseline_simplex_iterations"]/y["baseline_simplex_iterations"],
            "training_losses_over_0_5pp":int((train<-.5).sum()),
            "mean_training_difference_pp":float(train.mean()),"mean_test_difference_pp":float(test.mean()),
            "test_wins":int((test>1e-6).sum()),"test_ties":int((abs(test)<=1e-6).sum()),"test_losses":int((test<-1e-6).sum())}
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("directory");p.add_argument("--output",default="results/incremental-summary.json");a=p.parse_args()
    groups={k:[] for k in ("unit_independent","unit_correlated","heterogeneous_independent","heterogeneous_correlated","historical_modeled_historical")}
    errors=[];files=[];seen=set()
    for f in sorted(Path(a.directory).rglob("*.json")):
        j=json.loads(f.read_text());files.append(f.name);errors.extend(j["errors"])
        for r in j["results"]:
            group=r["capacity"]+"_"+r["workload"];key=(group,r["network"])
            if key in seen:raise ValueError("duplicate "+str(key))
            seen.add(key);groups[group].append(r)
    sets=[{r["network"] for r in v} for k,v in groups.items() if k!="historical_modeled_historical"]
    report={"files":files,"errors":errors,"regimes":{k:describe(v) for k,v in groups.items()},
            "same_topology_sets":all(v==sets[0] for v in sets)}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))
    if errors or any(len(v)!=(1 if k=="historical_modeled_historical" else 120) for k,v in groups.items()) or not report["same_topology_sets"]:
        raise SystemExit("Incomplete incremental benchmark")

if __name__=="__main__":main()
