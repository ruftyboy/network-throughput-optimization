"""Aggregate screening efficiency and solution quality; keep regimes separate."""
import argparse,json
from pathlib import Path
import numpy as np

def describe(rows):
    if not rows:return {"instances":0}
    names=("multicut","cardinality_screen","hybrid_screen")
    out={"instances":len(rows),"heldout_scenarios":sum(r["test_count"] for r in rows),"methods":{},"paired":{}}
    for name in names:
        a=[r["methods"][name] for r in rows]
        sizes=[v["mean_master_edge_count"]/v["initial_edges"] for v in a if v["mean_master_edge_count"] is not None]
        first=[v["first_master_edge_count"]/v["initial_edges"] for v in a if v["first_master_edge_count"] is not None]
        d={"mean_final_fraction_removed":float(np.mean([v["fraction_edges_removed"] for v in a])),
           "mean_fraction_removed_before_first_master":None if not first else float(1-np.mean(first)),
           "mean_fraction_removed_across_master_calls_per_instance":None if not sizes else float(1-np.mean(sizes)),
           "instances_with_no_integer_master":sum(v["first_master_edge_count"] is None for v in a),
           "instances_with_final_removal_at_least_80pct":sum(v["fraction_edges_removed"]>=.8 for v in a),
           "certified_within_0_5pp":sum(v["within_tolerance"] for v in a),
           "certificates_withheld":sum(v["certificate_withheld"] for v in a),
           "instances_with_numerical_incidents":sum(bool(v["numerical_incidents"]) for v in a),
           "mean_training_gap_pp":float(np.mean([v["train_gap_pct_points"] for v in a])),
           "mean_training_gain_pct":float(np.mean([v["train_lower_pct"] for v in a])),
           "median_total_selection_seconds":float(np.median([v["total_selection_seconds"] for v in a])),
           "total_selection_seconds":float(sum(v["total_selection_seconds"] for v in a)),
           "total_screening_seconds":float(sum(v["screening_seconds"] for v in a)),
           "mean_routing_lp_calls":float(np.mean([v["routing_lp_calls"] for v in a])),
           "mean_integer_master_calls":float(np.mean([v["master_calls"] for v in a])),
           "mean_screening_lp_calls":float(np.mean([v["screening_lp_calls"] for v in a]))}
        if all("test" in v for v in a):
            d.update({"mean_test_gain_pct":float(np.mean([v["test"]["mean_gain_pct"] for v in a])),
                      "mean_minimum_test_gain_pct":float(np.mean([v["test"]["minimum_gain_pct"] for v in a]))})
        out["methods"][name]=d
    for name in names[1:]:
        a=[r["methods"][name] for r in rows];b=[r["methods"]["multicut"] for r in rows]
        ratios=np.array([q["total_selection_seconds"]/max(v["total_selection_seconds"],1e-9) for v,q in zip(a,b)])
        diff=np.array([v["train_lower_pct"]-q["train_lower_pct"] for v,q in zip(a,b)])
        d={"total_time_speedup":out["methods"]["multicut"]["total_selection_seconds"]/max(out["methods"][name]["total_selection_seconds"],1e-9),
           "median_paired_speedup":float(np.median(ratios)),"faster_instances":int((ratios>1).sum()),
           "at_least_10x_faster_instances":int((ratios>=10).sum()),
           "mean_training_gain_difference_pp":float(diff.mean()),
           "training_losses_over_0_5pp":int((diff<-.5).sum()),
           "both_certified":sum(v["within_tolerance"] and q["within_tolerance"] for v,q in zip(a,b))}
        if all("test" in v for v in a+b):
            delta=np.array([v["test"]["mean_gain_pct"]-q["test"]["mean_gain_pct"] for v,q in zip(a,b)])
            d.update({"mean_test_difference_pp":float(delta.mean()),"test_wins":int((delta>1e-6).sum()),
                      "test_ties":int((abs(delta)<=1e-6).sum()),"test_losses":int((delta<-1e-6).sum())})
        out["paired"][name]=d
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("directory");ap.add_argument("--output",default="results/screening-summary.json");args=ap.parse_args()
    groups={k:[] for k in ("unit","heterogeneous","published_preinstalled","hypothetical_module")}
    errors=[];skips=[];files=[];seen=set()
    for f in sorted(Path(args.directory).rglob("*.json")):
        j=json.loads(f.read_text());files.append(f.name);errors.extend(j["errors"])
        if j["mode"]=="published":skips.extend(j["skipped"])
        for row in j["results"]:
            key=(row["capacity"],row["network"])
            if key in seen:raise ValueError("duplicate "+str(key))
            seen.add(key);groups[key[0]].append(row)
    report={"files":files,"errors":errors,"published_skipped":skips,
            "regimes":{k:describe(v) for k,v in groups.items()},
            "same_topology_set":{r["network"] for r in groups["unit"]}=={r["network"] for r in groups["heterogeneous"]}}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))
    if errors or any(len(groups[k])!=120 for k in ("unit","heterogeneous")) or not report["same_topology_set"] or not(groups["published_preinstalled"] or groups["hypothetical_module"]):
        raise SystemExit("Incomplete screening benchmark")

if __name__=="__main__":main()
