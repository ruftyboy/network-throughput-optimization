"""Aggregate matched-policy comparisons, retaining all missing/error counts."""
import argparse,json
from pathlib import Path
import numpy as np

def describe(rows):
    if not rows:return {"topologies":0}
    out={"topologies":len(rows),"held_out_scenarios":sum(r["test_count"] for r in rows),"policies":{},"comparisons":{}}
    for name in rows[0]["records"]:
        a=[r["records"][name] for r in rows]
        out["policies"][name]={
            "mean_test_gain_pct":float(np.mean([v["test"]["mean_gain_pct"] for v in a])),
            "mean_minimum_test_gain_pct":float(np.mean([v["test"]["minimum_gain_pct"] for v in a])),
            "mean_test_fraction_at_least_5pct":float(np.mean([v["test"]["fraction_at_least_5pct"] for v in a])),
            "mean_upgrades":float(np.mean([v["accounting"]["upgrades"] for v in a])),
            "mean_capacity_fraction":float(np.mean([v["accounting"]["added_capacity_fraction"] for v in a])),
            "mean_modeled_cost":None if a[0]["accounting"]["modeled_cost"] is None else float(np.mean([v["accounting"]["modeled_cost"] for v in a])),
            "training_closed_gaps":sum(v["closed_gap"] for v in a),
            "mean_training_gap_pct_points":float(np.mean([v["upper_pct"]-v["lower_pct"] for v in a])),
            "mean_retrospective_coverage_ceiling":float(np.mean([v["test_retrospective_threshold_bound"]["upper_count"]/v["test_retrospective_threshold_bound"]["scenarios"] for v in a]))
        }
    for name in ("count","cost"):
        a=[r["comparisons"][name] for r in rows]
        d=np.array([v["test_mean_difference_pct_points"] for v in a])
        out["comparisons"][name]={
            "mean_heldout_difference_pct_points":float(d.mean()),
            "wins":int((d>1e-6).sum()),"ties":int((abs(d)<=1e-6).sum()),"losses":int((d<-1e-6).sum()),
            "certified_strict_training_advantage":sum(v["certified_strict_training_advantage"] for v in a),
            "mean_training_advantage_lower_pct_points":float(np.mean([v["training_advantage_lower_pct_points"] for v in a])),
            "mean_training_advantage_upper_pct_points":float(np.mean([v["training_advantage_upper_pct_points"] for v in a])),
            "largest_heldout_difference_pct_points":float(d.max()),
            "largest_heldout_difference_network":rows[int(np.argmax(d))]["network"]
        }
    out["training_scenarios_with_adjacent_mandatory_bridges"]=sum(r["train_scenarios_adjacent_required_bridges"] for r in rows)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("directory");p.add_argument("--output",default="results/concentration-summary.json");a=p.parse_args()
    groups={"unit":[],"heterogeneous":[]};errors=[];seen=set();files=[]
    for f in sorted(Path(a.directory).rglob("*.json")):
        j=json.loads(f.read_text());files.append(f.name);errors.extend(j["errors"])
        for r in j["results"]:
            key=(r["capacity"],r["network"])
            if key in seen:raise ValueError("Duplicate result "+str(key))
            seen.add(key);groups[key[0]].append(r)
    report={"files":files,"errors":errors,"regimes":{k:describe(v) for k,v in groups.items()},
            "same_topology_set":{r["network"] for r in groups["unit"]}=={r["network"] for r in groups["heterogeneous"]}}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))
    if errors or any(len(v)!=120 for v in groups.values()) or not report["same_topology_set"]:raise SystemExit("Incomplete benchmark")

if __name__=="__main__":main()
