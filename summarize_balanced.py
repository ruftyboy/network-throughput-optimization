"""Descriptive, paired comparisons; never silently omit failures."""
import argparse
import json
from pathlib import Path
import numpy as np

def describe(rows):
    if not rows:
        return {"networks":0}
    names=sorted(set.intersection(*(set(r["plans"]) for r in rows)))
    out={"networks":len(rows),"holdout_scenarios":sum(r["test_count"] for r in rows),"methods":{}}
    for name in names:
        tests=[r["plans"][name]["test"] for r in rows]
        out["methods"][name]={
            "mean_gain_pct":float(np.mean([v["mean_gain_pct"] for v in tests])),
            "mean_network_min_gain_pct":float(np.mean([v["min_gain_pct"] for v in tests])),
            "mean_balanced_gain_pct":float(np.mean([v["balanced_gain_pct"] for v in tests])),
            "networks_all_positive":sum(v["min_gain_pct"]>1e-6 for v in tests),
            "mean_fraction_positive":float(np.mean([v["fraction_positive"] for v in tests])),
        }
        if name in rows[0]["optimizers"]:
            optim=[r["optimizers"][name] for r in rows]
            out["methods"][name].update({
                "closed_training_gaps":sum(o["closed_gap"] for o in optim),
                "mean_training_gap_ratio":float(np.mean([o["upper_bound"]-o["objective"] for o in optim])),
                "median_optimizer_seconds":float(np.median([o["seconds"] for o in optim])),
                "median_selection_seconds_with_shared_seeds":float(np.median([o["seconds"]+r["shared_seed_seconds"] for o,r in zip(optim,rows)])),
            })
    out["balanced_paired_comparisons"]={}
    for name in names:
        if name=="balanced":
            continue
        comparisons={}
        for key in ("mean_gain_pct","min_gain_pct","balanced_gain_pct"):
            d=np.array([r["plans"]["balanced"]["test"][key]-r["plans"][name]["test"][key] for r in rows])
            comparisons[key]={"mean_difference_percentage_points":float(d.mean()),"wins":int((d>1e-6).sum()),
                              "ties":int((abs(d)<=1e-6).sum()),"losses":int((d<-1e-6).sum())}
        out["balanced_paired_comparisons"][name]=comparisons
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--output",default="results/aggregate.json")
    args=ap.parse_args()
    groups={"zoo":[],"traces":[]};errors=[];seen=set();files=[]
    for path in sorted(Path(args.directory).rglob("*.json")):
        d=json.loads(path.read_text())
        mode=d["mode"]
        if mode not in groups:
            raise ValueError("Unexpected artifact mode: "+mode)
        files.append(path.name)
        errors.extend(d["errors"])
        for row in d["results"]:
            key=(mode,row["network"])
            if key in seen:
                raise ValueError("Duplicate network: "+str(key))
            seen.add(key);groups[mode].append(row)
    report={"files":files,"errors":errors,"synthetic":describe(groups["zoo"]),
            "historical_exploratory":describe(groups["traces"]),
            "historical_networks":[{"network":r["network"],"assumed_module_links":r["assumed_module_links"],
                                    "plans":r["plans"],"optimizers":r["optimizers"]} for r in groups["traces"]]}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!="historical_networks"},indent=2))
    if errors or len(groups["zoo"])!=120 or len(groups["traces"])!=3:
        raise SystemExit("Incomplete results: see aggregate.json")

if __name__=="__main__":
    main()
