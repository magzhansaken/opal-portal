"""Dataset statistics for Table 1 (Section 4.1) (writes results/dataset_stats.json and tables/datasets.tex)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from opal import data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
cs = data.load("oulad")
info = pd.read_parquet(os.path.join(data.DATA, "oulad", "studentInfo.parquet"))
sass = pd.read_parquet(os.path.join(data.DATA, "oulad", "studentAssessment.parquet"))
o = {"presentations": len(cs), "modules": len({c.name.split('-')[0] for c in cs}), "enrollments_raw": int(len(info)),
     "enrollments": int(sum(c.L for c in cs)), "resources": int(sum(c.R for c in cs)),
     "weeks_min": int(min(c.S for c in cs)), "weeks_max": int(max(c.S for c in cs)),
     "edges": int(sum(sum(a._nnz() for a in c.A) for c in cs)),
     "clicks": float(sum(sum(float(np.expm1(a.values()).sum()) for a in c.A) for c in cs)),
     "assess_rows": int(len(sass)), "x_dim": int(cs[0].x_seq.shape[-1]), "z_dim": int(cs[0].static_l.shape[1]),
     "d_dim": int(cs[0].static_r.shape[1])}
for sp, name in [(0, "train"), (1, "val"), (2, "test")]:
    n = sum(int((c.split == sp).sum()) for c in cs); pos = sum(float(c.y[c.split == sp].sum()) for c in cs)
    pres = sorted({c.name.split('-')[1] for c in cs if (c.split == sp).any()})
    elig = [sum(int(((c.split == sp) & c.eligible[:, h]).sum()) for c in cs) for h in range(len(cs[0].horizons))]
    o[name] = {"n": n, "pos_rate": pos / n, "presentations": pres, "n_cohorts": sum(1 for c in cs if (c.split == sp).any()), "eligible": elig}
out["oulad"] = o
m = data.load("act-mooc")[0]
a = {"users": m.L, "items": m.R, "days": m.S - 1, "edges": int(sum(x._nnz() for x in m.A)),
     "actions": float(sum(float(np.expm1(x.values()).sum()) for x in m.A)), "pos_rate": float(m.y.mean()),
     "x_dim": int(m.x_seq.shape[-1])}
for sp, name in [(0, "train"), (1, "val"), (2, "test")]:
    sel = m.split == sp
    a[name] = {"n": int(sel.sum()), "pos_rate": float(m.y[sel].mean()),
               "eligible": [int((sel & m.eligible[:, h]).sum()) for h in range(len(m.horizons))]}
out["act-mooc"] = a
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "dataset_stats.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
