"""Validation-only comparison of the recommendation-head variant (rec_hybrid) with the base head.

The selection score is the one used for early stopping: mean validation AUC plus half the mean validation
NDCG@10, taken at the best epoch.  Writes results/val_variant_hybrid.json (used in Supplementary Section S3).
"""
import glob
import json
import os
import pickle

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def best_epoch(path):
    hist = pickle.load(open(path, "rb"))["hist"]
    best, score = None, -1e9
    for h in hist:
        s = h["val_auc"] + 0.5 * h["val_ndcg10"]
        if s > score + 1e-4:          # same rule as in training
            best, score = h, s
    return [best["val_auc"], best["val_ndcg10"], score]


out = {}
for ds in ("oulad", "act-mooc"):
    base = np.array([best_epoch(f) for f in sorted(glob.glob(os.path.join(ROOT, ds, "opal", "seed*.pkl")))])
    hyb = os.path.join(ROOT, ds, "val_hybrid", "seed0.pkl")
    if len(base) and os.path.exists(hyb):
        out[ds] = {"base_seed0": base[0].tolist(), "base_mean": base.mean(0).tolist(), "base_sd": base.std(0).tolist(),
                   "hybrid_seed0": best_epoch(hyb)}
        print(ds, out[ds])
json.dump(out, open(os.path.join(ROOT, "val_variant_hybrid.json"), "w"), indent=1)

# validation scores of the full model and the ablation variants on the same seeds (Section 4.4, Table 4)
abl = {}
for name in ("opal", "abl_no_graph", "abl_no_coevo", "abl_no_peer", "abl_no_featpath", "abl_no_assess"):
    fs = sorted(glob.glob(os.path.join(ROOT, "oulad", name, "seed*.pkl")))[:3]
    if fs:
        abl[name] = [best_epoch(f)[2] for f in fs]
json.dump(abl, open(os.path.join(ROOT, "oulad", "ablation_val_scores.json"), "w"), indent=1)
print({k: round(float(np.mean(v)), 4) for k, v in abl.items()})
