"""List the runs that the article needs and report which ones are missing or unfinished."""
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
S5, S3 = range(5), range(3)
NEED = {("oulad", m): S5 for m in ["opal", "GRU", "Transformer", "R-GCN", "HGT", "LightGBM", "XGBoost", "RF", "HGB+Platt", "MLP"]}
NEED.update({("act-mooc", m): S5 for m in ["opal", "GRU", "Transformer", "R-GCN", "HGT", "LightGBM", "XGBoost", "RF", "HGB+Platt", "MLP"]})
NEED.update({("oulad", a): S3 for a in ["abl_no_graph", "abl_no_coevo", "abl_no_rec", "abl_no_peer", "abl_no_featpath", "abl_no_assess"]})
NEED.update({("oulad_lag14", "opal"): S3, ("oulad_lag14", "LightGBM"): S5, ("oulad", "LR"): [0], ("act-mooc", "LR"): [0]})
REC = {ds: [f"{m}_seed{s}" for m in ["SASRec", "GRU4Rec", "BPR-MF", "LightGCN"] for s in S3] + ["heur_seed0"] for ds in ("oulad", "act-mooc")}
missing = []
for (ds, m), seeds in sorted(NEED.items()):
    for s in seeds:
        if not os.path.exists(os.path.join(ROOT, ds, m, f"seed{s}.pkl")):
            ck = os.path.exists(os.path.join(ROOT, ds, m, f"seed{s}.ckpt"))
            missing.append(f"{ds}/{m}/seed{s}" + (" (checkpoint)" if ck else ""))
for ds, names in REC.items():
    for n in names:
        for ext in (".json", "_per.npz"):
            if not os.path.exists(os.path.join(ROOT, ds, "rec", n + ext)):
                missing.append(f"{ds}/rec/{n}{ext}")
for f in ("oulad/protocol_inflation.json", "val_variant_hybrid.json", "dataset_stats.json"):
    if not os.path.exists(os.path.join(ROOT, f)):
        missing.append(f)
print("complete" if not missing else f"{len(missing)} missing:\n  " + "\n  ".join(missing))
