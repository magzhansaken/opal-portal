"""Tabular baselines per horizon, 5 seeds for stochastic models; saves validation/test predictions."""
import argparse, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from opal import data
from opal.baselines import fit_tabular, tab_split
p = argparse.ArgumentParser(); p.add_argument("--dataset", default="oulad")
p.add_argument("--models", default="LR,RF,XGBoost,LightGBM,HGB+Platt,MLP"); p.add_argument("--seeds", default="0,1,2,3,4")
p.add_argument("--peer", action="store_true")
a = p.parse_args()
cs = data.load(a.dataset)
H = len(cs[0].horizons)
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", a.dataset)
for kind in a.models.split(","):
    seeds = [0] if kind == "LR" else [int(s) for s in a.seeds.split(",")]
    for seed in seeds:
        out = os.path.join(root, kind + ("+peer" if a.peer else ""), f"seed{seed}.pkl")
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        res = {"val": {}, "test": {}}; t0 = time.time(); tt = 0.0
        for h in range(H):
            Xtr, ytr, _, _ = tab_split(cs, h, 0, a.peer)
            Xva, yva, gva, nva = tab_split(cs, h, 1, a.peer)
            Xte, yte, gte, nte = tab_split(cs, h, 2, a.peer)
            m = fit_tabular(kind, Xtr, ytr, Xva, yva, seed=seed)
            pv = np.clip(m.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6)
            t1 = time.time(); pt = np.clip(m.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6); tt += time.time() - t1
            res["val"][h] = {"y": yva, "logit": np.log(pv / (1 - pv)), "groups": gva, "cohort": nva}
            res["test"][h] = {"y": yte, "logit": np.log(pt / (1 - pt)), "groups": gte, "cohort": nte}
        res["train_sec"] = time.time() - t0; res["infer_sec"] = tt
        pickle.dump(res, open(out, "wb"))
        from sklearn.metrics import roc_auc_score
        print(kind, seed, [round(roc_auc_score(res["test"][h]["y"], res["test"][h]["logit"]), 4) for h in range(H)],
              round(res["train_sec"], 1), flush=True)
