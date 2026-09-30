"""Static heterogeneous graph baselines (R-GCN, HGT); saves validation/test predictions per seed."""
import argparse, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from opal import data
from opal.baselines import train_graph_baseline
p = argparse.ArgumentParser(); p.add_argument("--dataset", default="oulad")
p.add_argument("--models", default="R-GCN,HGT"); p.add_argument("--seeds", default="0,1,2,3,4")
p.add_argument("--threads", type=int, default=2)
p.add_argument("--tag", default=""); p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--wd", type=float, default=1e-4); p.add_argument("--drop", type=float, default=0.1)
p.add_argument("--patience", type=int, default=6)
a = p.parse_args(); torch.set_num_threads(a.threads)
cs = data.load(a.dataset); H = len(cs[0].horizons)
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", a.dataset)
for kind in a.models.split(","):
    for seed in [int(s) for s in a.seeds.split(",")]:
        out = os.path.join(root, kind + a.tag, f"seed{seed}.pkl")
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t0 = time.time()
        model, run = train_graph_baseline(kind, cs, seed=seed, verbose=True, lr=a.lr, wd=a.wd, drop=a.drop,
                                          patience=a.patience, epochs=60)
        tr = time.time() - t0
        res = {"val": {}, "test": {}}
        for split, key in [(1, "val"), (2, "test")]:
            t1 = time.time(); o = run(split); dt = time.time() - t1
            for h in range(H):
                y = np.concatenate(o[h][0]); lg = np.concatenate(o[h][1])
                groups = {}
                names = []
                for ci, c in enumerate(cs):
                    sel = ((c.split == split) & c.eligible[:, h]).numpy()
                    if sel.any():
                        names += [c.name] * int(sel.sum())
                        for g, v in c.groups.items():
                            groups.setdefault(g, []).append(np.asarray(v)[sel])
                res[key][h] = {"y": y, "logit": lg, "groups": {g: np.concatenate(v) for g, v in groups.items()}, "cohort": np.array(names)}
            if key == "test":
                res["infer_sec"] = dt
        res["train_sec"] = tr; res["n_params"] = sum(p.numel() for p in model.parameters())
        pickle.dump(res, open(out, "wb"))
        from sklearn.metrics import roc_auc_score
        print(kind, seed, [round(roc_auc_score(res["test"][h]["y"], res["test"][h]["logit"]), 4) for h in range(H)], round(tr, 1), flush=True)
