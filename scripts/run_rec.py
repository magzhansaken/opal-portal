"""Recommendation baselines on test learners at each horizon; saves ranking metrics per seed."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from opal import data
from opal.baselines import rec_heuristics, fit_cf, fit_next_step, history
from opal.metrics import ranking_metrics
p = argparse.ArgumentParser(); p.add_argument("--dataset", default="oulad")
p.add_argument("--models", default="heur,BPR-MF,LightGCN,GRU4Rec,SASRec"); p.add_argument("--seeds", default="0,1,2,3,4")
p.add_argument("--threads", type=int, default=2)
a = p.parse_args(); torch.set_num_threads(a.threads)
cs = data.load(a.dataset); H = len(cs[0].horizons)
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", a.dataset, "rec")
os.makedirs(root, exist_ok=True)


def per_learner(parts):
    """Per-learner NDCG@10 (all and new resources) with (cohort, learner index) keys, for paired tests."""
    out = {}
    for mode in ["all", "new"]:
        vals, keys = [], []
        for sc, tg, se, cname, idx in parts:
            r = ranking_metrics(sc, tg, seen=(se if mode == "new" else None), per=True)
            vals.append(r["per_ndcg10"]); keys += [f"{cname}:{int(i)}" for i in np.asarray(idx)[r["kept"]]]
        out[mode] = (np.concatenate(vals) if vals else np.zeros(0), np.array(keys))
    return out


def pooled(parts):
    """Pool per-cohort ranking metrics weighted by number of evaluated learners."""
    parts = [p[:3] for p in parts]
    out = {}
    for mode in ["all", "new"]:
        rs = [ranking_metrics(sc, tg, seen=(se if mode == "new" else None)) for sc, tg, se in parts]
        n = sum(r["n"] for r in rs)
        out[mode] = {k: sum(r.get(k, 0) * r["n"] for r in rs) / max(n, 1) for k in rs[0] if k != "n"}
        out[mode]["n"] = n
    return out


def eval_sets(c, h):
    sel = (c.split == 2) & c.eligible[:, h]
    idx = torch.nonzero(sel).squeeze(1)
    tgt = (c.next_items[h].to_dense()[idx] > 0).numpy()
    st = c.state_idx[:, h]
    cum, _, _ = history(c, st)
    seen = (cum[idx] > 0).numpy()
    return idx, tgt, seen


for model in a.models.split(","):
    seeds = [0] if model == "heur" else [int(s) for s in a.seeds.split(",")]
    for seed in seeds:
        out = os.path.join(root, f"{model}_seed{seed}.json")
        out_per = out.replace(".json", "_per.npz")
        if os.path.exists(out) and os.path.exists(out_per):
            continue
        lock = out + ".lock"          # another process is already fitting this model and seed
        if os.path.exists(lock) and os.path.exists(f"/proc/{open(lock).read().strip()}"):
            print(f"skip {model} {seed}: running in process {open(lock).read().strip()}", flush=True)
            continue
        open(lock, "w").write(str(os.getpid()))
        t0 = time.time(); res = {}
        parts = {}
        for c in cs:
            if not (c.split == 2).any():
                continue
            for h in range(H):
                idx, tgt, seen = eval_sets(c, h)
                if model == "heur":
                    sc = rec_heuristics(c, h, idx)
                    for k, v in sc.items():
                        parts.setdefault((k, h), []).append((v, tgt, seen, c.name, idx.numpy()))
                elif model in ("BPR-MF", "LightGCN"):
                    s = fit_cf(model, c, h, seed=seed)
                    parts.setdefault((model, h), []).append((s[idx.numpy()], tgt, seen, c.name, idx.numpy()))
                else:
                    s = fit_next_step("gru" if model == "GRU4Rec" else "sasrec", c, h, seed=seed)
                    parts.setdefault((model, h), []).append((s[idx.numpy()], tgt, seen, c.name, idx.numpy()))
        for (name, h), pr in parts.items():
            res.setdefault(name, {})[h] = pooled(pr)
        res["_sec"] = time.time() - t0
        json.dump(res, open(out, "w"), indent=1)
        per = {}
        for (name, h), pr in parts.items():
            for mode, (v, keys) in per_learner(pr).items():
                per[f"{name}|{h}|{mode}|ndcg"] = v; per[f"{name}|{h}|{mode}|key"] = keys
        np.savez_compressed(out_per, **per)
        os.remove(lock)
        print(model, seed, round(res["_sec"], 1), {n: round(res[n][1]["all"]["NDCG@10"], 4) for n in res if n != "_sec"}, flush=True)
