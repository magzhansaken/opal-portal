"""Aggregate all runs of one dataset into results/<dataset>/summary.json.

Risk: AUC, PR-AUC, F1, MCC, Brier, ECE per decision point (F1/MCC at the validation-optimal threshold,
Brier/ECE after temperature scaling fitted on validation), mean +- sd over seeds, and the AUC of the
seed ensemble (mean logit).  Statistics: DeLong tests of OPAL against every baseline on the seed
ensembles with Holm's correction over the decision points, and paired bootstrap CIs (1000 resamples).
Ranking: HR/Recall/NDCG@K for all and for new resources; Wilcoxon signed-rank tests on per-learner
NDCG@10 between OPAL and the strongest recommenders; NDCG@10 of new resources for the learners flagged
by the top-20% alert policy.  Fairness: alert rate/TPR/FPR gaps and ABROCA.  Efficiency: training time,
time to score the test cohorts and parameter counts.

usage: python scripts/analyze.py oulad|act-mooc
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opal.metrics import (abroca, best_f1_threshold, bootstrap_diff, delong_test, fairness_gaps,  # noqa: E402
                          fit_temperature, holm, ranking_metrics, risk_metrics)

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
sig = lambda z: 1 / (1 + np.exp(-np.asarray(z)))  # noqa: E731
RISK_BASELINES = ["LR", "RF", "XGBoost", "LightGBM", "HGB+Platt", "MLP", "GRU", "Transformer", "R-GCN", "HGT"]


def run_files(ds, name):
    return sorted(glob.glob(os.path.join(ROOT, ds, name, "seed*.pkl")))


def load_runs(ds, name):
    """Load the runs of one model without the (large) ranking arrays, which are streamed seed by seed
    in opal_ranking(); only the learner keys of the test ranking are kept."""
    runs = []
    for f in run_files(ds, name):
        r = pickle.load(open(f, "rb"))
        for split in ("val", "test"):
            for h in r[split]:
                parts = r[split][h].pop("rank", None)
                if split == "test" and parts:
                    r[split][h]["has_rank"] = True
        runs.append(r)
    return runs


def risk_table(runs):
    H = len(runs[0]["test"])
    per_seed = []
    for r in runs:
        rows = []
        for h in range(H):
            v, t = r["val"][h], r["test"][h]
            T, b = fit_temperature(v["logit"], v["y"])
            pv, pt = sig(v["logit"] * T + b), sig(t["logit"] * T + b)
            m = risk_metrics(t["y"], pt, best_f1_threshold(v["y"], pv))
            m["ECE_raw"] = risk_metrics(t["y"], sig(t["logit"]))["ECE"]
            rows.append(m)
        per_seed.append(rows)
    out = {str(h): {k: [float(np.mean([s[h][k] for s in per_seed])), float(np.std([s[h][k] for s in per_seed]))]
                    for k in per_seed[0][h]} for h in range(H)}
    for k in per_seed[0][0]:
        vals = [np.mean([s[h][k] for h in range(H)]) for s in per_seed]
        out[f"mean_{k}"] = [float(np.mean(vals)), float(np.std(vals))]
    out["n_seeds"] = len(runs)
    out["train_sec"] = float(np.mean([r.get("train_sec", np.nan) for r in runs]))
    out["infer_sec"] = float(np.mean([r.get("infer_sec", np.nan) for r in runs]))
    out["n_params"] = runs[0].get("n_params")
    return out


def reliability(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [{"n": int((b == i).sum()), "p": float(p[b == i].mean()) if (b == i).any() else None,
             "y": float(y[b == i].mean()) if (b == i).any() else None} for i in range(bins)]


def ens(runs, split, h):
    return np.mean([r[split][h]["logit"] for r in runs], 0)


def calibrated_ensemble(runs, h):
    zv, zt = ens(runs, "val", h), ens(runs, "test", h)
    T, b = fit_temperature(zv, runs[0]["val"][h]["y"])
    return sig(zt * T + b)


def rank_parts_with_keys(run, h, ds):
    """OPAL ranking parts as (scores, targets, seen, keys); keys = 'cohort:learner-index'."""
    parts = run["test"][h].get("rank") or []
    out = []
    need = [p for p in parts if len(p) < 5]
    if need:
        from opal import data
        cs = {c.name: c for c in data.load(ds)}
        names = [n for n in dict.fromkeys(run["test"][h]["cohort"])]
        for p, nm in zip(parts, names):
            c = cs[nm]
            idx = np.nonzero(((c.split == 2) & c.eligible[:, h]).numpy())[0]
            out.append((p[0], p[1], p[2], nm, idx))
        return out
    return parts


def per_learner(parts, mode):
    vals, keys = [], []
    for sc, tg, se, nm, idx in parts:
        r = ranking_metrics(sc, tg, seen=(se if mode == "new" else None), per=True)
        vals.append(r["per_ndcg10"]); keys += [f"{nm}:{int(i)}" for i in np.asarray(idx)[r["kept"]]]
    return dict(zip(keys, np.concatenate(vals))) if vals else {}


def opal_ranking(ds, name, max_seeds=None):
    """Ranking metrics of a graph model, loading one seed at a time to bound the memory use.
    Returns the summary, per-learner NDCG@10 per seed, and the learner keys of the first seed."""
    res, per, keys0, agg = {}, {}, {}, {}
    for si, f in enumerate(run_files(ds, name)[:max_seeds]):
        r = pickle.load(open(f, "rb"))
        for h in range(len(r["test"])):
            agg.setdefault(h, {"all": [], "new": []}); per.setdefault(h, {"all": [], "new": []})
            parts = rank_parts_with_keys(r, h, ds)
            if not parts:
                continue
            if si == 0:
                keys0[h] = [f"{nm}:{int(i)}" for _, _, _, nm, idx in parts for i in idx]
            for mode in agg[h]:
                rs = [ranking_metrics(p[0], p[1], seen=(p[2] if mode == "new" else None)) for p in parts]
                n = sum(x["n"] for x in rs)
                agg[h][mode].append({k: sum(x.get(k, 0) * x["n"] for x in rs) / max(n, 1) for k in rs[0] if k != "n"})
                per[h][mode].append(per_learner(parts, mode))
        del r
    for h, a in agg.items():
        res[str(h)] = {m: {k: [float(np.mean([v[k] for v in vs])), float(np.std([v[k] for v in vs]))] for k in vs[0]}
                       for m, vs in a.items() if vs}
    return res, per, keys0


def rec_baselines(ds):
    out, per = {}, {}
    for f in sorted(glob.glob(os.path.join(ROOT, ds, "rec", "*.json"))):
        d = json.load(open(f))
        for name, hs in d.items():
            if name.startswith("_"):
                continue
            for h, modes in hs.items():
                for mode, mets in modes.items():
                    out.setdefault(name, {}).setdefault(h, {}).setdefault(mode, []).append(mets)
        fp = f.replace(".json", "_per.npz")
        if os.path.exists(fp):
            z = np.load(fp, allow_pickle=True)
            for key in z.files:
                if key.endswith("|ndcg"):
                    name, h, mode, _ = key.split("|")
                    keys = z[key.replace("|ndcg", "|key")]
                    per.setdefault(name, {}).setdefault(int(h), {}).setdefault(mode, []).append(dict(zip(keys, z[key])))
    res = {name: {h: {mode: {k: [float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))]
                             for k in ms[0] if k != "n"} for mode, ms in modes.items()} for h, modes in hs.items()}
           for name, hs in out.items()}
    return res, per


def seed_mean(dicts):
    keys = set.intersection(*[set(d) for d in dicts])
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def main(ds):
    names = sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(os.path.join(ROOT, ds, "*/")))
    names = [n for n in names if n not in ("rec",) and "_grid" not in n and not n.startswith(("val_", "rec_v1"))]
    runs = {n: load_runs(ds, n) for n in names}
    runs = {n: r for n, r in runs.items() if r}
    out = {"risk": {n: risk_table(r) for n, r in runs.items()}}
    H = len(runs["opal"][0]["test"]) if "opal" in runs else 0
    if "opal" in runs:
        yt = {h: runs["opal"][0]["test"][h]["y"] for h in range(H)}
        eo = {h: ens(runs["opal"], "test", h) for h in range(H)}
        same = lambda r: all(len(r[0]["test"][h]["y"]) == len(yt[h]) for h in range(H))  # noqa: E731
        out["ensemble_auc"] = {n: {str(h): float(roc_auc_score(yt[h], ens(r, "test", h))) for h in range(H)}
                               for n, r in runs.items() if same(r)}
        st = {}
        for n in RISK_BASELINES:
            if n not in runs or not same(runs[n]):
                continue
            dl = [delong_test(yt[h], eo[h], ens(runs[n], "test", h)) for h in range(H)]
            adj = holm([d["p"] for d in dl])
            st[n] = {str(h): {**dl[h], "p_holm": adj[h],
                              "boot": bootstrap_diff(yt[h], eo[h], ens(runs[n], "test", h), n=1000)} for h in range(H)}
        out["stats_vs_opal"] = st
        # fairness of the top-20% alert policy and ABROCA
        fair = {}
        for n in ["opal", "LightGBM", "XGBoost", "GRU", "HGB+Platt"]:
            if n not in runs or not runs[n][0]["test"][0]["groups"]:
                continue
            fair[n] = {}
            for h in range(H):
                g = runs[n][0]["test"][h]["groups"]
                p = calibrated_ensemble(runs[n], h)
                fg = fairness_gaps(yt[h], p, g)
                ab = {}
                for att, vals in g.items():
                    vals = np.asarray(vals)
                    keep = np.isin(vals, [v for v in np.unique(vals) if v not in ("unknown",)])
                    ab[att] = abroca(yt[h][keep], p[keep], vals[keep])
                fair[n][str(h)] = {"gaps": fg, "abroca": ab}
        out["fairness"] = fair
        # reliability diagrams (15 equal-width bins) of the calibrated seed ensembles, pooled over decision points
        rel = {}
        for n in ["opal", "LightGBM", "GRU", "HGB+Platt", "XGBoost"]:
            if n not in runs or not same(runs[n]):
                continue
            ps_ = np.concatenate([calibrated_ensemble(runs[n], h) for h in range(H)])
            ys_ = np.concatenate([yt[h] for h in range(H)])
            rel[n] = reliability(ys_, ps_)
            if n == "opal":
                raw = np.concatenate([sig(ens(runs[n], "test", h)) for h in range(H)])
                rel["opal_raw"] = reliability(ys_, raw)
        out["reliability"] = rel
        # ranking
        rk, per_opal, keys0 = opal_ranking(ds, "opal")
        out["opal_rank"] = rk
        rb, per_b = rec_baselines(ds)
        out["rec_baselines"] = rb
        wil, flag = {}, {}
        for h in range(H):
            for mode in ("all", "new"):
                if not per_opal[h][mode]:
                    continue
                po = seed_mean(per_opal[h][mode])
                for b in ("SASRec", "RecentPop", "GRU4Rec", "LightGCN"):
                    if b not in per_b or h not in per_b[b] or mode not in per_b[b][h]:
                        continue
                    pb = seed_mean(per_b[b][h][mode])
                    keys = sorted(set(po) & set(pb))
                    if len(keys) < 20:
                        continue
                    a, c = np.array([po[k] for k in keys]), np.array([pb[k] for k in keys])
                    w = stats.wilcoxon(a, c) if np.any(a != c) else None
                    wil.setdefault(b, {}).setdefault(str(h), {})[mode] = {
                        "n": len(keys), "opal": float(a.mean()), "base": float(c.mean()),
                        "p": float(w.pvalue) if w is not None else 1.0}
            # recommendation quality for learners flagged by OPAL's top-20% alert policy
            risk = ens(runs["opal"], "test", h)
            keys_all = keys0.get(h, [])
            if len(keys_all) == len(risk):
                thr = np.quantile(risk, 0.8)
                flagged = {k for k, p in zip(keys_all, risk) if p >= thr}
                po = seed_mean(per_opal[h]["new"]) if per_opal[h]["new"] else {}
                row = {}
                for label, src in [("OPAL", po)] + [(b, seed_mean(per_b[b][h]["new"])) for b in ("SASRec", "RecentPop")
                                                    if b in per_b and h in per_b[b] and "new" in per_b[b][h]]:
                    f = [v for k, v in src.items() if k in flagged]; o = [v for k, v in src.items() if k not in flagged]
                    row[label] = {"flagged": float(np.mean(f)) if f else None, "others": float(np.mean(o)) if o else None,
                                  "n_flagged": len(f), "n_others": len(o)}
                flag[str(h)] = row
        out["wilcoxon"] = wil
        out["flagged_rec"] = flag
    abl = {n: out["risk"][n] for n in runs if n.startswith("abl_")}
    out["ablation"] = {n: {"mean_AUC": t["mean_AUC"], "per_h": {h: t[h]["AUC"] for h in map(str, range(H))},
                           "n_seeds": t["n_seeds"]} for n, t in abl.items()}
    if abl and "opal" in runs:
        # the full model on the same seeds as the ablations (seeds 0-2), and DeLong tests of the seed ensembles
        k = max(t["n_seeds"] for t in abl.values())
        full = runs["opal"][:k]
        out["ablation_full"] = {"risk": risk_table(full), "n_seeds": len(full)}
        ef = {h: ens(full, "test", h) for h in range(H)}
        for n in abl:
            if not same(runs[n]):
                continue
            dl = [delong_test(yt[h], ens(runs[n], "test", h), ef[h]) for h in range(H)]
            adj = holm([d["p"] for d in dl])
            out["ablation"][n]["delong_vs_full"] = {str(h): {**dl[h], "p_holm": adj[h]} for h in range(H)}
        rk_full, _, _ = opal_ranking(ds, "opal", max_seeds=k)
        out["ablation_full"]["rank"] = rk_full
    if any(n.startswith("abl_") for n in runs):
        out["ablation_rank"] = {}
        for n in runs:
            if n.startswith("abl_") and runs[n][0]["test"][0].get("has_rank"):
                rk_n, _, _ = opal_ranking(ds, n)
                out["ablation_rank"][n] = rk_n
    pi = os.path.join(ROOT, ds, "protocol_inflation.json")
    if os.path.exists(pi):
        out["protocol"] = json.load(open(pi))
    json.dump(out, open(os.path.join(ROOT, ds, "summary.json"), "w"), indent=1, default=float)
    print("written", os.path.join(ROOT, ds, "summary.json"))
    for n, t in out["risk"].items():
        print(f"{n:16s} seeds={t['n_seeds']} " + " ".join(f"{t[str(h)]['AUC'][0]:.4f}" for h in range(H)) +
              f" | mean {t['mean_AUC'][0]:.4f}+-{t['mean_AUC'][1]:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "oulad")
