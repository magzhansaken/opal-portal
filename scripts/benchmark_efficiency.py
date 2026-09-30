"""Controlled efficiency benchmark (run on an otherwise idle machine, after all experiments).

For every model, seed 0 is retrained from scratch with one CPU thread, and the wall-clock training time, the
number of epochs, the time to score the whole test split at all decision points and the number of parameters
are recorded.  The retrained predictions are compared with the stored seed-0 predictions of the main runs,
which doubles as a reproducibility check.  Recommenders fitted on the test cohorts (SASRec, GRU4Rec) are timed
over all their fits (one per test cohort and decision point).

usage: python scripts/benchmark_efficiency.py [oulad|act-mooc ...]   ->  results/efficiency.json
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opal import data, train  # noqa: E402
from opal.baselines import fit_next_step, train_graph_baseline  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
torch.set_num_threads(1)
NEURAL = {"opal": ("opal", {"epochs": 60, "patience": 8, "deep_x": True}),
          "GRU": ("gru", {"epochs": 60, "patience": 8}),
          "Transformer": ("transformer", {"epochs": 60, "patience": 8})}
GRAPH = {"R-GCN": dict(lr=1e-3, wd=1e-4, drop=0.1), "HGT": dict(lr=1e-3, wd=1e-4, drop=0.3)}


def stored_logits(ds, name):
    f = os.path.join(ROOT, ds, name, "seed0.pkl")
    if not os.path.exists(f):
        return None
    r = pickle.load(open(f, "rb"))
    return {h: r["test"][h]["logit"] for h in r["test"]}


def compare(ds, name, logits):
    ref = stored_logits(ds, name)
    if ref is None:
        return None
    return float(max(np.max(np.abs(np.asarray(ref[h]) - np.asarray(logits[h]))) for h in ref))


def bench_neural(ds, cs, name):
    kind, cfg = NEURAL[name]
    cfg = dict(cfg)
    if ds == "act-mooc":
        cfg.update(steps_per_cohort=8, rec_steps=8, rec_learners=1024)
    t0 = time.time()
    model, hist = train.train(kind, cs, cfg, seed=0, verbose=False)
    tr = time.time() - t0
    rec = kind == "opal"
    times = []
    for _ in range(3):
        t1 = time.time(); ev = train.evaluate(model, cs, split=2, rec=rec); times.append(time.time() - t1)
    return {"train_sec": tr, "epochs": len(hist), "infer_sec": float(np.median(times)),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "max_abs_logit_diff_vs_main_run": compare(ds, name, {h: ev[h]["logit"] for h in ev})}


def bench_graph(ds, cs, name):
    t0 = time.time()
    model, run = train_graph_baseline(name, cs, seed=0, verbose=False, epochs=60, patience=8, **GRAPH[name])
    tr = time.time() - t0
    times = []
    for _ in range(3):
        t1 = time.time(); o = run(2); times.append(time.time() - t1)
    logits = {h: np.concatenate(o[h][1]) for h in o}
    return {"train_sec": tr, "infer_sec": float(np.median(times)),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "max_abs_logit_diff_vs_main_run": compare(ds, name, logits)}


def bench_seqrec(ds, cs, name):
    kind = "gru" if name == "GRU4Rec" else "sasrec"
    H = len(cs[0].horizons)
    t0 = time.time(); fits = 0
    for c in cs:
        if not (c.split == 2).any():
            continue
        for h in range(H):
            fit_next_step(kind, c, h, seed=0); fits += 1
    tr = time.time() - t0
    return {"train_sec": tr, "fits": fits, "sec_per_fit": tr / max(fits, 1)}


def main(datasets):
    path = os.path.join(ROOT, "efficiency.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for ds in datasets:
        cs = data.load(ds)
        res.setdefault(ds, {})
        jobs = [(n, lambda n=n: bench_neural(ds, cs, n)) for n in NEURAL] + \
               [(n, lambda n=n: bench_graph(ds, cs, n)) for n in GRAPH] + \
               [(n, lambda n=n: bench_seqrec(ds, cs, n)) for n in ("SASRec", "GRU4Rec")]
        for n, job in jobs:
            if n in res[ds]:
                continue
            print(ds, n, flush=True)
            res[ds][n] = job()
            print(res[ds][n], flush=True)
            json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["oulad", "act-mooc"])
