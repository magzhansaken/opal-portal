"""How much do common evaluation shortcuts inflate at-risk AUC on OULAD?

Five settings with the same LightGBM model and the same per-horizon features:
  * temporal          - train on 2013 presentations, early-stop on 2014B, test on 2014J (the paper's protocol);
  * random_same_size  - all eligible learners pooled and assigned at random to training, validation and test sets
                        of exactly the temporal sizes (isolates random assignment from the amount of data);
  * random            - the pooled learners split at random 70/15/15 (stratified), as in many OULAD studies;
  * within_presentation - only the 2014J presentations, split at random 70/15/15 (stratified), i.e. a random
                        split of a single course offering;
  * leaky             - temporal split, but features computed over the whole presentation (end-of-course
                        information), for the learners who are eligible at week 2.
Random settings are averaged over five split seeds.
Writes results/oulad/protocol_inflation.json.
"""
import copy
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opal import data  # noqa: E402
from opal.baselines import fit_tabular, tab_split  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "oulad")


def pooled(cohorts, h):
    parts = [tab_split(cohorts, h, sp) for sp in (0, 1, 2)]
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    return X, y


def main():
    cs = data.load("oulad")
    H = len(cs[0].horizons)
    path = os.path.join(ROOT, "protocol_inflation.json")
    out = {"temporal": [], "random": [], "random_sd": [], "random_same_size": [], "random_same_size_sd": [],
           "within_presentation": [], "within_presentation_sd": [], "leaky_temporal": None}
    for h in range(H):
        Xtr, ytr, _, _ = tab_split(cs, h, 0); Xva, yva, _, _ = tab_split(cs, h, 1); Xte, yte, _, _ = tab_split(cs, h, 2)
        m = fit_tabular("LightGBM", Xtr, ytr, Xva, yva, seed=0)
        out["temporal"].append(float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])))
        X, y = pooled(cs, h)
        aucs = []
        for r in range(5):
            Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.30, stratify=y, random_state=r)
            Xv, Xt, yv, yt = train_test_split(Xb, yb, test_size=0.50, stratify=yb, random_state=r)
            m = fit_tabular("LightGBM", Xa, ya, Xv, yv, seed=0)
            aucs.append(roc_auc_score(yt, m.predict_proba(Xt)[:, 1]))
        out["random"].append(float(np.mean(aucs))); out["random_sd"].append(float(np.std(aucs)))
        # same sizes as the temporal protocol, random assignment
        ntr, nva, nte = len(ytr), len(yva), len(yte)
        aucs = []
        for r in range(5):
            perm = np.random.RandomState(r).permutation(len(y))
            a_, b_, c_ = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:ntr + nva + nte]
            m = fit_tabular("LightGBM", X[a_], y[a_], X[b_], y[b_], seed=0)
            aucs.append(roc_auc_score(y[c_], m.predict_proba(X[c_])[:, 1]))
        out["random_same_size"].append(float(np.mean(aucs))); out["random_same_size_sd"].append(float(np.std(aucs)))
        # random split of the test presentations only (a single course offering)
        aucs = []
        for r in range(5):
            Xa, Xb, ya, yb = train_test_split(Xte, yte, test_size=0.30, stratify=yte, random_state=r)
            Xv, Xt, yv, yt = train_test_split(Xb, yb, test_size=0.50, stratify=yb, random_state=r)
            m = fit_tabular("LightGBM", Xa, ya, Xv, yv, seed=0)
            aucs.append(roc_auc_score(yt, m.predict_proba(Xt)[:, 1]))
        out["within_presentation"].append(float(np.mean(aucs))); out["within_presentation_sd"].append(float(np.std(aucs)))
        print(h, {k: round(v[-1], 4) for k, v in out.items() if isinstance(v, list) and v}, flush=True)
    # end-of-course features for learners eligible at the first horizon
    leaky = []
    for c in cs:
        c2 = copy.copy(c)
        c2.state_idx = torch.full_like(c.state_idx, c.S - 2)
        c2.eligible = c.eligible[:, :1].expand_as(c.eligible).clone()
        leaky.append(c2)
    Xtr, ytr, _, _ = tab_split(leaky, 0, 0); Xva, yva, _, _ = tab_split(leaky, 0, 1); Xte, yte, _, _ = tab_split(leaky, 0, 2)
    m = fit_tabular("LightGBM", Xtr, ytr, Xva, yva, seed=0)
    out["leaky_temporal"] = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    print("leaky", round(out["leaky_temporal"], 4))
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
