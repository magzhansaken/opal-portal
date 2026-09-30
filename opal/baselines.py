"""Baselines under the same leakage-free protocol and the same information as OPAL.

* tabular: LR, RF, XGBoost, LightGBM, HGB+Platt (calibrated gradient boosting, as in recent
  OULAD early-warning work), MLP - trained per horizon on aggregated features;
* static heterogeneous graphs: R-GCN and HGT on the learner-resource graph accumulated up to the cut-off;
* recommendation: Pop, RecentPop, Repeat, ItemKNN, BPR-MF, LightGCN, next-step GRU and SASRec-style models.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- tabular features
def tabular(c, h, peer=False):
    """Features known at the cut-off of horizon index h (steps 0..state_idx).
    With peer=True, cohort-standardised copies of all features are appended (same idea as OPAL's inputs)."""
    X = c.x_seq.numpy()
    st = c.state_idx[:, h].numpy()
    L, S, F_ = X.shape
    steps = np.arange(S)[None, :]
    m = (steps <= st[:, None]).astype(np.float32)[:, :, None]
    Xm = X * m
    cum = Xm.sum(1)
    last = X[np.arange(L), st]
    n_act = 20 if F_ > 22 else 2
    active = (np.abs(X[:, :, :n_act]).sum(2) > 0) & (m[:, :, 0] > 0)
    n_active = active.sum(1, keepdims=True)
    last_act = np.where(active.any(1), np.array([np.max(np.nonzero(r)[0]) if r.any() else 0 for r in active]), -1)
    since = (st - last_act)[:, None].astype(np.float32)
    mean_rel = cum[:, [22]] / np.maximum(st[:, None] + 1, 1) if F_ > 22 else cum[:, [6]] / np.maximum(st[:, None] + 1, 1)
    t0 = getattr(c, "t0", None)
    rel = st - (t0.numpy() if t0 is not None else 0)          # learner-relative step index
    feats = np.concatenate([cum, last, n_active, since, mean_rel, rel[:, None], c.static_l.numpy()], 1)
    if peer:
        beh = np.concatenate([cum, last, n_active, since, mean_rel], 1)
        z = np.clip((beh - beh.mean(0, keepdims=True)) / np.maximum(beh.std(0, keepdims=True), 1e-3), -5, 5)
        feats = np.concatenate([feats, z], 1)
    return feats.astype(np.float32)


def tab_split(cohorts, h, split, peer=False):
    Xs, ys, gs, names = [], [], {}, []
    for c in cohorts:
        sel = ((c.split == split) & c.eligible[:, h]).numpy()
        if not sel.any():
            continue
        Xs.append(tabular(c, h, peer)[sel]); ys.append(c.y.numpy()[sel]); names += [c.name] * int(sel.sum())
        for g, v in c.groups.items():
            gs.setdefault(g, []).append(np.asarray(v)[sel])
    return np.concatenate(Xs), np.concatenate(ys), {g: np.concatenate(v) for g, v in gs.items()}, np.array(names)


def fit_tabular(kind, Xtr, ytr, Xva, yva, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "LR":
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(Xtr, ytr)
    elif kind == "RF":
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=1, random_state=seed).fit(Xtr, ytr)
    elif kind == "MLP":
        m = make_pipeline(StandardScaler(), MLPClassifier((128, 64), early_stopping=True, max_iter=300,
                                                          random_state=seed)).fit(Xtr, ytr)
    elif kind == "HGB+Platt":
        base = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=seed)
        from sklearn.model_selection import StratifiedKFold
        m = CalibratedClassifierCV(base, method="sigmoid", cv=StratifiedKFold(5, shuffle=True, random_state=seed)).fit(Xtr, ytr)
    elif kind == "XGBoost":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                              early_stopping_rounds=100, eval_metric="auc", n_jobs=1, random_state=seed, verbosity=0)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif kind == "LightGBM":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        raise ValueError(kind)
    return m


# ----------------------------------------------------------------------------- static heterogeneous graphs
def static_graph(c, h):
    """Learner features, resource features and cumulative weighted edges up to the cut-off."""
    st = c.state_idx[:, h]
    xl = torch.tensor(tabular(c, h))
    rows, cols, vals = [], [], []
    for s in range(int(st.max()) + 1):
        A = c.A[s]
        if A._nnz() == 0:
            continue
        i, j = A.indices()
        keep = st[i] >= s
        rows.append(i[keep]); cols.append(j[keep]); vals.append(A.values()[keep])
    if rows:
        E = torch.sparse_coo_tensor(torch.stack([torch.cat(rows), torch.cat(cols)]), torch.cat(vals), (c.L, c.R)).coalesce()
    else:
        E = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (c.L, c.R)).coalesce()
    pop = torch.sparse.sum(E, 0).to_dense() if E._nnz() else torch.zeros(c.R)
    xr = torch.cat([c.static_r, torch.log1p(pop).unsqueeze(1) / 8], 1)
    return xl, xr, E


class RGCN(nn.Module):
    """Two-layer relational GCN on the learner-resource graph with one relation per resource type."""

    def __init__(self, fl, fr, n_types, d=64, drop=0.1):
        super().__init__()
        self.n_types = n_types
        self.l_in, self.r_in = nn.Linear(fl, d), nn.Linear(fr, d)
        self.r_upd = nn.Linear(2 * d, d)
        self.W_rel = nn.Parameter(torch.randn(n_types, d, d) * (1 / math.sqrt(d)))
        self.self_l = nn.Linear(d, d)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))
        self.drop = nn.Dropout(drop)

    def forward(self, xl, xr, E, rtype):
        hl, hr = F.gelu(self.l_in(xl)), F.gelu(self.r_in(xr))
        Et = E.t().coalesce()
        col = torch.sparse.sum(Et, 1).to_dense().unsqueeze(1).clamp_min(1e-6) if E._nnz() else torch.ones(hr.shape[0], 1)
        hr = F.gelu(self.r_upd(torch.cat([hr, torch.sparse.mm(Et, hl) / col if E._nnz() else torch.zeros_like(hr)], 1)))
        msg = torch.zeros_like(hl)
        if E._nnz():
            i, j = E.indices(); w = E.values()
            for t in range(self.n_types):
                sel = rtype[j] == t
                if not sel.any():
                    continue
                At = torch.sparse_coo_tensor(torch.stack([i[sel], j[sel]]), w[sel], E.shape).coalesce()
                row = torch.sparse.sum(At, 1).to_dense().unsqueeze(1).clamp_min(1e-6)
                msg = msg + (torch.sparse.mm(At, hr) / row) @ self.W_rel[t]
        h = F.gelu(self.self_l(hl) + msg)
        return self.out(self.drop(torch.cat([h, hl], 1))).squeeze(1)


class HGT(nn.Module):
    """Heterogeneous Graph Transformer (Hu et al., 2020) via PyTorch Geometric."""

    def __init__(self, fl, fr, d=64, heads=2, layers=2, drop=0.1):
        super().__init__()
        from torch_geometric.nn import HGTConv
        self.drop = nn.Dropout(drop)
        self.l_in, self.r_in = nn.Linear(fl, d), nn.Linear(fr, d)
        meta = (["learner", "resource"], [("learner", "uses", "resource"), ("resource", "used_by", "learner")])
        self.convs = nn.ModuleList([HGTConv(d, d, meta, heads=heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, xl, xr, E, rtype=None):
        h0 = F.gelu(self.l_in(xl))
        x = {"learner": h0, "resource": F.gelu(self.r_in(xr))}
        ei = E.indices()
        edges = {("learner", "uses", "resource"): ei, ("resource", "used_by", "learner"): ei.flip(0)}
        for conv in self.convs:
            x = {k: F.gelu(v) for k, v in conv(x, edges).items()}
        return self.out(torch.cat([x["learner"], h0], 1)).squeeze(1)


def train_graph_baseline(kind, cohorts, seed=0, epochs=40, patience=6, lr=2e-3, verbose=False, wd=1e-4, drop=0.1):
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed); np.random.seed(seed)
    H = len(cohorts[0].horizons)
    cache = {(ci, h): static_graph(c, h) for ci, c in enumerate(cohorts) for h in range(H)}
    fl = cache[(0, 0)][0].shape[1] + H
    fr = cache[(0, 0)][1].shape[1]
    rtypes = [c.static_r[:, :20].argmax(1) if c.name != "act-mooc" else torch.zeros(c.R, dtype=torch.long) for c in cohorts]
    n_types = int(max(int(r.max()) for r in rtypes)) + 1
    model = RGCN(fl, fr, n_types, drop=drop) if kind == "R-GCN" else HGT(fl, fr, drop=drop)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def inputs(ci, h):
        xl, xr, E = cache[(ci, h)]
        oh = torch.zeros(xl.shape[0], H); oh[:, h] = 1
        return torch.cat([xl, oh], 1), xr, E

    # standardise learner features with training statistics
    allx = torch.cat([inputs(ci, h)[0][(c.split == 0) & c.eligible[:, h]] for ci, c in enumerate(cohorts) for h in range(H)])
    mu, sd = allx.mean(0), allx.std(0).clamp_min(0.05)
    allr = torch.cat([cache[(ci, h)][1] for ci, c in enumerate(cohorts) if (c.split == 0).any() for h in range(H)])
    mur, sdr = allr.mean(0), allr.std(0).clamp_min(0.05)
    for key in list(cache):
        xl_, xr_, E_ = cache[key]
        cache[key] = (xl_, ((xr_ - mur) / sdr).clamp(-8, 8), E_)

    def run(split):
        model.eval(); out = {h: ([], [], []) for h in range(H)}
        with torch.no_grad():
            for ci, c in enumerate(cohorts):
                for h in range(H):
                    sel = (c.split == split) & c.eligible[:, h]
                    if not sel.any():
                        continue
                    xl, xr, E = inputs(ci, h)
                    lg = model(((xl - mu) / sd).clamp(-8, 8), xr, E, rtypes[ci])
                    out[h][0].append(c.y[sel].numpy()); out[h][1].append(lg[sel].numpy()); out[h][2].append(sel.numpy())
        return out

    best, best_state, bad = -1, None, 0
    tr = [(ci, h) for ci, c in enumerate(cohorts) if (c.split == 0).any() for h in range(H)]
    for ep in range(epochs):
        model.train()
        for k in np.random.permutation(len(tr)):
            ci, h = tr[k]; c = cohorts[ci]
            sel = (c.split == 0) & c.eligible[:, h]
            xl, xr, E = inputs(ci, h)
            lg = model(((xl - mu) / sd).clamp(-8, 8), xr, E, rtypes[ci])
            loss = F.binary_cross_entropy_with_logits(lg[sel], c.y[sel])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = run(1)
        auc = np.mean([roc_auc_score(np.concatenate(v[h][0]), np.concatenate(v[h][1])) for h in range(H)])
        if verbose:
            print(kind, ep, round(auc, 4), flush=True)
        if auc > best + 1e-4:
            best, best_state, bad = auc, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, run


# ----------------------------------------------------------------------------- recommendation baselines
def history(c, st):
    """Recency-weighted and cumulative learner x item matrices up to each learner's cut-off."""
    L, R = c.L, c.R
    cum = torch.zeros(L, R); rec = torch.zeros(L, R); last = torch.zeros(L, R)
    for s in range(int(st.max()) + 1):
        if c.A[s]._nnz() == 0:
            continue
        D = c.A[s].to_dense()
        keep = (st >= s).float().unsqueeze(1)
        cum += D * keep
        rec = rec * 0.7 + D * keep
        last = torch.where((st == s).unsqueeze(1), D, last)
    return cum, rec, last


def rec_windows(c, h):
    """Cut-off step st of every learner, the last step `end` whose interactions may be used for fitting, and
    the start step t0.  In a cohort that mixes training and test learners on one catalog (act-mooc), the
    training learners contribute their complete trajectories, as they do for OPAL; all other learners
    contribute only the steps up to their own cut-off.  For OULAD test cohorts end == st."""
    st = c.state_idx[:, h]
    t0 = c.t0 if getattr(c, "t0", None) is not None else torch.zeros(c.L, dtype=torch.long)
    shared = bool((c.split == 0).any()) and bool((c.split == 2).any())
    end = torch.where((c.split == 0) & shared, torch.full_like(st, c.S - 1), st)
    return st, end, t0


def fit_interactions(c, end):
    """Binary learner x item matrix of the interactions up to each learner's `end` step."""
    B = torch.zeros(c.L, c.R)
    for s in range(int(end.max()) + 1):
        if c.A[s]._nnz():
            B += (c.A[s].to_dense() > 0).float() * (end >= s).float().unsqueeze(1)
    return (B > 0).float()


def rec_heuristics(c, h, idx):
    """Pop, RecentPop, Repeat, ItemKNN scores for learners `idx` at horizon h."""
    st, end, _ = rec_windows(c, h)
    cum, rec, last = history(c, st)
    Bfit = fit_interactions(c, end)
    pop = Bfit.sum(0)
    # recent popularity: learners active in the cut-off step of each learner (same step for OULAD)
    recent = torch.zeros(c.L, c.R)
    for s in torch.unique(st[idx]).tolist():
        D = c.A[s].to_dense() if c.A[s]._nnz() else torch.zeros(c.L, c.R)
        rp = (D > 0).float().sum(0)
        recent[st == s] = rp
    B = Bfit
    norm = B.sum(0).clamp_min(1).sqrt()
    sim = (B.t() @ B) / (norm.unsqueeze(0) * norm.unsqueeze(1))
    sim.fill_diagonal_(1.0)
    knn = rec @ sim
    eps = 1e-3
    return {
        "Pop": pop.unsqueeze(0).expand(len(idx), -1).numpy(),
        "RecentPop": recent[idx].numpy(),
        "Repeat": (rec[idx] + eps * pop.unsqueeze(0) / pop.max().clamp_min(1)).numpy(),
        "ItemKNN": (knn[idx] + eps * pop.unsqueeze(0) / pop.max().clamp_min(1)).numpy(),
    }


def _bpr_data(c, h):
    _, end, _ = rec_windows(c, h)
    return fit_interactions(c, end)


class MF(nn.Module):
    def __init__(self, L, R, d=64):
        super().__init__()
        self.U = nn.Embedding(L, d); self.I = nn.Embedding(R, d)
        nn.init.normal_(self.U.weight, std=0.1); nn.init.normal_(self.I.weight, std=0.1)

    def emb(self, B=None):
        return self.U.weight, self.I.weight


class LightGCN(MF):
    def __init__(self, L, R, d=64, layers=2):
        super().__init__(L, R, d); self.layers = layers

    def emb(self, B):
        L, R = B.shape
        du = B.sum(1).clamp_min(1).rsqrt(); di = B.sum(0).clamp_min(1).rsqrt()
        Bn = du.unsqueeze(1) * B * di.unsqueeze(0)
        u, i = self.U.weight, self.I.weight
        us, is_ = [u], [i]
        for _ in range(self.layers):
            u, i = Bn @ i, Bn.t() @ u
            us.append(u); is_.append(i)
        return torch.stack(us).mean(0), torch.stack(is_).mean(0)


def fit_cf(kind, c, h, seed=0, epochs=30, lr=1e-2):
    """BPR-MF or LightGCN fitted on the cohort's own interactions up to the cut-off (transductive)."""
    torch.manual_seed(seed)
    B = _bpr_data(c, h)
    model = LightGCN(c.L, c.R) if kind == "LightGCN" else MF(c.L, c.R)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    u, i = torch.nonzero(B, as_tuple=True)
    if len(u) == 0:
        return np.zeros((c.L, c.R))
    for ep in range(epochs):
        perm = torch.randperm(len(u))
        for b in range(0, len(u), 8192):
            sel = perm[b:b + 8192]
            neg = torch.randint(0, c.R, (len(sel),))
            U, I = model.emb(B)
            pu, pi, ni = U[u[sel]], I[i[sel]], I[neg]
            loss = -F.logsigmoid((pu * pi).sum(1) - (pu * ni).sum(1)).mean() + 1e-5 * (pu.pow(2).sum() + pi.pow(2).sum()) / len(sel)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        U, I = model.emb(B)
        return (U @ I.t()).numpy()


class NextStep(nn.Module):
    """Next-step basket model over the sequence of per-step item sets: GRU4Rec-style or SASRec-style."""

    def __init__(self, R, d=64, kind="gru", max_len=64):
        super().__init__()
        self.kind = kind
        self.item = nn.Linear(R, d)
        if kind == "gru":
            self.rnn = nn.GRU(d, d, batch_first=True)
        else:
            self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 2, 2 * d, 0.1, batch_first=True, norm_first=True), 2)
        self.out = nn.Linear(d, R)

    def forward(self, seq):
        x = self.item(seq)
        if self.kind == "gru":
            o, _ = self.rnn(x)
        else:
            S = x.shape[1]
            o = self.enc(x + self.pos[:S], mask=torch.triu(torch.ones(S, S, dtype=torch.bool), 1))
        return self.out(o)


def fit_next_step(kind, c, h, seed=0, epochs=40, lr=3e-3):
    """GRU4Rec- or SASRec-style next-step model.  Sequences start at each learner's first step (t0), so that
    positions mean the same for every learner; the model is trained on the steps inside each learner's
    fitting window (rec_windows) and scores the step after the learner's cut-off from its prefix only."""
    torch.manual_seed(seed)
    st, end, t0 = rec_windows(c, h)
    rel_st, rel_end = st - t0, end - t0
    S_cal = int(end.max()) + 1
    full = torch.stack([(c.A[s].to_dense() > 0).float() if c.A[s]._nnz() else torch.zeros(c.L, c.R) for s in range(S_cal)], 1)
    Srel = int(rel_end.max()) + 1
    ar = torch.arange(Srel).unsqueeze(0)
    pos = (t0.unsqueeze(1) + ar).clamp(max=S_cal - 1)
    seq = full[torch.arange(c.L).unsqueeze(1), pos] * (ar <= rel_end.unsqueeze(1)).unsqueeze(2)
    del full
    model = NextStep(c.R, kind=kind)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # targets: next step inside the fitting window (s+1 <= end)
    tgt = torch.zeros_like(seq); tgt[:, :-1] = seq[:, 1:]
    tmask = (ar < rel_end.unsqueeze(1)) & (tgt.sum(2) > 0)
    for ep in range(epochs):
        for b in range(0, c.L, 1024):
            sl = slice(b, b + 1024)
            lg = model(seq[sl])
            m = tmask[sl]
            if m.sum() == 0:
                continue
            logp = F.log_softmax(lg[m], 1)
            t = tgt[sl][m]
            loss = -((logp * t).sum(1) / t.sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    del tgt
    model.eval()   # no dropout when scoring
    with torch.no_grad():
        out = torch.zeros(c.L, c.R)
        for b in range(0, c.L, 1024):   # score in chunks to bound the memory use
            sl = slice(b, b + 1024)
            prefix = seq[sl] * (ar <= rel_st[sl].unsqueeze(1)).unsqueeze(2)
            out[sl] = model(prefix)[torch.arange(prefix.shape[0]), rel_st[sl]]
        return out.numpy()
