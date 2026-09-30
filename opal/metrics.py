"""Evaluation metrics: discrimination, calibration, ranking, fairness, uncertainty."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


def ece(y, p, bins=15):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def risk_metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    yh = (p >= thr).astype(int)
    return {"AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p), "F1": f1_score(y, yh),
            "MCC": matthews_corrcoef(y, yh), "Brier": float(np.mean((p - y) ** 2)), "ECE": ece(y, p)}


def best_f1_threshold(y, p):
    ts = np.linspace(0.05, 0.95, 91)
    f = [f1_score(y, (p >= t).astype(int)) for t in ts]
    return float(ts[int(np.argmax(f))])


def fit_temperature(logits, y, iters=200):
    """Temperature scaling on validation logits (Guo et al., 2017)."""
    import torch
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([t, b], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg * torch.exp(t) + b, yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(t)), float(b)


def ranking_metrics(scores, targets, seen=None, ks=(5, 10), per=False):
    """scores: N x R, targets: N x R boolean. If `seen` is given, previously used items are removed
    from the candidates and from the targets (discovery of new resources). With per=True the result also
    holds the per-learner NDCG@10 ("per_ndcg10") and the boolean mask of evaluated rows ("kept")."""
    scores = np.array(scores, dtype=float, copy=True)
    targets = np.asarray(targets, dtype=bool).copy()
    if seen is not None:
        seen = np.asarray(seen, dtype=bool)
        scores[seen] = -np.inf
        targets &= ~seen
    keep = targets.any(1)
    scores, targets = scores[keep], targets[keep]
    out = {"n": int(keep.sum())}
    if per:
        out["kept"] = keep
    if out["n"] == 0:
        if per:
            out["per_ndcg10"] = np.zeros(0)
        return out
    order = np.argsort(-scores, 1)
    for k in ks:
        top = order[:, :k]
        hit = np.take_along_axis(targets, top, 1)
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = (hit * disc).sum(1)
        ideal = np.array([disc[:min(int(t), k)].sum() for t in targets.sum(1)])
        out[f"HR@{k}"] = float(hit.any(1).mean())
        out[f"Recall@{k}"] = float((hit.sum(1) / targets.sum(1)).mean())
        out[f"NDCG@{k}"] = float((dcg / ideal).mean())
        if per and k == 10:
            out["per_ndcg10"] = dcg / ideal
    return out


def fairness_gaps(y, p, groups, top_frac=0.2):
    """Alert policy: flag the top `top_frac` risk scores. Report per-group TPR, FPR and alert rate,
    and the largest gap across groups for each protected attribute."""
    y, p = np.asarray(y), np.asarray(p)
    thr = np.quantile(p, 1 - top_frac)
    a = p >= thr
    res = {}
    for name, g in groups.items():
        g = np.asarray(g)
        rows = {}
        for v in np.unique(g):
            m = g == v
            if m.sum() < 30:
                continue
            tpr = a[m & (y == 1)].mean() if (m & (y == 1)).any() else np.nan
            fpr = a[m & (y == 0)].mean() if (m & (y == 0)).any() else np.nan
            rows[str(v)] = {"n": int(m.sum()), "TPR": float(tpr), "FPR": float(fpr), "alert_rate": float(a[m].mean())}
        if len(rows) >= 2:
            res[name] = {"groups": rows,
                         "dTPR": float(np.nanmax([r["TPR"] for r in rows.values()]) - np.nanmin([r["TPR"] for r in rows.values()])),
                         "dFPR": float(np.nanmax([r["FPR"] for r in rows.values()]) - np.nanmin([r["FPR"] for r in rows.values()]))}
    return res


def bootstrap_diff(y, p1, p2, metric=roc_auc_score, n=1000, seed=0):
    """Paired bootstrap of metric(p1) - metric(p2) over learners; returns mean, 95% CI, one-sided p."""
    rng = np.random.RandomState(seed)
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    N = len(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, N, N)
        if y[i].min() == y[i].max():
            continue
        d.append(metric(y[i], p1[i]) - metric(y[i], p2[i]))
    d = np.array(d)
    return {"diff": float(metric(y, p1) - metric(y, p2)), "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)), "p_one_sided": float((d <= 0).mean())}


def delong_test(y, p1, p2):
    """Two-sided DeLong test for two correlated ROC AUCs (Sun & Xu fast implementation)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    y = y[order]
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    m = int(y.sum()); n = len(y) - m

    def midrank(x):
        j = np.argsort(x); z = x[j]; N = len(x); t = np.zeros(N); i = 0
        while i < N:
            k = i
            while k < N and z[k] == z[i]:
                k += 1
            t[i:k] = 0.5 * (i + k - 1) + 1
            i = k
        out = np.empty(N); out[j] = t
        return out
    tx = np.array([midrank(r[:m]) for r in preds]); ty = np.array([midrank(r[m:]) for r in preds])
    tz = np.array([midrank(r) for r in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def abroca(y, p, g, grid=1001):
    """Absolute between-ROC area (Gardner et al., 2019) between the two groups in g."""
    from sklearn.metrics import roc_curve
    y, p, g = np.asarray(y), np.asarray(p), np.asarray(g)
    vals = [v for v in np.unique(g)]
    if len(vals) != 2:
        return float("nan")
    xs = np.linspace(0, 1, grid)
    curves = []
    for v in vals:
        m = g == v
        if len(np.unique(y[m])) < 2:
            return float("nan")
        fpr, tpr, _ = roc_curve(y[m], p[m])
        curves.append(np.interp(xs, fpr, tpr))
    return float(np.trapezoid(np.abs(curves[0] - curves[1]), xs))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
