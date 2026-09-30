"""Training and inference for OPAL and the neural sequence baselines."""
from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import ranking_metrics
from .model import OPAL, SeqBaseline
from sklearn.metrics import roc_auc_score

ASSESS_COLS_OULAD = list(range(23, 31))
COHORT_COLS_OULAD = [22]
COHORT_COLS_MOOC = [6]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_model(kind, cohorts, cfg):
    c0 = cohorts[0]
    fx, fs, fr = c0.x_seq.shape[-1], c0.static_l.shape[1], c0.static_r.shape[1]
    d = cfg.get("d", 64)
    oulad = c0.name != "act-mooc"
    if kind == "opal":
        return OPAL(fx, fs, fr, d=d, drop=cfg.get("drop", 0.1), use_graph=cfg.get("use_graph", True),
                    use_coevo=cfg.get("use_coevo", True), use_rec=cfg.get("use_rec", True),
                    use_assess=cfg.get("use_assess", True), use_cohort=cfg.get("use_cohort", True),
                    assess_cols=ASSESS_COLS_OULAD if oulad else [],
                    cohort_cols=COHORT_COLS_OULAD if oulad else COHORT_COLS_MOOC,
                    n_act=20 if oulad else 2, use_wide=cfg.get("use_wide", True), peer=cfg.get("peer", True),
                    deep_x=cfg.get("deep_x", False), ctx=cfg.get("ctx", False),
                    grad_ckpt=cfg.get("grad_ckpt", False), rec_hybrid=cfg.get("rec_hybrid", False))
    return SeqBaseline(fx, fs, d=d, kind=kind, drop=cfg.get("drop", 0.1), n_act=20 if oulad else 2,
                       peer=cfg.get("peer", True))


def _forward(model, c):
    if isinstance(model, OPAL):
        return model(c)
    return model(c), None, None


def _rec_batch(c, split_mask, rng, n_steps=4, max_learners=512):
    """Sample (learner, step) pairs whose next step contains at least one resource."""
    L_sel, S_sel = [], []
    steps = rng.choice(np.arange(0, c.S - 1), size=min(n_steps, c.S - 1), replace=False)
    for s in steps:
        row = torch.sparse.sum(c.A[s + 1], 1).to_dense() if c.A[s + 1]._nnz() else torch.zeros(c.L)
        cand = torch.nonzero((row > 0) & split_mask).squeeze(1).numpy()
        if len(cand) == 0:
            continue
        if len(cand) > max_learners:
            cand = rng.choice(cand, max_learners, replace=False)
        L_sel.append(cand); S_sel.append(np.full(len(cand), s))
    if not L_sel:
        return None
    return torch.tensor(np.concatenate(L_sel)), torch.tensor(np.concatenate(S_sel))


def _targets(c, learners, steps):
    R = c.R
    T = torch.zeros(len(learners), R)
    for s in torch.unique(steps).tolist():
        sel = torch.nonzero(steps == s).squeeze(1)
        dense = c.A[s + 1].to_dense() if c.A[s + 1]._nnz() else torch.zeros(c.L, R)
        T[sel] = (dense[learners[sel]] > 0).float()
    return T


def evaluate(model, cohorts, split, rec=True):
    """Returns per-horizon arrays for risk and ranking evaluation on learners of `split`."""
    model.eval()
    H = len(cohorts[0].horizons)
    out = {h: {"y": [], "logit": [], "groups": {}, "scores": [], "targets": [], "seen": [], "cohort": [], "ridx": []} for h in range(H)}
    with torch.no_grad():
        for c in cohorts:
            m = c.split == split
            if not m.any():
                continue
            Hs, Rs, Ps = _forward(model, c)
            lg = model.risk_logits(c, Hs, c.state_idx)          # L x H
            for h in range(H):
                sel = m & c.eligible[:, h]
                idx = torch.nonzero(sel).squeeze(1)
                out[h]["y"].append(c.y[idx].numpy())
                out[h]["logit"].append(lg[idx, h].numpy())
                out[h]["cohort"] += [c.name] * len(idx)
                for g, vals in c.groups.items():
                    out[h]["groups"].setdefault(g, []).append(np.asarray(vals)[idx.numpy()])
                if rec and Rs is not None:
                    st = c.state_idx[idx, h]
                    sc = model.rec_scores(Hs, Rs, Ps, idx, st, c=c).numpy()
                    tgt = (c.next_items[h].to_dense()[idx] > 0).numpy()
                    seen = np.zeros_like(tgt)
                    upto = int(st.max()) + 1
                    acc = torch.zeros(c.L, c.R)
                    for s in range(upto):
                        if c.A[s]._nnz():
                            acc += c.A[s].to_dense()
                        rows = torch.nonzero((st == s)).squeeze(1)
                        if len(rows):
                            seen[rows.numpy()] = (acc[idx[rows]] > 0).numpy()
                    out[h]["scores"].append(sc); out[h]["targets"].append(tgt); out[h]["seen"].append(seen)
                    out[h]["ridx"].append((c.name, idx.numpy()))
    res = {}
    for h in range(H):
        o = out[h]
        r = {"y": np.concatenate(o["y"]), "logit": np.concatenate(o["logit"]), "cohort": np.array(o["cohort"]),
             "groups": {g: np.concatenate(v) for g, v in o["groups"].items()}}
        if o["scores"]:
            # cohorts have different catalogue sizes: evaluate ranking per cohort and pool per learner
            r["rank"] = [(sc, tg, se, nm, ix) for (sc, tg, se), (nm, ix) in zip(zip(o["scores"], o["targets"], o["seen"]), o["ridx"])]
        res[h] = r
    return res


def val_score(ev, rec=True):
    aucs = [roc_auc_score(ev[h]["y"], ev[h]["logit"]) for h in ev if len(np.unique(ev[h]["y"])) > 1]
    score = float(np.mean(aucs))
    nd = []
    if rec:
        for h in ev:
            if "rank" in ev[h]:
                vals, ns = [], []
                for part in ev[h]["rank"]:
                    sc, tg, se = part[:3]
                    m = ranking_metrics(sc, tg)
                    if m["n"]:
                        vals.append(m["NDCG@10"] * m["n"]); ns.append(m["n"])
                if ns:
                    nd.append(sum(vals) / sum(ns))
    return score, (float(np.mean(nd)) if nd else None)


def train(kind, cohorts, cfg, seed=0, verbose=True, ckpt=None):
    """Train with early stopping. If `ckpt` is a path, the full training state is saved after every
    epoch and training resumes from it after an interruption (same result as an uninterrupted run)."""
    set_seed(seed)
    rng = np.random.RandomState(seed)
    model = build_model(kind, cohorts, cfg)
    from .model import augment
    with torch.no_grad():
        xs = []
        for c in cohorts:
            m = c.split == 0
            if m.any():
                x = model._mask_inputs(c.x_seq) if isinstance(model, OPAL) else c.x_seq
                xs.append(augment(x, model.n_act, model.peer, getattr(model, "ctx", False))[m])
        model.norm.fit(xs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2e-3), weight_decay=cfg.get("wd", 1e-4))
    use_rec = isinstance(model, OPAL) and cfg.get("use_rec", True)
    train_cohorts = [c for c in cohorts if (c.split == 0).any()]
    best, best_state, bad = -1e9, None, 0
    hist = []
    start_ep = 0
    ema_decay = cfg.get("ema", 0.0)      # exponential moving average of the weights (0 = off)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if ema_decay else None
    if ckpt is not None and os.path.exists(ckpt):
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        best, best_state, bad, hist, start_ep = st["best"], st["best_state"], st["bad"], st["hist"], st["epoch"]
        if ema is not None:
            ema = st["ema"]
        rng.set_state(st["rng"]); torch.set_rng_state(st["torch_rng"]); random.setstate(st["py_rng"])
        if verbose:
            print(f"resumed from epoch {start_ep}", flush=True)
        if st.get("done"):
            model.load_state_dict(best_state)
            return model, hist
    steps_per_epoch = cfg.get("steps_per_cohort", 1)
    for ep in range(start_ep, cfg.get("epochs", 40)):
        model.train(); t0 = time.time(); losses = []
        order = rng.permutation(len(train_cohorts))
        for ci in order:
            c = train_cohorts[ci]
            trm = c.split == 0
            for _ in range(steps_per_epoch):
                Hs, Rs, Ps = _forward(model, c)
                T_max = cfg.get("train_steps", c.train_state.shape[1])
                lg = model.risk_logits(c, Hs, c.train_state[:, :T_max])
                mask = c.train_mask[:, :T_max] & trm.unsqueeze(1)
                yy = c.y.unsqueeze(1).expand_as(lg)
                l_risk = F.binary_cross_entropy_with_logits(lg[mask], yy[mask])
                loss = l_risk
                l_rec = torch.tensor(0.0)
                if use_rec:
                    b = _rec_batch(c, trm, rng, n_steps=cfg.get("rec_steps", 4), max_learners=cfg.get("rec_learners", 512))
                    if b is not None:
                        ls, ss = b
                        sc = model.rec_scores(Hs, Rs, Ps, ls, ss, c=c)
                        T = _targets(c, ls, ss)
                        logp = F.log_softmax(sc, 1)
                        l_rec = -((logp * T).sum(1) / T.sum(1)).mean()
                        lv = model.log_vars
                        loss = torch.exp(-lv[0]) * l_risk + lv[0] + torch.exp(-lv[1]) * l_rec * cfg.get("rec_weight", 1.0) + lv[1]
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if ema is not None:
                    with torch.no_grad():
                        for k, v in model.state_dict().items():
                            if v.dtype.is_floating_point:
                                ema[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)
                            else:
                                ema[k].copy_(v)
                losses.append((float(l_risk.detach()), float(l_rec.detach())))
        if (ep + 1) % cfg.get("eval_every", 1) == 0:
            if ema is not None:          # validate (and keep) the averaged weights
                live = copy.deepcopy(model.state_dict()); model.load_state_dict(ema)
            ev = evaluate(model, cohorts, split=1, rec=use_rec)
            cur_state = copy.deepcopy(model.state_dict())
            if ema is not None:
                model.load_state_dict(live)
            auc, nd = val_score(ev, rec=use_rec)
            score = auc + (cfg.get("val_rec_weight", 0.5) * nd if nd is not None else 0)
            hist.append({"epoch": ep + 1, "loss_risk": float(np.mean([a for a, _ in losses])),
                         "loss_rec": float(np.mean([b for _, b in losses])), "val_auc": auc, "val_ndcg10": nd,
                         "sec": round(time.time() - t0, 1)})
            if verbose:
                print(hist[-1], flush=True)
            done = False
            if score > best + 1e-4:
                best, best_state, bad = score, cur_state, 0
            else:
                bad += 1
                done = bad >= cfg.get("patience", 6)
            if ckpt is not None:
                tmp = ckpt + ".tmp"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "best": best, "best_state": best_state,
                            "bad": bad, "hist": hist, "epoch": ep + 1, "rng": rng.get_state(),
                            "torch_rng": torch.get_rng_state(), "py_rng": random.getstate(), "done": done, "ema": ema}, tmp)
                os.replace(tmp, ckpt)
            if done:
                break
    model.load_state_dict(best_state)
    return model, hist
