"""Leakage-free temporal cohorts for OULAD and SNAP act-mooc.

A *cohort* is a group of learners that share one resource catalogue and one clock:
an OULAD module presentation (e.g. BBB-2014J) or the whole act-mooc course.
Time is discretised into steps (weeks for OULAD, days for act-mooc). Every tensor
indexed by step ``s`` only contains events that happened inside step ``s``; a model
that reads steps ``0..k`` therefore never sees anything after the prediction cut-off.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "data", "cache")

OULAD_TYPES = ["resource", "subpage", "oucontent", "url", "forumng", "quiz", "page", "oucollaborate",
               "questionnaire", "ouwiki", "dataplus", "externalquiz", "homepage", "glossary",
               "ouelluminate", "dualpane", "repeatactivity", "htmlactivity", "sharedsubpage", "folder"]
MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
EDU = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent", "HE Qualification",
       "Post Graduate Qualification"]
REGIONS = ["East Anglian Region", "East Midlands Region", "Ireland", "London Region", "North Region",
           "North Western Region", "Scotland", "South East Region", "South Region", "South West Region",
           "Wales", "West Midlands Region", "Yorkshire Region"]
OULAD_HORIZONS = [2, 4, 6, 8, 12]
MOOC_HORIZONS = [1, 3, 5, 7]


@dataclass
class Cohort:
    name: str
    L: int
    R: int
    S: int
    static_l: torch.Tensor          # L x Fs   learner context (no protected attributes)
    static_r: torch.Tensor          # R x Fr   resource descriptors
    x_seq: torch.Tensor             # L x S x Fx  per-step learner behaviour
    A: list                         # S sparse L x R (log1p clicks inside the step)
    y: torch.Tensor                 # L   1 = at risk (Fail/Withdrawn) or dropout
    split: torch.Tensor             # L   0 train, 1 validation, 2 test
    state_idx: torch.Tensor         # L x H   step index whose *end* is the cut-off of horizon h
    eligible: torch.Tensor          # L x H   learner can be scored at horizon h (still enrolled/started)
    next_items: list                # H lists: for each horizon, sparse L x R targets (items in the following step)
    groups: dict = field(default_factory=dict)   # protected attributes for auditing only
    ids: np.ndarray | None = None
    horizons: list = field(default_factory=list)
    train_state: torch.Tensor | None = None   # L x T  step indices used for the anytime risk loss
    train_mask: torch.Tensor | None = None    # L x T  valid (learner still enrolled / started)
    t0: torch.Tensor | None = None            # L  first step of each learner (self-paced courses); None = 0


def _sparse(rows, cols, vals, L, R):
    if len(rows) == 0:
        return torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (L, R)).coalesce()
    idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32), (L, R)).coalesce()


# --------------------------------------------------------------------------- OULAD
def _oulad_tables():
    import pyarrow.dataset as ds
    base = os.path.join(DATA, "oulad")
    info = pd.read_parquet(f"{base}/studentInfo.parquet")
    reg = pd.read_parquet(f"{base}/studentRegistration.parquet")
    courses = pd.read_parquet(f"{base}/courses.parquet")
    ass = pd.read_parquet(f"{base}/assessments.parquet")
    sass = pd.read_parquet(f"{base}/studentAssessment.parquet")
    vle = pd.read_parquet(f"{base}/vle.parquet")
    svle = ds.dataset(f"{base}/studentVle_ds", format="parquet", partitioning="hive").to_table().to_pandas()
    for c in ["code_module", "code_presentation"]:
        svle[c] = svle[c].astype(str)
    return info, reg, courses, ass, sass, vle, svle


def oulad_step(day):
    """Step 0 = before the official start (negative days); step s>=1 covers days [7(s-1), 7s)."""
    day = np.asarray(day, dtype=float)
    return np.where(day < 0, 0, np.floor(day / 7) + 1).astype(int)


def build_oulad(val_frac=0.15, seed=0, max_steps=41, score_lag_days=0):
    info, reg, courses, ass, sass, vle, svle = _oulad_tables()
    key = ["code_module", "code_presentation", "id_student"]
    df = info.merge(reg, on=key, how="left")
    df = df[~(df["date_unregistration"].notna() & (df["date_unregistration"] <= 0))].reset_index(drop=True)
    df["y"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    # split: 2014J = test (future presentation); 2014B learners -> 85% train / 15% validation; 2013 -> train
    # temporal protocol: 2013 presentations train, 2014B presentations validate, 2014J presentations test
    df["split"] = 0
    df.loc[df.code_presentation == "2014B", "split"] = 1
    df.loc[df.code_presentation == "2014J", "split"] = 2
    svle = svle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    sass = sass.merge(ass[["id_assessment", "code_module", "code_presentation", "assessment_type", "date", "weight"]],
                      on="id_assessment", how="left")
    cohorts = []
    for (mod, pres), d in df.groupby(["code_module", "code_presentation"], sort=True):
        d = d.reset_index(drop=True)
        L = len(d)
        length = float(courses[(courses.code_module == mod) & (courses.code_presentation == pres)].module_presentation_length.iloc[0])
        S = min(max_steps, int(math.ceil(length / 7)) + 2)
        lidx = {s: i for i, s in enumerate(d.id_student.values)}
        # resources of this presentation
        res = vle[(vle.code_module == mod) & (vle.code_presentation == pres)].reset_index(drop=True)
        ridx = {s: j for j, s in enumerate(res.id_site.values)}
        R = len(res)
        # static resource descriptors
        tr = np.zeros((R, len(OULAD_TYPES) + 5), dtype=np.float32)
        for j, t in enumerate(res.activity_type.values):
            tr[j, OULAD_TYPES.index(t)] = 1.0
        wf = res.week_from.values.astype(float)
        wt = res.week_to.values.astype(float)
        tr[:, -5] = np.isfinite(wf)
        tr[:, -4] = np.nan_to_num(wf, nan=0.0) / 40
        tr[:, -3] = np.nan_to_num(wt, nan=0.0) / 40
        tr[:, -2] = length / 270
        tr[:, -1] = float(pres.endswith("J"))
        # clicks
        v = svle[(svle.code_module == mod) & (svle.code_presentation == pres)]
        v = v[v.id_student.isin(lidx) & v.id_site.isin(ridx)]
        v = v.assign(step=oulad_step(v.date.values))
        v = v[v.step < S]
        li = v.id_student.map(lidx).values
        rj = v.id_site.map(ridx).values
        st = v.step.values
        # per-step learner x resource matrix (sum of clicks)
        g = pd.DataFrame({"l": li, "r": rj, "s": st, "c": v.sum_click.values, "day": v.date.values})
        agg = g.groupby(["s", "l", "r"]).c.sum().reset_index()
        A = []
        for s in range(S):
            a = agg[agg.s == s]
            A.append(_sparse(a.l.values, a.r.values, np.log1p(a.c.values), L, R))
        # learner step features: clicks per type (log1p), active days, distinct resources
        tcode = v.activity_type.map({t: i for i, t in enumerate(OULAD_TYPES)}).values
        X = np.zeros((L, S, len(OULAD_TYPES) + 12), dtype=np.float32)
        np.add.at(X, (li, st, tcode), v.sum_click.values)
        X[:, :, :len(OULAD_TYPES)] = np.log1p(X[:, :, :len(OULAD_TYPES)])
        ad = g.groupby(["l", "s"]).day.nunique().reset_index()
        X[ad.l.values, ad.s.values, len(OULAD_TYPES)] = ad.day.values / 7.0
        nr = g.groupby(["l", "s"]).r.nunique().reset_index()
        X[nr.l.values, nr.s.values, len(OULAD_TYPES) + 1] = np.log1p(nr.r.values)
        tot = np.zeros((L, S), dtype=np.float32)
        np.add.at(tot, (li, st), v.sum_click.values)
        ltot = np.log1p(tot)
        # cohort-relative engagement (uses only the same step of peers)
        X[:, :, len(OULAD_TYPES) + 2] = ltot - ltot.mean(0, keepdims=True)
        # assessments: submissions inside the step, and deadlines missed inside the step
        a_all = ass[(ass.code_module == mod) & (ass.code_presentation == pres)]
        sa = sass[(sass.code_module == mod) & (sass.code_presentation == pres) & sass.id_student.isin(lidx)].copy()
        sa["step"] = oulad_step(sa.date_submitted.values)
        # a score becomes known when the submission is marked: immediately for computer-marked (CMA)
        # assessments and `score_lag_days` after submission for tutor-marked ones (0 in the main runs)
        lag = np.where(sa.assessment_type.values == "TMA", score_lag_days, 0)
        sa["sstep"] = oulad_step(sa.date_submitted.values + lag)
        sa["l"] = sa.id_student.map(lidx)
        sa["late"] = ((sa.date_submitted - sa.date) > 0).astype(float)
        sa["dlate"] = np.clip((sa.date_submitted - sa.date).fillna(0), 0, 60) / 30.0
        sa["score01"] = sa.score.fillna(0) / 100.0
        base = len(OULAD_TYPES) + 3
        n_scored = np.zeros((L, S), dtype=np.float32)
        for (l, s), gg in sa[sa.step < S].groupby(["l", "step"]):
            X[l, s, base + 0] += len(gg)
            X[l, s, base + 2] = gg.late.mean()
            X[l, s, base + 3] = gg.dlate.mean()
            X[l, s, base + 4] = (gg.assessment_type == "TMA").sum()
            X[l, s, base + 5] = (gg.assessment_type == "CMA").sum()
        for (l, s), gg in sa[sa.sstep < S].groupby(["l", "sstep"]):
            X[l, s, base + 1] = gg.score01.mean()
            n_scored[l, s] = len(gg)
        # missed deadline: no submission on or before the deadline day. Only submissions made by the
        # deadline count, so a later (late) submission cannot leak into the step of the deadline.
        on_time = set(zip(sa.l.values[sa.date_submitted.values <= sa.date.values],
                          sa.id_assessment.values[sa.date_submitted.values <= sa.date.values]))
        for _, arow in a_all.iterrows():
            if pd.isna(arow.date) or arow.assessment_type == "Exam":
                continue
            s_dead = int(oulad_step([arow.date])[0])
            if s_dead >= S:
                continue
            for l in range(L):
                if (l, arow.id_assessment) not in on_time:
                    X[l, s_dead, base + 6] += 1.0      # deadline passed without an on-time submission
        # running mean score so far (known at the end of each step)
        cum_sum = np.cumsum(X[:, :, base + 1] * n_scored, axis=1)
        cum_n = np.cumsum(n_scored, axis=1)
        X[:, :, base + 7] = np.where(cum_n > 0, cum_sum / np.maximum(cum_n, 1), 0)
        X[:, :, base + 8] = np.array([s / S for s in range(S)])[None, :]   # course progress
        # static learner context (protected attributes are *not* inputs)
        stl = np.zeros((L, len(EDU) + len(REGIONS) + len(MODULES) + 5), dtype=np.float32)
        for i, e in enumerate(d.highest_education.values):
            if e in EDU:
                stl[i, EDU.index(e)] = 1
        for i, rgn in enumerate(d.region.values):
            if rgn in REGIONS:
                stl[i, len(EDU) + REGIONS.index(rgn)] = 1
        stl[:, len(EDU) + len(REGIONS) + MODULES.index(mod)] = 1
        o = len(EDU) + len(REGIONS) + len(MODULES)
        stl[:, o] = np.minimum(d.num_of_prev_attempts.values, 3) / 3
        stl[:, o + 1] = np.log1p(d.studied_credits.values) / 6
        stl[:, o + 2] = np.nan_to_num(d.date_registration.values.astype(float), nan=0) / 100
        stl[:, o + 3] = float(pres.endswith("J"))
        stl[:, o + 4] = length / 270
        # horizons: state after step k (data < 7k days); eligible = not unregistered before cut-off
        state_idx = torch.tensor(np.tile(np.array(OULAD_HORIZONS), (L, 1)), dtype=torch.long)
        unreg = d.date_unregistration.values.astype(float)
        elig = np.stack([~(np.isfinite(unreg) & (unreg < 7 * k)) for k in OULAD_HORIZONS], 1)
        next_items = []
        for k in OULAD_HORIZONS:
            nxt = A[k + 1] if k + 1 < S else _sparse([], [], [], L, R)
            next_items.append(nxt)
        groups = {
            "gender": d.gender.values,
            "age_band": np.where(d.age_band.values == "0-35", "0-35", "35+"),
            "imd_band": np.where(d.imd_band.isin(["0-10%", "10-20", "20-30%", "30-40%", "40-50%"]).values, "low (0-50%)",
                                 np.where(d.imd_band.isna().values, "unknown", "high (50-100%)")),
            "disability": d.disability.values,
        }
        ks = np.arange(1, 21)
        tstate = np.tile(np.minimum(ks, S - 2), (L, 1))
        tmask = np.stack([~(np.isfinite(unreg) & (unreg < 7 * k)) for k in ks], 1) & (ks[None, :] <= S - 2)
        cohorts.append(Cohort(name=f"{mod}-{pres}", L=L, R=R, S=S, train_state=torch.tensor(tstate), train_mask=torch.tensor(tmask),
                              static_l=torch.tensor(stl), static_r=torch.tensor(tr), x_seq=torch.tensor(X), A=A,
                              y=torch.tensor(d.y.values, dtype=torch.float32), split=torch.tensor(d.split.values),
                              state_idx=state_idx, eligible=torch.tensor(elig), next_items=next_items,
                              groups=groups, ids=d.id_student.values, horizons=list(OULAD_HORIZONS)))
    return cohorts


# --------------------------------------------------------------------------- act-mooc
def build_mooc(frac=(0.70, 0.15), day_len=86400.0):
    base = os.path.join(DATA, "act-mooc")
    a = pd.read_csv(f"{base}/mooc_actions.tsv", sep="\t")
    lab = pd.read_csv(f"{base}/mooc_action_labels.tsv", sep="\t")
    fea = pd.read_csv(f"{base}/mooc_action_features.tsv", sep="\t")
    a = a.merge(lab, on="ACTIONID").merge(fea, on="ACTIONID").sort_values(["TIMESTAMP", "ACTIONID"])
    users = a.groupby("USERID").TIMESTAMP.min().sort_values(kind="stable")
    order = users.index.values                      # users ordered by first activity (new cohorts arrive later)
    L = len(order)
    lidx = {u: i for i, u in enumerate(order)}
    R = int(a.TARGETID.max()) + 1
    a["l"] = a.USERID.map(lidx)
    a["s"] = np.floor(a.TIMESTAMP.values / day_len).astype(int)
    S = int(a.s.max()) + 2
    y = a.groupby("l").LABEL.max().reindex(range(L)).values.astype(np.float32)
    n_tr, n_va = int(frac[0] * L), int(frac[1] * L)
    split = np.array([0] * n_tr + [1] * n_va + [2] * (L - n_tr - n_va))
    agg = a.groupby(["s", "l", "TARGETID"]).size().reset_index(name="c")
    A = []
    for s in range(S):
        g = agg[agg.s == s]
        A.append(_sparse(g.l.values, g.TARGETID.values, np.log1p(g.c.values), L, R))
    X = np.zeros((L, S, 10), dtype=np.float32)
    cnt = a.groupby(["l", "s"]).size()
    X[cnt.index.get_level_values(0), cnt.index.get_level_values(1), 0] = np.log1p(cnt.values)
    nt = a.groupby(["l", "s"]).TARGETID.nunique()
    X[nt.index.get_level_values(0), nt.index.get_level_values(1), 1] = np.log1p(nt.values)
    for f_i, f in enumerate(["FEATURE0", "FEATURE1", "FEATURE2", "FEATURE3"]):
        m = a.groupby(["l", "s"])[f].mean()
        X[m.index.get_level_values(0), m.index.get_level_values(1), 2 + f_i] = m.values
    ltot = X[:, :, 0]
    X[:, :, 6] = ltot - ltot.mean(0, keepdims=True)
    # (course progress is set below, relative to each learner's first day)
    hours = a.assign(h=(a.TIMESTAMP % day_len) / 3600.0).groupby(["l", "s"]).h.nunique()
    X[hours.index.get_level_values(0), hours.index.get_level_values(1), 8] = np.log1p(hours.values)
    start = np.floor(users.values / day_len).astype(int)
    for l in range(L):
        X[l, start[l]:, 9] = 1.0                    # enrolled flag (course started for this learner)
    # learner-relative clock: in a self-paced course each learner's time starts at the first action
    X[:, :, 7] = np.clip(np.arange(S)[None, :] - start[:, None], 0, None) / 30.0
    # static context: time of day of the first action only (the calendar start day is excluded because
    # the split is chronological and the start day would be pure extrapolation for new cohorts)
    static_l = torch.zeros((L, 2))
    tod = (users.values % day_len) / day_len
    static_l[:, 0] = torch.tensor(np.sin(2 * np.pi * tod), dtype=torch.float32)
    static_l[:, 1] = torch.tensor(np.cos(2 * np.pi * tod), dtype=torch.float32)
    static_r = torch.eye(R)
    H = len(MOOC_HORIZONS)
    state_idx = np.zeros((L, H), dtype=np.int64)
    elig = np.zeros((L, H), dtype=bool)
    for h, k in enumerate(MOOC_HORIZONS):
        si = start + k - 1                          # last step (day) included; data < start_day + k days
        state_idx[:, h] = np.minimum(si, S - 2)
        elig[:, h] = si <= S - 2
    next_items = []
    for h, k in enumerate(MOOC_HORIZONS):
        nxt_step = state_idx[:, h] + 1
        g = agg.merge(pd.DataFrame({"l": np.arange(L), "ns": nxt_step}), on="l")
        g = g[g.s == g.ns]
        next_items.append(_sparse(g.l.values, g.TARGETID.values, np.log1p(g.c.values), L, R))
    ds_ = np.arange(1, 11)
    tstate = start[:, None] + ds_[None, :] - 1
    tmask = tstate <= S - 2
    tstate = np.minimum(tstate, S - 2)
    return [Cohort(name="act-mooc", L=L, R=R, S=S, static_l=static_l, static_r=static_r,
                   train_state=torch.tensor(tstate), train_mask=torch.tensor(tmask),
                   x_seq=torch.tensor(X), A=A, y=torch.tensor(y), split=torch.tensor(split),
                   state_idx=torch.tensor(state_idx), eligible=torch.tensor(elig), next_items=next_items,
                   groups={}, ids=order, horizons=list(MOOC_HORIZONS), t0=torch.tensor(start, dtype=torch.long))]


def load(dataset: str):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{dataset}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    if dataset == "oulad":
        cohorts = build_oulad()
    elif dataset.startswith("oulad_lag"):                 # sensitivity: tutor-marked scores known later
        cohorts = build_oulad(score_lag_days=int(dataset[len("oulad_lag"):]))
    else:
        cohorts = build_mooc()
    torch.save(cohorts, path)
    return cohorts
