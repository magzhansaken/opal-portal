"""Tests that run without the datasets (used in continuous integration).

They check causality of the model on a synthetic cohort, the evaluation metrics, and that training
resumed from a checkpoint reproduces an uninterrupted run.
"""
import copy
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opal import train  # noqa: E402
from opal.data import Cohort  # noqa: E402
from opal.metrics import (abroca, ece, fit_temperature, holm, ranking_metrics)  # noqa: E402
from opal.model import OPAL, augment, time_encoding_rel  # noqa: E402


def synthetic_cohort(name="SYN-2013J", L=60, R=12, S=10, split=0, seed=0, fx=32, t0=None):
    g = torch.Generator().manual_seed(seed)
    A = []
    for _ in range(S):
        dense = (torch.rand(L, R, generator=g) < 0.15).float() * torch.rand(L, R, generator=g) * 3
        A.append(dense.to_sparse().coalesce())
    x = torch.rand(L, S, fx, generator=g)
    y = (x[:, :4, :20].mean((1, 2)) < 0.5).float()
    H = [2, 4]
    next_items = [A[k + 1] for k in H]
    return Cohort(name=name, L=L, R=R, S=S, static_l=torch.rand(L, 30, generator=g), static_r=torch.rand(R, 25, generator=g),
                  x_seq=x, A=A, y=y, split=torch.full((L,), split), state_idx=torch.tensor(H).repeat(L, 1),
                  eligible=torch.ones(L, len(H), dtype=torch.bool), next_items=next_items, groups={},
                  ids=np.arange(L), horizons=H, train_state=torch.arange(1, 7).clamp(max=S - 2).repeat(L, 1),
                  train_mask=torch.ones(L, 6, dtype=torch.bool), t0=t0)


def test_augment_is_causal():
    x = torch.rand(7, 9, 5)
    a = augment(x, 3)
    x2 = x.clone(); x2[:, 5:] = torch.rand(7, 4, 5) * 10
    assert torch.allclose(a[:, :5], augment(x2, 3)[:, :5])


@pytest.mark.parametrize("relative", [False, True])
def test_model_states_do_not_see_the_future(relative):
    torch.manual_seed(0)
    c = synthetic_cohort(t0=torch.randint(0, 4, (60,)) if relative else None)
    m = OPAL(32, 30, 25, d=16).eval()
    with torch.no_grad():
        H1, R1, _ = m(c)
        k = 4
        c2 = copy.deepcopy(c)
        for s in range(k + 1, c.S):
            c2.x_seq[:, s] = torch.rand_like(c2.x_seq[:, s]) * 5
            c2.A[s] = (c2.A[s] * 7).coalesce()
        H2, R2, _ = m(c2)
    assert torch.allclose(H1[: k + 1], H2[: k + 1], atol=1e-5)
    assert torch.allclose(R1[: k + 2], R2[: k + 2], atol=1e-5)


def test_relative_time_encoding():
    enc = time_encoding_rel(5, torch.tensor([0, 5, 9]), 20)
    assert torch.allclose(enc[1], enc[2])          # both learners are at relative step 0
    assert not torch.allclose(enc[0], enc[1])


def test_ece_and_temperature():
    rng = np.random.RandomState(0)
    p = rng.rand(20000); y = (rng.rand(20000) < p).astype(int)
    assert ece(y, p) < 0.02
    z = np.log(p / (1 - p)) * 3                        # over-confident logits
    T, b = fit_temperature(z, y)
    assert 2.0 < 1 / T < 4.0 and abs(b) < 0.1         # recovers the scale of 3


def test_ranking_metrics_values():
    s = np.array([[0.9, 0.8, 0.1, 0.2] + [0.0] * 8])
    t = np.zeros((1, 12), bool); t[0, [1, 3]] = True
    r = ranking_metrics(s, t, ks=(5, 10))
    ideal = 1 + 1 / np.log2(3)
    assert abs(r["NDCG@5"] - (1 / np.log2(3) + 1 / np.log2(4)) / ideal) < 1e-9
    assert r["HR@5"] == 1.0 and r["Recall@5"] == 1.0
    seen = np.zeros((1, 12), bool); seen[0, 1] = True
    r2 = ranking_metrics(s, t, seen=seen, ks=(5,))
    assert abs(r2["NDCG@5"] - 1 / np.log2(3)) < 1e-9   # item 3 moves to rank 2 after removing the seen item


def test_holm_and_abroca():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 4000); p = rng.rand(4000) + 0.5 * y
    g = np.where(rng.rand(4000) < 0.5, "a", "b")
    assert abroca(y, p, g) < 0.03                       # same model quality in both groups
    p2 = p.copy(); p2[g == "b"] = rng.rand((g == "b").sum())
    assert abroca(y, p2, g) > 0.2


def test_training_resumes_exactly(tmp_path):
    cohorts = [synthetic_cohort("SYN-2013B", seed=1, split=0), synthetic_cohort("SYN-2013J", seed=2, split=0),
               synthetic_cohort("SYN-2014B", seed=3, split=1), synthetic_cohort("SYN-2014J", seed=4, split=2)]
    cfg = {"epochs": 4, "patience": 10, "d": 16, "rec_learners": 32}
    full, _ = train.train("opal", cohorts, cfg, seed=0, verbose=False)
    ck = str(tmp_path / "run.ckpt")
    train.train("opal", cohorts, dict(cfg, epochs=2), seed=0, verbose=False, ckpt=ck)   # "interrupted" after 2 epochs
    resumed, _ = train.train("opal", cohorts, cfg, seed=0, verbose=False, ckpt=ck)
    for (k, a), (_, b) in zip(full.state_dict().items(), resumed.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-6), k


def test_gradient_checkpointing_is_exact():
    """Recomputing the steps in the backward pass (used to save memory on act-mooc) changes nothing."""
    c = synthetic_cohort(t0=torch.randint(0, 4, (60,)))
    torch.manual_seed(0)
    a = OPAL(32, 30, 25, d=16, deep_x=True)
    b = OPAL(32, 30, 25, d=16, deep_x=True, grad_ckpt=True); b.load_state_dict(a.state_dict())
    grads = []
    for mdl in (a, b):
        mdl.train(); torch.manual_seed(5)
        H, R, P = mdl(c)
        loss = mdl.risk_logits(c, H, c.train_state).sum() + mdl.rec_scores(H, R, P, torch.arange(10), torch.full((10,), 3)).sum()
        loss.backward()
        grads.append([p.grad for p in mdl.parameters() if p.grad is not None])
    assert all(torch.equal(x, y) for x, y in zip(*grads))


@pytest.mark.parametrize("kind", ["gru", "sasrec", "BPR-MF", "heur"])
def test_rec_baselines_use_only_the_prefix_of_test_learners(kind):
    """In a cohort with a shared catalog (act-mooc), the recommenders may learn from the complete trajectories
    of training learners, but the scores of a test learner must not depend on that learner's future steps."""
    from opal import baselines
    c = synthetic_cohort(L=60, S=10, seed=3, t0=torch.randint(0, 3, (60,)))
    c.split = torch.tensor([0] * 40 + [2] * 20)
    h = 0
    test = torch.arange(40, 60)

    def scores(coh):
        if kind in ("gru", "sasrec"):
            return baselines.fit_next_step(kind, coh, h, seed=0, epochs=3)[test.numpy()]
        if kind == "BPR-MF":
            return baselines.fit_cf(kind, coh, h, seed=0, epochs=2)[test.numpy()]
        return baselines.rec_heuristics(coh, h, test)["ItemKNN"]
    base = scores(c)
    c2 = copy.deepcopy(c)
    st = c.state_idx[:, h]
    for s in range(c.S):
        D = c2.A[s].to_dense()
        future = (st < s) & (c.split == 2)              # steps after the cut-off of test learners
        D[future] = torch.rand(int(future.sum()), c.R) * 5
        c2.A[s] = D.to_sparse().coalesce()
    assert np.allclose(base, scores(c2), atol=1e-6)


def test_hybrid_recommendation_head_is_causal():
    """Basket-transition and personal-recency terms at step k use only steps up to k."""
    torch.manual_seed(0)
    c = synthetic_cohort()
    m = OPAL(32, 30, 25, d=16, rec_hybrid=True).eval()
    learners, steps = torch.arange(20), torch.full((20,), 4)
    with torch.no_grad():
        H, R, P = m(c)
        s1 = m.rec_scores(H, R, P, learners, steps, c=c)
        c2 = copy.deepcopy(c)
        for s in range(5, c.S):
            c2.x_seq[:, s] = torch.rand_like(c2.x_seq[:, s]) * 5
            c2.A[s] = (c2.A[s] * 7).coalesce()
        H2, R2, P2 = m(c2)
        s2 = m.rec_scores(H2, R2, P2, learners, steps, c=c2)
    assert torch.allclose(s1, s2, atol=1e-5)


def test_training_with_weight_averaging_resumes_exactly(tmp_path):
    cohorts = [synthetic_cohort("SYN-2013B", seed=1, split=0), synthetic_cohort("SYN-2014B", seed=3, split=1),
               synthetic_cohort("SYN-2014J", seed=4, split=2)]
    cfg = {"epochs": 4, "patience": 10, "d": 16, "rec_learners": 32, "ema": 0.9}
    full, _ = train.train("opal", cohorts, cfg, seed=0, verbose=False)
    ck = str(tmp_path / "run.ckpt")
    train.train("opal", cohorts, dict(cfg, epochs=2), seed=0, verbose=False, ckpt=ck)
    resumed, _ = train.train("opal", cohorts, cfg, seed=0, verbose=False, ckpt=ck)
    for (k, a), (_, b) in zip(full.state_dict().items(), resumed.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-6), k
