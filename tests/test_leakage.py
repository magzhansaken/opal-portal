"""Leakage and correctness tests for the OPAL pipeline (run: python -m pytest -q tests)."""
import copy
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opal import data  # noqa: E402
from opal.metrics import delong_test, ranking_metrics  # noqa: E402
from opal.model import OPAL, augment  # noqa: E402


HAVE_OULAD = os.path.exists(os.path.join(data.DATA, "oulad", "studentInfo.parquet"))
pytestmark = pytest.mark.skipif(not HAVE_OULAD, reason="OULAD not downloaded (run scripts/download_data.py)")


@pytest.fixture(scope="module")
def oulad():
    return data.load("oulad")


def test_step_mapping():
    assert list(data.oulad_step([-10, -1, 0, 6, 7, 13, 14])) == [0, 0, 1, 1, 2, 2, 3]


def test_splits(oulad):
    for c in oulad:
        pres = c.name.split("-")[1]
        sp = set(c.split.tolist())
        if pres == "2014J":
            assert sp == {2}
        elif pres == "2014B":
            assert sp <= {0, 1}
        else:
            assert sp == {0}


def test_clicks_before_cutoff_match_raw(oulad):
    import pyarrow.dataset as ds
    c = [x for x in oulad if x.name == "AAA-2014J"][0]
    raw = ds.dataset(os.path.join(data.DATA, "oulad", "studentVle_ds"), format="parquet", partitioning="hive").to_table(
        filter=(ds.field("code_module") == "AAA") & (ds.field("code_presentation") == "2014J")).to_pandas()
    rng = np.random.RandomState(0)
    for li in rng.choice(c.L, 20, replace=False):
        sid = c.ids[li]
        for k in data.OULAD_HORIZONS:
            expect = raw[(raw.id_student == sid) & (raw.date < 7 * k)].sum_click.sum()
            got = sum(float(torch.expm1(c.A[s].to_dense()[li]).sum()) for s in range(k + 1))
            # A stores log1p of per-step sums per resource; expm1 recovers the click counts
            assert abs(got - expect) < 1e-3 * max(expect, 1)


def test_eligibility_excludes_withdrawn_before_cutoff(oulad):
    reg = pd.read_parquet(os.path.join(data.DATA, "oulad", "studentRegistration.parquet"))
    c = [x for x in oulad if x.name == "BBB-2014J"][0]
    r = reg[(reg.code_module == "BBB") & (reg.code_presentation == "2014J")].set_index("id_student")
    for h, k in enumerate(data.OULAD_HORIZONS):
        for li in range(0, c.L, 50):
            un = r.loc[c.ids[li], "date_unregistration"]
            expected = not (pd.notna(un) and un < 7 * k)
            assert bool(c.eligible[li, h]) == expected


def test_augment_is_causal():
    x = torch.rand(5, 10, 6)
    a = augment(x, 3)
    x2 = x.clone(); x2[:, 6:] = torch.rand(5, 4, 6) * 10
    b = augment(x2, 3)
    assert torch.allclose(a[:, :6], b[:, :6])


def test_model_state_is_causal(oulad):
    torch.manual_seed(0)
    c = copy.deepcopy([x for x in oulad if x.name == "AAA-2014J"][0])
    m = OPAL(c.x_seq.shape[-1], c.static_l.shape[1], c.static_r.shape[1], d=16).eval()
    with torch.no_grad():
        H1, R1, _ = m(c)
        k = 4
        for s in range(k + 1, c.S):
            c.x_seq[:, s] = torch.rand_like(c.x_seq[:, s]) * 5
            c.A[s] = (c.A[s] * 3).coalesce()
        H2, R2, _ = m(c)
    assert torch.allclose(H1[: k + 1], H2[: k + 1], atol=1e-5)
    assert torch.allclose(R1[: k + 2], R2[: k + 2], atol=1e-5)   # resource state for step k+1 uses data <= k


def test_ranking_metrics_toy():
    s = np.array([[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]])
    t = np.array([[True, False, False], [True, False, False]])
    r = ranking_metrics(s, t, ks=(1,))
    assert r["HR@1"] == 0.5 and r["n"] == 2


def test_delong_matches_auc():
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 500); p1 = rng.rand(500) + y * 0.3; p2 = rng.rand(500)
    r = delong_test(y, p1, p2)
    assert abs(r["auc1"] - roc_auc_score(y, p1)) < 1e-9 and abs(r["auc2"] - roc_auc_score(y, p2)) < 1e-9


def test_assessment_features_are_causal(oulad):
    """Missed-deadline flags and scores up to a cut-off must be reproducible from submissions made
    before that cut-off only (a late submission after the cut-off must not change them)."""
    ass = pd.read_parquet(os.path.join(data.DATA, "oulad", "assessments.parquet"))
    sass = pd.read_parquet(os.path.join(data.DATA, "oulad", "studentAssessment.parquet"))
    c = [x for x in oulad if x.name == "CCC-2014J"][0]
    base = len(data.OULAD_TYPES) + 3
    a = ass[(ass.code_module == "CCC") & (ass.code_presentation == "2014J") & (ass.assessment_type != "Exam")]
    a = a[a.date.notna()]
    s = sass[sass.id_assessment.isin(a.id_assessment)].merge(a[["id_assessment", "date"]], on="id_assessment")
    rng = np.random.RandomState(1)
    for li in rng.choice(c.L, 40, replace=False):
        sid = c.ids[li]
        mine = s[s.id_student == sid]
        for k in data.OULAD_HORIZONS:
            # expected number of missed deadlines in steps 1..k from on-time submissions only
            due = a[data.oulad_step(a.date.values) <= k]
            on_time = set(mine[mine.date_submitted <= mine.date].id_assessment)
            expect = sum(1 for i in due.id_assessment if i not in on_time)
            got = float(c.x_seq[li, :k + 1, base + 6].sum())
            assert abs(got - expect) < 1e-6
            # number of submissions counted up to the cut-off equals submissions dated before 7k
            n_sub = int((mine.date_submitted < 7 * k).sum())
            assert abs(float(c.x_seq[li, :k + 1, base + 0].sum()) - n_sub) < 1e-6
