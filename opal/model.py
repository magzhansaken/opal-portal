"""OPAL: temporal heterogeneous learner-resource graph network for joint early warning
and learning-resource recommendation in an open-university portal.

At every step s the model
  1. updates each resource from the learners who used it in step s-1 (resource <- learner),
  2. aggregates the resources a learner used in step s (learner <- resource),
  3. fuses this message with the learner's own behaviour, assessments and cohort-relative
     engagement through a gated recurrent update,
so learner and resource states co-evolve through the course. Two heads read the learner
state at the cut-off: an at-risk classifier and a next-step resource ranker.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def mlp(i, h, o, drop=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, o))


def time_encoding(s, S):
    """8-d encoding of step s: 3 sine/cosine pairs, course progress and log step."""
    f = torch.tensor([1.0, 0.3, 0.1])
    enc = torch.cat([torch.sin(s * f), torch.cos(s * f),
                     torch.tensor([s / max(S - 1, 1), math.log1p(s) / 4])])
    return enc.unsqueeze(0)


def time_encoding_rel(s, t0, S):
    """Per-learner 8-d encoding of the learner-relative step max(s - t0, 0) (self-paced courses)."""
    r = (s - t0).clamp_min(0).float().unsqueeze(1)
    f = torch.tensor([1.0, 0.3, 0.1]).unsqueeze(0)
    return torch.cat([torch.sin(r * f), torch.cos(r * f), r / max(S - 1, 1), torch.log1p(r) / 4], 1)


def augment(x, n_act, peer=True, ctx=False):
    """Per-step inputs + running means + inactivity streak (+ peer-normalised copies).
    Peer normalisation standardises every feature against the learner's own cohort at the same step,
    using only information available at that step, which removes presentation-level shifts."""
    L, S, F_ = x.shape
    steps = torch.arange(1, S + 1, dtype=x.dtype).view(1, S, 1)
    run_mean = torch.cumsum(x, 1) / steps
    active = (x[:, :, :n_act].abs().sum(2) > 0).float()
    streak = torch.zeros(L, S)
    cur = torch.zeros(L)
    for s in range(S):
        cur = (cur + 1) * (1 - active[:, s])
        streak[:, s] = cur
    out = torch.cat([x, run_mean, torch.log1p(streak).unsqueeze(2)], 2)
    base = out
    if peer:
        mu = base.mean(0, keepdim=True)
        sd = base.std(0, keepdim=True).clamp_min(1e-3)
        out = torch.cat([out, ((base - mu) / sd).clamp(-5, 5)], 2)
    if ctx:
        # cohort context: mean behaviour of the whole cohort at the same step (known at that step)
        out = torch.cat([out, base.mean(0, keepdim=True).expand(L, -1, -1)], 2)
    return out


class Normalizer(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.register_buffer("mu", torch.zeros(f)); self.register_buffer("sd", torch.ones(f))

    def fit(self, xs):
        x = torch.cat([t.reshape(-1, t.shape[-1]) for t in xs])
        self.mu.copy_(x.mean(0)); self.sd.copy_(x.std(0).clamp_min(0.05))

    def forward(self, x):
        return ((x - self.mu) / self.sd).clamp(-8, 8)


class OPAL(nn.Module):
    def __init__(self, fx, fs, fr, d=64, drop=0.1, use_graph=True, use_coevo=True, use_rec=True,
                 use_assess=True, use_cohort=True, assess_cols=None, cohort_cols=None, n_act=20, use_wide=True, peer=True, deep_x=False, ctx=False, grad_ckpt=False, rec_hybrid=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.rec_hybrid = rec_hybrid
        self.d, self.use_graph, self.use_coevo, self.use_rec = d, use_graph, use_coevo, use_rec
        self.use_wide = use_wide
        self.use_assess, self.use_cohort = use_assess, use_cohort
        self.assess_cols = assess_cols or []
        self.cohort_cols = cohort_cols or []
        self.n_act = n_act
        self.peer, self.ctx = peer, ctx
        fa = (2 * fx + 1) * ((2 if peer else 1) + (1 if ctx else 0))
        self.norm = Normalizer(fa)
        self.inp = mlp(fa + 8, d, d, drop)
        self.wide = nn.Linear(fa + fs, 1)
        self.h0 = nn.Sequential(nn.Linear(fs, d), nn.Tanh())
        self.s_emb = nn.Linear(fs, d)
        self.r_static = nn.Linear(fr, d)
        self.r_dyn = nn.Linear(3, d)
        self.r_from_l = nn.Linear(d, d, bias=False)
        self.r_norm = nn.LayerNorm(d)
        self.msg = nn.Linear(d + 1, d)
        self.gate = nn.Linear(2 * d, d)
        self.cell = nn.GRUCell(2 * d, d)
        self.drop = nn.Dropout(drop)
        self.deep_x = deep_x
        self.x_proj = nn.Linear(fa, d) if deep_x else None
        self.risk = mlp(3 * d if deep_x else 2 * d, d, 1, drop)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.pop_w = nn.Parameter(torch.tensor(1.0))
        if rec_hybrid:   # basket-transition and personal-recency terms of the recommendation head
            self.w_basket = nn.Linear(d, d, bias=False)
            self.rep_w = nn.Parameter(torch.tensor(1.0))
        self.log_vars = nn.Parameter(torch.zeros(2))

    def _mask_inputs(self, x):
        if not self.use_assess and self.assess_cols:
            x = x.clone(); x[..., self.assess_cols] = 0
        if not self.use_cohort and self.cohort_cols:
            x = x.clone(); x[..., self.cohort_cols] = 0
        return x

    def inputs(self, c):
        return self.norm(augment(self._mask_inputs(c.x_seq), self.n_act, self.peer, self.ctx))

    def forward(self, c, upto=None):
        """Run the co-evolution over steps 0..upto-1. Returns learner states H (S,L,d),
        resource states Rs (S+1,R,d) where Rs[s] is the resource state known at the start of step s,
        and popularity logits P (S+1,R)."""
        S = c.S if upto is None else upto
        L, R, d = c.L, c.R, self.d
        x_all = self.inputs(c)
        self._last_inputs = x_all
        h = self.h0(c.static_l)
        r_base = self.r_static(c.static_r)
        agg = torch.zeros(R, d)
        cum_pop = torch.zeros(R)
        last_pop = torch.zeros(R)
        n_seen = torch.zeros(R)
        Hs, Rs, Ps, Bs = [], [], [], []
        bb = torch.zeros(L, d)
        t0 = getattr(c, "t0", None)
        ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
        for s in range(S):
            dyn = torch.stack([torch.log1p(cum_pop) / 8, n_seen / max(L, 1), torch.log1p(last_pop) / 8], 1)
            A = c.A[s]
            te = time_encoding(s, c.S).expand(L, -1) if t0 is None else time_encoding_rel(s, t0, c.S)
            if ckpt:   # recompute the step in the backward pass instead of storing its activations
                h, agg, r, b = checkpoint(self._step, h, agg, r_base, dyn, x_all[:, s], te, A, use_reentrant=False)
            else:
                h, agg, r, b = self._step(h, agg, r_base, dyn, x_all[:, s], te, A)
            Rs.append(r); Ps.append(dyn); Hs.append(h)
            if self.rec_hybrid:
                bb = 0.5 * bb + b          # decayed basket: resource states opened in the recent steps
                Bs.append(bb)
            # popularity statistics of the resources for the next step (no parameters involved)
            if A._nnz() > 0:
                ones = torch.sparse_coo_tensor(A.indices(), torch.ones(A._nnz()), A.shape)
                users = torch.sparse.sum(ones, 0).to_dense()
                last_pop = users
                cum_pop = cum_pop + users
                n_seen = n_seen + users
            else:
                last_pop = torch.zeros(R)
        dyn = torch.stack([torch.log1p(cum_pop) / 8, n_seen / max(L, 1), torch.log1p(last_pop) / 8], 1)
        r = r_base + self.r_dyn(dyn) + (self.r_from_l(agg) if self.use_coevo else 0)
        Rs.append(self.r_norm(torch.tanh(r))); Ps.append(dyn)
        self._last_basket = torch.stack(Bs) if self.rec_hybrid else None
        return torch.stack(Hs), torch.stack(Rs), torch.stack(Ps)

    def _step(self, h, agg, r_base, dyn, xs, te, A):
        """One step of the co-evolution: resource states from the previous step, learner update, and the
        learner -> resource aggregate used by the next step."""
        r = r_base + self.r_dyn(dyn)
        if self.use_coevo:
            r = r + self.r_from_l(agg)
        r = self.r_norm(torch.tanh(r))
        x = self.inp(torch.cat([xs, te], 1))
        L, d = h.shape
        b = torch.zeros(L, d)
        if A._nnz() > 0 and (self.use_graph or self.rec_hybrid):
            row = torch.sparse.sum(A, 1).to_dense().unsqueeze(1)
            b = torch.sparse.mm(A, r) / row.clamp_min(1e-6)     # click-weighted mean of the opened resources
        if self.use_graph and A._nnz() > 0:
            m = self.msg(torch.cat([b, torch.log1p(row) / 4], 1))
            g = torch.sigmoid(self.gate(torch.cat([x, m], 1)))
            m = g * m
        else:
            m = torch.zeros(L, d)
        h = self.cell(self.drop(torch.cat([x, m], 1)), h)
        if A._nnz() > 0:
            At = A.t().coalesce()
            col = torch.sparse.sum(At, 1).to_dense().unsqueeze(1)
            agg = torch.sparse.mm(At, h) / col.clamp_min(1e-6)
        return h, agg, r, b

    def risk_logits(self, c, H, state_idx):
        """state_idx: L x T step indices -> logits L x T (deep head + wide path over known features)."""
        L, T = state_idx.shape
        ar = torch.arange(L).repeat_interleave(T)
        h = H[state_idx.reshape(-1), ar].reshape(L, T, -1)
        s = self.s_emb(c.static_l).unsqueeze(1).expand(-1, T, -1)
        xa = self._last_inputs[ar, state_idx.reshape(-1)].reshape(L, T, -1)
        parts = [h, s] + ([torch.relu(self.x_proj(xa))] if self.deep_x else [])
        out = self.risk(torch.cat(parts, -1)).squeeze(-1)
        if self.use_wide:
            out = out + self.wide(torch.cat([xa, c.static_l.unsqueeze(1).expand(-1, T, -1)], -1)).squeeze(-1)
        return out

    def rec_scores(self, H, Rs, Ps, learners, steps, c=None):
        """Scores for predicting the resources used in step s+1 from the state after step s.
        Base head: state matching plus popularity.  Hybrid head (rec_hybrid): adds a basket-transition term,
        which matches the resources opened in the recent steps with each candidate in the space of resource
        states, and a personal-recency term over the learner's own earlier clicks.  Both terms are inductive:
        they never use resource identifiers, so they transfer to the new catalog of the next presentation."""
        q = self.q(H[steps, learners])                       # B x d
        k = self.k(Rs[steps + 1])                            # B x R x d
        pop = Ps[steps + 1][:, :, 2] + Ps[steps + 1][:, :, 0]
        score = (k @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d) + self.pop_w * pop
        if self.rec_hybrid:
            tb = self.w_basket(self._last_basket[steps, learners])
            score = score + (k @ tb.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d)
            score = score + self.rep_w * personal_recency(c, learners, steps)
        return score


def personal_recency(c, learners, steps, decay=0.7):
    """Recency-weighted log clicks of each learner on each resource up to and including its step."""
    out = torch.zeros(len(learners), c.R)
    rec = torch.zeros(c.L, c.R)
    for s in range(int(steps.max()) + 1):
        rec = rec * decay
        if c.A[s]._nnz():
            rec = rec + torch.log1p(c.A[s].to_dense().clamp_min(0))
        sel = steps == s
        if sel.any():
            out[sel] = rec[learners[sel]]
    return out


class SeqBaseline(nn.Module):
    """GRU or Transformer over the same per-step learner features (no graph, no resources)."""

    def __init__(self, fx, fs, d=64, kind="gru", drop=0.1, max_len=64, n_act=20, peer=True):
        super().__init__()
        self.kind, self.n_act, self.peer = kind, n_act, peer
        fa = (2 * fx + 1) * (2 if peer else 1)
        self.norm = Normalizer(fa)
        self.inp = mlp(fa, d, d, drop)
        self.s_emb = nn.Linear(fs, d)
        self.h0 = nn.Sequential(nn.Linear(fs, d), nn.Tanh())
        if kind == "gru":
            self.rnn = nn.GRU(d, d, batch_first=True)
        else:
            self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
        self.risk = mlp(2 * d, d, 1, drop)

    def inputs(self, c):
        return self.norm(augment(c.x_seq, self.n_act, self.peer))

    def forward(self, c):
        x = self.inp(self.inputs(c))                         # L x S x d
        if self.kind == "gru":
            out, _ = self.rnn(x, self.h0(c.static_l).unsqueeze(0))
        else:
            S = x.shape[1]
            mask = torch.triu(torch.ones(S, S, dtype=torch.bool), 1)   # causal: step s sees only <= s
            out = self.enc(x + self.pos[:S] + self.h0(c.static_l).unsqueeze(1), mask=mask)
        return out.transpose(0, 1)                           # S x L x d

    def risk_logits(self, c, H, state_idx):
        L, T = state_idx.shape
        h = H[state_idx.reshape(-1), torch.arange(L).repeat_interleave(T)].reshape(L, T, -1)
        s = self.s_emb(c.static_l).unsqueeze(1).expand(-1, T, -1)
        return self.risk(torch.cat([h, s], -1)).squeeze(-1)
