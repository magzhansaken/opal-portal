"""Build the result tables (LaTeX) and figures of Section 4 and of the Supplementary Materials from results/<dataset>/summary.json.

usage: python scripts/make_results.py [paper_dir]
Tables are written to <paper_dir>/tables, figures to figures/ (PDF, 900-dpi TIFF, PNG) and copied to
<paper_dir>/figures.  Every number printed in the paper comes from these files.
"""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opal import plotstyle as ps  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper")
TAB = os.path.join(PAPER, "tables"); os.makedirs(TAB, exist_ok=True)
FIG = os.environ.get("OPAL_FIG_DIR", os.path.join(ROOT, "figures")); os.makedirs(FIG, exist_ok=True)
RISK_ORDER = ["LR", "RF", "XGBoost", "LightGBM", "HGB+Platt", "MLP", "GRU", "Transformer", "R-GCN", "HGT"]
LABEL = {"HGB+Platt": "HGB + Platt", "opal": "OPAL (ours)"}
REC_ORDER = ["Pop", "RecentPop", "Repeat", "ItemKNN", "BPR-MF", "LightGCN", "GRU4Rec", "SASRec"]
DSNAME = {"oulad": "the 2014J presentations of OULAD", "act-mooc": "act-mooc (last 15\\% of learners)"}
UP, DOWN = "$\\uparrow$", "$\\downarrow$"
NEWSUP = "\\textsuperscript{new}"
HLAB = {"oulad": ["Week 2", "Week 4", "Week 6", "Week 8", "Week 12"], "act-mooc": ["Day 1", "Day 3", "Day 5", "Day 7"]}


def load(ds):
    p = os.path.join(ROOT, "results", ds, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(m, sd, bold=False, star=False, d=3):
    core = f"\\textbf{{{m:.{d}f}}}" if bold else f"{m:.{d}f}"
    s = core if sd is None else core + f"$\\pm${sd:.{d}f}"
    return s + ("$^{*}$" if star else "")


FAMILY = [("Tabular models", ["LR", "RF", "XGBoost", "LightGBM", "HGB+Platt", "MLP"]),
          ("Sequence models", ["GRU", "Transformer"]), ("Static graph models", ["R-GCN", "HGT"])]


def table_risk_auc(ds, S):
    risk = S["risk"]; H = len(HLAB[ds])
    rows = [n for n in RISK_ORDER if n in risk] + (["opal"] if "opal" in risk else [])
    cols = [str(h) for h in range(H)] + ["mean"]
    val = lambda n, c: risk[n]["mean_AUC"] if c == "mean" else risk[n][c]["AUC"]  # noqa: E731
    best = {c: max(val(n, c)[0] for n in rows) for c in cols}
    st = S.get("stats_vs_opal", {})
    nc = len(cols) + 1
    lines = [r"\begin{table}[H]", r"\tablesize{\footnotesize}",
             "\\caption{Test AUC of at-risk prediction on " + DSNAME[ds] +
             " at each decision point (mean $\\pm$ standard deviation over seeds); the best mean in each column is in bold.}",
             f"\\label{{tab:auc-{ds}}}", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabularx}{\textwidth}{@{}l" + "C" * len(cols) + "@{}}", r"\toprule",
             r"\textbf{Model} & " + " & ".join(f"\\textbf{{{x}}}" for x in HLAB[ds] + ["Mean"]) + r"\\", r"\midrule"]

    def row(n):
        cells = []
        for c in cols:
            m, sd = val(n, c)
            star = (n != "opal" and c != "mean" and n in st and st[n][c]["p_holm"] < 0.05)
            cells.append(fmt(m, sd if risk[n]["n_seeds"] > 1 else None, bold=abs(m - best[c]) < 5e-4, star=star))
        return f"{LABEL.get(n, n)} & " + " & ".join(cells) + r"\\"
    first = True
    for fam, names in FAMILY:
        present = [n for n in names if n in risk]
        if not present:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(f"\\multicolumn{{{nc}}}{{@{{}}l}}{{\\textit{{{fam}}}}}\\\\")
        lines += [row(n) for n in present]
    if "opal" in risk:
        lines += [r"\midrule", row("opal")]
    lines += [r"\bottomrule", r"\end{tabularx}",
              r"\footnotesize{$^{*}$Seed-ensemble AUC differs from that of OPAL (DeLong test, Holm-adjusted $p<0.05$); "
              r"LR is deterministic and was trained once.}",
              r"\end{table}"]
    open(os.path.join(TAB, f"auc_{ds}.tex"), "w").write("\n".join(lines) + "\n")


def table_risk_other(S_all):
    mets = [("PR_AUC", "PR-AUC", 1), ("F1", "F1", 1), ("MCC", "MCC", 1), ("Brier", "Brier", -1), ("ECE", "ECE", -1)]
    lines = [r"\begin{table}[H]", r"\tablesize{\footnotesize}",
             r"\caption{Threshold and calibration metrics averaged over the decision points (mean over seeds). F1 and MCC use the "
             r"threshold that maximizes F1 on the validation set; Brier score and ECE are computed after temperature scaling. "
             r"Arrows show whether higher or lower is better.}",
             r"\label{tab:risk-other}", r"\setlength{\tabcolsep}{3pt}",
             
             r"\begin{tabularx}{\textwidth}{@{}l" + "C" * 10 + "@{}}", r"\toprule",
             r" & \multicolumn{5}{c}{\textbf{OULAD}} & \multicolumn{5}{c}{\textbf{act-mooc}}\\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
             r"\textbf{Model} & " + " & ".join([lab + " " + (UP if d > 0 else DOWN) for _, lab, d in mets] * 2) + r"\\",
             r"\midrule"]
    names = RISK_ORDER + ["opal"]
    best = {}
    for ds in ("oulad", "act-mooc"):
        S = S_all.get(ds)
        for key, _, d in mets:
            vals = [S["risk"][n][f"mean_{key}"][0] for n in names if S and n in S["risk"]]
            best[(ds, key)] = (max(vals) if d > 0 else min(vals)) if vals else None
    for n in names:
        if n == "opal":
            lines.append(r"\midrule")
        cells = []
        for ds in ("oulad", "act-mooc"):
            S = S_all.get(ds)
            for key, _, d in mets:
                if S and n in S["risk"]:
                    v = S["risk"][n][f"mean_{key}"][0]
                    s = f"{v:.3f}"
                    cells.append(f"\\textbf{{{s}}}" if abs(v - best[(ds, key)]) < 5e-4 else s)
                else:
                    cells.append("--")
        lines.append(f"{LABEL.get(n, n)} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    open(os.path.join(TAB, "risk_other.tex"), "w").write("\n".join(lines) + "\n")


def rec_mean(entry, mode, metric, H):
    vals = [entry[str(h)][mode][metric][0] for h in range(H) if str(h) in entry and mode in entry[str(h)]]
    return float(np.mean(vals)) if vals else None


def table_rec(S_all):
    mets = [("all", "HR@10"), ("all", "Recall@10"), ("all", "NDCG@10"), ("new", "Recall@10"), ("new", "NDCG@10")]
    head = ["HR@10", "R@10", "N@10", "R@10 (new)", "N@10 (new)"]
    lines = [r"\begin{table}[H]", r"\tablesize{\footnotesize}",
             r"\caption{Next-step resource recommendation averaged over the decision points (mean over seeds). "
             r"HR, R and N denote the hit rate, recall and NDCG at rank 10; ``new'' marks metrics over resources that the learner "
             r"has not opened before; the best value in each column is in bold.}",
             r"\label{tab:rec}", r"\setlength{\tabcolsep}{2.5pt}", 
             r"\begin{tabularx}{\textwidth}{@{}l" + "C" * 10 + "@{}}", r"\toprule",
             r" & \multicolumn{5}{c}{\textbf{OULAD}} & \multicolumn{5}{c}{\textbf{act-mooc}}\\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
             r"\textbf{Model} & " + " & ".join([h.replace(" (new)", NEWSUP) for h in head] * 2) + r"\\", r"\midrule"]
    table = {}
    for ds in ("oulad", "act-mooc"):
        S = S_all.get(ds)
        if not S:
            continue
        H = len(HLAB[ds])
        for n in REC_ORDER:
            if n in S.get("rec_baselines", {}):
                table[(n, ds)] = [rec_mean(S["rec_baselines"][n], mo, me, H) for mo, me in mets]
        if S.get("opal_rank"):
            table[("opal", ds)] = [rec_mean(S["opal_rank"], mo, me, H) for mo, me in mets]
    best = {}
    for ds in ("oulad", "act-mooc"):
        for j in range(len(mets)):
            vals = [v[j] for (n, d), v in table.items() if d == ds and v[j] is not None]
            best[(ds, j)] = max(vals) if vals else None
    for n in REC_ORDER + ["opal"]:
        if n == "opal":
            lines.append(r"\midrule")
        cells = []
        for ds in ("oulad", "act-mooc"):
            v = table.get((n, ds))
            for j in range(len(mets)):
                if v is None or v[j] is None:
                    cells.append("--"); continue
                s = f"{v[j]:.3f}"
                cells.append(f"\\textbf{{{s}}}" if abs(v[j] - best[(ds, j)]) < 5e-4 else s)
        lines.append(f"{LABEL.get(n, n)} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    open(os.path.join(TAB, "rec.tex"), "w").write("\n".join(lines) + "\n")


def fig_auc(S_all):
    """(a, b) test AUC per decision point (mean and s.d. over seeds); (c, d) AUC difference between the seed
    ensembles of OPAL and of the strongest baselines with paired-bootstrap 95% confidence intervals."""
    import matplotlib.pyplot as plt
    ps.apply()
    show = [("opal", "OPAL", 0, "o"), ("LightGBM", "LightGBM", 1, "s"), ("XGBoost", "XGBoost", 2, "^"),
            ("GRU", "GRU", 3, "D"), ("Transformer", "Transformer", 4, "v"), ("HGT", "HGT", 6, "P")]
    fig, axes = plt.subplots(2, 2, figsize=(ps.FULL_W, 11.0 * ps.CM))
    for j, ds in enumerate(("oulad", "act-mooc")):
        S = S_all.get(ds)
        if not S:
            continue
        H = len(HLAB[ds]); xs = np.arange(H)
        ax = axes[0, j]
        for key, name, ci, mk in show:
            if key not in S["risk"]:
                continue
            m = [S["risk"][key][str(h)]["AUC"][0] for h in range(H)]
            sd = [S["risk"][key][str(h)]["AUC"][1] for h in range(H)]
            ax.errorbar(xs, m, yerr=sd, color=ps.PALETTE[ci], marker=mk, lw=2.0 if key == "opal" else 1.1,
                        ms=4.5 if key == "opal" else 3.5, capsize=2, elinewidth=0.7, label=name,
                        zorder=3 if key == "opal" else 2)
        ax.set_xticks(xs); ax.set_xticklabels(HLAB[ds]); ax.set_ylabel("Test AUC")
        ax.set_title("(" + "ab"[j] + ") " + ("OULAD" if ds == "oulad" else "act-mooc") + ": AUC per decision point",
                     fontsize=8.5, loc="left", fontweight="bold")
        ax = axes[1, j]
        st = S.get("stats_vs_opal", {})
        comp = [(k, n, c, mk) for k, n, c, mk in show[1:] if k in st]
        off = np.linspace(-0.24, 0.24, max(len(comp), 1))
        for (key, name, ci, mk), o in zip(comp, off):
            d = [st[key][str(h)]["boot"] for h in range(H)]
            mid = np.array([x["diff"] for x in d]) * 100
            lo = mid - np.array([x["ci_low"] for x in d]) * 100
            hi = np.array([x["ci_high"] for x in d]) * 100 - mid
            ax.errorbar(xs + o, mid, yerr=[lo, hi], color=ps.PALETTE[ci], marker=mk, ls="none", ms=3.5, capsize=2,
                        elinewidth=0.8, label=name)
        ax.axhline(0, color=ps.GREY, lw=0.8, zorder=1)
        ax.set_xticks(xs); ax.set_xticklabels(HLAB[ds])
        ax.set_ylabel("$\\Delta$AUC, OPAL $-$ model (points)")
        ax.set_title("(" + "cd"[j] + ") " + ("OULAD" if ds == "oulad" else "act-mooc") + ": difference with 95% CI",
                     fontsize=8.5, loc="left", fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.4)
    ps.save(fig, os.path.join(FIG, "fig3_auc"))
    plt.close(fig)


def fig_rec(S_all):
    import matplotlib.pyplot as plt
    ps.apply()
    show = [("opal", "OPAL", 0, "o"), ("SASRec", "SASRec", 1, "s"), ("GRU4Rec", "GRU4Rec", 2, "^"),
            ("RecentPop", "RecentPop", 3, "D")]
    fig, axes = plt.subplots(2, 2, figsize=(ps.FULL_W, 9.5 * ps.CM))
    axes = axes.ravel()
    k = 0
    for ds in ("oulad", "act-mooc"):
        S = S_all.get(ds)
        for mode, title in (("all", "all resources"), ("new", "new resources")):
            ax = axes[k]; k += 1
            if not S:
                continue
            H = len(HLAB[ds]); xs = np.arange(H)
            for key, name, ci, mk in show:
                src = S.get("opal_rank") if key == "opal" else S.get("rec_baselines", {}).get(key)
                if not src:
                    continue
                m = [src[str(h)][mode]["NDCG@10"][0] if str(h) in src and mode in src[str(h)] else np.nan for h in range(H)]
                ax.plot(xs, m, color=ps.PALETTE[ci], marker=mk, lw=2.0 if key == "opal" else 1.1,
                        ms=4.5 if key == "opal" else 3.5, label=name, zorder=3 if key == "opal" else 2)
            ax.set_xticks(xs); ax.set_xticklabels([x.split()[1] for x in HLAB[ds]])
            ax.set_xlabel("Week" if ds == "oulad" else "Day")
            ax.set_title("(" + "abcd"[k - 1] + ") " + ("OULAD" if ds == "oulad" else "act-mooc") + ", " + title,
                         fontsize=8.5, loc="left", fontweight="bold")
            ax.set_ylabel("NDCG@10")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.2)
    ps.save(fig, os.path.join(FIG, "fig4_rec"))
    plt.close(fig)


def table_fairness(S):
    """Fairness of the top-20% alert policy on OULAD, averaged over the decision points."""
    fa = S.get("fairness", {})
    models = [n for n in ("opal", "LightGBM", "XGBoost", "HGB+Platt", "GRU") if n in fa]
    atts = [("gender", "Gender"), ("age_band", "Age band"), ("imd_band", "IMD band"), ("disability", "Disability")]
    H = len(HLAB["oulad"])
    val = {}
    for n in models:
        for a, _ in atts:
            g = [fa[n][str(h)]["gaps"][a] for h in range(H)]
            val[(n, a)] = (np.mean([x["dTPR"] for x in g]), np.mean([x["dFPR"] for x in g]),
                           np.mean([fa[n][str(h)]["abroca"][a] for h in range(H)]))
    lines = [r"\begin{table}[H]", r"\tablesize{\footnotesize}",
             r"\caption{Comparison of group fairness for an alert policy that flags the 20\% of eligible OULAD learners with the "
             r"highest predicted risk. $\Delta$TPR and $\Delta$FPR are the largest differences in true and false positive rate "
             r"between groups, and ABR is the ABROCA, the absolute area between the groups' ROC curves; all values are averaged over the "
             r"five decision points (lower is fairer).}",
             r"\label{tab:fairness}", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabularx}{\textwidth}{@{}l" + "C" * (3 * len(atts)) + "@{}}", r"\toprule",
             r" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{lab}}}}}" for _, lab in atts) + r"\\",
             "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(atts))),
             r"\textbf{Model} & " + " & ".join([r"$\Delta$TPR & $\Delta$FPR & ABR"] * len(atts)) + r"\\", r"\midrule"]
    for n in models:
        cells = []
        for a, _ in atts:
            for j in range(3):
                v = val[(n, a)][j]
                best = min(val[(m_, a)][j] for m_ in models)
                cells.append(f"\\textbf{{{v:.3f}}}" if abs(v - best) < 5e-4 else f"{v:.3f}")
        lines.append(f"{LABEL.get(n, n)} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    open(os.path.join(TAB, "fairness.tex"), "w").write("\n".join(lines) + "\n")


def fig_calibration(S_all):
    """Reliability diagrams of the calibrated seed ensembles, pooled over the decision points."""
    import matplotlib.pyplot as plt
    ps.apply()
    show = [("opal_raw", "OPAL, before scaling", 0, "o", "--"), ("opal", "OPAL", 0, "o", "-"),
            ("LightGBM", "LightGBM", 1, "s", "-"), ("GRU", "GRU", 3, "D", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 7.0 * ps.CM))
    for ax, ds, lab in zip(axes, ("oulad", "act-mooc"), "ab"):
        S = S_all.get(ds)
        ax.plot([0, 1], [0, 1], color=ps.GREY, lw=0.8, ls=":", zorder=1)
        if S and S.get("reliability"):
            for key, name, ci, mk, ls_ in show:
                r = S["reliability"].get(key)
                if not r:
                    continue
                pts = [(b["p"], b["y"]) for b in r if b["n"] >= 30]
                ax.plot([p for p, _ in pts], [y for _, y in pts], color=ps.PALETTE[ci], marker=mk, ls=ls_,
                        lw=1.8 if key == "opal" else 1.1, ms=3.5, label=name, mfc="white" if key == "opal_raw" else None)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("Mean predicted risk"); ax.set_ylabel("Observed rate")
        ax.set_title(f"({lab}) " + ("OULAD" if ds == "oulad" else "act-mooc"), fontsize=9, loc="left", fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    ps.save(fig, os.path.join(FIG, "fig5_calibration"))
    plt.close(fig)


def table_efficiency(S_all):
    """Parameters, training time and scoring time.  Uses results/efficiency.json (controlled benchmark) when it
    exists, otherwise the times logged by the main runs."""
    import glob
    p = os.path.join(ROOT, "results", "efficiency.json")
    eff = json.load(open(p)) if os.path.exists(p) else {}
    rows = [("opal", "OPAL (ours)", "Risk + rec."), ("GRU", "GRU", "Risk"), ("Transformer", "Transformer", "Risk"),
            ("R-GCN", "R-GCN", "Risk"), ("HGT", "HGT", "Risk"), ("LightGBM", "LightGBM", "Risk"),
            ("XGBoost", "XGBoost", "Risk"), ("SASRec", "SASRec", "Rec."), ("GRU4Rec", "GRU4Rec", "Rec.")]

    def get(ds, n):
        S = S_all.get(ds) or {}
        e = eff.get(ds, {}).get(n)
        if e:
            return e.get("n_params"), e["train_sec"], e.get("infer_sec"), e.get("fits")
        if n in S.get("risk", {}):
            r = S["risk"][n]
            return r.get("n_params"), r["train_sec"], r["infer_sec"], None
        fs = sorted(glob.glob(os.path.join(ROOT, "results", ds, "rec", f"{n}_seed*.json")))
        if fs:
            return None, float(np.mean([json.load(open(x))["_sec"] for x in fs])), None, None
        return None
    lines = [r"\begin{table}[H]", r"\tablesize{\footnotesize}",
             r"\caption{Computational cost: number of parameters, training time and the time to score all test learners at every "
             r"decision point, measured with one CPU thread on an otherwise idle machine (seed 0). For SASRec and GRU4Rec, the "
             r"training time covers all fits on the test cohorts (one per test presentation and decision point); the tree "
             r"ensembles, fitted once per decision point, are timed in the main runs.}",
             r"\label{tab:efficiency}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabularx}{\textwidth}{@{}llCCCCC@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{\textbf{OULAD}} & \multicolumn{2}{c}{\textbf{act-mooc}}\\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"\textbf{Model} & \textbf{Service} & \textbf{Parameters}$^{a}$ & Train (min) & Score (s) & Train (min) & Score (s)\\",
             r"\midrule"]
    for n, lab, svc in rows:
        cells = []
        par = None
        for ds in ("oulad", "act-mooc"):
            g = get(ds, n)
            if g is None:
                cells += ["--", "--"]; continue
            par = par or g[0]
            small = lambda v: "$<$0.1" if v < 0.05 else f"{v:.1f}"  # noqa: E731
            cells += [small(g[1] / 60), small(g[2]) if g[2] is not None else "--"]
        lines.append(f"{lab} & {svc} & " + (f"{int(par):,}" if par else "--") + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\footnotesize{$^{a}$Trainable parameters of the OULAD model.}", r"\end{table}"]
    open(os.path.join(TAB, "efficiency.tex"), "w").write("\n".join(lines) + "\n")


def fig_protocol_effect():
    """AUC of the same LightGBM model under different evaluation protocols (OULAD)."""
    import matplotlib.pyplot as plt
    p = os.path.join(ROOT, "results", "oulad", "protocol_inflation.json")
    if not os.path.exists(p):
        return
    pr = json.load(open(p))
    ps.apply()
    xs = np.arange(len(HLAB["oulad"]))
    fig, ax = plt.subplots(figsize=(ps.FULL_W * 0.75, 7.6 * ps.CM))
    show = [("temporal", None, "Next-cohort protocol (this paper)", 0, "o"),
            ("random_same_size", "random_same_size_sd", "Random assignment, same set sizes", 1, "s"),
            ("random", "random_sd", "Random 70/15/15, all presentations", 2, "^"),
            ("within_presentation", "within_presentation_sd", "Random 70/15/15 within 2014J", 3, "D")]
    for key, sdk, lab, ci, mk in show:
        if not pr.get(key):
            continue
        ax.errorbar(xs, pr[key], yerr=pr[sdk] if sdk else None, color=ps.PALETTE[ci], marker=mk,
                    lw=2.0 if key == "temporal" else 1.1, ms=4, capsize=2, elinewidth=0.7, label=lab,
                    zorder=3 if key == "temporal" else 2)
    ax.plot([0], [pr["leaky_temporal"]], marker="*", ms=9, ls="none", color=ps.PALETTE[7],
            label="End-of-course features (week-2 learners)")
    ax.set_xticks(xs); ax.set_xticklabels([x.split()[1] for x in HLAB["oulad"]])
    ax.set_xlabel("Week of the decision point"); ax.set_ylabel("Test AUC of LightGBM")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
    fig.tight_layout()
    ps.save(fig, os.path.join(FIG, "fig6_protocol"))
    plt.close(fig)


ABL = [("abl_no_graph", "w/o resource$\\to$learner messages"), ("abl_no_coevo", "w/o learner$\\to$resource updates"),
       ("abl_no_rec", "w/o recommendation head"), ("abl_no_peer", "w/o peer normalization"),
       ("abl_no_featpath", "w/o feature path of the risk head"), ("abl_no_assess", "w/o assessment features")]


def table_ablation(S):
    """Ablation study on OULAD: every variant against the full model on the same seeds."""
    H = len(HLAB["oulad"])
    if not S.get("ablation_full"):              # placeholder until the ablation runs are analysed
        S = dict(S, ablation={}, ablation_full={"risk": S["risk"]["opal"], "rank": S.get("opal_rank", {})})
    sg = lambda x: "0.0" if round(x, 1) == 0 else ("$+$" if x > 0 else "$-$") + f"{abs(x):.1f}"  # noqa: E731
    full = S["ablation_full"]
    fa = full["risk"]["mean_AUC"]
    fr = full.get("rank", {})
    fn = {m_: float(np.mean([fr[str(h)][m_]["NDCG@10"][0] for h in range(H)])) if fr else None for m_ in ("all", "new")}
    lines = [r"\begin{table}[H]", r"\tablesize{\footnotesize}",
             r"\caption{Ablation study of OPAL on OULAD (mean $\pm$ standard deviation over seeds 0--2; $\Delta$ is the "
             r"difference to the full model on the same seeds, in points; ``Sig.'' counts the decision points at which the "
             r"ensemble AUC differs from the full model, DeLong test, Holm-adjusted $p<0.05$).}",
             r"\label{tab:ablation}", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabularx}{\textwidth}{@{}lCCCCCC@{}}", r"\toprule",
             r"\textbf{Variant} & \textbf{Mean AUC} & $\boldsymbol{\Delta}$\textbf{AUC} & \textbf{Sig.} & "
             r"\textbf{N@10} & \textbf{N@10}\textsuperscript{\textbf{new}} & $\boldsymbol{\Delta}$\textbf{N@10}\textsuperscript{\textbf{new}}\\",
             r"\midrule",
             f"OPAL (full) & {fa[0]:.3f}$\\pm${fa[1]:.3f} & -- & -- & "
             + (f"{fn['all']:.3f} & {fn['new']:.3f} & --" if fn["all"] is not None else "-- & -- & --") + r"\\"]
    for key, lab in ABL:
        a = S["ablation"].get(key)
        if not a:
            lines.append(f"{lab} & \\multicolumn{{6}}{{c}}{{\\textit{{running}}}}\\\\"); continue
        m_, sd = a["mean_AUC"]
        dl = a.get("delong_vs_full", {})
        nsig = sum(1 for h in range(H) if str(h) in dl and dl[str(h)]["p_holm"] < 0.05)
        rk = S.get("ablation_rank", {}).get(key)
        if rk:
            na = float(np.mean([rk[str(h)]["all"]["NDCG@10"][0] for h in range(H)]))
            nn = float(np.mean([rk[str(h)]["new"]["NDCG@10"][0] for h in range(H)]))
            rec = f"{na:.3f} & {nn:.3f} & {sg(100 * (nn - fn['new']))}"
        else:
            rec = "-- & -- & --"
        lines.append(f"{lab} & {m_:.3f}$\\pm${sd:.3f} & {sg(100 * (m_ - fa[0]))} & {nsig}/{H} & {rec}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    open(os.path.join(TAB, "ablation.tex"), "w").write("\n".join(lines) + "\n")


def main():
    S_all = {ds: load(ds) for ds in ("oulad", "act-mooc")}
    for ds, S in S_all.items():
        if S and "risk" in S:
            table_risk_auc(ds, S)
    table_risk_other(S_all)
    table_rec(S_all)
    table_efficiency(S_all)
    if S_all.get("oulad"):
        table_ablation(S_all["oulad"])
    fig_protocol_effect()
    fig_auc(S_all)
    fig_rec(S_all)
    if S_all.get("oulad") and S_all["oulad"].get("fairness"):
        table_fairness(S_all["oulad"])
    if any(S and S.get("reliability") for S in S_all.values()):
        fig_calibration(S_all)
    os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
    for f in os.listdir(FIG):
        if f.endswith(".pdf"):
            shutil.copy(os.path.join(FIG, f), os.path.join(PAPER, "figures", f))
    print("tables:", sorted(os.listdir(TAB)))


if __name__ == "__main__":
    main()
