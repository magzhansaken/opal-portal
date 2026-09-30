"""Insert the main results (from results/<dataset>/summary.json) into README.md between the markers
<!-- RESULTS START --> and <!-- RESULTS END -->.

usage: python scripts/update_readme.py [README.md]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "README.md")
NAMES = {"opal": "**OPAL**", "HGB+Platt": "HGB + Platt"}
ORDER = ["opal", "LightGBM", "XGBoost", "RF", "HGB+Platt", "GRU", "Transformer", "R-GCN", "HGT", "MLP", "LR"]
REC = ["opal", "SASRec", "GRU4Rec", "RecentPop", "Repeat", "LightGCN", "BPR-MF"]


def load(ds):
    p = os.path.join(ROOT, "results", ds, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def risk_rows(S, labels):
    out = ["| Model | " + " | ".join(labels) + " | Mean |", "|---|" + "---|" * (len(labels) + 1)]
    for n in ORDER:
        if n not in S["risk"]:
            continue
        r = S["risk"][n]
        vals = [f"{r[str(h)]['AUC'][0]:.3f}" for h in range(len(labels))]
        out.append(f"| {NAMES.get(n, n)} | " + " | ".join(vals) + f" | {r['mean_AUC'][0]:.3f} ± {r['mean_AUC'][1]:.3f} |")
    return out


def rec_rows(S, H):
    out = ["| Model | NDCG@10 (all) | NDCG@10 (new) | Recall@10 (new) |", "|---|---|---|---|"]
    for n in REC:
        src = S.get("opal_rank") if n == "opal" else S.get("rec_baselines", {}).get(n)
        if not src:
            continue
        mean = lambda mode, met: np.mean([src[str(h)][mode][met][0] for h in range(H)])  # noqa: E731
        out.append(f"| {NAMES.get(n, n)} | {mean('all', 'NDCG@10'):.3f} | {mean('new', 'NDCG@10'):.3f} | {mean('new', 'Recall@10'):.3f} |")
    return out


lines = []
So, Sm = load("oulad"), load("act-mooc")
if So:
    lines += ["### Early warning, OULAD (test presentations 2014J, AUC, mean over seeds)", ""]
    lines += risk_rows(So, ["Week 2", "Week 4", "Week 6", "Week 8", "Week 12"]) + [""]
    lines += ["### Next-week recommendation, OULAD (mean over decision points)", ""] + rec_rows(So, 5) + [""]
if Sm:
    lines += ["### Early warning, act-mooc (last 15 % of learners, AUC)", ""]
    lines += risk_rows(Sm, ["Day 1", "Day 3", "Day 5", "Day 7"]) + [""]
    lines += ["### Next-day recommendation, act-mooc", ""] + rec_rows(Sm, 4) + [""]
if So and So.get("ablation_full"):
    AB = [("abl_no_graph", "w/o resource-to-learner messages"), ("abl_no_coevo", "w/o learner-to-resource updates"),
          ("abl_no_rec", "w/o recommendation head"), ("abl_no_peer", "w/o peer normalization"),
          ("abl_no_featpath", "w/o feature path of the risk head"), ("abl_no_assess", "w/o assessment features")]
    fa = So["ablation_full"]["risk"]["mean_AUC"][0]
    fr = So["ablation_full"].get("rank", {})
    fnew = np.mean([fr[str(h)]["new"]["NDCG@10"][0] for h in range(5)]) if fr else None
    lines += ["### Ablation study, OULAD (seeds 0-2)", "", "| Variant | Mean AUC | ΔAUC (points) | NDCG@10 new |", "|---|---|---|---|",
              f"| OPAL (full) | {fa:.3f} | – | {fnew:.3f} |" if fnew is not None else f"| OPAL (full) | {fa:.3f} | – | – |"]
    for key, lab in AB:
        a = So["ablation"].get(key)
        if not a:
            continue
        rk = So.get("ablation_rank", {}).get(key)
        nn = f"{np.mean([rk[str(h)]['new']['NDCG@10'][0] for h in range(5)]):.3f}" if rk else "–"
        lines.append(f"| {lab} | {a['mean_AUC'][0]:.3f} | {100 * (a['mean_AUC'][0] - fa):+.1f} | {nn} |")
    lines.append("")
pp = os.path.join(ROOT, "results", "oulad", "protocol_inflation.json")
if os.path.exists(pp):
    pr = json.load(open(pp))
    lines += ["### Effect of the evaluation protocol (LightGBM, OULAD, AUC at weeks 2 / 12)", "",
              "| Protocol | Week 2 | Week 12 |", "|---|---|---|",
              f"| Next-cohort (this work) | {pr['temporal'][0]:.3f} | {pr['temporal'][-1]:.3f} |",
              f"| Random 70/15/15, all presentations | {pr['random'][0]:.3f} | {pr['random'][-1]:.3f} |"]
    if pr.get("within_presentation"):
        lines.append(f"| Random 70/15/15 within 2014J | {pr['within_presentation'][0]:.3f} | {pr['within_presentation'][-1]:.3f} |")
    lines += [f"| End-of-course features (week-2 learners) | {pr['leaky_temporal']:.3f} | – |", ""]
text = open(README).read()
a, b = "<!-- RESULTS START -->", "<!-- RESULTS END -->"
if a in text and b in text:
    text = text[:text.index(a) + len(a)] + "\n\n" + "\n".join(lines) + "\n" + text[text.index(b):]
    open(README, "w").write(text)
    print("README updated")
else:
    print("\n".join(lines))
