"""Supplementary Fig. S1: evaluation protocol. (a) presentation-level split of OULAD on the calendar;
(b) information available at a weekly cutoff k and the targets of the two tasks."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from opal import plotstyle as ps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
st = json.load(open(os.path.join(ROOT, "results", "dataset_stats.json")))["oulad"]
ps.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(ps.FULL_W, 7.0 * ps.CM), gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.55})
C_TR, C_VA, C_TE, C_OBS, C_FUT = ps.PALETTE[0], ps.PALETTE[3], ps.PALETTE[1], "#cfe0f5", "#f2f2f2"

# (a) calendar of presentations: start months Feb (B) and Oct (J); ~9 months long
pres = [("2013B", 2013 + 1 / 12, C_TR, "Training", st["train"]["n_cohorts"]),
        ("2013J", 2013 + 9 / 12, C_TR, "Training", None),
        ("2014B", 2014 + 1 / 12, C_VA, "Validation", st["val"]["n_cohorts"]),
        ("2014J", 2014 + 9 / 12, C_TE, "Test", st["test"]["n_cohorts"])]
for i, (name, x0, col, role, _) in enumerate(pres):
    y = 0.62 if i % 2 == 0 else 0.12
    tint = {C_TR: "#cfe0f7", C_VA: "#fbe3b0", C_TE: "#f9d3c3"}[col]
    ax1.add_patch(FancyBboxPatch((x0, y), 0.72, 0.34, boxstyle="round,pad=0,rounding_size=0.02", fc=tint, ec=col, lw=1.2))
    ax1.text(x0 + 0.36, y + 0.17, f"{name} ({role.lower()})", ha="center", va="center", color="black", fontsize=8, fontweight="bold")
ax1.set_xlim(2012.95, 2016.35); ax1.set_ylim(0, 1.05)
ax1.set_yticks([]); ax1.grid(False); ax1.spines["left"].set_visible(False)
ax1.set_xticks([2013, 2013.5, 2014, 2014.5, 2015, 2015.5])
ax1.set_xticklabels(["Jan 2013", "Jul 2013", "Jan 2014", "Jul 2014", "Jan 2015", "Jul 2015"])
ax1.set_xlabel("Calendar time (presentations start in February, B, or October, J)")
txt = (f"Training: {st['train']['n']:,} learners\nValidation: {st['val']['n']:,} learners\n"
       f"Test: {st['test']['n']:,} learners")
ax1.text(2015.72, 0.52, txt, ha="left", va="center", fontsize=8)
ps.panel_label(ax1, "a")

# (b) weekly timeline of one presentation with a cutoff at week k
k = 6; W = 10
for w in range(0, W + 1):
    col = C_OBS if w <= k else C_FUT
    ax2.add_patch(Rectangle((w, 0.35), 0.92, 0.4, fc=col, ec="white", lw=0.8))
    ax2.text(w + 0.46, 0.55, str(w), ha="center", va="center", fontsize=8)
ax2.text(W + 1.25, 0.55, "...", ha="center", va="center", fontsize=8)
ax2.add_patch(Rectangle((W + 1.7, 0.35), 1.9, 0.4, fc=C_FUT, ec="white", lw=0.8))
ax2.text(W + 2.65, 0.55, "end", ha="center", va="center", fontsize=8)
ax2.text(W + 2.65, 0.95, r"final result $\rightarrow y_u$", ha="center", va="bottom", fontsize=8)
ax2.add_patch(Rectangle((k + 1, 0.35), 0.92, 0.4, fc="none", ec=C_TE, lw=1.4))
ax2.axvline(k + 0.96, ymin=0.25, ymax=0.9, color="black", lw=1.0, ls=(0, (3, 2)))
ax2.text(k + 0.96, 0.98, f"cutoff $k$ (here $k$ = {k})", ha="center", va="bottom", fontsize=8)
ax2.annotate("", xy=(0.0, 0.2), xytext=(k + 0.92, 0.2), arrowprops=dict(arrowstyle="<->", lw=0.8))
ax2.text((k + 0.92) / 2, 0.04, r"observed $\mathcal{O}^{(\leq k)}$: own and peer activity, submissions", ha="center", va="center", fontsize=8)
ax2.text(k + 1.0, -0.14, r"targets $\mathcal{T}_u^{(k)}$: resources opened in week $k+1$", ha="left", va="center", fontsize=8, color="#b9481c")
ax2.set_xlim(-1.2, W + 3.8); ax2.set_ylim(-0.26, 1.15)
ax2.set_yticks([]); ax2.set_xticks([]); ax2.grid(False)
for sp in ("left", "bottom"):
    ax2.spines[sp].set_visible(False)
ax2.text(-0.15, 0.55, "week", ha="right", va="center", fontsize=8)
ps.panel_label(ax2, "b")
out = os.path.join(ROOT, "figures", "fig1_protocol")
ps.save(fig, out)
print("saved", out)
