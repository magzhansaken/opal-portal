"""Fig. 1 of the article: OPAL architecture for one weekly step and its place in the portal."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from opal import plotstyle as ps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ps.apply()
plt.rcParams["axes.grid"] = False
fig, ax = plt.subplots(figsize=(ps.FULL_W, 8.8 * ps.CM))
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
BLUE, ORANGE, GREEN, YELLOW, PINK = ps.PALETTE[0], ps.PALETTE[1], ps.PALETTE[2], ps.PALETTE[3], ps.PALETTE[4]
LB, LY, GR = "#e6effa", "#fdf3dc", "#f0f0f0"
# light fills with a coloured outline and black text for the processing blocks (legible in print and greyscale)
T_BLUE, T_YEL, T_GREEN, T_PINK = "#cfe0f7", "#fbe3b0", "#cdeee0", "#f8d9e5"
DARK_OR = "#b9481c"


def box(x, y, w, h, text, fc, tc="black", fs=8, ec=None, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=fc, ec=ec or fc, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.3)


def arrow(p1, p2, col="black", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=8, color=col, lw=0.9, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# top and bottom input bars
box(14, 48.5, 36, 6.5, "Resource inputs: descriptor $\\mathbf{d}_r$,\npopularity, users of $r$ in week $s{-}1$", LY, fs=8)
box(1, 1.0, 49, 6.5, "Learner inputs of week $s$: behavior $\\mathbf{x}_u^{(s)}$, running\nmeans, inactivity streak, peer $z$-scores", LB, fs=8)
# resource row
box(1, 31, 12, 11, "$\\boldsymbol{\\rho}_r^{(s-1)}$", LY, fs=9)
box(20, 31, 24, 11, "Resource update", T_YEL, fs=8.5, ec=YELLOW, lw=1.2)
box(51, 31, 11, 11, "$\\boldsymbol{\\rho}_r^{(s)}$", LY, fs=9)
# learner row
box(1, 12, 12, 11, "$\\mathbf{h}_u^{(s-1)}$", LB, fs=9)
box(20, 12, 24, 11, "Learner update\n(gated GRU cell)", T_BLUE, fs=8.5, ec=BLUE, lw=1.2)
box(51, 12, 11, 11, "$\\mathbf{h}_u^{(s)}$", LB, fs=9)
arrow((13.4, 36.5), (19.6, 36.5)); arrow((44.4, 36.5), (50.6, 36.5))
arrow((13.4, 17.5), (19.6, 17.5)); arrow((44.4, 17.5), (50.6, 17.5))
arrow((32, 48.1), (32, 42.4)); arrow((32, 7.9), (32, 11.6))
# cross messages between the rows
arrow((7, 23.4), (22, 30.6), col=ORANGE, rad=-0.25)
ax.text(13.2, 25.6, "users of $r$ in $s{-}1$", fontsize=8, color=DARK_OR, ha="left", va="center")
arrow((56, 30.6), (42, 23.4), col=ORANGE, rad=-0.2)
ax.text(35.0, 28.4, "resources opened in $s$", fontsize=8, color=DARK_OR, ha="left", va="center")
ax.text(52.2, 5.0, "every week, for all\nenrolled learners", fontsize=8, color="#555555", ha="left", va="center")
# heads and portal
box(68, 29, 31, 13, "Risk head\nMLP + wide linear path\n$\\rightarrow$ calibrated risk $\\hat{p}_u^{(k)}$", T_GREEN, fs=8, ec=GREEN, lw=1.2)
box(68, 10, 31, 14, "Recommendation head\n$\\mathbf{q}(\\mathbf{h}_u^{(k)})^{\\top}\\mathbf{k}(\\boldsymbol{\\rho}_r^{(k+1)})$ + popularity\n$\\rightarrow$ top-$K$ resources", T_PINK, fs=8, ec=PINK, lw=1.2)
box(68, 47, 31, 8, "Portal: alerts for tutors,\nnext-week resources for learners", GR, fs=8)
arrow((62.4, 19.5), (67.6, 33.5), rad=0.15); arrow((62.4, 16.5), (67.6, 16.5))
arrow((62.4, 36.5), (67.6, 20.5), col=ORANGE, rad=-0.1)
arrow((83.5, 42.4), (83.5, 46.6))
ax.text(83.5, 4.8, "joint training with an\nuncertainty-weighted loss", fontsize=8, color="#555555", ha="center", va="center")
out = os.path.join(ROOT, "figures", "fig2_architecture")
ps.save(fig, out)
print("saved", out)
