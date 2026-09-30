"""Generate LaTeX tables that are filled from result files (so the text never disagrees with the data)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)
st = json.load(open(os.path.join(ROOT, "results", "dataset_stats.json")))
o, a = st["oulad"], st["act-mooc"]
f = lambda n: f"{int(round(n)):,}"
p = lambda x: f"{100 * x:.1f}"
rows = [
    ("Cohorts; time step", f"{o['presentations']} presentations of {o['modules']} modules; week", "1 course; day"),
    ("Enrollments; resources", f"{f(o['enrollments'])}; {f(o['resources'])}", f"{f(a['users'])}; {f(a['items'])}"),
    ("Clicks or actions; learner--resource edges", f"{f(o['clicks'])}; {f(o['edges'])}", f"{f(a['actions'])}; {f(a['edges'])}"),
    ("Features $\\mathbf{x}_u^{(s)}$ / $\\mathbf{z}_u$ / $\\mathbf{d}_r$", f"{o['x_dim']} / {o['z_dim']} / {o['d_dim']}", f"{a['x_dim']} / 2 / {a['items']} (one-hot)"),
    ("Training / validation / test learners", " / ".join(f(o[k]['n']) for k in ("train", "val", "test")),
     " / ".join(f(a[k]['n']) for k in ("train", "val", "test"))),
    ("Positive \\% (training / validation / test)", " / ".join(p(o[k]['pos_rate']) for k in ("train", "val", "test")),
     " / ".join(p(a[k]['pos_rate']) for k in ("train", "val", "test"))),
    ("Split", "2013B+2013J / 2014B / 2014J", "70\\% / 15\\% / 15\\% of learners by first action"),
]
lines = [r"\begin{table}[H]", r"\tablesize{\footnotesize}",
         r"\caption{Statistics of the two datasets after preprocessing; positive \% is the share of learners with $y_u=1$.}",
         r"\label{tab:datasets}", r"\renewcommand{\arraystretch}{1.1}",
         r"\begin{tabularx}{\textwidth}{@{}p{5.2cm}LL@{}}", r"\toprule",
         r"\textbf{Property} & \textbf{OULAD} & \textbf{act-mooc}\\", r"\midrule"]
lines += [f"{r} & {x} & {y}\\\\" for r, x, y in rows]
lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
open(os.path.join(OUT, "datasets.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
