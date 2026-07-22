# -*- coding: utf-8 -*-
"""
plot_scalability.py  ->  figures/scalability.(png|pdf)
Figura de escalabilidad a partir del CSV nuevo (con screening_active,
n_solved_screened, time_screened). Dos paneles:
  (a) tiempo de pared vs |R|   (mediana + banda min-max)
  (b) subproblemas resueltos e iteraciones de Dinkelbach vs |R|  (medianas)
Uso:  python plot_scalability.py [ruta_csv]
"""
import os, sys, csv, statistics as st
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_CSV = os.path.join(_REPO_ROOT, "output", "scal_final.csv")
_FIGURES_DIR = os.path.join(_REPO_ROOT, "figures")

CSV = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_CSV
FORMOSE_R = 38

rows = list(csv.DictReader(open(CSV, newline="")))
sizes = []
for r in rows:
    key = int(r["n_reactions"])
    if key not in sizes:
        sizes.append(key)
sizes.sort()

def agg(col, fn=st.median):
    xs, ys, lo, hi = [], [], [], []
    for R in sizes:
        vals = [float(r[col]) for r in rows if int(r["n_reactions"]) == R]
        xs.append(R); ys.append(fn(vals)); lo.append(min(vals)); hi.append(max(vals))
    return xs, ys, lo, hi

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) tiempo
x, y, lo, hi = agg("time_screened")
ax1.plot(x, y, "o-", color="#1a5e9e", zorder=3, label="median")
ax1.fill_between(x, lo, hi, color="#1a5e9e", alpha=0.18, label="min--max")
ax1.axvline(FORMOSE_R, color="#888", ls=":", zorder=0)
ax1.annotate("formose", xy=(FORMOSE_R, ax1.get_ylim()[1]), xytext=(3, -10),
             textcoords="offset points", color="#666", fontsize=8)
ax1.set_yscale("log")   # rango 0.3 s -- 10345 s: log obligatorio
ax1.set_xlabel(r"number of reactions $|\mathcal{R}|$")
ax1.set_ylabel("wall-clock time (s)")
ax1.set_title("(a) Runtime")
ax1.legend(fontsize=8)

# (b) subproblemas e iteraciones (medianas)
_, ysub, *_ = agg("n_solved_screened")
_, yit,  *_ = agg("iter_dinkelbach")
ax2.plot(sizes, ysub, "o-", color="#1a5e9e", label="subproblems solved")
ax2.plot(sizes, yit,  "s--", color="#c0392b", label="Dinkelbach iterations")
ax2.axvline(FORMOSE_R, color="#888", ls=":", zorder=0)
ax2.set_xlabel(r"number of reactions $|\mathcal{R}|$")
ax2.set_ylabel("count (median)")
ax2.set_title("(b) Subproblems and iterations")
ax2.legend(fontsize=8)

fig.tight_layout()
os.makedirs(_FIGURES_DIR, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(_FIGURES_DIR, f"scalability.{ext}"), dpi=200, bbox_inches="tight")
print(f"figuras -> {os.path.join(_FIGURES_DIR, 'scalability.png')} y .pdf")