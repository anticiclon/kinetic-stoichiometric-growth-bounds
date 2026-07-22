# -*- coding: utf-8 -*-
"""
drawing_demo.py
===============
Standalone usage examples for src/drawing.py (network diagrams and Pareto
frontier plots). Not referenced by any figure in the paper; kept separately
so that src/drawing.py can be imported as a clean plotting library.

Run from the repository root:
    python experiments/drawing_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np

from drawing import plotOptimalSubnetwork, plot_pareto

import warnings
try:
    from algorithm_3 import computeParametricSweep
    from scenario_generator import CORES, _block_diagonal, _sample_k
except ImportError:
    raise SystemExit("Necesitas algorithm_3.py y scenario_generator.py.")

SWEEP = dict(decimals=1, max_points=300, time_limit_iteration=60,
             max_steps=100, accuracy=1e-5)

# --- Ejemplo 1: mutual2 + ruido [5 esp × 10 rxn] ---
print("Ejemplo 1: mutual2 + ruido  [5 esp × 10 rxn]")
rng = np.random.default_rng(17)
Sm_c = CORES["mutual2"]["S_minus"].astype(float)
Sp_c = CORES["mutual2"]["S_plus"].astype(float)
n_cs, n_cr = Sm_c.shape
n_ts, n_tr = 5, 10
Sm = np.zeros((n_ts, n_tr)); Sp = np.zeros((n_ts, n_tr))
Sm[:n_cs, :n_cr] = Sm_c;    Sp[:n_cs, :n_cr] = Sp_c
for a in range(n_cr, n_tr):
    for mat in (Sm, Sp):
        sv = rng.choice(n_ts, size=rng.integers(1, 4), replace=False)
        for s in sv:
            mat[s, a] = rng.integers(1, 3)
k = _sample_k(n_tr, (0.2, 4.0), "loguniform", rng)

with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    best, _, _ = computeParametricSweep(Sm, Sp, k, **SWEEP)

plotOptimalSubnetwork(
    Sm, Sp, best,
    simplify=True,
    save_path=os.path.join("..", "figures", "network_mutual2_noise.png"),
)
print("  Guardado en network_mutual2_noise.png")

# --- Ejemplo 2: benchmark de Pareto (bloque cuádruple) ---
print("Ejemplo 2: bloque cuádruple  [7 esp × 6 rxn]")
rng2 = np.random.default_rng(7)
quad_blocks = [
    ("v_faster", (0.10, 0.30)),
    ("i_simple", (1.00, 1.50)),
    ("cross2",   (2.00, 3.00)),
    ("mutual2",  (4.00, 6.00)),
]
parts_m, parts_p, ks = [], [], []
for cn, kr in quad_blocks:
    c   = CORES[cn]
    n_r = c["S_minus"].shape[1]
    parts_m.append(c["S_minus"].astype(float))
    parts_p.append(c["S_plus"].astype(float))
    ks.append(rng2.uniform(*kr, size=n_r))
Sm2, Sp2 = _block_diagonal(parts_m, parts_p)
k2 = np.concatenate(ks)

with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    best2, _, _ = computeParametricSweep(Sm2, Sp2, k2, **SWEEP)

# Nombres explícitos para los 4 bloques
sp_names  = ["A", "B", "C", "D", "E", "F", "G"]
rxn_names = ["r_vf", "r_is", "r_c1", "r_c2", "r_m1", "r_m2"]
plotOptimalSubnetwork(
    Sm2, Sp2, best2,
    species_names=sp_names,
    reaction_names=rxn_names,
    simplify=False,        # bloque-diagonal → no simplificar
    save_path=os.path.join("..", "figures", "network_quad.png"),
)
print("  Guardado en network_quad.png")

plt.show()

try:
    from scenario_generator import CORES, _block_diagonal, _sample_k
    from algorithm_3 import computeParametricSweep
except ImportError:
    raise SystemExit("Necesitas algorithm_3.py y scenario_generator.py en el path.")
 
SWEEP_MEDIUM = dict(decimals=1, max_points=400,
                    time_limit_iteration=60, max_steps=100, accuracy=1e-5)
 
def run_sweep(Sm, Sp, k, **kw):
    cfg = {**SWEEP_MEDIUM, **kw}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        best, results, exact = computeParametricSweep(Sm, Sp, k, **cfg)
    if not exact:
        print("    [aviso] malla no exacta")
    return best, results
 
# ---------------------------------------------------------------
# Ejemplo 1: mutual2 + 8 reacciones aleatorias (5 especies, 10 rxn)
# ---------------------------------------------------------------
print("\n=== Ejemplo 1: mutual2 + ruido  [5 esp × 10 rxn] ===")
rng = np.random.default_rng(17)
Sm_c = CORES["mutual2"]["S_minus"].astype(float)   # 2 esp × 2 rxn
Sp_c = CORES["mutual2"]["S_plus"].astype(float)
n_cs, n_cr = Sm_c.shape
n_ts, n_tr = 5, 10
Sm1 = np.zeros((n_ts, n_tr))
Sp1 = np.zeros((n_ts, n_tr))
Sm1[:n_cs, :n_cr] = Sm_c
Sp1[:n_cs, :n_cr] = Sp_c
for a in range(n_cr, n_tr):
    for mat in (Sm1, Sp1):
        sv = rng.choice(n_ts, size=rng.integers(1, 4), replace=False)
        for s in sv:
            mat[s, a] = rng.integers(1, 3)
k1 = _sample_k(n_tr, (0.2, 4.0), "loguniform", rng)
 
best1, res1 = run_sweep(Sm1, Sp1, k1)
print(f"  factibles={sum(r['feasible'] for r in res1)}  "
      f"best.obj={best1['obj']:.4f}  alpha*={best1['alpha_star']:.4f}")
plot_pareto(res1, best1,
            title=r"mutual2 + ruido  [5 esp $\times$ 10 rxn]",
            show_obj_curve=True,
            save_path=os.path.join("..", "figures", "pareto_mutual2_noise.png"))
 
# ---------------------------------------------------------------
# Ejemplo 2: cycle3 + 12 reacciones aleatorias (6 especies, 15 rxn)
# ---------------------------------------------------------------
print("\n=== Ejemplo 2: cycle3 + ruido  [6 esp × 15 rxn] ===")
rng = np.random.default_rng(99)
Sm_c = CORES["cycle3"]["S_minus"].astype(float)   # 3 esp × 3 rxn
Sp_c = CORES["cycle3"]["S_plus"].astype(float)
n_cs, n_cr = Sm_c.shape
n_ts, n_tr = 6, 15
Sm2 = np.zeros((n_ts, n_tr))
Sp2 = np.zeros((n_ts, n_tr))
Sm2[:n_cs, :n_cr] = Sm_c
Sp2[:n_cs, :n_cr] = Sp_c
for a in range(n_cr, n_tr):
    for mat in (Sm2, Sp2):
        sv = rng.choice(n_ts, size=rng.integers(1, 4), replace=False)
        for s in sv:
            mat[s, a] = rng.integers(1, 4)
k2 = _sample_k(n_tr, (0.1, 8.0), "loguniform", rng)
 
best2, res2 = run_sweep(Sm2, Sp2, k2)
print(f"  factibles={sum(r['feasible'] for r in res2)}  "
      f"best.obj={best2['obj']:.4f}  alpha*={best2['alpha_star']:.4f}")
plot_pareto(res2, best2,
            title=r"cycle3 + ruido  [6 esp $\times$ 15 rxn]",
            show_obj_curve=True,
            save_path=os.path.join("..", "figures", "pareto_cycle3_noise.png"))
 
# ---------------------------------------------------------------
# Ejemplo 3: bloque cuádruple (todos los núcleos de 2 esp, k mixto)
# Bloques: v_faster / mutual2 / cross2 / i_simple
# Diseñado para tener 4 regiones bien separadas en la frontera
# ---------------------------------------------------------------
print("\n=== Ejemplo 3: bloque cuádruple  [7 esp × 6 rxn] ===")
rng = np.random.default_rng(7)
quad_blocks = [
    ("v_faster", (0.10, 0.30)),   # MAF=3, k lento  → obj bajo
    ("i_simple", (1.00, 1.50)),   # MAF=3/2, k medio
    ("cross2",   (2.00, 3.00)),   # MAF=2, k medio-rápido
    ("mutual2",  (4.00, 6.00)),   # MAF=3, k rápido → obj alto
]
parts_m, parts_p, ks = [], [], []
for cn, kr in quad_blocks:
    c   = CORES[cn]
    n_r = c["S_minus"].shape[1]
    parts_m.append(c["S_minus"].astype(float))
    parts_p.append(c["S_plus"].astype(float))
    ks.append(rng.uniform(*kr, size=n_r))
Sm3, Sp3 = _block_diagonal(parts_m, parts_p)
k3 = np.concatenate(ks)
 
best3, res3 = run_sweep(Sm3, Sp3, k3)
print(f"  factibles={sum(r['feasible'] for r in res3)}  "
      f"best.obj={best3['obj']:.4f}  alpha*={best3['alpha_star']:.4f}")
plot_pareto(res3, best3,
            title=r"Bloque cuádruple  [7 esp $\times$ 6 rxn]"
                  "\n"
                  r"v\_faster | i\_simple | cross2 | mutual2",
            show_obj_curve=True,
            show_infeasible=True,
            save_path=os.path.join("..", "figures", "pareto_quad.png"))
 

 
plt.show()