# -*- coding: utf-8 -*-
"""
pareto_formose.py — Frontera de Pareto (alpha* vs ||S^-||_k) de la formosa.
Un solo barrido con un vector de k plausible y reproducible (semilla fija),
para exhibir la MESETA de alpha* (~1.138) que explica por que la cota deja de
discriminar en esta red.
"""
import os, warnings
import numpy as np

from experiment_4_formose_ensemble import (
    load_formose, classify_reactions, sample_k, effective_k, C1_REF, SWEEP)
from algorithm_3 import computeParametricSweep
from drawing import plot_pareto

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FIGURES_DIR = os.path.join(_REPO_ROOT, "figures")
os.makedirs(_FIGURES_DIR, exist_ok=True)

# --- red + un vector de k reproducible (primera muestra de la semilla 0) ---
S_minus, S_plus, food_rxn, food_sp, int_sp = load_formose()
labels = classify_reactions(S_minus, S_plus, food_rxn)
rng = np.random.default_rng(0)          # misma semilla que el ensemble
k_base, _ = sample_k(labels, rng)
k = effective_k(k_base, food_rxn, C1_REF)

# --- un barrido completo ---
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    best, results, exact = computeParametricSweep(
            S_minus, S_plus, k, **SWEEP, screening=False)
    
print(f"malla exacta={exact}  puntos={len(results)}  "
      f"factibles={sum(r['feasible'] for r in results)}")
if best is not None:
    print(f"best: mu={best['mu']:.4f}  alpha*={best['alpha_star']:.4f}  "
          f"obj={best['obj']:.4f}")


import pickle
_cache_path = os.path.join(_FIGURES_DIR, "pareto_formose_cache.pkl")
with open(_cache_path, "wb") as f:
    pickle.dump({"results": results, "best": best}, f)
print(f"Cache -> {_cache_path}")


# --- Pareto: solo subredes auto-amplificantes (alpha* > 1) ---
_pareto_path = os.path.join(_FIGURES_DIR, "formose_pareto.png")
plot_pareto(results, best,
            show_obj_curve=True,
            show_infeasible=False,
            alpha_min=1.0,
            title="Formose: Pareto frontier",
            save_path=_pareto_path)
print(f"Figura -> {_pareto_path}")