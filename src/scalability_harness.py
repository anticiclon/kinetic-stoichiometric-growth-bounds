# -*- coding: utf-8 -*-
"""
scalability_harness.py
======================
Estudio de escalabilidad del Algoritmo 3 con recorrido DESCENDENTE y CRIBADO
por incumbente.

Genera instancias sintéticas de tamaño creciente (con un núcleo autocatalítico
verificado embebido => factibilidad garantizada) y, sobre cada una, corre
computeParametricSweep en DOS modos para poder cuantificar el efecto del cribado:

  * modo SCREENED (screening=True)  -> el método propuesto: descendente + poda.
  * modo FULL     (screening=False) -> barrido completo (línea base). Es caro, así
    que solo se ejecuta hasta un tamaño de corte (BASELINE_MAX_REACTIONS); por
    encima de él se mide únicamente el método propuesto.

Registra, por instancia:
  - tamaño               : |S| x |R|
  - screening_active     : True si el cribado estuvo REALMENTE activo (el MAF
                           global se certifico). False => el sweep cayo al
                           barrido completo y las metricas de esa fila NO miden
                           el cribado. Es la variable que separa los dos
                           regimenes; NO la infieras de proven_all, que se
                           refiere a los subproblemas P(mu), no al MAF global.
  - n_attainable         : nº de normas cinéticas alcanzables (línea base FULL)
  - n_solved_screened    : nº de subproblemas P(mu) resueltos bajo cribado
  - n_feasible_screened  : de esos, cuántos fueron factibles
  - iter_dinkelbach      : iteraciones de Dinkelbach del modo SCREENED
                           (NOTA: excluye la resolución única del MAF global)
  - time_screened        : segundos de pared del método propuesto (incluye MAF global)
  - time_full            : segundos de pared del barrido completo (NaN si se omitió)
  - speedup              : time_full / time_screened  (NaN si no hay línea base)
  - exact                : True si la malla recorrió el rango entero
  - proven_all           : True si todos los P(mu) factibles se probaron a optimalidad
  - obj                  : valor óptimo (debe coincidir en ambos modos)

La comparación n_attainable vs n_solved_screened muestra cuántos subproblemas
elimina el cribado; time_full vs time_screened, la aceleración resultante.

Salida:
  - output/scalability_results.csv
  - figures/scalability.png

Dependencias del repo: algorithm_3.py, scenario_generator.py (+ matplotlib).
"""

import os
import csv
import time
import warnings
from collections import defaultdict

import numpy as np

from algorithm_3 import computeParametricSweep
from scenario_generator import scenarioFromCore

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCENARIOS_DIR = os.path.join(_REPO_ROOT, "scenarios")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "output")
FIGURES_DIR = os.path.join(_REPO_ROOT, "figures")
os.makedirs(SCENARIOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuración del estudio
# ---------------------------------------------------------------------------
SIZES = [(5, 7), (8, 11), (12, 16), (16, 21), (20, 26),
         (28, 38), (36, 48), (45, 60)]
FORMOSE_SIZE = (28, 38)          # se marca en la figura

SEEDS = list(range(5))           # nº de instancias por tamaño (dispersión)
_ai = os.environ.get("PBS_ARRAYID", os.environ.get("PBS_ARRAY_INDEX"))
OUT_SUFFIX = ""
if _ai is not None:
    SIZES = [SIZES[int(_ai)]]      # esta tarea corre solo su tamaño
    OUT_SUFFIX = f"_{int(_ai)}"

CORE = "v_simple"                # núcleo embebido (A->2A): garantiza factibilidad
K_RANGE = (0.5, 5.0)             # evita constantes que redondeen a ~0 al escalar
K_DIST = "loguniform"

# Ajustes comunes del barrido (SIN la clave 'screening': se pasa por separado).
# time_limit_global: limite SOLO para el MAF global que habilita el cribado.
# Se resuelve UNA vez por instancia, asi que no debe heredar los 30 s de los
# subproblemas. Si no se certifica en ese tiempo el cribado se desactiva y el
# barrido pasa a ser completo (exacto, pero lento). None = sin limite (arriesga
# el walltime); 1800 s acota la apuesta.
SWEEP_COMMON = dict(decimals=1, max_points=2000, time_limit_iteration=30,
                    max_steps=200, accuracy=1e-5, time_limit_global=1800)

# La línea base FULL (screening=False) resuelve TODAS las normas alcanzables y es
# cara en instancias grandes: se ejecuta solo hasta este nº de reacciones. Por
# encima, solo se mide el método propuesto (SCREENED). Súbelo si tu máquina aguanta.
RUN_BASELINE = False
BASELINE_MAX_REACTIONS = 38      # hasta el tamaño de la formosa


# ---------------------------------------------------------------------------
def _load_scenario(name):
    """Carga S^-, S^+ y k de ./scenarios/ a partir del nombre del generador."""
    Sm = np.loadtxt(os.path.join(SCENARIOS_DIR, f"{name}_minus.txt"), dtype=float)
    Sp = np.loadtxt(os.path.join(SCENARIOS_DIR, f"{name}_plus.txt"), dtype=float)
    k = np.loadtxt(os.path.join(SCENARIOS_DIR, f"{name}_k.txt"))
    if Sm.ndim == 1:                 # np.loadtxt colapsa 1xN a vector
        Sm = Sm.reshape(1, -1)
        Sp = Sp.reshape(1, -1)
    return Sm, Sp, np.atleast_1d(k)


def _run_one(Sm, Sp, k, screening):
    """Corre un barrido y devuelve (dict de métricas de esa corrida, segundos)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        best, results, exact = computeParametricSweep(
            Sm, Sp, k, **SWEEP_COMMON, screening=screening)
        elapsed = time.time() - t0

    # Se pidio cribado pero el MAF global no se certifico? El sweep avisa y cae
    # al barrido completo. OJO: se detecta por el TEXTO del warning de
    # algorithm_3.py; si ese texto cambia, hay que ajustar esta cadena.
    fallback = any("DESACTIVA el cribado" in str(w.message) for w in caught)
    screening_active = bool(screening) and not fallback

    feas = [r for r in results if r["feasible"]]
    iters = sum(r["info"]["steps"] for r in feas
                if r.get("info") is not None and "steps" in r["info"])
    return dict(
        screening_active=screening_active,
        n_visited=len(results),
        n_feasible=len(feas),
        iter_dinkelbach=iters,
        exact=bool(exact),
        proven_all=(all(r.get("proven", False) for r in feas) if feas else False),
        feasible=(best is not None),
        obj=(best["obj"] if best is not None else float("nan")),
    ), elapsed


NAN = float("nan")


def _error_row(n_s, n_r, seed):
    return dict(n_species=n_s, n_reactions=n_r, seed=seed,
                screening_active=False, n_attainable=-1, n_solved_screened=-1, n_feasible_screened=-1,
                iter_dinkelbach=-1, time_screened=NAN, time_full=NAN,
                speedup=NAN, exact=False, proven_all=False,
                feasible=False, obj=NAN)


# ---------------------------------------------------------------------------
def run_scalability():
    rows = []
    print("=" * 82)
    print("Estudio de escalabilidad — Algoritmo 3 (descendente + cribado)")
    print("=" * 82)
    print(f"{'|S|':>4} {'|R|':>4} {'seed':>4} {'scr?':>5} {'solved':>7} "
          f"{'t_scr':>10} {'exact':>6} {'prov':>5}")

    for (n_s, n_r) in SIZES:
        do_baseline = RUN_BASELINE and n_r <= BASELINE_MAX_REACTIONS
        for seed in SEEDS:
            name = scenarioFromCore(
                CORE,
                n_extra_species=max(0, n_s - 1),
                n_extra_reactions=max(0, n_r - 1),
                seed=seed, version=seed,
                k_range=K_RANGE, k_distribution=K_DIST,
            )
            Sm, Sp, k = _load_scenario(name)

            try:
                # --- método propuesto: SCREENED (siempre) ---
                scr, t_scr = _run_one(Sm, Sp, k, screening=True)

                # --- línea base: FULL (solo hasta el corte de tamaño) ---
                if do_baseline:
                    full, t_full = _run_one(Sm, Sp, k, screening=False)
                    n_attainable = full["n_feasible"]
                    speedup = (t_full / t_scr) if t_scr > 0 else NAN
                else:
                    t_full = NAN
                    n_attainable = -1        # no medido a este tamaño
                    speedup = NAN

                row = dict(
                    n_species=n_s, n_reactions=n_r, seed=seed,
                    screening_active=scr["screening_active"],
                    n_attainable=n_attainable,
                    n_solved_screened=scr["n_visited"],
                    n_feasible_screened=scr["n_feasible"],
                    iter_dinkelbach=scr["iter_dinkelbach"],
                    time_screened=t_scr, time_full=t_full, speedup=speedup,
                    exact=scr["exact"], proven_all=scr["proven_all"],
                    feasible=scr["feasible"], obj=scr["obj"],
                )
            except Exception as e:
                print(f"  [{n_s}x{n_r} seed={seed}] EXCEPCIÓN: {e}", flush=True)
                row = _error_row(n_s, n_r, seed)

            rows.append(row)
            print(f"{n_s:>4} {n_r:>4} {seed:>4} "
                  f"{str(row['screening_active']):>5} "
                  f"{row['n_solved_screened']:>7} "
                  f"{row['time_screened']:>10.2f} "
                  f"{str(row['exact']):>6} {str(row['proven_all']):>5}", flush=True)

    _write_csv(rows, path=os.path.join(OUTPUT_DIR, f"scalability_results{OUT_SUFFIX}.csv"))
    if OUT_SUFFIX == "":
        _plot(rows)
        
    return rows


def _write_csv(rows, path=None):
    if path is None:
        path = os.path.join(OUTPUT_DIR, "scalability_results.csv")
    keys = ["n_species", "n_reactions", "seed", "screening_active",
            "n_attainable", "n_solved_screened", "n_feasible_screened",
            "iter_dinkelbach", "time_screened", "time_full", "speedup",
            "exact", "proven_all", "feasible", "obj"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV -> {path}")


def _agg(rows, xkey, ykey):
    """Mediana y (min, max) de ykey agrupando por xkey, descartando error/nan."""
    g = defaultdict(list)
    for r in rows:
        v = r[ykey]
        if v is None or (isinstance(v, float) and np.isnan(v)) or v < 0:
            continue
        g[r[xkey]].append(v)
    xs = sorted(g)
    return (xs,
            [float(np.median(g[x])) for x in xs],
            [float(np.min(g[x])) for x in xs],
            [float(np.max(g[x])) for x in xs])


def _plot(rows, path=None):
    if path is None:
        path = os.path.join(FIGURES_DIR, "scalability.png")
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.warn(f"matplotlib not available, no figure generated: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) wall-clock time: screened (proposed) vs full (baseline)
    xs, med, lo, hi = _agg(rows, "n_reactions", "time_screened")
    if xs:
        ax1.plot(xs, med, "o-", color="#1a5e9e", zorder=3, label="screened (proposed)")
        ax1.fill_between(xs, lo, hi, color="#1a5e9e", alpha=0.18)
    xsf, medf, *_ = _agg(rows, "n_reactions", "time_full")
    if xsf:
        ax1.plot(xsf, medf, "s--", color="#c0392b", zorder=3, label="full (baseline)")
    ax1.axvline(FORMOSE_SIZE[1], color="#888", ls=":", zorder=0)
    ax1.annotate("formose", xy=(FORMOSE_SIZE[1], 0), xytext=(2, 4),
                 textcoords="offset points", color="#666", fontsize=8)
    ax1.set_xlabel(r"number of reactions $|\mathcal{R}|$")
    ax1.set_ylabel("wall-clock time (s)")
    ax1.set_title("(a) Runtime: screening vs full")
    ax1.legend(fontsize=8)

    # (b) subproblems: attainable norms (baseline) vs solved under screening
    xsa, meda, *_ = _agg(rows, "n_reactions", "n_attainable")
    xss, meds, *_ = _agg(rows, "n_reactions", "n_solved_screened")
    if xsa:
        ax2.plot(xsa, meda, "s--", color="#c0392b", label="attainable norms")
    if xss:
        ax2.plot(xss, meds, "o-", color="#1a5e9e", label="solved (screening)")
    ax2.axvline(FORMOSE_SIZE[1], color="#888", ls=":", zorder=0)
    ax2.set_xlabel(r"number of reactions $|\mathcal{R}|$")
    ax2.set_ylabel("subproblems (median)")
    ax2.set_title("(b) Subproblems: screening effect")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  figura -> {path}")


if __name__ == "__main__":
    run_scalability()
    print("\nEstudio de escalabilidad terminado.")