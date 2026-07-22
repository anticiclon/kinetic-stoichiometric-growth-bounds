# -*- coding: utf-8 -*-
"""
experiment_4_formose_ensemble.py
================================
CASO DE ESTUDIO — Formosa (red completa), ROBUSTEZ POR ENSEMBLE.

La red de la formosa (Müller, Flamm & Stadler, J. Cheminform. 2022) es un objeto
ESTRUCTURAL: trae las matrices estequiométricas pero NO constantes cinéticas.
No existen k medidas para una red completa de formosa, y no hay un vector único
de k que copiar. En lugar de inventar uno, parametrizamos la cinética como un
ENSEMBLE de vectores plausibles y reportamos la DISTRIBUCIÓN de resultados: con
qué frecuencia el método selecciona el mismo núcleo, cómo de holgada queda la
cota, y con qué frecuencia (si alguna) la cota misselecciona (elige un núcleo
que crece menos que el del MAF).

CÓMO SE GENERA CADA VECTOR DE k (TST por clase mecanística)
-----------------------------------------------------------
1) Cada reacción se clasifica por su forma estequiométrica (sin etiquetar
   especies a mano):
     - aldólica       : consume formaldehído (food) y produce 1 especie  (C-C up)
     - retro-aldol    : 1 reactivo -> 2 moléculas producidas (C-C scission, cierre)
     - isomerización  : 1 -> 1 interna, no consume food
     - pérdida        : produce una especie terminal (nunca reconsumida)
2) A cada clase le corresponde un rango de barrera de energía libre DG! (kcal/mol)
   en un RÉGIMEN CATALIZADO COHERENTE (un solo conjunto de condiciones). Los
   rangos son estrechos a propósito: las barreras anchas de la literatura
   (~0-45 kcal/mol) mezclan condiciones catalíticas distintas, no pasos bajo una
   misma condición; en un ciclo que opera, los pasos difieren solo unos pocos
   kcal/mol. Orden anclado en la literatura (Kua et al.; Venturini et al.):
       aldólica      [4,  7]   (formación C-C, rápida)
       isomerización [5,  8]   (tautomerización)
       retro-aldol   [7, 10]   (cierre del ciclo, limitante pero modesto)
       pérdida       [8, 11]   (Cannizzaro/drenaje, lento)
   El spread resultante es ~4 órdenes de magnitud en k: heterogéneo pero
   computable (con rangos más anchos, el retro-aldol hunde Lambda a cero).
3) Por cada muestra del ensemble se sortea DG!_r ~ Uniforme(rango de su clase) y
   se convierte a constante por teoría del estado de transición (TST):
       k_r = (kB T / h) * exp(-DG!_r / RT)
   El vector se NORMALIZA (mediana -> 1): al método solo le importan las k
   RELATIVAS y el control por [C1], no la escala absoluta de tiempo. Así se
   preserva la heterogeneidad real (orden y dispersión entre clases) sin que la
   escala distorsione.

Las k del MAF son independientes de k (el MAF es puramente estequiométrico), así
que el núcleo del Algoritmo 2 se calcula UNA vez y se reutiliza como referencia
para el test de misselección en cada muestra.

Dependencias del repo: algorithm_2.py, algorithm_3.py, dynamics.py.
Ficheros: formose_minus.txt, formose_plus.txt.
"""

import os
import warnings
from collections import Counter
import numpy as np

import dynamics as dyn

try:
    from algorithm_2 import computeSubhypergraphWithGreaterMAF
    from algorithm_3 import computeParametricSweep
except Exception as e:
    raise SystemExit("Necesito algorithm_2 y algorithm_3: " + str(e))

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCENARIOS_DIR = os.path.join(_REPO_ROOT, "scenarios")
_OUTPUT_DIR = os.path.join(_REPO_ROOT, "output")
_FIGURES_DIR = os.path.join(_REPO_ROOT, "figures")
os.makedirs(_OUTPUT_DIR, exist_ok=True)
os.makedirs(_FIGURES_DIR, exist_ok=True)
MINUS_PATH = os.path.join(_SCENARIOS_DIR, "formose_minus.txt")
PLUS_PATH = os.path.join(_SCENARIOS_DIR, "formose_plus.txt")

# ----- parámetros del ensemble -----
N_SAMPLES = 50           # nº de vectores de k muestreados
C1_REF = 1.0             # concentración de formaldehído de referencia
SEED = 0                 # reproducibilidad
TL = 120
SWEEP = dict(decimals=2, max_points=600, time_limit_iteration=TL,
             max_steps=300, accuracy=1e-5)

# TST a 25 °C:  RT = 0.593 kcal/mol ;  prefactor kB T / h ~ 6.21e12 s^-1
RT = 0.593
PREFACTOR = 6.21e12

# rangos de barrera DG! (kcal/mol) por clase, RÉGIMEN CATALIZADO COHERENTE.
# Estrechos a propósito: las barreras "anchas" de la literatura mezclan
# condiciones catalíticas distintas (agua / NH3 / HCOOH como proxies), no pasos
# bajo UNA misma condición. En un ciclo que opera bajo un régimen fijo los pasos
# difieren solo unos pocos kcal/mol (el retro-aldol es limitante, pero modesto;
# si estuviera ~18 kcal/mol por debajo de los aldoles el ciclo no giraría y
# Lambda se hundiría a cero numérico). Spread resultante ~4 órdenes en k:
# heterogéneo pero computable. Orden preservado: aldol < iso < retro < pérdida.
BARRIER_RANGES = {
    "aldolica":      (4.0, 7.0),
    "isomerizacion": (5.0, 8.0),
    "retroaldol":    (7.0, 10.0),
    "perdida":       (8.0, 11.0),
}


# ===========================================================================
# Carga y clasificación
# ===========================================================================
def load_formose(minus_path=MINUS_PATH, plus_path=PLUS_PATH):
    Sm = np.loadtxt(minus_path)
    Sp = np.loadtxt(plus_path)
    nS, nR = Sm.shape
    produced = Sp.sum(axis=1) > 0
    consumed = Sm.sum(axis=1) > 0
    food_sp = [s for s in range(nS) if consumed[s] and not produced[s]]
    int_sp = [s for s in range(nS) if s not in food_sp]
    food_rxn = sorted({r for r in range(nR) for s in food_sp if Sm[s, r] > 0})
    return Sm[int_sp], Sp[int_sp], food_rxn, food_sp, int_sp


def classify_reactions(S_minus, S_plus, food_rxn):
    """Etiqueta cada reacción por su forma estequiométrica."""
    nS, nR = S_minus.shape
    consumed_ever = (S_minus > 0).any(axis=1)
    terminal = {s for s in range(nS) if not consumed_ever[s]}
    food = set(food_rxn)
    labels = []
    for r in range(nR):
        prods = [s for s in range(nS) if S_plus[s, r] > 0]
        prod_total = S_plus[:, r].sum()           # cuenta multiplicidad (2C2 -> 2)
        if any(s in terminal for s in prods):
            labels.append("perdida")
        elif prod_total >= 2:                     # 1 -> 2 : ruptura C-C
            labels.append("retroaldol")
        elif r in food:
            labels.append("aldolica")
        else:
            labels.append("isomerizacion")
    return labels


# ===========================================================================
# Muestreo de k por TST
# ===========================================================================
def sample_k(labels, rng):
    """Sortea DG! por clase, lo pasa a k por TST y normaliza (máximo -> 1).

    Se normaliza por el MÁXIMO (no la mediana): con un spread de ~12 órdenes
    de magnitud, normalizar por la mediana deja la cola rápida en valores
    enormes, la norma ||S^-||_k se dispara y el barrido de mu del Algoritmo 3
    (que recorre enteros = mu * 10^decimals) excede max_points y deja de ser
    exacto. Con el máximo a 1, ||S^-||_k queda acotada en torno a [C1] y el
    barrido es pequeño y exacto. La escala absoluta es irrelevante para el
    método, así que ambas normalizaciones son legítimas; el máximo es la sana.
    """
    dG = np.array([rng.uniform(*BARRIER_RANGES[c]) for c in labels])
    k = PREFACTOR * np.exp(-dG / RT)
    k = k / k.max()                               # escala absoluta irrelevante
    return k, dG


def effective_k(k_base, food_rxn, C1):
    k = k_base.copy()
    for r in food_rxn:
        k[r] *= C1
    return k


# ===========================================================================
# Utilidades de subred
# ===========================================================================
def binding_support(S_minus, S_plus, k, arcs, nodes):
    species = list(nodes) if nodes else list(range(S_minus.shape[0]))
    if not arcs or not species:
        return list(arcs), list(species), None
    consum = {s: sum(S_minus[s, r] * k[r] for r in arcs) for s in species}
    s_star = max(consum, key=consum.get)
    comp_sp, comp_rx, changed = {s_star}, set(), True
    while changed:
        changed = False
        for r in arcs:
            if r in comp_rx:
                continue
            touches = [s for s in species if S_minus[s, r] > 0 or S_plus[s, r] > 0]
            if any(s in comp_sp for s in touches):
                comp_rx.add(r)
                for s in touches:
                    if s not in comp_sp:
                        comp_sp.add(s)
                changed = True
    return sorted(comp_rx), sorted(comp_sp), s_star


def nodes_of(S_minus, S_plus, arcs):
    nS = S_minus.shape[0]
    return sorted({s for s in range(nS) for r in arcs
                   if S_minus[s, r] > 0 or S_plus[s, r] > 0})


def lambda_of(S_minus, S_plus, k, arcs):
    if not arcs:
        return 0.0
    nd = nodes_of(S_minus, S_plus, arcs)
    L, _, _ = dyn.growth_rate_of_subnetwork(S_minus, S_plus, k, list(arcs), nd)
    return L


# ===========================================================================
# Ensemble
# ===========================================================================
def run_ensemble(n_samples=N_SAMPLES, C1=C1_REF, seed=SEED):
    S_minus, S_plus, food_rxn, food_sp, int_sp = load_formose()
    nS, nR = S_minus.shape
    labels = classify_reactions(S_minus, S_plus, food_rxn)

    print("=" * 78)
    print("Ensemble TST sobre la red de la formosa")
    print("=" * 78)
    print(f"  {len(int_sp)} especies internas, {nR} reacciones; "
          f"food (formaldehído): especie(s) {food_sp}")
    print("  Clasificación de reacciones:", dict(Counter(labels)))
    print("  Rangos de barrera DG! (kcal/mol), régimen catalizado:")
    for c, rng in BARRIER_RANGES.items():
        print(f"    {c:14s} {rng}")
    print(f"  Muestras={n_samples}  [C1]={C1}  semilla={seed}")

    # Núcleo MAF (Algoritmo 2): es estequiométrico, NO depende de k -> una vez.
    maf, _ = computeSubhypergraphWithGreaterMAF(
        S_minus, S_plus, "formose_maf", TL, max_steps=2000, accuracy=1e-6)
    maf_arcs = sorted(maf["z"])
    print(f"\n  Núcleo MAF (Alg.2, referencia fija): alpha*={maf['alpha']:.4f}, "
          f"arcos={maf_arcs}")

    rng = np.random.default_rng(seed)
    full_arcs = list(range(nR))

    sel_cores = []        # núcleo cota-óptimo (tupla de arcos) por muestra
    ratios = []           # Lambda/cota del núcleo cota-óptimo
    miss = 0              # nº de muestras con misselección
    lam_gap = []          # Lambda(MAF) - Lambda(cota) por muestra (relativo)
    n_ok = 0

    for i in range(n_samples):
        k_base, _ = sample_k(labels, rng)
        k = effective_k(k_base, food_rxn, C1)

        best, results, _ = computeParametricSweep(
            S_minus, S_plus, k, **SWEEP, screening=True) 
        if best is None:
            continue
        info = best["info"]
        bind_rx, bind_sp, _ = binding_support(
            S_minus, S_plus, k, info["act_arcs"], info["act_nodes"])
        mu = dyn.kinetic_norm(S_minus, k, bind_rx, bind_sp)
        bound = (best["alpha_star"] - 1.0) * mu
        L_cota = lambda_of(S_minus, S_plus, k, bind_rx)
        L_maf = lambda_of(S_minus, S_plus, k, maf_arcs)

        sel_cores.append(tuple(bind_rx))
        if bound > 1e-12:
            ratios.append(L_cota / bound)
        # misselección: la cota elige un núcleo que crece menos que el del MAF
        if L_maf > L_cota * (1.0 + 1e-3):
            miss += 1
        denom = max(L_maf, L_cota, 1e-12)
        lam_gap.append((L_maf - L_cota) / denom)
        n_ok += 1

        if (i + 1) % 25 == 0:
            print(f"    ... {i + 1}/{n_samples} muestras")

    # -------- agregación --------

    print("\n" + "-" * 78)
    print("Resultados del ensemble")
    print("-" * 78)
    print(f"  muestras factibles: {n_ok}/{n_samples}")

    core_freq = Counter(sel_cores)
    print(f"\n  Núcleos cota-óptimos distintos: {len(core_freq)}")
    print("  Top núcleos seleccionados (frecuencia):")
    for core, c in core_freq.most_common(5):
        print(f"    {100*c/n_ok:5.1f}%  ({len(core)} arcos)  {list(core)}")

    ratios = np.array(ratios)
    print(f"\n  Holgura Lambda/cota:  min={ratios.min():.3f}  "
          f"mediana={np.median(ratios):.3f}  max={ratios.max():.3f}")
    print(f"  (todas <= 1: {np.all(ratios <= 1 + 1e-6)})")

    print(f"\n  Misselección (cota elige núcleo más lento que el MAF): "
          f"{100*miss/n_ok:.1f}% de las muestras")
    lam_gap = np.array(lam_gap)
    print(f"  Brecha relativa de crecimiento (Lambda_MAF - Lambda_cota)/max:  "
          f"mediana={np.median(lam_gap):+.3f}  max={lam_gap.max():+.3f}")
    
    import pickle
    with open(os.path.join(_OUTPUT_DIR, "ensemble_cache.pkl"), "wb") as f:
        pickle.dump({"ratios": ratios, "core_freq": core_freq, "n_ok": n_ok}, f)
    print(f"  Cache -> {os.path.join(_OUTPUT_DIR, 'ensemble_cache.pkl')}")

    _plots(ratios, core_freq, n_ok)
    return dict(core_freq=core_freq, ratios=ratios, miss_rate=miss / n_ok,
                lam_gap=lam_gap, labels=labels)


def _plots(ratios, core_freq, n_ok):
    try:
        import matplotlib.pyplot as plt
        # # histograma de holgura
        # fig, ax = plt.subplots(figsize=(6.4, 4.0))
        # ax.hist(ratios, bins=20, color="#1a5e9e", alpha=0.8, edgecolor="white")
        # ax.axvline(1.0, color="#c0392b", ls="--", label=r"$\Lambda/$cota$=1$")
        # ax.set_xlabel(r"$\Lambda_{\rm obs}/$cota")
        # ax.set_ylabel("nº de muestras")
        # ax.set_title("Formosa: holgura de la cota sobre el ensemble")
        # ax.legend(fontsize=9)
        # fig.tight_layout()
        # fig.savefig("figures/exp4_ensemble_tightness.png", dpi=150,
        #             bbox_inches="tight")
        # histograma de holgura (escala log en x)
        r = np.clip(ratios, 1e-4, None)        # piso para log; los ~0 caen al borde izq.
        bins = np.logspace(np.log10(r.min()), 0.0, 25)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.hist(r, bins=bins, color="#1a5e9e", alpha=1, edgecolor="white")
        ax.set_xscale("log")
        ax.axvline(1.0, color="#c0392b", ls="--", label=r"$\Lambda/=1$")
        ax.set_xlabel(r"$\Lambda_{\rm obs}/$bound (log scale)")
        ax.set_ylabel("number of samples")
        ax.set_title("Formose: bound tightness over the ensemble")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(_FIGURES_DIR, "exp4_ensemble_tightness.png"), dpi=150, bbox_inches="tight")
        print(f"\n  Figura -> {os.path.join(_FIGURES_DIR, 'exp4_ensemble_tightness.png')}")


        # core frequency (top 6)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        top = core_freq.most_common(6)
        # labels_ = [f"{len(c)} arcs" for c, _ in top]
        labels_ = [f"core {i+1}" for i in range(len(top))]
        freqs = [100 * n / n_ok for _, n in top]
        ax.bar(range(len(top)), freqs, color="#e07b39", alpha=1)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(labels_, rotation=20, ha="right", fontsize=12)
        ax.set_ylabel("% of samples", fontsize=15)
        ax.set_title("Formose: selected cores over the ensemble", fontsize=15)
        fig.tight_layout()
        fig.savefig(os.path.join(_FIGURES_DIR, "exp4_ensemble_cores.png"), dpi=150,
                    bbox_inches="tight")
        print(f"  Figure -> {os.path.join(_FIGURES_DIR, 'exp4_ensemble_cores.png')}")
    except Exception as e:
        warnings.warn(f"plot: {e}")

        


if __name__ == "__main__":
    res = run_ensemble()
    np.savetxt("formose_ratios.txt", res["ratios"])
    print("\nEnsemble terminado.")