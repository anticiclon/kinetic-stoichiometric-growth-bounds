# -*- coding: utf-8 -*-
"""
rtca_experiment.py — Construcción y análisis de dos redes rTCA (reductive TCA)
para el paper de optimización cinético-estequiométrica.

  (A) rTCA COMPLETO   : 11 especies internas, 11 reacciones. Red real, grande.
  (B) rTCA MÍNIMO     : 3 especies (C4/C6/C2), 3 reacciones. Núcleo autocatalítico
                        lumpeado que conserva la topología rTCA (doblado de C4).

Modelo escalable: CO2, reductor [H], ATP, CoA, H2O se tratan como FOOD (externas,
buffer ambiental), absorbidas en constantes efectivas. Exactamente el mismo
preprocesado que la formosa con el formaldehído.

Este script NO usa Gurobi. Calcula, sin solver:
  * alpha* de cada núcleo, por la ecuación característica (Perron/Collatz-Wielandt,
    igual que la cúbica del Oregonator en el paper), verificada por root-finding.
  * norma cinética  ||S^-||_k  (Def. 2.5).
  * tasa de crecimiento Lambda = autovalor de Perron de M = S diag(k) P (Sec. 5.1).
  * ensemble TST por clase de reacción -> distribución de tau = Lambda/cota.

La SELECCIÓN (qué subred maximiza la cota) requiere el sweep MILP (Gurobi) sobre
la red completa; aquí se reporta el ajuste del ciclo canónico. Los .txt exportados
permiten lanzar computeParametricSweep tal cual.
"""

import os
import numpy as np

RT = 0.593          # kcal/mol a 25 C
PREFACTOR = 6.21e12  # kB T / h  [s^-1]
np.set_printoptions(precision=4, suppress=True)


# ======================================================================
# Utilidades (reimplementación mínima de dynamics.py)
# ======================================================================
def kinetic_norm(S_minus, k, arcs, nodes):
    """Def. 2.5: max_s sum_r S^-_{s,r} k_r sobre las reacciones activas."""
    return max(sum(S_minus[s, r] * k[r] for r in arcs) for s in nodes)


def growth_operator(S_minus, S_plus, k, arcs, nodes):
    """M = S diag(k) P restringida a (nodes, arcs). Cada reacción escalable
    consume exactamente 1 especie interna (mult. 1)."""
    S = S_plus - S_minus
    idx = {s: i for i, s in enumerate(nodes)}
    M = np.zeros((len(nodes), len(nodes)))
    for r in arcs:
        reactants = [s for s in nodes if S_minus[s, r] > 0]
        assert len(reactants) == 1, f"reacción {r} no es de primer orden interno"
        s = reactants[0]
        for v in nodes:
            M[idx[v], idx[s]] += S[v, r] * k[r]
    return M


def growth_rate(S_minus, S_plus, k, arcs, nodes):
    """Lambda = mayor parte real del espectro de M (autovalor de Perron)."""
    M = growth_operator(S_minus, S_plus, k, arcs, nodes)
    ev = np.linalg.eigvals(M)
    return float(np.max(ev.real))


def norm_species(S_minus, k, arcs, nodes):
    """Especie que alcanza la norma cinética (consumo máximo)."""
    cons = {s: sum(S_minus[s, r] * k[r] for r in arcs) for s in nodes}
    return max(cons, key=cons.get), cons


# ======================================================================
# (A) rTCA COMPLETO — 11 especies, 11 reacciones
# ======================================================================
# Especies:  0 OAA  1 MAL  2 FUM  3 SUC  4 SCoA  5 AKG  6 ICIT  7 CIT
#            8 ACoA  9 PYR 10 PEP
# Reacción i (col i) consume la especie i  ->  S_minus = I_11
SP_FULL = ["OAA", "MAL", "FUM", "SUC", "SCoA", "AKG", "ICIT", "CIT",
           "ACoA", "PYR", "PEP"]
RX_FULL = ["R1 OAA->MAL", "R2 MAL->FUM", "R3 FUM->SUC", "R4 SUC->SCoA",
           "R5 SCoA->AKG(+CO2)", "R6 AKG->ICIT(+CO2)", "R7 ICIT->CIT",
           "R8 CIT->OAA+ACoA", "R9 ACoA->PYR(+CO2)", "R10 PYR->PEP",
           "R11 PEP->OAA(+CO2)"]

nS_full = 11
S_minus_full = np.eye(nS_full)
S_plus_full = np.zeros((nS_full, nS_full))
prod = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7],
        7: [0, 8], 8: [9], 9: [10], 10: [0]}   # reacción -> especies producidas
for r, outs in prod.items():
    for s in outs:
        S_plus_full[s, r] += 1

# clases de reacción (para el ensemble)
CLASS_FULL = {
    0: "reduccion",       # R1 malato DH reductiva
    1: "deshidratacion",  # R2 fumarasa
    2: "reduccion",       # R3 fumarato reductasa
    3: "ligacion",        # R4 succinil-CoA sintetasa
    4: "carboxilacion",   # R5 (rate-limiting)
    5: "carboxilacion",   # R6
    6: "isomerizacion",   # R7 aconitasa
    7: "escision",        # R8 ATP-citrato liasa (rápida, ATP-driven)
    8: "carboxilacion",   # R9
    9: "ligacion",        # R10 PEP sintasa
    10: "carboxilacion",  # R11
}
# food: toda reacción que fija CO2 o consume una externa. Aquí TODAS las k son
# efectivas; marcamos como "food_rxn" las que absorben una concentración externa
# variable ([CO2]). Para el sweep, food_rxn = reacciones de carboxilación.
FOOD_FULL = [r for r, c in CLASS_FULL.items() if c == "carboxilacion"]


# ======================================================================
# (B) rTCA MÍNIMO — 3 especies (C4, C6, C2), 3 reacciones
# ======================================================================
# 0 C4(OAA)  1 C6(citrato)  2 C2(acetil-CoA)
# Ra: C4->C6  (cadena carboxilativa reductiva lumpeada, LENTA)
# Rb: C6->C4+C2  (escisión, RÁPIDA)
# Rc: C2->C4  (brazo acetil, carboxilación, LENTA)
SP_MIN = ["C4", "C6", "C2"]
RX_MIN = ["Ra C4->C6", "Rb C6->C4+C2", "Rc C2->C4"]
S_minus_min = np.array([[1., 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
S_plus_min = np.array([[0., 1, 1],
                       [1, 0, 0],
                       [0, 1, 0]])
CLASS_MIN = {0: "carboxilacion", 1: "escision", 2: "carboxilacion"}
FOOD_MIN = [0, 2]


# ======================================================================
# alpha* por ecuación característica (verificado por root-finding)
# ======================================================================
def alpha_star_from_poly(coeffs, name):
    """Mayor raíz real > 1 del polinomio característico."""
    roots = np.roots(coeffs)
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 1.0)
    a = real[-1] if real else float("nan")
    return a


# Mínimo:  alpha^3 - alpha - 1 = 0   (número plástico)
ALPHA_MIN = alpha_star_from_poly([1, 0, -1, -1], "min")
# Completo: alpha^11 - alpha^3 - 1 = 0
ALPHA_FULL = alpha_star_from_poly(
    [1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1], "full")


# ======================================================================
# Ensemble TST
# ======================================================================
BARRIER = {   # kcal/mol, régimen catalizado COHERENTE (mismo criterio que formosa:
              # los pasos difieren pocos kcal/mol; carboxilación limitante pero
              # MODESTA, análoga al retro-aldol de la formosa). Rangos anchos
              # (>6 kcal/mol de hueco) hunden Lambda a cero: el ciclo no giraría.
    "carboxilacion":  (7.0, 10.0),   # fijación de CO2: limitante pero modesta
    "ligacion":       (6.0, 9.0),
    "reduccion":      (6.0, 9.0),
    "isomerizacion":  (5.0, 8.0),
    "deshidratacion": (4.0, 7.0),
    "escision":       (4.0, 7.0),
}


def sample_k(classes, rng):
    dG = np.array([rng.uniform(*BARRIER[classes[r]]) for r in range(len(classes))])
    k = PREFACTOR * np.exp(-dG / RT)
    return k / k.max()          # escala absoluta irrelevante


def run_ensemble(S_minus, S_plus, classes, alpha_star, name,
                 n=200, seed=0):
    nS, nR = S_minus.shape
    arcs = list(range(nR))
    nodes = list(range(nS))
    rng = np.random.default_rng(seed)

    taus, lams, norms = [], [], []
    normspec_counter = {}
    for _ in range(n):
        k = sample_k(classes, rng)
        mu = kinetic_norm(S_minus, k, arcs, nodes)
        L = growth_rate(S_minus, S_plus, k, arcs, nodes)
        bound = (alpha_star - 1.0) * mu
        if bound > 1e-12:
            taus.append(L / bound)
        lams.append(L)
        norms.append(mu)
        sstar, _ = norm_species(S_minus, k, arcs, nodes)
        normspec_counter[sstar] = normspec_counter.get(sstar, 0) + 1

    taus = np.array(taus)
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(f"  especies={nS}  reacciones={nR}  alpha*={alpha_star:.4f}")
    print(f"  cota valida (Lambda<=cota) en todas: {np.all(taus <= 1+1e-6)}")
    print(f"  tau = Lambda/cota :  min={taus.min():.2e}  "
          f"mediana={np.median(taus):.2e}  max={taus.max():.2e}")
    print(f"  Lambda (norm.)    :  mediana={np.median(lams):.2e}")
    print(f"  ||S^-||_k (norm.) :  mediana={np.median(norms):.4f}")
    frac_tiny = np.mean(taus < 0.05)
    print(f"  fracción con tau<0.05 : {100*frac_tiny:.0f}%")
    print(f"  especie que alcanza la norma (consumo máx), frecuencia:")
    names = SP_FULL if nS == 11 else SP_MIN
    for s, c in sorted(normspec_counter.items(), key=lambda t: -t[1]):
        print(f"      {names[s]:6s} {100*c/n:5.0f}%")
    return taus


# ======================================================================
# Verificación de autocatálisis (flujo balanceado con S x > 0)
# ======================================================================
def check_autocatalysis(S_minus, S_plus, alpha_star, name):
    """Construye el flujo geométrico x_i = alpha^{-(pos)} y comprueba S x > 0
    en la componente amplificada (chequeo cualitativo de Def. 2.2)."""
    S = S_plus - S_minus
    nS, nR = S_minus.shape
    # flujo de prueba: unidad en la primera reacción, geométrico según cadena
    # (para redes cíclicas simples basta un x>0 genérico y comprobar signo neto)
    # Usamos x uniforme perturbado y verificamos que existe x con Sx>0 via LP-free:
    # para estos ciclos, x_i proporcional a alpha^{-i} funciona.
    x = np.array([alpha_star ** (-i) for i in range(nR)])
    net = S @ x
    print(f"  [{name}] net = S x (x geométrico): "
          f"min={net.min():+.4f}  (>0 en {np.sum(net>1e-9)}/{nS} especies)")
    return net


# ======================================================================
if __name__ == "__main__":
    print("VERIFICACIÓN DE AUTOCATÁLISIS")
    print(f"  rTCA mínimo : alpha^3-alpha-1=0 -> alpha*={ALPHA_MIN:.6f} "
          f"(número plástico)")
    print(f"  rTCA completo: alpha^11-alpha^3-1=0 -> alpha*={ALPHA_FULL:.6f}")
    check_autocatalysis(S_minus_min, S_plus_min, ALPHA_MIN, "mínimo")
    check_autocatalysis(S_minus_full, S_plus_full, ALPHA_FULL, "completo")

    run_ensemble(S_minus_min, S_plus_min, CLASS_MIN, ALPHA_MIN,
                 "rTCA MÍNIMO (núcleo lumpeado C4/C6/C2)")
    run_ensemble(S_minus_full, S_plus_full, CLASS_FULL, ALPHA_FULL,
                 "rTCA COMPLETO (11 especies, 11 reacciones)")

    # export para el sweep (Gurobi) del usuario
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _scenarios_dir = os.path.join(_repo_root, "scenarios")
    os.makedirs(_scenarios_dir, exist_ok=True)
    np.savetxt(os.path.join(_scenarios_dir, "rtca_min_minus.txt"), S_minus_min, fmt="%d")
    np.savetxt(os.path.join(_scenarios_dir, "rtca_min_plus.txt"), S_plus_min, fmt="%d")
    np.savetxt(os.path.join(_scenarios_dir, "rtca_full_minus.txt"), S_minus_full, fmt="%d")
    np.savetxt(os.path.join(_scenarios_dir, "rtca_full_plus.txt"), S_plus_full, fmt="%d")
    print(f"\n  food_rxn (mínimo)  = {FOOD_MIN}")
    print(f"  food_rxn (completo) = {FOOD_FULL}")
    print(f"  Matrices exportadas en: {_scenarios_dir}")
    print("    rtca_min_minus.txt, rtca_min_plus.txt, "
          "rtca_full_minus.txt, rtca_full_plus.txt")
