# -*- coding: utf-8 -*-
"""
reproduce_examples.py
=====================
Reproduce las cifras de las dos tablas de la Seccion "Illustrative examples":
  * tab:tesis              (instancia disenada de dos bloques)
  * tab:oregonator_resultados  (Oregonator podado)

Depende SOLO de: numpy, gurobipy y tu algorithm_3.py (computeParametricSweep).
NO necesita dynamics.py ni algorithm_2.py: la norma cinetica y la tasa de
crecimiento Lambda se calculan aqui (Lambda = autovalor dominante de
M = Q diag(k) P, la eq:operador-M del paper).

Uso:  python reproduce_examples.py
"""

import numpy as np
from algorithm_3 import computeParametricSweep


# ----------------------------------------------------------------------
# Ayudantes (verificados): norma cinetica y tasa de crecimiento observada
# ----------------------------------------------------------------------
def _source(Sm, a):
    """Especie reactivo (unica, bajo cinetica escalable de 1er orden) del arco a."""
    idx = np.where(Sm[:, a] > 0)[0]
    return int(idx[0])


def _incident_species(Sm, Sp, arcs):
    """Especies tocadas por los arcos activos (reactivo o producto)."""
    sp = set()
    for a in arcs:
        for s in range(Sm.shape[0]):
            if Sm[s, a] > 0 or Sp[s, a] > 0:
                sp.add(s)
    return sorted(sp)


def kinetic_norm(Sm, k, arcs):
    """||S^-|_A'||_k = max_v sum_a S^-_{va} k_a  (sobre arcos activos)."""
    species = sorted({s for a in arcs for s in range(Sm.shape[0]) if Sm[s, a] > 0})
    return max(sum(Sm[s, a] * k[a] for a in arcs) for s in species)


def growth_rate(Sm, Sp, k, arcs):
    """Lambda = autovalor de Perron de M = Q diag(k) P restringida al subgrafo."""
    arcs = sorted(arcs)
    Q = Sp - Sm
    species = _incident_species(Sm, Sp, arcs)
    P = np.zeros((len(arcs), Sm.shape[0]))
    for i, a in enumerate(arcs):
        P[i, _source(Sm, a)] = 1.0
    M = Q[:, arcs] @ np.diag(k[arcs]) @ P          # |N| x |N|
    Msub = M[np.ix_(species, species)]
    return float(max(np.linalg.eigvals(Msub).real))


# ----------------------------------------------------------------------
# 1) Instancia disenada (tab:tesis)
# ----------------------------------------------------------------------
def designed_instance():
    print("=" * 66)
    print("INSTANCIA DISENADA  (tab:tesis)")
    print("=" * 66)
    # Bloque A: A->3A (k=0.3) ; Bloque B: B->2B (k=1.0)  (bloque-diagonal)
    Sm = np.array([[1.0, 0.0], [0.0, 1.0]])   # reactivos  S^-
    Sp = np.array([[3.0, 0.0], [0.0, 2.0]])   # productos  S^+
    k = np.array([0.3, 1.0])
    names_arc = ["rA (A->3A)", "rB (B->2B)"]

    # Filas de la tabla: cada bloque por separado (formas cerradas via ayudantes)
    print(f"  {'Subred':14s} {'alpha*':>7s} {'||S^-||k':>9s} "
          f"{'cota':>7s} {'Lambda':>8s}  seleccionada por")
    for a, m in [(0, 3.0), (1, 2.0)]:      # alpha* = m para A->mA
        mu = kinetic_norm(Sm, k, [a])
        lam = growth_rate(Sm, Sp, k, [a])
        sel = "MAF" if a == 0 else "Alg.3"
        print(f"  {names_arc[a]:14s} {m:7.3f} {mu:9.3f} "
              f"{(m-1)*mu:7.3f} {lam:8.4f}  {sel}")

    # Confirmacion por el barrido: Alg.3 debe elegir el objetivo de Bloque B (=1.0)
    best, _, exact = computeParametricSweep(
        Sm, Sp, k, decimals=1, max_points=400,
        time_limit_iteration=60, max_steps=200, accuracy=1e-5)
    arcs = sorted(best["info"]["act_arcs"])
    print(f"\n  Barrido (Alg.3): malla exacta={exact}  "
          f"obj*={best['obj']:.4f}  alpha*={best['alpha_star']:.4f}  "
          f"arcos={arcs}")
    print("  (En bloque-diagonal el optimo es DEGENERADO: el bloque lento puede "
          "absorberse\n   sin cambiar obj=max(0.6,1.0)=1.0; el bloque VINCULANTE "
          "es rB.)")


# ----------------------------------------------------------------------
# 2) Oregonator (tab:oregonator_resultados)
# ----------------------------------------------------------------------
def oregonator():
    print("\n" + "=" * 66)
    print("OREGONATOR  (tab:oregonator_resultados)")
    print("=" * 66)
    # Filas X,Y,Z ; columnas R1 (Y->X), R3 (X->2X+2Z), R5 (Z->Y)
    Sm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float)
    Sp = np.array([[1, 2, 0], [0, 0, 1], [0, 2, 0]], float)
    k = np.array([0.0768, 2.016, 0.05])       # k efectivas

    best, results, exact = computeParametricSweep(
        Sm, Sp, k, decimals=2, max_points=400,
        time_limit_iteration=60, max_steps=200, accuracy=1e-5)
    arcs = sorted(best["info"]["act_arcs"])
    alpha = best["alpha_star"]

    # Verdad de referencia analitica: raiz real >2 de alpha^3 - 2 alpha^2 - 2 = 0
    roots = np.roots([1.0, -2.0, 0.0, -2.0])
    alpha_an = max(r.real for r in roots if abs(r.imag) < 1e-9)

    print(f"  Barrido (Alg.3): malla exacta={exact}  arcos recuperados={arcs}")
    print(f"  alpha* recuperado = {alpha:.5f}   analitico = {alpha_an:.5f}   "
          f"|err| = {abs(alpha-alpha_an):.2e}\n")

    # Filas de la tabla
    mu_cycle = kinetic_norm(Sm, k, arcs)
    lam_cycle = growth_rate(Sm, Sp, k, arcs)
    mu_r3 = kinetic_norm(Sm, k, [1])
    lam_r3 = growth_rate(Sm, Sp, k, [1])

    print(f"  {'Subred':26s} {'alpha*':>7s} {'||S^-||k':>9s} "
          f"{'cota':>7s} {'Lambda':>8s}  ajuste")
    print(f"  {'R3 aislada (X->2X+2Z)':26s} {2.0:7.3f} {mu_r3:9.3f} "
          f"{(2-1)*mu_r3:7.3f} {lam_r3:8.4f}  tight")
    print(f"  {'ciclo {R1,R3,R5} (Alg.3)':26s} {alpha:7.3f} {mu_cycle:9.3f} "
          f"{(alpha-1)*mu_cycle:7.3f} {lam_cycle:8.4f}  loose")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    designed_instance()
    oregonator()