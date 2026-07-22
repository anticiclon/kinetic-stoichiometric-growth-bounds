# -*- coding: utf-8 -*-
"""
verify_cores.py
===============
Comprueba, con el barrido paramétrico (SIN cribado, para enumerar TODAS las
normas alcanzables), la afirmación de la subsección de rTCA/glioxilato:

    "el sweep identifica un único NÚCLEO AUTO-AMPLIFICANTE" (alpha* > 1,
    cf. Teorema thm:maf_char), aunque puede haber además otros niveles de
    mu factibles para el MILP con alpha*=1 exacto (ciclos triviales,
    produccion=consumo, que NO son autoamplificantes por definicion del
    paper y por tanto no cuentan para esta afirmación).

Para cada red imprime:
  - número de normas factibles para el MILP vs. número de ellas que son
    genuinamente auto-amplificantes (alpha* > 1),
  - el núcleo recuperado en cada una y su alpha*,
  - el núcleo óptimo global y si coincide con el esperado.

Nota sobre k: se usa un vector genérico de primos SOLO para separar núcleos
distintos; el número de normas alcanzables es una propiedad estructural
(estequiométrica), independiente de la elección concreta de k. alpha* tampoco
depende de k, así que debe reproducir el valor esperado.

Requisitos: numpy, gurobipy y el algorithm_3.py nuevo (en la misma carpeta).
Uso:  python verify_cores.py
"""

import numpy as np
from algorithm_3 import computeParametricSweep

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


def build_glyoxylate():
    Sm = np.zeros((7, 7))
    Sp = np.zeros((7, 7))
    for s, r in {0: 0, 1: 1, 2: 2, 3: 6, 4: 3, 5: 4, 6: 5}.items():
        Sm[s, r] = 1
    for r, o in {0: [1], 1: [2], 2: [3, 4], 3: [5], 4: [0], 5: [6], 6: [5]}.items():
        for s in o:
            Sp[s, r] += 1
    return Sm, Sp, "glyoxylate (7 reactions)", [0, 1, 2, 3, 4, 6], 1.1487


def build_rtca_full():
    Sm = np.eye(11)
    Sp = np.zeros((11, 11))
    for r, o in {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6],
                 6: [7], 7: [0, 8], 8: [9], 9: [10], 10: [0]}.items():
        for s in o:
            Sp[s, r] += 1
    return Sm, Sp, "rTCA full (11 reactions)", list(range(11)), 1.0764


def verify(Sm, Sp, name, expected_core, expected_alpha):
    n_arcs = Sm.shape[1]
    k = np.array(PRIMES[:n_arcs], dtype=float)

    best, results, exact = computeParametricSweep(
        Sm, Sp, k,
        decimals=0,               # k ya entero (primos): malla exacta
        max_points=5000,
        time_limit_iteration=120,
        max_steps=300,
        accuracy=1e-6,
        screening=False,          # SIN cribado: se visitan TODAS las normas
    )

    feas = [r for r in results if r["feasible"]]
    # Por Teorema thm:maf_char del paper, un subhipergrafo es AUTO-AMPLIFICANTE
    # solo si 1 < alpha* < infinito (estricto). Un subproblema P(mu) puede ser
    # factible para el MILP (satisface auto-suficiencia) con alpha*=1 exacto
    # sin ser autoamplificante: es un ciclo trivial (produccion=consumo), no
    # crecimiento neto. Por eso filtramos por alpha* > 1 + tolerancia antes de
    # contar "normas alcanzables" en el sentido de la Seccion 5.2.
    amp_tol = 1e-6
    self_amplifying = [r for r in feas if r["alpha_star"] > 1.0 + amp_tol]

    print(f"\n===== {name} =====")
    print(f"  malla exacta                 : {exact}")
    print(f"  normas factibles (MILP)       : {len(feas)}  "
          f"(incluye niveles triviales con alpha*=1, no autoamplificantes)")
    print(f"  normas auto-amplificantes     : {len(self_amplifying)}  "
          f"(alpha* > 1, cf. Teorema de caracterizacion)")
    for r in feas:
        tag = "" if r["alpha_star"] > 1.0 + amp_tol else "  [trivial, alpha*=1]"
        print(f"     mu={r['mu']:.0f}  alpha*={r['alpha_star']:.4f}  "
              f"núcleo(arcos)={sorted(r['info']['act_arcs'])}{tag}")

    if best is None:
        print("  [!] ningún subproblema factible: revisa la red.")
        return

    core = sorted(best["info"]["act_arcs"])
    print(f"  núcleo óptimo recuperado     : {core}")
    print(f"  alpha* óptimo                : {best['alpha_star']:.4f} "
          f"(esperado ~{expected_alpha})")
    print(f"  -> UN solo núcleo auto-amplificante : {len(self_amplifying) == 1}")
    print(f"  -> núcleo == esperado {sorted(expected_core)}: "
          f"{core == sorted(expected_core)}")
    print(f"  -> alpha* == esperado        : "
          f"{abs(best['alpha_star'] - expected_alpha) < 1e-2}")


if __name__ == "__main__":
    for builder in (build_glyoxylate, build_rtca_full):
        verify(*builder())