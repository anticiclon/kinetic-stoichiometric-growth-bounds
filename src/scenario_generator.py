# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:53:34 2026

@author: Trabajador
"""

# -*- coding: utf-8 -*-
"""
scenario_generator.py
Generador de escenarios de prueba para el barrido paramétrico (Algorithm 3).

Cuatro funciones principales:

  scenarioFromCore   – núcleo autocatalítico verificado + reacciones aleatorias.
                       Garantiza al menos un subhipergrafo auto-amplificante.
  scenarioRandom     – red totalmente aleatoria (puede o no ser auto-amplificante).
                       Útil para testear infactibilidad y escalabilidad.
  scenarioPareto     – red bloque-diagonal con tres subredes de perfiles distintos
                       (α* alto/k lento, α* medio/k medio, α* bajo/k rápido),
                       diseñada para que el máximo de (α*-1)·‖S⁻‖_k NO esté
                       en el bloque con mayor α*.
  generateBatch      – genera una cuadrícula sistemática de escenarios.

Todas las funciones guardan {name}_minus.txt, {name}_plus.txt y {name}_k.txt
en SCENARIOS_DIR (./scenarios/ por defecto).

Uso rápido:
    python scenario_generator.py          # genera todos los escenarios de demo
    python scenario_generator.py --list   # lista los núcleos disponibles
"""

import argparse
import numpy as np
import os
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCENARIOS_DIR = os.path.join(_REPO_ROOT, "scenarios")


# ===========================================================================
# Utilidades internas
# ===========================================================================

def _ensure_dir(path: str = SCENARIOS_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def _save(name: str,
          m_minus: np.ndarray,
          m_plus:  np.ndarray,
          k:       Optional[np.ndarray] = None) -> str:
    """Guarda las matrices y, opcionalmente, las constantes cinéticas."""
    _ensure_dir()
    np.savetxt(os.path.join(SCENARIOS_DIR, f"{name}_minus.txt"), m_minus, fmt="%d")
    np.savetxt(os.path.join(SCENARIOS_DIR, f"{name}_plus.txt"),  m_plus,  fmt="%d")
    if k is not None:
        np.savetxt(os.path.join(SCENARIOS_DIR, f"{name}_k.txt"), k)
    n_s, n_r = m_minus.shape
    k_info = f"  k={k.min():.3f}..{k.max():.3f}" if k is not None else ""
    print(f"  Guardado '{name}'  [{n_s} esp × {n_r} rxn]{k_info}")
    return name


def _sample_k(n: int,
              k_range: Tuple[float, float],
              distribution: str,
              rng: np.random.Generator) -> np.ndarray:
    """
    Muestrea n constantes cinéticas.

    distribution:
      'uniform'    – k ~ U(lo, hi)
      'loguniform' – k ~ exp(U(log lo, log hi)); recomendado para cubrir
                     varios órdenes de magnitud con sesgo neutro.
      'mixed'      – mitad lenta (0–30 % del rango), mitad rápida (70–100 %);
                     diseñado para crear fronteras de Pareto interesantes.
    """
    lo, hi = k_range
    if lo <= 0:
        raise ValueError("k_range debe tener lo > 0.")
    if distribution == "uniform":
        return rng.uniform(lo, hi, size=n)
    elif distribution == "loguniform":
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
    elif distribution == "mixed":
        cut = lo + 0.3 * (hi - lo)
        half = n // 2
        slow = rng.uniform(lo,  cut, size=half)
        fast = rng.uniform(lo + 0.7 * (hi - lo), hi, size=n - half)
        k = np.concatenate([slow, fast])
        rng.shuffle(k)
        return k
    else:
        raise ValueError(
            f"Distribución desconocida: '{distribution}'. "
            "Usa 'uniform', 'loguniform' o 'mixed'."
        )


def _block_diagonal(parts_minus: List[np.ndarray],
                    parts_plus:  List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Ensambla matrices bloque-diagonales a partir de bloques individuales."""
    n_s = sum(m.shape[0] for m in parts_minus)
    n_r = sum(m.shape[1] for m in parts_minus)
    S_m = np.zeros((n_s, n_r), dtype=int)
    S_p = np.zeros((n_s, n_r), dtype=int)
    rs = cs = 0
    for sm, sp in zip(parts_minus, parts_plus):
        nr, nc = sm.shape
        S_m[rs:rs + nr, cs:cs + nc] = sm
        S_p[rs:rs + nr, cs:cs + nc] = sp
        rs += nr
        cs += nc
    return S_m, S_p


# ===========================================================================
# Catálogo de núcleos autocatalíticos verificados
# ===========================================================================
# Convención: S_minus[s, r] = multiplicidad de la especie s como reactivo de r.
#             S_plus [s, r] = multiplicidad de la especie s como producto de r.
#
# MAF exacto: α* = sup_{x>0} min_v [(S⁺x)_v / (S⁻x)_v].
# Verificación analítica en los comentarios de cada entrada.

CORES: Dict[str, Dict] = {
    # -------------------------------------------------------------------
    # Una especie
    # -------------------------------------------------------------------
    "v_simple": {
        # r: A → 2A.  α* = 2/1 = 2.  (MAF exacto = 2)
        "S_minus": np.array([[1]], dtype=int),
        "S_plus":  np.array([[2]], dtype=int),
        "maf": 2.0,
        "desc": "A → 2A  (MAF = 2)"
    },
    "v_faster": {
        # r: A → 3A.  α* = 3/1 = 3.  (MAF exacto = 3)
        "S_minus": np.array([[1]], dtype=int),
        "S_plus":  np.array([[3]], dtype=int),
        "maf": 3.0,
        "desc": "A → 3A  (MAF = 3)"
    },
    "i_simple": {
        # r: 2A → 3A.  α* = 3/2.  (MAF exacto = 3/2)
        "S_minus": np.array([[2]], dtype=int),
        "S_plus":  np.array([[3]], dtype=int),
        "maf": 1.5,
        "desc": "2A → 3A  (MAF = 3/2)"
    },
    # -------------------------------------------------------------------
    # Dos especies
    # -------------------------------------------------------------------
    "cross2": {
        # r1: A → A+B,  r2: B → A+B.
        # α_A(x) = (x1+x2)/x1,  α_B(x) = (x1+x2)/x2.
        # min = 1 + min(x2/x1, x1/x2), maximizado en x1=x2: α* = 2.
        "S_minus": np.array([[1, 0],
                              [0, 1]], dtype=int),
        "S_plus":  np.array([[1, 1],
                              [1, 1]], dtype=int),
        "maf": 2.0,
        "desc": "A→A+B, B→A+B  (MAF = 2)"
    },
    "mutual2": {
        # r1: A → 2A+B,  r2: B → A+2B.
        # α_A(x) = (2x1+x2)/x1 = 2+x2/x1,  α_B(x) = (x1+2x2)/x2 = x1/x2+2.
        # α = 2 + min(x2/x1, x1/x2), maximizado en x1=x2: α* = 3.
        "S_minus": np.array([[1, 0],
                              [0, 1]], dtype=int),
        "S_plus":  np.array([[2, 1],
                              [1, 2]], dtype=int),
        "maf": 3.0,
        "desc": "A→2A+B, B→A+2B  (MAF = 3)"
    },
    # -------------------------------------------------------------------
    # Tres especies
    # -------------------------------------------------------------------
    "cycle3": {
        # r1: A→A+B,  r2: B→B+C,  r3: C→C+A.
        # α_A = (x1+x3)/x1 = 1+x3/x1,  α_B = (x1+x2)/x2,  α_C = (x2+x3)/x3.
        # En x1=x2=x3: α*=2.  (verificado: t³=1 ⟹ t=1, α=1+1=2)
        "S_minus": np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]], dtype=int),
        "S_plus":  np.array([[1, 0, 1],
                              [1, 1, 0],
                              [0, 1, 1]], dtype=int),
        "maf": 2.0,
        "desc": "Ciclo 3: A→A+B, B→B+C, C→C+A  (MAF = 2)"
    },
}


def list_cores() -> List[str]:
    """Imprime y devuelve los nombres de los núcleos disponibles."""
    print("Núcleos autocatalíticos disponibles:")
    for name, info in CORES.items():
        print(f"  {name:12s}  MAF={info['maf']:.4f}   {info['desc']}")
    return list(CORES.keys())


# ===========================================================================
# 1. scenarioFromCore
# ===========================================================================

def scenarioFromCore(core_name:          str   = "v_simple",
                     n_extra_species:    int   = 2,
                     n_extra_reactions:  int   = 5,
                     max_stoich:         int   = 2,
                     seed:               int   = 42,
                     version:            int   = 0,
                     k_range:            Tuple[float, float] = (0.1, 5.0),
                     k_distribution:     str   = "loguniform") -> str:
    """
    Genera un escenario embebiendo un núcleo autocatalítico verificado en
    una red mayor con reacciones aleatorias adicionales.

    Estructura resultante de la matriz (bloque + extra):
      · Columnas 0…n_core_rxns-1 : reacciones del núcleo (auto-amplificantes).
      · Columnas siguientes       : reacciones aleatorias adicionales.
      · Filas 0…n_core_sp-1      : especies del núcleo.
      · Filas siguientes          : especies extra.

    Al ser el núcleo un subhipergrafo embebido, la red entera tiene siempre
    al menos un subhipergrafo auto-amplificante, garantizando que el barrido
    paramétrico encuentre al menos un P(μⱼ) factible.

    Parámetros
    ----------
    core_name         : nombre en CORES (ver list_cores())
    n_extra_species   : número de especies adicionales al núcleo
    n_extra_reactions : número de reacciones aleatorias adicionales
    max_stoich        : coeficiente estequiométrico máximo en reacciones extra
    seed              : semilla aleatoria
    version           : sufijo para el nombre del fichero
    k_range           : (k_min, k_max) para el muestreo de constantes cinéticas
    k_distribution    : 'uniform', 'loguniform' o 'mixed' (ver _sample_k)
    """
    if core_name not in CORES:
        raise ValueError(f"Núcleo desconocido '{core_name}'. Usa list_cores().")

    core = CORES[core_name]
    Sm_c = core["S_minus"]
    Sp_c = core["S_plus"]
    n_cs, n_cr = Sm_c.shape            # core: n_cs especies, n_cr reacciones
    n_ts = n_cs + n_extra_species      # total especies
    n_tr = n_cr + n_extra_reactions    # total reacciones

    rng = np.random.default_rng(seed)

    S_minus = np.zeros((n_ts, n_tr), dtype=int)
    S_plus  = np.zeros((n_ts, n_tr), dtype=int)

    # Incrustar el núcleo en el bloque superior-izquierdo
    S_minus[:n_cs, :n_cr] = Sm_c
    S_plus[:n_cs,  :n_cr] = Sp_c

    # Reacciones adicionales aleatorias (sparse)
    for a in range(n_cr, n_tr):
        n_react   = rng.integers(1, min(3, n_ts) + 1)
        n_prod    = rng.integers(1, min(3, n_ts) + 1)
        reactants = rng.choice(n_ts, size=n_react, replace=False)
        products  = rng.choice(n_ts, size=n_prod,  replace=False)
        for s in reactants:
            S_minus[s, a] = rng.integers(1, max_stoich + 1)
        for s in products:
            S_plus[s, a]  = rng.integers(1, max_stoich + 1)

    k = _sample_k(n_tr, k_range, k_distribution, rng)
    name = f"core_{core_name}_s{n_ts}_r{n_tr}_v{version}"
    return _save(name, S_minus, S_plus, k)


# ===========================================================================
# 2. scenarioRandom
# ===========================================================================

def scenarioRandom(n_species:    int,
                   n_reactions:  int,
                   sparsity:     float = 0.35,
                   max_stoich:   int   = 3,
                   seed:         int   = 42,
                   version:      int   = 0,
                   k_range:      Tuple[float, float] = (0.1, 5.0),
                   k_distribution: str = "loguniform") -> str:
    """
    Genera una red CRN aleatoria pura sin garantía de autocatálisis.

    Útil para:
      · Testear el comportamiento del código cuando todos los P(μⱼ) son
        infactibles (ninguna subred auto-amplificante).
      · Pruebas de escalabilidad con matrices de gran tamaño.
      · Comparar con resultados de scenarioFromCore (mismo tamaño, sin núcleo).

    Parámetros
    ----------
    n_species      : número de especies (filas)
    n_reactions    : número de reacciones (columnas)
    sparsity       : fracción esperada de entradas no nulas en cada matriz
    max_stoich     : coeficiente estequiométrico máximo
    seed           : semilla aleatoria
    version        : sufijo para el nombre del fichero
    k_range        : rango para las constantes cinéticas
    k_distribution : 'uniform', 'loguniform' o 'mixed'
    """
    rng = np.random.default_rng(seed)
    shape = (n_species, n_reactions)
    mask_m = rng.random(shape) < sparsity
    mask_p = rng.random(shape) < sparsity
    vals   = rng.integers(1, max_stoich + 1, shape)
    S_minus = (mask_m * vals).astype(int)
    S_plus  = (mask_p * vals).astype(int)
    k = _sample_k(n_reactions, k_range, k_distribution, rng)
    name = f"random_s{n_species}_r{n_reactions}_sp{int(sparsity * 100)}_v{version}"
    return _save(name, S_minus, S_plus, k)


# ===========================================================================
# 3. scenarioPareto
# ===========================================================================

def scenarioPareto(seed: int = 42) -> str:
    """
    Genera una red bloque-diagonal diseñada para tener una frontera de Pareto
    (α*, ‖S⁻‖_k) no trivial, replicando la situación de la Tabla de la
    Sección 4 de la propuesta:

      Bloque 1  v_faster: MAF=3,   k pequeño  → alto α*, bajo  ‖S⁻‖_k, obj moderado
      Bloque 2  cross2:  MAF=2,   k medio    → α* y ‖S⁻‖_k equilibrados, obj medio
      Bloque 3  i_simple: MAF=3/2, k grande   → bajo α*, alto  ‖S⁻‖_k, obj mayor

    El objetivo (α*-1)·‖S⁻‖_k es máximo en el Bloque 3, aunque este tiene
    el MAF más bajo: eso ilustra que maximizar solo α* es insuficiente.

    La red completa tiene 4 especies y 4 reacciones.
    También guarda {name}_k.txt con las k sugeridas para el main.

    Devuelve el nombre del escenario.
    """
    rng = np.random.default_rng(seed)

    # (núcleo, rango k)  → estimaciones de ‖S⁻‖_k y obj en el bloque
    #   Bloque 1: ‖S⁻‖_k = 1·k ≈ 0.15,  obj = (3-1)·0.15 = 0.30
    #   Bloque 2: ‖S⁻‖_k = max(k₁,k₂) ≈ 1.0,  obj = (2-1)·1.0 = 1.00
    #   Bloque 3: ‖S⁻‖_k = 2·k ≈ 8.0,  obj = (1.5-1)·8.0 = 4.00  ← ganador
    blocks = [
        ("v_faster",  (0.10, 0.20)),
        ("cross2",    (0.80, 1.20)),
        ("i_simple",  (3.50, 4.50)),
    ]

    parts_m, parts_p, ks = [], [], []
    for core_name, k_range in blocks:
        core = CORES[core_name]
        n_r  = core["S_minus"].shape[1]
        parts_m.append(core["S_minus"])
        parts_p.append(core["S_plus"])
        ks.append(rng.uniform(*k_range, size=n_r))

    S_minus, S_plus = _block_diagonal(parts_m, parts_p)
    k = np.concatenate(ks)

    name = "pareto_bench_3blocks"
    _save(name, S_minus, S_plus, k)

    # Tabla de perfiles esperados
    print("  Perfiles esperados (bloque-diagonal):")
    print(f"  {'Bloque':8s}  {'Núcleo':12s}  {'MAF':>5s}  "
          f"{'k_range':>14s}  {'‖S⁻‖_k est.':>12s}  {'obj est.':>9s}")
    for i, (core_name, k_range) in enumerate(blocks):
        core  = CORES[core_name]
        k_mid = 0.5 * (k_range[0] + k_range[1])
        # ‖S⁻‖_k = max_s sum_r S^-_{sr}·k_r·z_r para z_r=1 (subred completa)
        norm_est = float(np.max(np.sum(core["S_minus"], axis=1))) * k_mid
        obj_est  = (core["maf"] - 1.0) * norm_est
        flag = "  ← ganador esperado" if i == len(blocks) - 1 else ""
        print(f"  {i+1:<8d}  {core_name:12s}  {core['maf']:>5.3f}  "
              f"{str(k_range):>14s}  {norm_est:>12.3f}  {obj_est:>9.3f}{flag}")
    return name


# ===========================================================================
# 4. generateBatch
# ===========================================================================

def generateBatch(sizes:      Optional[List[Tuple[int, int]]] = None,
                  n_versions: int  = 3,
                  seed_base:  int  = 0) -> List[str]:
    """
    Genera una cuadrícula sistemática de escenarios para pruebas exhaustivas.

    Crea dos familias para cada tamaño en `sizes`:
      · Con núcleo embebido (v_simple) → siempre factible.
      · Aleatorio puro → puede ser infactible.

    Parámetros
    ----------
    sizes       : lista de (n_especies, n_reacciones); None usa la cuadrícula
                  predefinida (pequeño / mediano / grande).
    n_versions  : versiones aleatorias distintas por tamaño.
    seed_base   : semilla base; la versión v usa seed_base + v.

    Devuelve la lista de nombres generados.
    """
    if sizes is None:
        sizes = [
            (2,  4),    # mínimo
            (3,  6),    # pequeño
            (4, 10),    # pequeño-mediano
            (5, 15),    # mediano
            (6, 20),    # mediano-grande
        ]

    names = []

    print("=== Batch: núcleo v_simple embebido ===")
    for (n_s, n_r) in sizes:
        for v in range(n_versions):
            # Reservamos 1 especie y 1 reacción para el núcleo v_simple
            name = scenarioFromCore(
                "v_simple",
                n_extra_species=max(0, n_s - 1),
                n_extra_reactions=max(0, n_r - 1),
                seed=seed_base + v,
                version=v,
            )
            names.append(name)

    print("\n=== Batch: redes aleatorias puras ===")
    for (n_s, n_r) in sizes:
        for v in range(n_versions):
            name = scenarioRandom(n_s, n_r, seed=seed_base + v + 1000, version=v)
            names.append(name)

    return names


# ===========================================================================
# CLI
# ===========================================================================

def _run_demo() -> None:
    print("=" * 60)
    print("DEMO: generando escenarios de prueba")
    print("=" * 60)

    print("\n--- 1. Un escenario por núcleo ---")
    for core in list(CORES.keys()):
        scenarioFromCore(core, n_extra_species=2, n_extra_reactions=4,
                         seed=0, version=0)

    print("\n--- 2. Redes aleatorias (sin garantía de autocatálisis) ---")
    for (s, r) in [(3, 6), (4, 10), (5, 15)]:
        scenarioRandom(s, r, seed=7, version=0)

    print("\n--- 3. Benchmark de Pareto (3 bloques con perfiles distintos) ---")
    scenarioPareto(seed=42)

    print("\n--- 4. Batch sistemático (tamaños pequeños, 2 versiones) ---")
    generateBatch(sizes=[(3, 6), (4, 8)], n_versions=2)

    print("\nFicheros generados en:", os.path.abspath(SCENARIOS_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de escenarios para Algorithm 3.")
    parser.add_argument("--list",  action="store_true", help="Lista los núcleos disponibles.")
    parser.add_argument("--demo",  action="store_true", help="Genera todos los escenarios de demo (por defecto).")
    parser.add_argument("--pareto", action="store_true", help="Solo el benchmark de Pareto.")
    args = parser.parse_args()

    if args.list:
        list_cores()
    elif args.pareto:
        scenarioPareto()
    else:
        _run_demo()
































