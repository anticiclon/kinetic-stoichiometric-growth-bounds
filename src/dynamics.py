# -*- coding: utf-8 -*-
"""
dynamics.py
============
Crecimiento balanceado bajo cinética escalable de primer orden.

Para una subred (S^-, S^+, k) en la que cada reacción es de PRIMER ORDEN
EXACTO en su única especie interna reactiva, la dinámica

        ṅ = S_net · J(n),     J_r(n) = k_r · n_{s(r)}

es LINEAL:  ṅ = M n,  con  M = S_net · diag(k) · P,  donde P[r, s(r)] = 1
y s(r) es la especie reactiva de r.

La tasa de crecimiento balanceado Λ es el autovalor dominante (mayor parte
real) de M. Como M es una matriz de Metzler (entradas fuera de la diagonal
no negativas: solo hay producción), Λ es real y su autovector asociado es
no negativo: es la composición de crecimiento balanceado n_0.

Este Λ es el "Λ observado" que el Experimento 2 contrasta contra la cota
cinético-estequiométrica  (α* - 1) · ||S^-||_k.

Solo usa numpy (la solución del sistema lineal se obtiene por descomposición
espectral, sin necesidad de un integrador).
"""

import warnings
import numpy as np


# ---------------------------------------------------------------------------
def _touched_species(S_minus, S_plus, active_arcs):
    touched = set()
    for r in active_arcs:
        touched |= set(np.where(S_minus[:, r] > 0)[0].tolist())
        touched |= set(np.where(S_plus[:, r] > 0)[0].tolist())
    return sorted(touched)


# ---------------------------------------------------------------------------
def build_linear_operator(S_minus, S_plus, k,
                          active_arcs=None, active_nodes=None, strict=False):
    """
    Construye el operador lineal M de la subred indicada por active_arcs,
    restringido a las especies relevantes.

    Parámetros
    ----------
    S_minus, S_plus : matrices estequiométricas (|S| x |R|), filas=especies.
    k               : vector de constantes cinéticas (|R|).
    active_arcs     : reacciones activas (None -> todas).
    active_nodes    : especies a las que restringir M (None -> las tocadas
                      por active_arcs).
    strict          : si True, lanza error cuando una reacción no es de
                      primer orden exacto; si False, solo avisa.

    Devuelve (M, info) con info["species_index"] (qué especies son las filas
    de M) e info["warnings"].
    """
    S_minus = np.asarray(S_minus, dtype=float)
    S_plus = np.asarray(S_plus, dtype=float)
    k = np.asarray(k, dtype=float)
    n_s, n_r = S_minus.shape
    S_net = S_plus - S_minus

    if active_arcs is None:
        active_arcs = list(range(n_r))
    active_arcs = [int(a) for a in active_arcs]

    P = np.zeros((n_r, n_s))
    msgs = []
    for r in active_arcs:
        reac = np.where(S_minus[:, r] > 0)[0]
        if len(reac) == 0:
            msgs.append(f"R{r}: sin reactivo interno (reacción de entrada/food); "
                        f"se omite del operador.")
            continue
        if len(reac) > 1:
            msgs.append(f"R{r}: {len(reac)} reactivos internos -> no es de grado 1 "
                        f"(cinética NO escalable; la ODE lineal no es fiel).")
        s = int(reac[0])
        if not np.isclose(S_minus[s, r], 1.0):
            msgs.append(f"R{r}: multiplicidad {S_minus[s, r]:.0f} en especie {s} -> "
                        f"no es de primer orden exacto (la ODE lineal no es fiel).")
        P[r, s] = 1.0  # primer orden en la especie reactiva

    K = np.zeros((n_r, n_r))
    for r in active_arcs:
        K[r, r] = k[r]

    M_full = S_net @ K @ P  # (|S| x |R|)(|R| x |R|)(|R| x |S|) = |S| x |S|

    if strict and msgs:
        raise ValueError("Red incompatible con cinética escalable:\n" + "\n".join(msgs))
    for m in msgs:
        warnings.warn(m)

    if active_nodes is None:
        active_nodes = _touched_species(S_minus, S_plus, active_arcs)
    idx = sorted(set(int(v) for v in active_nodes))
    if len(idx) == 0:
        return np.zeros((0, 0)), {"species_index": [], "warnings": msgs}
    M = M_full[np.ix_(idx, idx)]
    return M, {"species_index": idx, "warnings": msgs}


# ---------------------------------------------------------------------------
def balanced_growth_rate(M):
    """
    Λ = mayor parte real de los autovalores de M (autovalor de Perron, M de
    Metzler). Devuelve (Lambda, n0) con n0 el autovector dominante, orientado
    a no negativo y normalizado.
    """
    if M.size == 0:
        return float("nan"), np.zeros(0)
    w, V = np.linalg.eig(M)
    idx = int(np.argmax(w.real))
    Lambda = float(w[idx].real)
    v = V[:, idx].real
    if np.sum(v) < 0:
        v = -v
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v = v / nrm
    return Lambda, v


# ---------------------------------------------------------------------------
def growth_rate_of_subnetwork(S_minus, S_plus, k,
                              active_arcs=None, active_nodes=None,
                              strict=True, tol=1e-12):
    """Atajo: Λ de la subred activa. Devuelve (Lambda, M, info)."""
    M, info = build_linear_operator(S_minus, S_plus, k, active_arcs,
                                    active_nodes, strict=strict)
    Lambda, n0 = balanced_growth_rate(M)
    info["n0"] = n0
    info["M"] = M
    # Hipótesis del Teorema 2.8: composición de crecimiento estrictamente positiva
    info["perron_positive"] = bool(n0.size > 0 and np.all(n0 > tol))
    if strict and not info["perron_positive"]:
        raise ValueError("El autovector dominante no es estrictamente positivo: "
                         "la subred no admite crecimiento balanceado con n0 >> 0.")
    return Lambda, M, info

# ---------------------------------------------------------------------------
def kinetic_norm(S_minus, k, active_arcs=None, active_nodes=None):
    """
    ||S^-|_{A'}||_k = max_{s} sum_{r in A'} (S^-)[s,r] k_r   (sobre arcos
    activos; máximo sobre especies activas si se indican).
    """
    S_minus = np.asarray(S_minus, dtype=float)
    k = np.asarray(k, dtype=float)
    n_s, n_r = S_minus.shape
    if active_arcs is None:
        active_arcs = list(range(n_r))
    species = range(n_s) if active_nodes is None else active_nodes
    consum = [sum(S_minus[s, r] * k[r] for r in active_arcs) for s in species]
    return float(max(consum)) if consum else 0.0


# ---------------------------------------------------------------------------
def simulate(M, n0, t_grid):
    """
    Solución exacta del sistema lineal ṅ = M n por descomposición espectral:
        n(t) = sum_i c_i e^{λ_i t} v_i.
    Devuelve N (len(t_grid) x dim). Útil para mostrar la trayectoria
    transitoria; la tasa asintótica coincide con balanced_growth_rate(M).
    """
    w, V = np.linalg.eig(M)
    try:
        c = np.linalg.solve(V, n0.astype(complex))
    except np.linalg.LinAlgError:
        c, *_ = np.linalg.lstsq(V, n0.astype(complex), rcond=None)
    N = np.array([(V @ (c * np.exp(w * t))).real for t in t_grid])
    return N


# ---------------------------------------------------------------------------
def empirical_growth_rate(t_grid, N):
    """
    Tasa de crecimiento empírica d/dt log||n(t)|| medida en la segunda mitad
    del tramo (régimen asintótico). Sirve para verificar que coincide con Λ.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    norms = np.linalg.norm(N, axis=1)
    mask = norms > 0
    lg, tt = np.log(norms[mask]), t_grid[mask]
    if len(tt) < 2:
        return float("nan")
    half = len(tt) // 2
    A = np.vstack([tt[half:], np.ones(len(tt[half:]))]).T
    slope, _ = np.linalg.lstsq(A, lg[half:], rcond=None)[0]
    return float(slope)