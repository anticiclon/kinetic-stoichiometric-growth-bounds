# -*- coding: utf-8 -*-
"""
plot_network.py
Dibujado de la subred óptima devuelta por computeParametricSweep.

Adapta el estilo de visualización del paper anterior (grafo bipartito,
layout Graphviz, colormap de flujos, nodos activos resaltados) al formato
de S⁻/S⁺ y al vector de flujos opt_x del Algoritmo 3.

Función principal:
    plotOptimalSubnetwork(S_minus, S_plus, best, ...)

Auxiliares expuestas:
    buildNetworkGraph(...)     – construye el DiGraph bipartito
    simplifyGraph(G)           – elimina nodos reacción de grado (1,1)
    plotNetworkGraph(G, ...)   – dibuja (si ya tienes el grafo construido)
"""

import warnings
from typing import List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.ticker as mticker


try:
    from networkx.drawing.nx_pydot import graphviz_layout
    _HAS_GRAPHVIZ = True
except Exception:
    _HAS_GRAPHVIZ = False
    warnings.warn(
        "graphviz / pydot no disponibles; se usará spring_layout como fallback. "
        "Instala con:  pip install pygraphviz  o  pip install pydot"
    )


# ---------------------------------------------------------------------------
# Colores y estilo (mismo espíritu que el paper)
# ---------------------------------------------------------------------------
_NET_STYLE = dict(
    cmap             = "plasma",       # colormap de flujos (igual que el paper)
    node_size_sp     = 2800,           # tamaño nodo especie
    node_size_rxn    = 400,            # tamaño nodo reacción
    active_alpha     = 0.25,           # transparencia del halo dorado
    active_color     = "gold",
    inactive_ec      = "black",        # borde nodos inactivos
    active_rxn_lw    = 7,              # grosor borde reacciones activas
    inactive_rxn_lw  = 1,
    active_edge_w    = 8,              # grosor aristas activas
    inactive_edge_w  = 1,
    active_edge_a    = 0.85,
    font_size        = 28,
    arrowsize        = 40,
    cbar_label       = "Normalized flow $x_r / x_{\\rm max}$",
    graphviz_prog    = "neato",        # programa de layout (neato, dot, fdp, sfdp)
)


# ===========================================================================
# 1. Construcción del grafo bipartito
# ===========================================================================

def buildNetworkGraph(
    S_minus:        np.ndarray,
    S_plus:         np.ndarray,
    opt_x:          np.ndarray,
    act_arcs:       List[int],
    act_nodes:      List[int],
    species_names:  Optional[List[str]] = None,
    reaction_names: Optional[List[str]] = None,
) -> nx.DiGraph:
    """
    Construye el grafo bipartito dirigido de la CRN.

    Nodos especie   → círculo  (type='species')
    Nodos reacción  → cuadrado (type='reaction')

    Aristas:
      especie → reacción : la especie es reactivo (S⁻[s,r] > 0)
      reacción → especie : la especie es producto  (S⁺[s,r] > 0)

    Atributos de nodo:
      active : bool   (está en act_nodes / act_arcs)
      flow   : float  (opt_x[r] normalizado, solo en reacciones activas)

    Atributos de arista:
      weight : coeficiente estequiométrico
      active : bool
      flow   : igual que el nodo reacción correspondiente (si activa)
    """
    n_s, n_r = S_minus.shape
    act_set_arcs  = set(act_arcs)
    act_set_nodes = set(act_nodes)

    if species_names is None:
        species_names   = [f"S{i+1}"  for i in range(n_s)]
    if reaction_names is None:
        reaction_names  = [f"R{r+1}"  for r in range(n_r)]

    # Flujos normalizados (como en el paper: x / max(x))
    x_max = max(opt_x[a] for a in act_arcs) if act_arcs else 1.0
    norm_flow = {a: float(opt_x[a]) / x_max for a in act_arcs}

    G = nx.DiGraph()

    # --- Nodos especie ---
    for i, name in enumerate(species_names):
        G.add_node(name,
                   type="species",
                   idx=i,
                   active=(i in act_set_nodes))

    # --- Nodos reacción ---
    for r, name in enumerate(reaction_names):
        attrs = dict(type="reaction", idx=r, active=(r in act_set_arcs))
        if r in act_set_arcs:
            attrs["flow"] = norm_flow[r]
        G.add_node(name, **attrs)

    # --- Aristas reactivos (especie → reacción) ---
    for r, rname in enumerate(reaction_names):
        is_active_r = r in act_set_arcs
        flow_r      = norm_flow.get(r, None)
        for s in range(n_s):
            coeff = float(S_minus[s, r])
            if coeff > 0:
                attrs = dict(weight=coeff,
                             active=is_active_r and (s in act_set_nodes))
                if is_active_r:
                    attrs["flow"] = flow_r
                G.add_edge(species_names[s], rname, **attrs)

    # --- Aristas productos (reacción → especie) ---
    for r, rname in enumerate(reaction_names):
        is_active_r = r in act_set_arcs
        flow_r      = norm_flow.get(r, None)
        for s in range(n_s):
            coeff = float(S_plus[s, r])
            if coeff > 0:
                attrs = dict(weight=coeff,
                             active=is_active_r and (s in act_set_nodes))
                if is_active_r:
                    attrs["flow"] = flow_r
                G.add_edge(rname, species_names[s], **attrs)

    return G


# ===========================================================================
# 2. Simplificación: eliminar nodos reacción de grado (1 entrada, 1 salida)
# ===========================================================================

def simplifyGraph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Equivalente a replaceReactionNodesWithAdj2WithFlows del paper.

    Elimina los nodos reacción con exactamente 1 predecesor y 1 sucesor,
    reemplazándolos por una arista directa especie→especie.
    Los atributos (flow, active) se copian a la arista directa.
    """
    H = G.copy()
    changed = True
    while changed:
        changed = False
        for node in list(H.nodes):
            if H.nodes[node].get("type") != "reaction":
                continue
            preds = list(H.predecessors(node))
            succs = list(H.successors(node))
            if len(preds) == 1 and len(succs) == 1:
                pred, succ = preds[0], succs[0]
                attrs = {k: v for k, v in H.nodes[node].items()
                         if k not in ("type", "idx")}
                # Coeficiente = producto de los dos coeficientes
                w_in  = H[pred][node].get("weight", 1.0)
                w_out = H[node][succ].get("weight", 1.0)
                attrs["weight"] = w_in * w_out
                H.add_edge(pred, succ, **attrs)
                H.remove_node(node)
                changed = True
                break   # reiniciar tras modificar
    return H


# ===========================================================================
# 3. Dibujado principal
# ===========================================================================

def plotNetworkGraph(
    G:               nx.DiGraph,
    act_nodes_names: List[str],
    *,
    title:           Optional[str]          = None,
    style:           Optional[dict]         = None,
    save_path:       Optional[str]          = None,
    dpi:             int                    = 150,
    # figsize:         Tuple[float, float]    = (14, 9),
    figsize:         Tuple[float, float]    = (18, 12),
    ax:              Optional[plt.Axes]     = None,
) -> plt.Axes:
    """
    Dibuja el grafo bipartito con el mismo estilo que el paper:
      - Nodos especie: círculos.
      - Nodos reacción: cuadrados.  Activos: borde coloreado por flujo.
      - Aristas activas: coloreadas por flujo con colormap.
      - Aristas inactivas: negras, delgadas.
      - Halo dorado sobre nodos especie activos.
      - Colorbar de flujo.
    """
    s = {**_NET_STYLE, **(style or {})}
    cmap = cm.get_cmap(s["cmap"])

    # --- Layout ---
    if _HAS_GRAPHVIZ:
        pos = graphviz_layout(G, prog=s["graphviz_prog"])
    else:
        pos = nx.spring_layout(G, seed=42)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    species_nodes  = [n for n in G.nodes if G.nodes[n].get("type") == "species"]
    reaction_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "reaction"]

    # Nodos con flujo definido (reacciones activas, o aristas directas tras simplify)
    def _flow(node):
        return G.nodes[node].get("flow", None)

    active_rxn   = [n for n in reaction_nodes if _flow(n) is not None]
    inactive_rxn = [n for n in reaction_nodes if _flow(n) is None]

    # Normalización del colormap sobre flujos en [0, 1] (ya normalizados)
    all_flows = [_flow(n) for n in active_rxn] if active_rxn else [0.0, 1.0]
    norm = mcolors.Normalize(vmin=min(all_flows), vmax=max(all_flows))
    flow_color = {n: cmap(norm(_flow(n))) for n in active_rxn}

    node_sizes = {n: (s["node_size_sp"]  if G.nodes[n].get("type") == "species"
                      else s["node_size_rxn"])
                  for n in G.nodes}

    # ---- Halos dorados sobre nodos activos ----
    act_sp_names = [n for n in species_nodes if n in set(act_nodes_names)]
    if act_sp_names:
        nx.draw_networkx_nodes(
            G, pos, nodelist=act_sp_names,
            node_shape="o", node_color=s["active_color"],
            edgecolors="none",
            node_size=s["node_size_sp"], alpha=s["active_alpha"], ax=ax,
        )

    # ---- Nodos especie (todos, borde negro) ----
    nx.draw_networkx_nodes(
        G, pos, nodelist=species_nodes,
        node_shape="o", node_color="none",
        edgecolors="black", linewidths=3.5,
        node_size=s["node_size_sp"], ax=ax,
    )

    # ---- Nodos reacción inactivos ----
    if inactive_rxn:
        nx.draw_networkx_nodes(
            G, pos, nodelist=inactive_rxn,
            node_shape="s", node_color="none",
            edgecolors=s["inactive_ec"], linewidths=s["inactive_rxn_lw"],
            node_size=s["node_size_rxn"], ax=ax,
        )

    # ---- Nodos reacción activos (borde coloreado) ----
    if active_rxn:
        nx.draw_networkx_nodes(
            G, pos, nodelist=active_rxn,
            node_shape="s", node_color="none",
            edgecolors=[flow_color[n] for n in active_rxn],
            linewidths=s["active_rxn_lw"],
            node_size=s["node_size_rxn"], ax=ax,
        )

    all_sizes = [node_sizes[n] for n in G.nodes]

    # ---- Aristas activas ----
    active_edges = [(u, v) for u, v in G.edges
                    if G[u][v].get("active", False)
                    and G[u][v].get("flow") is not None]
    if active_edges:
        edge_colors_a = [cmap(norm(G[u][v]["flow"])) for u, v in active_edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=active_edges,
            edge_color=edge_colors_a,
            node_size=all_sizes,
            width=s["active_edge_w"],
            alpha=s["active_edge_a"],
            arrowsize=s["arrowsize"],
            connectionstyle="arc3,rad=0.05",
            ax=ax,
        )

    # ---- Aristas inactivas ----
    inactive_edges = [(u, v) for u, v in G.edges
                      if not G[u][v].get("active", False)]
    if inactive_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=inactive_edges,
            edge_color="black",
            node_size=all_sizes,
            width=s["inactive_edge_w"],
            alpha=0.4,
            arrowsize=s["arrowsize"],
            connectionstyle="arc3,rad=0.05",
            ax=ax,
        )

    # ---- Etiquetas ----
    nx.draw_networkx_labels(
        G, pos,
        labels={n: n for n in species_nodes},
        font_size=s["font_size"], ax=ax,
    )

    # ---- Colorbar ----
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(all_flows)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(s["cbar_label"], fontsize=25)
    cbar.ax.tick_params(labelsize=22)

    # ---- Decoración ----
    if title:
        ax.set_title(title, fontsize=13, pad=12)
    ax.axis("off")

    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    pad    = 0.05 * max(max(x_vals) - min(x_vals),
                        max(y_vals) - min(y_vals), 1.0)
    ax.set_xlim(min(x_vals) - pad, max(x_vals) + pad)
    ax.set_ylim(min(y_vals) - pad, max(y_vals) + pad)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return ax


# ===========================================================================
# 4. Función de alto nivel
# ===========================================================================

def plotOptimalSubnetwork(
    S_minus:         np.ndarray,
    S_plus:          np.ndarray,
    best:            dict,
    *,
    species_names:   Optional[List[str]]    = None,
    reaction_names:  Optional[List[str]]    = None,
    simplify:        bool                   = True,
    title:           Optional[str]          = None,
    style:           Optional[dict]         = None,
    figsize:         Tuple[float, float]    = (14, 9),
    save_path:       Optional[str]          = None,
    dpi:             int                    = 150,
    ax:              Optional[plt.Axes]     = None,
) -> Optional[plt.Axes]:
    """
    Dibuja la subred óptima devuelta por computeParametricSweep.

    Parámetros
    ----------
    S_minus, S_plus  : matrices de la CRN completa (n_esp × n_rxn).
    best             : dict devuelto por computeParametricSweep como 'best'.
                       Debe contener best["info"]["opt_x"], ["act_arcs"],
                       ["act_nodes"].
    species_names    : lista de nombres de especies (None → S1, S2, ...).
    reaction_names   : lista de nombres de reacciones (None → R1, R2, ...).
    simplify         : si True, elimina nodos reacción de grado (1,1).
    title            : título del gráfico.
    style            : overrides sobre _NET_STYLE.
    figsize, save_path, dpi, ax : opciones de figura estándar.

    Devuelve ax (None si best es None).
    """
    if best is None:
        print("[plot_network] best=None: la red no tiene subhipergrafo factible.")
        return None

    info       = best.get("info")
    if info is None:
        print("[plot_network] best['info'] es None.")
        return None

    opt_x      = np.array(info["opt_x"])
    act_arcs   = info["act_arcs"]
    act_nodes  = info["act_nodes"]

    n_s, n_r = S_minus.shape
    if species_names is None:
        species_names  = [f"S{i+1}"  for i in range(n_s)]
    if reaction_names is None:
        reaction_names = [f"R{r+1}"  for r in range(n_r)]

    G = buildNetworkGraph(
        S_minus, S_plus, opt_x, act_arcs, act_nodes,
        species_names, reaction_names,
    )

    if simplify:
        G = simplifyGraph(G)

    # Nombres de las especies activas (para el halo dorado)
    act_sp_names = [species_names[i] for i in act_nodes if i < len(species_names)]

    # Título automático si no se pasa
    if title is None:
        alpha = best.get("alpha_star", float("nan"))
        mu    = best.get("mu",         float("nan"))
        obj   = best.get("obj",        float("nan"))
        title = (rf"Subred óptima   "
                 rf"$\alpha^*={alpha:.3g}$,  "
                 rf"$\mu={mu:.3g}$,  "
                 rf"cota $\Lambda \leq {obj:.3g}$")

    return plotNetworkGraph(
        G, act_sp_names,
        title=title, style=style,
        figsize=figsize, save_path=save_path, dpi=dpi, ax=ax,
    )







# ---------------------------------------------------------------------------
# Colores y estilo
# ---------------------------------------------------------------------------
_STYLE = dict(
    color_feasible   = "#1a5e9e",   # azul oscuro – puntos factibles
    color_infeasible = "#cccccc",   # gris claro  – barras de infactibles (opcional)
    color_best       = "#d62728",   # rojo         – punto óptimo
    color_obj        = "#e07b39",   # naranja      – curva obj = (α*-1)·μ
    color_frontier   = "#1a5e9e",   # igual que feasible para la línea escalonada
    alpha_scatter    = 0.75,
    alpha_obj        = 0.85,
    lw_frontier      = 1.6,
    lw_obj           = 1.8,
    ms_feasible      = 48,          # tamaño del scatter de puntos factibles
    ms_best          = 130,
)


# ===========================================================================
# Función principal
# ===========================================================================

def plot_pareto(
    results:           list,
    best:              Optional[dict]   = None,
    *,
    # --- Contenido ---
    show_obj_curve:    bool             = True,
    show_infeasible:   bool             = False,
    show_frontier:     bool             = True,
    label_best:        bool             = True,
    # --- Filtros ---
    alpha_min:         float            = -np.inf,   # ocultar puntos con α* < umbral
    # --- Estética ---
    title:             Optional[str]    = None,
    xlabel: str = r"$\|\mathbb{S}^-\|_k$  (kinetic norm $\mu$)",
    ylabel: str = r"$\alpha^*$  (amplification factor)",
    figsize:           Tuple[float, float] = (9.0, 6.0),
    ax:                Optional[plt.Axes]  = None,
    style:             Optional[dict]   = None,
    # --- Salida ---
    save_path:         Optional[str]    = None,
    dpi:               int              = 150,
) -> plt.Axes:
    """
    Dibuja la frontera de Pareto (μ, α*) generada por computeParametricSweep.

    Parámetros
    ----------
    results         : lista de dicts devuelta por computeParametricSweep.
    best            : dict del punto óptimo (puede ser None si la red es infactible).
    show_obj_curve  : superponer la curva (α*-1)·μ en un eje Y secundario.
    show_infeasible : marcar con líneas verticales grises los μⱼ infactibles.
    show_frontier   : dibujar la línea escalonada que une los puntos factibles.
    label_best      : anotar el punto óptimo con su valor de obj.
    alpha_min       : ocultar puntos con α* < alpha_min (p.ej. 1.0 para mostrar
                      solo subredes auto-amplificantes).
    title           : título del gráfico.
    xlabel/ylabel   : etiquetas de los ejes.
    figsize         : tamaño de la figura en pulgadas.
    ax              : eje matplotlib existente (si None, se crea uno nuevo).
    style           : dict de overrides sobre _STYLE.
    save_path       : si se indica, guarda la figura en esa ruta.
    dpi             : resolución de guardado.

    Devuelve
    --------
    ax : el eje principal.
    """
    s = {**_STYLE, **(style or {})}

    # ---- Separar puntos factibles / infactibles ----
    feas   = [r for r in results if r["feasible"]
              and r["alpha_star"] >= alpha_min]
    infeas = [r for r in results if not r["feasible"]]

    # ---- Crear figura si no se pasó eje ----
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ---- Líneas verticales de infactibles (opcional, muy sutiles) ----
    if show_infeasible and infeas:
        for r in infeas:
            ax.axvline(r["mu"], color=s["color_infeasible"],
                       linewidth=0.5, zorder=0)

    # ---- Línea escalonada de la frontera ----
    if show_frontier and feas:
        xs = [r["mu"]         for r in feas]
        ys = [r["alpha_star"] for r in feas]
        ax.step(xs, ys, where="mid",
                color=s["color_frontier"], linewidth=s["lw_frontier"],
                alpha=0.5, zorder=2, linestyle="--")

    # ---- Scatter de puntos factibles ----
    if feas:
        xs = np.array([r["mu"]         for r in feas])
        ys = np.array([r["alpha_star"] for r in feas])
        sc = ax.scatter(xs, ys,
                        s=s["ms_feasible"], zorder=3,
                        color=s["color_feasible"], alpha=s["alpha_scatter"],
                        edgecolors="white", linewidths=0.5,
                        label=r"feasible $P(\mu_j)$")
    else:
        ax.text(0.5, 0.5, "No feasible subnetworks",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=13, color="gray")

    # ---- Punto óptimo ----
    if best is not None and best["alpha_star"] >= alpha_min:
        ax.scatter(best["mu"], best["alpha_star"],
                   s=s["ms_best"], zorder=5,
                   color=s["color_best"], marker="*",
                   edgecolors="white", linewidths=0.6,
                   label=rf"Optimum  $\mu^*$={best['mu']:.3g}, "
                   rf"$\alpha^*$={best['alpha_star']:.3g}")
        if label_best:
            obj_val = best["obj"]
            ax.annotate(
                rf"obj = {obj_val:.4g}",
                xy=(best["mu"], best["alpha_star"]),
                xytext=(-60, -50), textcoords="offset points",
                fontsize=11, color=s["color_best"],
                arrowprops=dict(arrowstyle="-", color=s["color_best"],
                                lw=0.8, alpha=0.6),
            )

    # ---- Curva obj = (α*-1)·μ en eje secundario ----
    ax2 = None
    if show_obj_curve and feas:
        ax2 = ax.twinx()
        xs_obj = np.array([r["mu"]  for r in feas])
        ys_obj = np.array([r["obj"] for r in feas])
        ax2.plot(xs_obj, ys_obj,
                 color=s["color_obj"], linewidth=s["lw_obj"],
                 alpha=s["alpha_obj"], zorder=2,
                 label=r"objective $=(\alpha^{-1})\cdot\mu$")
        ax2.set_ylabel(r"$(\alpha^{-1})\cdot\|\mathbb{S}^-\|_k$  (objetive)",
               color=s["color_obj"], fontsize=14)
        ax2.tick_params(axis="y", labelcolor=s["color_obj"], labelsize=2)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        # Marcar el máximo en el eje secundario
        if best is not None and best["alpha_star"] >= alpha_min:
            ax2.axhline(best["obj"], color=s["color_obj"],
                        linewidth=0.8, linestyle=":", alpha=0.5)

    # ---- Decoración del eje principal ----
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    if title:
        ax.set_title(title, fontsize=15, pad=12)

    # Línea α*=1 (umbral de auto-amplificación)
    ax.axhline(1.0, color="#888888", linewidth=0.9,
               linestyle=":", alpha=0.6, zorder=1,
               label=r"$\alpha^*=1$  (threshold)")

    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

    # Leyenda combinada si hay eje secundario
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels  += l2
    ax.legend(handles, labels, fontsize=12, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc")

    # Márgenes
    if feas:
        x_vals = [r["mu"] for r in feas]
        pad_x  = 0.05 * (max(x_vals) - min(x_vals) + 1e-9)
        ax.set_xlim(min(x_vals) - pad_x, max(x_vals) + pad_x)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return ax







