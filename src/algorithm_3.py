# -*- coding: utf-8 -*-
"""
Barrido paramétrico sobre la frontera de Pareto (alpha*, ||S^-||_k),
con GENERACIÓN DESCENDENTE de mu y CRIBADO POR INCUMBENTE (modelo de Víctor).

Idea (Algoritmo 3 del paper):
  * Se recorren los valores de la norma cinética mu en orden DESCENDENTE.
  * Se mantiene el mejor objetivo hallado, LB = max (alpha*-1)*mu.
  * Antes de resolver P(mu) se aplica un test de cribado con una cota superior
    válida alpha_glob de alpha* (aquí, el MAF GLOBAL, constante):
        si (alpha_glob - 1)*mu <= LB  =>  P(mu) no puede mejorar el incumbente.
    Como (alpha_glob-1)*mu es CRECIENTE en mu, en orden descendente ese test,
    una vez satisfecho, lo cumplen todos los mu menores: se convierte en una
    REGLA DE PARADA exacta (break).  El barrido sigue siendo globalmente óptimo
    siempre que la malla cubra los valores alcanzables y alpha_glob sea válida.

Notas de exactitud:
  * alpha_glob (MAF global) es una cota superior válida de P(mu) para todo mu
    SOLO si está CERTIFICADO (todos los MILP probados a optimalidad y salida
    del bucle por rho*~0). Si no lo está, es una cota INFERIOR y usarla podaría
    el óptimo verdadero: por eso, si no se certifica, se DESACTIVA el cribado
    y se hace barrido completo (más lento, pero sigue siendo exacto).
    Como el MAF global se resuelve UNA sola vez por instancia, conviene darle
    un límite de tiempo propio y generoso (time_limit_global; None = sin
    límite), independiente del de los subproblemas (time_limit_iteration).
  * (detalle) alpha_glob (MAF global) es cota superior de P(mu) para todo mu,
    porque P(mu) = max{alpha* : norma == mu} <= max{alpha* : cualquier subred}.
    Se calcula una vez con la misma maquinaria fraccional, SIN restricción de
    norma (enforce_norm_eq=False).  Si no se prueba a optimalidad, se DESACTIVA
    el cribado y se hace barrido completo (que sigue siendo exacto).
  * El cribado sólo descarta mu PROVADAMENTE dominados; no altera el óptimo.

Semántica de la norma: se mide SOLO sobre reacciones con flujo positivo
(z_a = 1 <=> x_a >= eps), coherente con Z1+Z2 y con la Definición del MAF.

Limitación conocida (malla de paso fijo): con `decimals` finitos, la norma real
puede no caer sobre un múltiplo de la malla (p.ej. 2.016 con decimals=2 -> 2.02),
de modo que 'obj' en unidades de malla puede diferir del exacto. Para cifras
exactas, recalcula norma y cota con las k reales sobre el soporte recuperado.

Para MAPEAR la frontera de Pareto completa (todos los mu alcanzables), usa
screening=False: el cribado, por diseño, se salta subredes dominadas.
"""

import numpy as np
import gurobipy as gb
import time
import warnings
from math import gcd
from fractions import Fraction


# ----------------------------------------------------------------------
def toIntegerK(k, decimals=4):
    """
    Convierte k (racional) a entero multiplicando por el m.c.m. de los
    denominadores tras redondear a `decimals` decimales.
    Devuelve (k_int, scale) con k_int = k * scale (entero) y scale = m.c.m.
    """
    fracs = [Fraction(round(float(ki), decimals)).limit_denominator(10 ** decimals)
             for ki in k]
    lcm_den = fracs[0].denominator
    for f in fracs[1:]:
        lcm_den = lcm_den * f.denominator // gcd(lcm_den, f.denominator)
    k_int = np.array([int(f * lcm_den) for f in fracs], dtype=np.int64)
    return k_int, lcm_den


# ----------------------------------------------------------------------
def computeMuGrid(input_matrix, k_int, max_points=None):
    """
    Malla de valores de la norma cinética en unidades ESCALADAS (k_int entero).
    Como k_int es entero y las multiplicidades son enteras, todos los valores
    alcanzables de ||S^-||_k son enteros, así que step=1 los recorre todos.

    Devuelve (grid, exact):
      grid  : lista de enteros mu_j (ASCENDENTE)
      exact : True si step=1 (recorre todos los enteros del rango); False si se
              submuestreó por superar max_points (entonces el barrido NO es exacto).
    """
    num_nodes, num_arcs = input_matrix.shape

    # mu_min: menor norma posible (una sola reacción activa)
    mu_min = min(
        max(int(round(input_matrix[s, a] * k_int[a])) for s in range(num_nodes))
        for a in range(num_arcs)
        if any(input_matrix[s, a] > 0 for s in range(num_nodes))
    )
    # mu_max: mayor norma posible (todas las reacciones activas)
    mu_max = max(
        sum(int(round(input_matrix[s, a] * k_int[a])) for a in range(num_arcs))
        for s in range(num_nodes)
    )

    total = mu_max - mu_min + 1
    if max_points is None or total <= max_points:
        grid = list(range(mu_min, mu_max + 1))           # step = 1 -> exacto
        exact = True
    else:
        step = max(1, total // max_points)
        grid = list(range(mu_min, mu_max + 1, step))
        if grid[-1] != mu_max:
            grid.append(mu_max)
        exact = False
        warnings.warn(
            f"Rango de mu ({total} enteros) > max_points ({max_points}); "
            f"se usa step={step}. El barrido NO es exacto: puede saltarse "
            f"valores alcanzables. Sube max_points o reduce 'decimals' en toIntegerK."
        )
    return grid, exact


# ----------------------------------------------------------------------
def _milp_fixed(output_matrix, input_matrix, k_int, mu_j,
                alpha_prev, upper_bound, time_limit, enforce_norm_eq=True):
    """
    Un MILP de Dinkelbach con alpha_prev fijo.
      enforce_norm_eq=True : norma == mu_j (SUP + INF)  -> subproblema P(mu_j).
      enforce_norm_eq=False: norma <= mu_j (solo SUP)   -> con mu_j holgado,
                             equivale a MAF GLOBAL (sin restricción de norma).
    Devuelve (status, data) donde:
      status in {'optimal', 'timelimit', 'infeasible', 'other'}
      data   = (opt_x, opt_rho, act_nodes, act_arcs, gap)  o  None si no hay incumbente.
    """
    num_nodes, num_arcs = output_matrix.shape
    nodes = range(num_nodes)
    arcs = range(num_arcs)

    # big-M ajustado al rango real de (T x)_v - alpha_prev (S x)_v sobre x in [0, U]
    maxT = max(float(np.sum(output_matrix[v, :])) for v in nodes)
    maxS = max(float(np.sum(input_matrix[v, :])) for v in nodes)
    big_M = upper_bound * (maxT + alpha_prev * maxS) + 1.0

    m = gb.Model("P_mu")
    m.Params.OutputFlag = 0
    if time_limit is not None:          # None => sin límite de tiempo
        m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 1e-6            # ya no 0.0
    # OptimalityTol se deja en su valor por defecto (1e-6); 1e-9 daba problemas
    # numéricos con la big-M.

    x = m.addVars(num_arcs, lb=0, ub=upper_bound, name="x")
    rho = m.addVar(lb=-gb.GRB.INFINITY, name="rho")   # margen maximin (libre)
    y = m.addVars(num_nodes, vtype=gb.GRB.BINARY, name="y")
    z = m.addVars(num_arcs, vtype=gb.GRB.BINARY, name="z")
    w = m.addVars(num_nodes, vtype=gb.GRB.BINARY, name="w")

    m.setObjective(rho, gb.GRB.MAXIMIZE)

    # --- Dinkelbach con big-M (linealización correcta del auxiliar de Alg. 2) ---
    m.addConstrs(
        (rho <= gb.quicksum(output_matrix[v, a] * x[a] for a in arcs)
              - alpha_prev * gb.quicksum(input_matrix[v, a] * x[a] for a in arcs)
              + big_M * (1 - y[v])
         for v in nodes), name="dinkelbach")

    # --- Autosuficiencia / no vacuidad (eq. 2-6) ---
    m.addConstrs(
        (y[v] <= gb.quicksum(x[a] for a in arcs if output_matrix[v, a] > 0)
         for v in nodes), name="ss_target")
    m.addConstrs(
        (y[v] <= gb.quicksum(x[a] for a in arcs if input_matrix[v, a] > 0)
         for v in nodes), name="ss_source")
    m.addConstrs(
        (x[a] <= upper_bound * gb.quicksum(y[v] for v in nodes if output_matrix[v, a] > 0)
         for a in arcs), name="ss_node_target")
    m.addConstrs(
        (x[a] <= upper_bound * gb.quicksum(y[v] for v in nodes if input_matrix[v, a] > 0)
         for a in arcs), name="ss_node_source")
    m.addConstr(gb.quicksum(y[v] for v in nodes) >= 1, name="nonempty_M")
    m.addConstr(gb.quicksum(x[a] for a in arcs) >= 1, name="nonempty_x")

    # --- z = indicador de flujo positivo (Z1, Z2) ---
    m.addConstrs((x[a] <= upper_bound * z[a] for a in arcs), name="Z1")
    eps = 1.0 / upper_bound
    m.addConstrs((x[a] >= eps * z[a] for a in arcs), name="Z2")
    
    # --- Autonomía (eq:P-autonomy): todo nodo incidente a un arco
    #     seleccionado pertenece a M  =>  M = N' ---
    m.addConstrs(
        (y[v] >= z[a]
         for a in arcs for v in nodes
         if input_matrix[v, a] > 0 or output_matrix[v, a] > 0),
        name="autonomy")    

    # --- SUP: ||S^-||_k <= mu_j  (con enforce_norm_eq=False y mu_j holgado
    #          esta cota es inactiva => MAF global) ---
    m.addConstrs(
        (gb.quicksum(int(round(input_matrix[s, a])) * int(k_int[a]) * z[a] for a in arcs) <= mu_j
         for s in nodes), name="SUP")

    # --- INF: al menos una especie alcanza mu_j (=> norma == mu_j) ---
    #     Solo cuando se fuerza la igualdad de norma (subproblema P(mu_j)).
    if enforce_norm_eq:
        m.addConstr(gb.quicksum(w[s] for s in nodes) >= 1, name="INF_sum")
        m.addConstrs(
            (gb.quicksum(int(round(input_matrix[s, a])) * int(k_int[a]) * z[a] for a in arcs) >= mu_j * w[s]
             for s in nodes), name="INF")

    m.optimize()
    st = m.Status

    if st in (gb.GRB.INFEASIBLE, gb.GRB.INF_OR_UNBD):
        return "infeasible", None
    if m.SolCount == 0:
        # sin incumbente (p.ej. TIME_LIMIT sin solución, o NUMERIC)
        return ("timelimit" if st == gb.GRB.TIME_LIMIT else "other"), None

    # hay incumbente
    opt_x = np.array([x[a].X for a in arcs])
    opt_rho = rho.X
    act_nodes = [v for v in nodes if y[v].X > 0.5]
    act_arcs = [a for a in arcs if z[a].X > 0.5]
    status = ("optimal" if st == gb.GRB.OPTIMAL
              else "timelimit" if st == gb.GRB.TIME_LIMIT
              else "suboptimal")
    return status, (opt_x, opt_rho, act_nodes, act_arcs, m.MIPGap)


# ----------------------------------------------------------------------
def computeCineticInSubhypergraph(output_matrix, input_matrix, k_int, mu_j,
                                  t_max, time_limit_iteration, accuracy=1e-6,
                                  enforce_norm_eq=True):
    """
    Dinkelbach para hallar max alpha* sobre subredes con:
      enforce_norm_eq=True : ||S^-||_k == mu_j   (subproblema P(mu_j))
      enforce_norm_eq=False: sin restricción de norma (mu_j holgado) -> MAF global.

    Devuelve (status, alpha, info):
      status in {'optimal', 'timelimit', 'infeasible', 'other'}
      alpha  : alpha* (None si no hubo solución)
      info   : dict con datos de la solución (None si no hubo solución)
    """
    num_nodes, num_arcs = output_matrix.shape
    arcs = range(num_arcs)
    upper_bound = float(np.sum(input_matrix) + np.sum(output_matrix))

    current_alpha = 1.0
    prev_alpha = 0.0
    stall = 0
    step = 0
    timed_out = False
    last = None
    last_rho = None
    rho_converged = False
    alpha_stable = False

    while True:
        st, data = _milp_fixed(output_matrix, input_matrix, k_int, mu_j,
                               current_alpha, upper_bound, time_limit_iteration,
                               enforce_norm_eq=enforce_norm_eq)

        if data is None:
            if step == 0:
                # En la primera iteración: si no hay subred admisible, es
                # infactible; si fue timeout sin incumbente, 'other'/'timelimit'.
                return st, None, None
            else:
                # No debería ocurrir (la factibilidad no depende de alpha), pero
                # por robustez: cerramos con el último alpha bueno.
                break

        opt_x, opt_rho, act_nodes, act_arcs, gap = data
        last = (opt_x, act_nodes, act_arcs, gap)
        last_rho = opt_rho
        rho_converged = abs(opt_rho) < accuracy
        if st != "optimal":
            timed_out = True

        if act_nodes:
            current_alpha = float(np.min([
                sum(output_matrix[v, a] * opt_x[a] for a in arcs) /
                sum(input_matrix[v, a] * opt_x[a] for a in arcs)
                for v in act_nodes
            ]))

        stall = stall + 1 if abs(current_alpha - prev_alpha) < accuracy else 0
        prev_alpha = current_alpha
        alpha_stable = stall >= 3

        converged = (abs(opt_rho) < accuracy or alpha_stable
                     or step >= t_max or len(act_nodes) < 1)
        if converged:
            break
        step += 1

    if last is None:
        return "other", None, None

    opt_x, act_nodes, act_arcs, gap = last
    # Certificacion: current_alpha es demostrablemente el MAXIMO (y por tanto
    # una cota superior VALIDA) si ningun MILP agoto tiempo Y el bucle alcanzo
    # un punto fijo, sea por rho*~0 o por alpha estable. Justificacion del
    # segundo caso: si rho*(a)>0 el flujo optimo cumple (Tf)_v > a (Sf)_v en
    # todo nodo seleccionado, luego el nuevo alpha = min_v (Tf)_v/(Sf)_v > a;
    # por contrarreciproco, si alpha deja de crecer entonces rho*(a)<=0. Como
    # alpha siempre es un ratio ALCANZADO (alpha <= alpha*), se tiene rho*(a)>=0,
    # luego rho*(a)=0 y alpha=alpha*. En cambio, salir por agotar t_max SIN punto
    # fijo deja alpha como mera cota INFERIOR: ahi NO se certifica.
    certified = (not timed_out) and (rho_converged or alpha_stable)
    info = {"opt_x": opt_x, "act_nodes": act_nodes, "act_arcs": act_arcs,
            "gap": gap, "steps": step, "rho": last_rho,
            "certified": certified}
    return ("timelimit" if timed_out else "optimal"), current_alpha, info


# ----------------------------------------------------------------------
def computeGlobalMAF(output_matrix, input_matrix, k_int, mu_max_int,
                     t_max, time_limit_global=None, accuracy=1e-6):
    """
    MAF global alpha*_glob = max_{subred auto-amplificante} alpha*  (sin restringir
    la norma).  Cota superior VÁLIDA y constante de P(mu) para todo mu.

    Se usa una cota de norma holgada (>= cualquier norma alcanzable) para que la
    restricción SUP quede inactiva; alpha* no depende de k, así que el valor es
    independiente de la escala.
    """
    num_nodes, num_arcs = output_matrix.shape
    # Cota holgada: suma de TODAS las consumiciones (>= norma de cualquier subred).
    mu_slack = int(sum(int(round(input_matrix[s, a])) * int(k_int[a])
                       for s in range(num_nodes) for a in range(num_arcs)))
    mu_slack = max(mu_slack, int(mu_max_int))
    return computeCineticInSubhypergraph(
        output_matrix, input_matrix, k_int, mu_slack,
        t_max, time_limit_global, accuracy, enforce_norm_eq=False)







def _next_attainable_norm(output_matrix, input_matrix, k_int, mu_upper,
                          time_limit=60):
    """
    Mayor norma cinetica ENTERA mu <= mu_upper alcanzable por alguna subred
    admisible (mismas condiciones estructurales que P(mu): autosuficiencia,
    autonomia, no vacuidad). Devuelve (mu, certificado):
      - (mu, True)          : mu es la mayor norma alcanzable <= mu_upper.
      - (None, True)        : no hay ninguna norma alcanzable <= mu_upper.
      - (mu_upper, False)   : el MILP no se probo optimo en el tiempo dado;
                              se devuelve mu_upper para no saltarse ninguna
                              norma alcanzable (el barrido sigue siendo exacto).
    """
    num_nodes, num_arcs = input_matrix.shape
    nodes, arcs = range(num_nodes), range(num_arcs)
    if mu_upper < 1:
        return None, True
    m = gb.Model("next_norm")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    y = m.addVars(num_nodes, vtype=gb.GRB.BINARY, name="y")
    z = m.addVars(num_arcs, vtype=gb.GRB.BINARY, name="z")
    w = m.addVars(num_nodes, vtype=gb.GRB.BINARY, name="w")
    mu = m.addVar(lb=0, ub=mu_upper, vtype=gb.GRB.INTEGER, name="mu")
    m.setObjective(mu, gb.GRB.MAXIMIZE)

    # Autosuficiencia expresada sobre z (equivalente a la de P(mu), donde
    # x_a > 0 si y solo si z_a = 1)
    m.addConstrs((y[v] <= gb.quicksum(z[a] for a in arcs if output_matrix[v, a] > 0)
                  for v in nodes), name="ss_target")
    m.addConstrs((y[v] <= gb.quicksum(z[a] for a in arcs if input_matrix[v, a] > 0)
                  for v in nodes), name="ss_source")
    m.addConstrs((z[a] <= gb.quicksum(y[v] for v in nodes if output_matrix[v, a] > 0)
                  for a in arcs), name="ss_node_target")
    m.addConstrs((z[a] <= gb.quicksum(y[v] for v in nodes if input_matrix[v, a] > 0)
                  for a in arcs), name="ss_node_source")
    m.addConstrs((y[v] >= z[a] for a in arcs for v in nodes
                  if input_matrix[v, a] > 0 or output_matrix[v, a] > 0),
                 name="autonomy")
    m.addConstr(gb.quicksum(y[v] for v in nodes) >= 1, name="nonempty_M")
    m.addConstr(gb.quicksum(z[a] for a in arcs) >= 1, name="nonempty_z")

    # Norma cinetica: L_s <= mu para todo s, y L_s >= mu para alguna s (w_s = 1)
    L = {s: gb.quicksum(int(round(input_matrix[s, a])) * int(k_int[a]) * z[a]
                        for a in arcs) for s in nodes}
    m.addConstrs((L[s] <= mu for s in nodes), name="SUP")
    m.addConstr(gb.quicksum(w[s] for s in nodes) >= 1, name="INF_sum")
    m.addConstrs((L[s] >= mu - mu_upper * (1 - w[s]) for s in nodes), name="INF")

    m.optimize()
    if m.Status == gb.GRB.INFEASIBLE:
        return None, True
    if m.Status == gb.GRB.OPTIMAL:
        return int(round(mu.X)), True
    return int(mu_upper), False
























# ----------------------------------------------------------------------
def computeParametricSweep(input_matrix, output_matrix, k,
                           decimals=4, max_points=500,
                           time_limit_iteration=300, max_steps=1000,
                           accuracy=1e-6, screening=True,
                           time_limit_global=None):
    """
    Barrido paramétrico sobre la frontera de Pareto alpha* vs ||S^-||_k.

    screening=True  (modelo de Víctor): recorre mu en orden DESCENDENTE y usa el
        cribado por incumbente con cota constante alpha_glob (regla de parada
        exacta).  Resuelve muchos menos subproblemas.
    screening=False (barrido completo, ASCENDENTE): resuelve todos los P(mu_j);
        útil para MAPEAR la frontera de Pareto entera.

    Devuelve (best, results, exact):
      best    : dict del mejor subhipergrafo (max (alpha*-1)*mu), o None si todo infactible.
      results : lista (ASCENDENTE en mu) con un dict por mu_j VISITADO.
      exact   : True si la malla recorrió todos los valores alcanzables (paso 1).

    time_limit_global: límite (s) SOLO para el MAF global que usa el cribado.
        None (por defecto) = sin límite. Se resuelve una vez por instancia; si
        no se certifica, el cribado se desactiva y el barrido es completo.
        time_limit_iteration sigue aplicándose a los subproblemas P(mu).

    Nota: 'mu' y 'obj' se reportan en unidades REALES (ya des-escaladas).
    Con screening=True, `results` puede no contener todos los mu alcanzables
    (los dominados se podan); `best` sí es el óptimo global (malla exacta + cota válida).
    """
    k_int, scale = toIntegerK(k, decimals)
    computeParametricSweep.last_n_norm_milps = 0
    mu_grid, exact = computeMuGrid(input_matrix, k_int, max_points)
    if not mu_grid:
        warnings.warn("Malla de mu vacía: revisa los datos de entrada.")
        return None, [], exact

    # --- Cota superior constante para el cribado: MAF global -------------------
    alpha_glob = None
    if screening:
        st_g, alpha_glob, info_g = computeGlobalMAF(
            output_matrix, input_matrix, k_int, mu_grid[-1],
            max_steps, time_limit_global, accuracy)
        # El cribado SOLO es valido si alpha_glob esta CERTIFICADO como maximo.
        # Si el bucle no probo rho*~0 (o algun MILP agoto tiempo), alpha_glob es
        # una cota INFERIOR del MAF global y usarla como cota superior podria
        # podar el optimo verdadero: en ese caso se desactiva el cribado.
        if (alpha_glob is None or info_g is None
                or not info_g.get("certified", False)):
            warnings.warn(
                "MAF global no certificado; se DESACTIVA el cribado "
                "(barrido completo, sigue siendo exacto). Sube "
                "time_limit_global para intentar certificarlo.")
            screening = False
        else:
            print(f"[cribado] MAF global alpha*_glob = {alpha_glob:.6f} "
                  f"(cota superior constante)")

    # Orden de recorrido: descendente si hay cribado, ascendente si no.
    order = list(reversed(mu_grid)) if screening else list(mu_grid)

    results = {}
    LB = -np.inf
    best = None
    n_solved = n_feasible = n_infeas = n_unproven = 0
    stopped = False

    n_norm_milps = [0]

    def _mu_sequence():
        """Normas a visitar. Con cribado: solo las ALCANZABLES, en orden
        descendente, obtenidas con _next_attainable_norm. Sin cribado: la malla."""
        if not screening:
            yield from order
            return
        cur = mu_grid[-1]
        while cur >= 1:
            nxt, _ = _next_attainable_norm(output_matrix, input_matrix, k_int,
                                           cur, time_limit_iteration)
            n_norm_milps[0] += 1
            if nxt is None:
                return
            yield nxt
            cur = nxt - 1

    if screening:
        exact = True   # se enumeran todas las normas alcanzables, sin malla

    for mu_j in _mu_sequence():
        mu_real = mu_j / scale

        # --- Cribado por incumbente (cota constante, orden descendente) --------
        # (alpha_glob-1)*mu es creciente en mu: una vez que no supera a LB,
        # ningún mu menor lo hará => PARADA exacta.
        if screening and (alpha_glob - 1.0) * mu_real <= LB + 1e-9:
            print(f"[stop] mu={mu_real:.4f}: (alpha_glob-1)*mu="
                  f"{(alpha_glob - 1.0) * mu_real:.4f} <= LB={LB:.4f}; "
                  f"resto de mu podado.")
            stopped = True
            break

        st, alpha, info = computeCineticInSubhypergraph(
            output_matrix, input_matrix, k_int, mu_j,
            max_steps, time_limit_iteration, accuracy, enforce_norm_eq=True)
        n_solved += 1

        if alpha is None:
            n_infeas += 1
            results[mu_j] = {"mu": mu_real, "mu_int": mu_j, "alpha_star": None,
                             "obj": -np.inf, "feasible": False,
                             "proven": False, "status": st, "info": None}
            tag = "infactible" if st == "infeasible" else f"sin solución ({st})"
            print(f"[soln] mu={mu_real:.4f}: {tag}")
        else:
            obj = (alpha - 1.0) * mu_real
            proven = (st == "optimal")
            n_feasible += 1
            n_unproven += (not proven)
            rec = {"mu": mu_real, "mu_int": mu_j, "alpha_star": alpha,
                   "obj": obj, "feasible": True,
                   "proven": proven, "status": st, "info": info}
            results[mu_j] = rec
            if obj > LB:
                LB = obj
                best = rec
            flag = "" if proven else "  [no probado: timeout]"
            print(f"[soln] mu={mu_real:.4f}  alpha*={alpha:.4f}  obj={obj:.4f}{flag}")

    results_list = [results[mu_j] for mu_j in sorted(results)]

    if best is None:
        warnings.warn("Ningún subproblema P(mu_j) resultó factible: revisa la "
                      "malla (toIntegerK/decimals) o los datos de entrada.")
        return None, results_list, exact
    if n_unproven:
        warnings.warn(f"{n_unproven}/{n_feasible} subproblemas factibles no se "
                      f"resolvieron a optimalidad (timeout); el óptimo global "
                      f"podría no ser exacto.")
        
    computeParametricSweep.last_n_norm_milps = n_norm_milps[0]
    print(f"[resumen] subproblemas P(mu) resueltos={n_solved} "
          f"(factibles={n_feasible}, infactibles={n_infeas}); "
          f"MILP de norma resueltos={n_norm_milps[0]}; "
          f"{'PARADA por cribado' if stopped else 'recorrido completo'}; "
          f"malla exacta={exact}")
    return best, results_list, exact


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo de §3:  r1: 2A->3A (k=0.5),  r2: A->2A (k=2.0)
    input_matrix = np.array([[2.0, 1.0]])    # reactivos  (S^-)
    output_matrix = np.array([[3.0, 2.0]])   # productos  (S^+)
    k = [0.5, 2.0]
    best, results, exact = computeParametricSweep(
        input_matrix, output_matrix, k,
        decimals=4, max_points=500, time_limit_iteration=60, screening=True)
    print("\nMalla exacta:", exact)
    if best is not None:
        print(f"Mejor: mu={best['mu']:.4f}  alpha*={best['alpha_star']:.4f}  "
              f"cota={best['obj']:.4f}  arcos activos={best['info']['act_arcs']}")