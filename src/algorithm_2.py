# -*- coding: utf-8 -*-
"""
algorithm_2.py
==============
MAF-only optimization framework (gonzalez2024maf): identifies the
self-amplifying subhypergraph with maximum amplification factor, used
throughout the paper as the baseline compared against the proposed
kinetic--stoichiometric growth-bound criterion (algorithm_3.py).
"""

# import networkx as nx
import numpy as np
import gurobipy as gb
import time
from typing import Any
import auxiliary_functions as aux
import math
import os

# =============================================================================
def computeSubhypergraphWithGreaterMAF(
        input_matrix: Any,
        output_matrix: Any,
        scenario_name: str,
        time_limit_iteration: int,
        max_steps = 1000,
        name_nodes = "",
        accuracy = 1e-6
    ) -> None:
    """
    Calculate the amplification factor of subgraphs and log the results to a file.

    Args:
        input_matrix: The input incidence matrix of the hypergraph.
        output_matrix: The output incidence matrix of the hypergraph.
        time_limit_iteration: Time limit for each algorithm iteration.
        max_steps: The maximum number of iterations.
        scenario_name: Name for the output file.
        accuracy: Precision of the solution
    """
        
    # Give name to the nodes
    if name_nodes == "":
        name_nodes = ["s" + str(idx) for idx in range(input_matrix.shape[0])]

    
    solution = computeMAFinSubhypergraph(
                output_matrix, input_matrix, max_steps, time_limit_iteration, accuracy
            )

    
    # Find the solution with the MAF (alpha)
    best_solution = {
        "x": solution[0],
        "alpha": solution[1],
        "step": solution[2],
        "alphaDict": solution[3],
        "y": solution[4],
        "z": solution[5],
        "time": solution[6],
        "nodes": range(output_matrix.shape[0]),
        "arcs": range(output_matrix.shape[1]),
        "dict_sol": solution[7],
    }

    # Prepare output data
    output_lines = []
    output_lines.append(f"Complete solution:\n{best_solution['dict_sol']}\n")
    output_lines.append(f"Total time:\n{best_solution['time']}\n")
    output_lines.append(f"Q dimension:\n{input_matrix.shape}\n")
    output_lines.append(f"self-sufficiency Q:\n{aux.checkSelfSufficiently(input_matrix, output_matrix)[2]}\n")
    output_lines.append(f"Number of self-amplifying nodes (y):\n{len(best_solution['y'])}\n")
    output_lines.append(
        f"Self-amplifying nodes (y):\n{' '.join([str(name_nodes[best_solution['nodes'][i]]) for i in best_solution['y']])}\n"
        )
    # name_nodes
    output_lines.append(f"nodes (numbers):\n{[best_solution['nodes'][i] for i in best_solution['y']]}\n")
    output_lines.append(f"arcs (numbers):\n{[best_solution['arcs'][z] for z in best_solution['z']]}\n")
    output_lines.append(f"MAF:\n{best_solution['alpha']}\n")
    
    # Record the resulting arcs
    selected_arcs = [best_solution["arcs"][z] for z in best_solution["z"]]
    output_ssp = output_matrix[:, selected_arcs]
    output_ssm = input_matrix[:, selected_arcs]
    
    # Map the rows and columns for the final output
    new_column_mapping = {i: selected_arcs[i] for i in range(len(selected_arcs))}
    alter_column_mapping = {i: best_solution["z"][i] for i in range(len(best_solution["z"]))}
    row_mapping = {i: i for i in range(output_ssm.shape[0])}
        
    if name_nodes == "":
        arc_details = aux.recordArcsAltered(output_ssp, output_ssm, best_solution["x"], row_mapping, new_column_mapping, alter_column_mapping)
    else:
        arc_details = aux.recordArcsAlteredWithNames(output_ssp, output_ssm, best_solution["x"], row_mapping, new_column_mapping, alter_column_mapping, name_nodes)

    output_lines.append("Q_result arcs:\n")
    output_lines.extend(arc_details)
    
    # Write output to a file
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _output_dir = os.path.join(_repo_root, "output")
    os.makedirs(_output_dir, exist_ok=True)
    output_file = os.path.join(_output_dir, f"{scenario_name}_algorithm_2.txt")
    try:
        with open(output_file, 'w') as f:
            f.writelines(output_lines)
    except Exception as e:
        raise RuntimeError(f"Error writing to file {output_file}: {e}")
        
    return best_solution, arc_details
# =============================================================================


# =============================================================================
def computeMAFinSubhypergraph(output_matrix, input_matrix, t_max, time_limit_iteration, accuracy = 1e-6):
    """
    Calculate the MAF for a subhypergraph.
    
    Parameters:
    - output_matrix (np.ndarray): The output incidence matrix of the hypergraph.
    - input_matrix (np.ndarray): The input incidence matrix of the hypergraph.
    - t_max (int): Maximum number of iterations for convergence.
    - time_limit_iteration (float): Time limit for each optimization step in seconds.
    - accuracy (float, optional): precision solution
    
    Returns:
    - Tuple[np.ndarray, float, int, dict, list, list, float, dict]: 
        (optimal_intensity, final_amplification_factor, steps, amplification_factors, active_nodes, active_arcs, total_time, iteration_data)
    """
    # Parameters
    # --------------------------- 
    # Number Nodes (int) and Number Arcs (int)
    num_nodes, num_arcs = output_matrix.shape
    # Nodes (list)
    nodes = range(num_nodes)
    # Arcs (list)
    arcs = range(num_arcs)
    # Alpha_0 (float)
    x_0 = np.ones(num_arcs)
    # Vectorized computation of initial alpha
    initial_alpha_vector = np.sum(output_matrix * x_0, axis=1) / np.maximum(np.sum(input_matrix * x_0, axis=1), 1e-15)
    alpha_0 = np.min(initial_alpha_vector)
    # --------------------------------------
    # Parameters

    def computeIntensityUpperBound(input_matrix, output_matrix):
        return float(np.sum(input_matrix) + np.sum(output_matrix))
    upper_bound = computeIntensityUpperBound(input_matrix, output_matrix)
    
    if alpha_0 == 0:
        big_M = sum(sum(input_matrix[v, :]) for v in nodes) * upper_bound
    else:
        big_M = math.ceil(alpha_0)*sum(sum(input_matrix[v, :]) for v in nodes) * upper_bound
    # -------------------------------------------------------------------------
    def computeMAFinSubhypergraphFixed(alpha_0, time_limit_iteration):

        # Initialize model
        model = gb.Model("Amplification_Factor_Model_SN")
        model.Params.OutputFlag = 0  # Suppress console output
        model.Params.TimeLimit = time_limit_iteration
        model.Params.MIPGap = 0.00  # Set to zero for exact solutions, might be too strict for practical use
        model.Params.OptimalityTol = 1e-9
    
        # Define variables
        x = model.addVars(num_arcs, lb=0, ub=upper_bound, name="arc_intensities")
        alpha = model.addVar(name = "amplification_factor")
        y = model.addVars(num_nodes, vtype = gb.GRB.BINARY, name = "is_self_amplifying")

        # Objective function
        model.setObjective(alpha, gb.GRB.MAXIMIZE)
        # Constraints
        # --------------------------------------
        model.addConstrs(
            (
            alpha <= gb.quicksum(output_matrix[v, a] * x[a] 
                                  for a in arcs)
                    - alpha_0 * gb.quicksum(input_matrix[v, a] * x[a] 
                                          for a in arcs)
                    + big_M * (1 - y[v])
                    for v in nodes), name = "definition")
        # 
        model.addConstrs(
            (y[v] <= gb.quicksum(x[a] for a in arcs if output_matrix[v, a] > 0)
             for v in nodes), name = "self_sufficient_arc_target")
        # 
        model.addConstrs(
            (y[v] <= gb.quicksum(x[a] for a in arcs if input_matrix[v, a] > 0)
             for v in nodes), name = "self_sufficient_arc_source")
        #
        model.addConstrs(
            (x[a] <= upper_bound * gb.quicksum(y[v] for v in nodes if output_matrix[v, a] > 0)
             for a in arcs), name = "self_sufficient_node_target")
        # 
        model.addConstrs(
            (x[a] <= upper_bound * gb.quicksum(y[v] for v in nodes if input_matrix[v, a] > 0)
             for a in arcs), name = "self_sufficient_node_source")
        #   
        model.addConstr(gb.quicksum(y[v] for v in nodes) >= 1, 
                    name = "subset_nodes_not_empty")
        #
        model.addConstr(gb.quicksum(x[a] for a in arcs) >= 1,
                    name = "at_least_one_hyperarc")
        # Autonomía: todo nodo incidente a un arco con flujo positivo pertenece a M
        model.addConstrs(
            (x[a] <= upper_bound * y[v]
             for a in arcs for v in nodes
             if input_matrix[v, a] > 0 or output_matrix[v, a] > 0),
            name="autonomy")
        # --------------------------------------      
  
        # Solve the model
        model.optimize()
        
    
        if model.status != gb.GRB.OPTIMAL:
            model.computeIIS()
            model.write("infeasibility_report.ILP")
            print("Model infeasible. Check infeasibility report.")
            raise ValueError("Optimization failed: model is infeasible")
    
        # Extract solution
        optimal_intensities = np.array([x[r].X for r in range(num_arcs)])
        
        optimal_amplification_factor = alpha.X
        active_nodes = [v for v in range(num_nodes) if y[v].X > 0.5]
        active_arcs = [a for a in range(num_arcs) if x[a].X > 0.5]
    
        return optimal_intensities, optimal_amplification_factor, active_nodes, active_arcs, model.MIPGap, model.NumVars, model.NumConstrs
    # -------------------------------------------------------------------------
        
    stop = False
    step = 0
    alpha_dict = {}
    current_alpha = alpha_0

    previous_alpha = 0  # Starting with 0 for comparison
    start_time=time.time()
    
    iteration_data = {}
    
    cumulative_time = 0
    while not stop:
        iteration_start_time = time.time()

        # Solve optimization model for this iteration
        optimal_intensities, alpha_bar, active_nodes, active_arcs, gap, num_vars, num_constrs = computeMAFinSubhypergraphFixed(current_alpha, time_limit_iteration)
        
        # Update current alpha considering only active nodes
        current_alpha = np.min([sum(output_matrix[v, a] * optimal_intensities[a]
                        for a in arcs)
                        /
                        sum(input_matrix[v, a] * optimal_intensities[a] 
                        for a in arcs) 
                    for v in active_nodes])

        
        iteration_end_time = time.time()
        
        it_time = iteration_end_time - iteration_start_time
        cumulative_time = cumulative_time + it_time
        
        iteration_data[step] = {
            "intensities": optimal_intensities,
            "active_nodes": active_nodes,
            "active_arcs": active_arcs,
            "alpha_bar": alpha_bar,
            "gap": gap,
            "variables": num_vars,
            "constraints": num_constrs,
            "step": step,
            "alpha": current_alpha,
            "time": it_time
        }
        
        print("step:", step + 1, "alpha:", round(current_alpha, 3), "it_time:", round(it_time, 3), "total_time:", round(cumulative_time, 3))


        if (len(active_nodes) < 1 or 
            np.abs(alpha_bar) < accuracy or
            step >= t_max):
            stop = True
            alpha_dict[step] = current_alpha
            return optimal_intensities, current_alpha, step, alpha_dict, active_nodes, active_arcs, time.time() - start_time, iteration_data
        else:
            alpha_dict[step] = current_alpha
            previous_alpha = current_alpha
            step += 1
# =============================================================================

