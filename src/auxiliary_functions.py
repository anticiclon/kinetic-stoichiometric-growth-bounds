# -*- coding: utf-8 -*-
"""
auxiliary_functions.py
=======================
Helper routines used by algorithm_2.py: self-sufficiency checks, removal of
null rows/columns, weakly-connected-component decomposition, and formatted
arc reporting.
"""
import numpy as np
import os
from typing import Tuple
import networkx as nx

# =============================================================================
def readScenario(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read input and output matrices for a given scenario from files.

    Parameters:
    - name (str): The name of the scenario without file extensions.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the input (mMinus) and output (mPlus) matrices.

    Raises:
    - FileNotFoundError: If either file cannot be found.
    - ValueError: If matrices read do not have the same dimensions.
    """
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_path = os.path.join(_repo_root, "scenarios")
    
    # Construct file paths using os.path for better cross-platform compatibility
    minus_file = os.path.join(base_path, f"{name}_minus.txt")
    plus_file = os.path.join(base_path, f"{name}_plus.txt")

    # Use try-except for error handling
    try:
        # Read matrices with error checking for file existence
        m_minus = np.loadtxt(minus_file, dtype=int)
        m_plus = np.loadtxt(plus_file, dtype=int)
        
        # Check if matrices have the same dimension
        if m_minus.shape != m_plus.shape:
            raise ValueError(f"Matrices from {name} have different dimensions: {m_minus.shape} vs {m_plus.shape}")
        
        return m_minus, m_plus

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        raise
    except ValueError as e:
        print(f"Error reading or processing matrix data: {e}")
        raise
# =============================================================================


# =============================================================================
def checkSelfSufficiently(input_matrix, output_matrix, epsilon=1e-7):
    """Check if the network is self-sufficiently."""
    nodes_self_sufficiently = np.all((np.sum(input_matrix, axis=1) > epsilon) & (np.sum(output_matrix, axis=1) > epsilon))
    arc_self_sufficiently = np.all((np.sum(input_matrix, axis=0) > epsilon) & (np.sum(output_matrix, axis=0) > epsilon))
    return nodes_self_sufficiently, arc_self_sufficiently, nodes_self_sufficiently and arc_self_sufficiently
# =============================================================================


# =============================================================================
def removeNullRowsAndColumns(matrix1, matrix2, tolerance=1e-10):
    """
    Remove rows and columns from two matrices where all elements are below a tolerance in either matrix.

    Parameters:
    - matrix1, matrix2 (np.ndarray): Input matrices to be modified.
    - tolerance (float): Threshold below which elements are considered zero.

    Returns:
    - Tuple[np.ndarray, np.ndarray, List[int], List[int]]: 
        Modified matrices, removed row indices, removed column indices.

    Raises:
    - ValueError: If matrices do not have the same dimensions or if index tracking fails.
    """
    # Check if the matrices have the same dimensions
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same dimensions")

    # Original list of rows and columns
    original_rows = np.arange(matrix1.shape[0]) 
    original_columns = np.arange(matrix1.shape[1])

    # Initialize lists to store indices of null rows and columns
    rows_removed = []
    columns_removed = []
    
    # Initialize lists of indices of rows and columns that are kept in the solution
    rows_kept = np.arange(matrix1.shape[0]) 
    columns_kept = np.arange(matrix1.shape[1])
        
    while True:

        null_rows = []
        null_columns = []
        
        rows_to_remove = []
        columns_to_remove = []

        # Check for null rows in either matrix
        for idx, value in enumerate(rows_kept):
            if (np.all(np.abs(matrix1[idx, :]) < tolerance) 
                or np.all(np.abs(matrix2[idx, :]) < tolerance)):
                null_rows.append(value)
                rows_to_remove.append(idx)

        # Check for null columns in either matrix
        for idx, value in enumerate(columns_kept):
            if (np.all(np.abs(matrix1[:, idx]) < tolerance) 
                or np.all(np.abs(matrix2[:, idx]) < tolerance)):
                print()
                null_columns.append(value)
                columns_to_remove.append(idx)

        # If no new null rows or columns are found, exit the loop
        if not null_rows and not null_columns:
            break
        
        # Update the lists of rows and columns removed
        rows_removed.extend(null_rows)
        columns_removed.extend(null_columns)
        
        # Update the lists of rows and columns kept
        rows_kept = sorted(list(set(rows_kept) - set(null_rows)))
        columns_kept = sorted(list(set(columns_kept) - set(null_columns)))

        # Remove null rows and columns from both matrices
        matrix1 = np.delete(matrix1, rows_to_remove, axis = 0)  # Remove null rows
        matrix1 = np.delete(matrix1, columns_to_remove, axis = 1)  # Remove null columns

        matrix2 = np.delete(matrix2, rows_to_remove, axis = 0)  # Remove null rows
        matrix2 = np.delete(matrix2, columns_to_remove, axis = 1)  # Remove null columns
    
    # Verify that the final lists of rows and columns match the original dimensions
    total_rows = list(rows_kept) + list(rows_removed)
    total_rows.sort()
    total_columns = list(columns_kept) + list(columns_removed)
    total_columns.sort()
    # Check if the lists of rows and columns are identical to the original indices
    check_rows = all(x == y for x, y in zip(original_rows, total_rows))
    check_columns = all(x == y for x, y in zip(original_columns, total_columns))
    if not check_rows:
        raise ValueError("Error: Row indices do not match the original matrix.")
    if not check_columns:
        raise ValueError("Error: Column indices do not match the original matrix.")

    return matrix1, matrix2, sorted(rows_removed), sorted(columns_removed)
# =============================================================================


# =============================================================================
def mappingRowsAndColumns(matrix, rows_to_remove, columns_to_remove):
    """
    Generate mappings for row and column indices after removing specified rows and columns.
    
    Parameters:
    - matrix (np.ndarray): The original matrix.
    - rows_to_remove (list): Indices of rows to be removed.
    - columns_to_remove (list): Indices of columns to be removed.
    
    Returns:
    - Tuple[dict, dict]: 
        - row_mapping: Maps new row indices to old row indices.
        - column_mapping: Maps new column indices to old column indices.
    """
    # Create the row mapping from old row indices to new row indices
    # -----------
    row_mapping = {}
    current_row = 0
    for old_row in range(matrix.shape[0]):
        if old_row not in rows_to_remove:
            row_mapping[old_row] = current_row
            current_row += 1
    # Reverse the row mapping: new_row -> old_row
    row_mapping = {new_row: old_row 
                      for old_row, new_row in row_mapping.items()}
    # -----------

    # Create the column mapping from old column indices to new column indices
    # -----------
    column_mapping = {}
    current_column = 0
    for old_column in range(matrix.shape[1]):
        if old_column not in columns_to_remove:
            column_mapping[old_column] = current_column
            current_column += 1
    # Reverse the row mapping: new_row -> old_row
    column_mapping = {new_column: old_column 
                         for old_column, new_column in column_mapping.items()}
    # -----------
    return row_mapping, column_mapping
# =============================================================================


# =============================================================================
def giveMeMatrixByComponent(input_matrix, output_matrix):
    """
    Construct a bipartite graph from input and output matrices and decompose it into weakly connected components.
    
    Parameters:
    - input_matrix (np.ndarray): Matrix representing source set coefficients.
    - output_matrix (np.ndarray): Matrix representing target set coefficients.
    
    Returns:
    - List[dict]: Each dict represents a component with keys for input and output matrices, 
      nodes indices, and arc indices.
    """
    # Create the graph
    G = nx.DiGraph()
    
    # Get the number of nodes and arcs
    num_nodes, num_arcs = input_matrix.shape
    
    # Add nodes nodes
    for i in range(num_nodes):
        G.add_node(f'Nodes_{i+1}', bipartite=0)
    
    # Add arc nodes
    for j in range(num_arcs):
        G.add_node(f'Arc_{j+1}', bipartite=1)
    
    # Add edges from nodes to arcs (input matrix)
    for i in range(num_nodes):
        for j in range(num_arcs):
            if input_matrix[i, j] > 0:
                G.add_edge(f'Nodes_{i+1}', f'Arc_{j+1}')
    
    # Add edges from arcs to nodes (output matrix)
    for i in range(num_nodes):
        for j in range(num_arcs):
            if output_matrix[i, j] > 0:
                G.add_edge(f'Arc_{j+1}', f'Nodes_{i+1}')
    
    # Find the weakly connected components
    components = list(nx.weakly_connected_components(G))
    
    # Create matrices and mappings for each component
    components_all = []
    for component in components:
        nodes_indices = []
        arc_indices = []
    
        for node in component:
            if node.startswith('Nodes'):
                nodes_indices.append(int(node.split('_')[1]) - 1)  # Convert to zero-based index
            elif node.startswith('Arc'):
                arc_indices.append(int(node.split('_')[1]) - 1)  # Convert to zero-based index
    
        # Create the input and output matrices for this component
        input_submatrix = input_matrix[np.ix_(nodes_indices, arc_indices)]
        output_submatrix = output_matrix[np.ix_(nodes_indices, arc_indices)]
    
        # Store the submatrices and mappings
        components_all.append({'input': input_submatrix, 'output': output_submatrix, 'nodes': nodes_indices, 'arcs': arc_indices})

    return components_all
# =============================================================================



# =========================================================================
def recordArcsAltered(output_matrix, input_matrix, x, row_mapping, column_mapping, column_mapping_alter):
    """
    Generate string representations of hypergraph arcs.

    Parameters:
    - output_matrix (np.ndarray): Output incidence matrix.
    - input_matrix (np.ndarray): Input incidence matrix.
    - x (np.ndarray): Arc flows.
    - row_mapping (dict): Mapping from old nodes indices to new indices for string representation.
    - column_mapping (dict): Mapping from old arc indices to new indices (unused in this function).
    - column_mapping_alter (dict): Alternative mapping for arc indices for flow annotation.

    Returns:
    - List[str]: List of arc strings.
    """
    num_nodes, num_arcs = input_matrix.shape
    arcs = []

    for j in range(num_arcs):
        source_set = []
        target_set = []
        
        # Gather source set
        for i in range(num_nodes):
            if input_matrix[i, j] > 0.5:
                coef = int(input_matrix[i, j]) if input_matrix[i, j] > 1 else ''
                source_set.append(f"{coef}s{row_mapping[i] + 1}")
        
        # Gather target set
        for i in range(num_nodes):
            if output_matrix[i, j] > 0.5:
                coef = int(output_matrix[i, j]) if output_matrix[i, j] > 1 else ''
                target_set.append(f"{coef}s{row_mapping[i] + 1}")
        
        # Construct arc string
        source_set_str = ' + '.join(source_set) if source_set else ''
        target_set_str = ' + '.join(target_set) if target_set else ''
        arc = f"{source_set_str} -> {target_set_str} {x[column_mapping_alter[j]]}"
        arcs.append(arc + '\n')

    return arcs
# =========================================================================


# =========================================================================
def recordArcsAlteredWithNames(output_matrix, input_matrix, x, row_mapping, column_mapping, column_mapping_alter, name_nodes):
    """
    Generate string representations of hypergraph arcs with nodes names.

    Parameters:
    - output_matrix (np.ndarray): Output incidence matrix.
    - input_matrix (np.ndarray): Input incidence matrix.
    - x (np.ndarray): Arc intensities.
    - row_mapping (dict): Mapping from old nodes indices to new indices for string representation.
    - column_mapping (dict): Mapping from old arc indices to new indices (unused in this function).
    - column_mapping_alter (dict): Alternative mapping for arc indices for flow annotation.
    - name_nodes (list): Names of nodes for arc representation.

    Returns:
    - List[str]: List of arc strings with nodes names.
    """
    num_nodes, num_arcs = input_matrix.shape
    arcs = []

    for j in range(num_arcs):
        source_set = []
        target_set = []
        
        # Gather source_set
        for i in range(num_nodes):
            if input_matrix[i, j] > 0.5:
                coef = int(input_matrix[i, j]) if input_matrix[i, j] > 1 else ''
                source_set.append(f"{coef}{name_nodes[row_mapping[i]]}")
        
        # Gather target_set
        for i in range(num_nodes):
            if output_matrix[i, j] > 0.5:
                coef = int(output_matrix[i, j]) if output_matrix[i, j] > 1 else ''
                target_set.append(f"{coef}{name_nodes[row_mapping[i]]}")
        
        # Construct arc string
        source_set_str = ' + '.join(source_set) if source_set else ''
        target_set_str = ' + '.join(target_set) if target_set else ''
        arc = f"{source_set_str} -> {target_set_str} {x[column_mapping_alter[j]]}"
        arcs.append(arc + '\n')

    return arcs
# =============================================================================









