from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm


def edge_mask_from_e(e: Any, num_rows: int, num_columns: int) -> torch.Tensor:
    """Convert a supported E encoding to a 2-D boolean adjacency mask.

    ``True`` at ``[i, j]`` means that node i and node j are connected.  For a
    bipartite graph, the row and column nodes belong to U and V respectively.
    """

    e_tensor = torch.as_tensor(e).detach().cpu()

    if e_tensor.ndim < 2:
        raise ValueError(
            f"E must have at least 2 dimensions, but its shape is "
            f"{tuple(e_tensor.shape)}"
        )
    if tuple(e_tensor.shape[:2]) != (num_rows, num_columns):
        raise ValueError(
            "The first two dimensions of E have the wrong size: "
            f"got E.shape={tuple(e_tensor.shape)}, expected "
            f"({num_rows}, {num_columns}, ...)"
        )

    nonzero = e_tensor.ne(0)
    if e_tensor.ndim == 2:
        # Single-label/multiclass: label 0 means no connection.
        return nonzero

    # Multi-label: an all-zero attribute vector means no connection.  Taking
    # any over every trailing dimension also supports non-flat label tensors.
    label_dimensions = tuple(range(2, e_tensor.ndim))
    return nonzero.any(dim=label_dimensions)


def _dfs_reaches_all(adjacency: list[list[int]]) -> bool:
    """Check undirected connectivity of an adjacency list using DFS."""

    if not adjacency:
        return False

    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append(neighbour)

    return len(visited) == len(adjacency)


def is_valid_bipartite_graph(graph: dict[str, Any]) -> bool:
    """Return whether all U and V nodes belong to one connected component.

    Edges are treated as undirected for the connectivity check.  A one-node
    graph is connected; an empty graph is treated as invalid.
    """

    try:
        u_nodes = graph["U"]
        v_nodes = graph["V"]
        e = graph["E"]
    except KeyError as exc:
        raise KeyError(f"Graph is missing the required key {exc.args[0]!r}") from exc

    num_u = len(u_nodes)
    num_v = len(v_nodes)
    num_nodes = num_u + num_v

    if num_nodes == 0:
        return False

    edge_mask = edge_mask_from_e(e, num_u, num_v)

    # Use indices 0..num_u-1 for U and num_u..num_u+num_v-1 for V.
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for u_index, v_index in edge_mask.nonzero(as_tuple=False).tolist():
        shifted_v_index = num_u + v_index
        adjacency[u_index].append(shifted_v_index)
        adjacency[shifted_v_index].append(u_index)

    # For an undirected graph, reaching every node from any one node is
    # equivalent to the graph having exactly one connected component.
    return _dfs_reaches_all(adjacency)


def is_valid_general_graph(graph: dict[str, Any]) -> bool:
    """Return whether all real nodes of a non-bipartite graph are connected.

    Padded nodes for which ``node_mask`` is false are ignored.  Connectivity is
    checked as undirected/weak connectivity: either E[i,j] or E[j,i] creates an
    undirected edge.  This is also robust if a future dataset stores each edge
    in only one direction.
    """

    try:
        x_nodes = graph["X"]
        e = graph["E"]
    except KeyError as exc:
        raise KeyError(f"Graph is missing the required key {exc.args[0]!r}") from exc

    max_nodes = len(x_nodes)
    edge_mask = edge_mask_from_e(e, max_nodes, max_nodes)

    if "node_mask" in graph:
        node_mask = torch.as_tensor(graph["node_mask"]).detach().cpu()
        if node_mask.ndim != 1 or len(node_mask) != max_nodes:
            raise ValueError(
                "node_mask must be one-dimensional with len(node_mask) == len(X): "
                f"got node_mask.shape={tuple(node_mask.shape)}, len(X)={max_nodes}"
            )
        active_indices = node_mask.bool().nonzero(as_tuple=False).flatten()
    else:
        active_indices = torch.arange(max_nodes)

    num_active_nodes = int(active_indices.numel())
    if num_active_nodes == 0:
        return False

    # Drop padded rows/columns, then ignore edge direction for connectivity.
    active_edges = edge_mask.index_select(0, active_indices).index_select(
        1, active_indices
    )
    active_edges = active_edges | active_edges.T

    adjacency: list[list[int]] = [[] for _ in range(num_active_nodes)]
    for source, target in active_edges.nonzero(as_tuple=False).tolist():
        if source != target:
            adjacency[source].append(target)

    return _dfs_reaches_all(adjacency)


def is_valid_graph(graph: dict[str, Any]) -> bool:
    """Dispatch a bipartite or general graph to its connectivity checker."""

    if "U" in graph and "V" in graph:
        return is_valid_bipartite_graph(graph)
    if "X" in graph:
        return is_valid_general_graph(graph)
    raise ValueError("Graph must contain either U/V/E or X/E fields")


def valid_graph_statistics(graphs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Calculate valid/invalid counts, rate, and invalid zero-based indices."""

    valid_count = 0
    invalid_indices: list[int] = []

    for index, graph in enumerate(graphs):
        try:
            is_valid = is_valid_graph(graph)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid graph at index {index}: {exc}") from exc

        if is_valid:
            valid_count += 1
        else:
            invalid_indices.append(index)

    total_count = valid_count + len(invalid_indices)
    valid_rate = valid_count / total_count if total_count else 0.0
    return {
        "total_count": total_count,
        "valid_count": valid_count,
        "invalid_count": len(invalid_indices),
        "valid_rate": valid_rate,
        "invalid_indices": invalid_indices,
    }

def is_valid_DAG(g, subg=True):
    # Check if the given igraph g is a valid DAG computation graph
    # first need to have no directed cycles
    # second need to have no zero-indegree nodes except input
    # third need to have no zero-outdegree nodes except output
    # i.e., ensure nodes are connected
    # fourth need to have exactly one input node
    # finally need to have exactly one output node
    if subg:
        START_TYPE=0
        END_TYPE=1
    else:
        START_TYPE=8 
        END_TYPE=9
    res = g.is_dag()
    #return res
    n_start, n_end = 0, 0
    for v in g.vs:
        if v['type'] == START_TYPE:
            n_start += 1
        elif v['type'] == END_TYPE:
            n_end += 1
        if v.outdegree() == 0 and v['type'] != END_TYPE:
            return False
    return res and n_start == 1 and n_end == 1

def load_bipartite_graphs(dataset_path: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    with path.open("rb") as file:
        dataset = pickle.load(file)

    if isinstance(dataset, list):
        return dataset
    elif isinstance(dataset, dict) and "G" in dataset:
        return list(dataset["G"])
    else:
        raise ValueError(f"Dataset at {path} is not a valid bipartite graph dataset.")

def load_OCB101_graphs(dataset_path: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    with path.open("rb") as file:
        dataset = pickle.load(file)

    G = [g_pair[0] for g_pair in dataset[0]]
    return G


def ratio_same_DAG(G0, G1):
    """Calculates the ratio of graphs in G1 that also appear in G0.
    
    Useful for measuring novelty: if comparing generated circuits (G1) against
    training set (G0), a high ratio indicates more novel/diverse generation.
    
    Args:
        G0: Reference graph list (e.g., training set).
        G1: Query graph list (e.g., generated circuits).
        
    Returns:
        float: Ratio of G1 graphs found in G0, in range [0, 1].
               Higher values indicate more novelty.
               
    Notes:
        - Uses is_same_DAG for exact graph matching
        - O(|G0| * |G1|) complexity - can be slow for large datasets
    """
    # how many G1 are in G0
    res = 0
    for g1 in tqdm(G1, desc="Comparing graphs"):
        for g0 in G0:
            if is_same_DAG(g1, g0):
                res += 1
                break
                
    return 1 - (res / len(G1))


def is_same_DAG(g0, g1):
    """Checks if two graphs are identical (same structure and node types).
    
    Two graphs are considered the same if:
    1. Same number of vertices
    2. Same node type at each position
    3. Same incoming edges for each node
    
    Args:
        g0: First igraph.Graph with 'type' vertex attribute.
        g1: Second igraph.Graph with 'type' vertex attribute.
        
    Returns:
        bool: True if graphs are structurally identical.
        
    Notes:
        - Does NOT check for graph isomorphism (different node orderings)
        - Assumes node ordering is canonical and consistent
        - Only checks incoming edges (sufficient for DAGs)
    """
    # note that it does not check isomorphism
    if g0.vcount() != g1.vcount():
        return False
    for vi in range(g0.vcount()):
        if g0.vs[vi]['type'] != g1.vs[vi]['type']:
            return False
        if set(g0.neighbors(vi, 'in')) != set(g1.neighbors(vi, 'in')):  # compare the vertice set of in neighbors between generate and gnd-truth
            return False
    return True