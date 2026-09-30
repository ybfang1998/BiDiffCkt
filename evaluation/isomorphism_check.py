"""OCB graph isomorphism, uniqueness, and novelty checks.

Isomorphism is checked in two steps:

1. Compare normalized representations in the current U/V node order.
2. If step 1 differs, compare exact BLISS canonical representations that are
   invariant to independent permutations of U and V nodes.

Node partitions, node types, edge labels, and isolated active nodes are all
preserved. Masked padding in the source database is ignored.

Run from the project root with::

    python BiDiffCkt/ocb_isomorphism_check.py

Only load pickle files that you trust.
"""

from __future__ import annotations
import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import igraph as ig
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_GENERATED = (
    BASE_DIR / "generated_data" / "BiDiffCkt_OCB101_generated_1000_results.pkl"
)
DEFAULT_REFERENCE = BASE_DIR / "raw_proceed" / "OCB_in_Bipartite.pkl"

Graph = Mapping[str, Any]
GraphSource = Iterable[Graph] | Mapping[str, Any] | str | Path


class Representation(NamedTuple):
    """Immutable OCB representation in the graph's current node order."""

    u_types: tuple[int, ...]
    v_types: tuple[int, ...]
    edges: tuple[tuple[int, int, int], ...]


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _check_nonnegative_integers(array: np.ndarray, name: str) -> None:
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite nonnegative integers.")
    if np.any(array < 0) or np.any(array != np.floor(array)):
        raise ValueError(f"{name} must contain finite nonnegative integers.")


def _active_partition(sample: Graph, name: str, capacity: int):
    """Return active node types and their positions in a padded array."""
    values = _as_numpy(sample[name])
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1-D type-ID array, got {values.shape}.")

    mask_value = sample.get(f"{name}_mask")
    if mask_value is None:
        if len(values) != capacity:
            raise ValueError(
                f"{name} has {len(values)} types but E has capacity {capacity}; "
                f"provide {name}_mask for padded E."
            )
        indices = np.arange(capacity)
    else:
        mask = _as_numpy(mask_value)
        if mask.shape != (capacity,) or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name}_mask must be binary with shape ({capacity},).")
        indices = np.flatnonzero(mask)
        if len(values) == capacity:
            values = values[indices]
        elif len(values) != len(indices):
            raise ValueError(
                f"{name} must have either {capacity} padded types or "
                f"{len(indices)} active types, got {len(values)}."
            )

    _check_nonnegative_integers(values, name)
    return tuple(int(value) for value in values), indices


def _get_representation(sample: Graph) -> Representation:
    """Normalize one OCB graph while retaining its current U/V ordering."""
    e = _as_numpy(sample["E"])
    if e.ndim != 2:
        raise ValueError(f"OCB E must have shape [U,V], got {e.shape}.")

    u_types, u_indices = _active_partition(sample, "U", e.shape[0])
    v_types, v_indices = _active_partition(sample, "V", e.shape[1])
    e = e[u_indices][:, v_indices]
    _check_nonnegative_integers(e, "active E")

    edges = tuple(
        (int(u), int(v), int(e[u, v])) for u, v in np.argwhere(e != 0)
    )
    return Representation(u_types, v_types, edges)


def _canonical_topology_key(representation: Representation) -> tuple:
    """Create an exact canonical key independent of U/V node ordering."""
    # U and V nodes use different partition labels even if their type IDs match.
    labels = (
        [("U", node_type) for node_type in representation.u_types]
        + [("V", node_type) for node_type in representation.v_types]
    )
    graph_edges = []

    # Turn every labeled U--V edge into U--edge_vertex--V. This lets BLISS
    # preserve the complete edge label through ordinary vertex coloring.
    for u_index, v_index, edge_label in representation.edges:
        edge_vertex = len(labels)
        labels.append(("E", edge_label))
        graph_edges.extend(
            (
                (u_index, edge_vertex),
                (edge_vertex, len(representation.u_types) + v_index),
            )
        )

    palette = {label: index for index, label in enumerate(sorted(set(labels)))}
    colors = [palette[label] for label in labels]
    graph = ig.Graph(n=len(labels), edges=graph_edges, directed=False)
    graph.vs["semantic_label"] = labels
    canonical = graph.permute_vertices(graph.canonical_permutation(color=colors))

    # Return semantic labels and edges, not a digest, to avoid hash collisions.
    return (
        tuple(canonical.vs["semantic_label"]),
        tuple(
            sorted(
                (min(source, target), max(source, target))
                for source, target in canonical.get_edgelist()
            )
        ),
    )


def _graphs(source: GraphSource) -> Iterable[Graph]:
    """Accept a graph iterable, a ``{'G': ...}`` dataset, or a pickle path."""
    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as file:
            source = pickle.load(file)
    if isinstance(source, Mapping):
        if "G" not in source:
            raise ValueError("A dataset mapping must contain a 'G' graph list.")
        source = source["G"]
    if isinstance(source, (str, bytes, Mapping)) or not isinstance(source, Iterable):
        raise TypeError("Expected an iterable of graphs, a {'G': graphs} dataset, or a path.")
    return source


def _canonical_keys(
    source: GraphSource,
    dataset_name: str,
    cache: dict[Representation, tuple] | None = None,
) -> list[tuple]:
    """Build one canonical key per sample using the two-step shortcut."""
    if cache is None:
        cache = {}

    keys = []
    for index, graph in enumerate(_graphs(source)):
        try:
            representation = _get_representation(graph)
            # Step 1: identical normalized representations reuse their result.
            key = cache.get(representation)
            if key is None:
                # Step 2: canonicalize distinct ordered representations.
                key = _canonical_topology_key(representation)
                cache[representation] = key
            keys.append(key)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid {dataset_name} graph at index {index}: {exc}"
            ) from exc
    return keys


def check_isomorphism(graph_a: Graph, graph_b: Graph) -> bool:
    """Return whether two typed OCB bipartite graphs are isomorphic."""
    representation_a = _get_representation(graph_a)
    representation_b = _get_representation(graph_b)

    # Step 1: same normalized graph in the same U/V order.
    if representation_a == representation_b:
        return True

    # Step 2: allow independent permutations within the U and V partitions.
    return (
        _canonical_topology_key(representation_a)
        == _canonical_topology_key(representation_b)
    )


def check_uniqueness(graphs: GraphSource) -> float:
    """Return uniqueness as a percentage in [0, 100].

    uniqueness = number of isomorphism classes / number of generated samples.
    Empty input returns 0.0.
    """
    keys = _canonical_keys(graphs, "generated")
    return 100.0 * len(set(keys)) / len(keys) if keys else 0.0


def check_novelty(
    generated_graphs: GraphSource,
    reference_graphs: GraphSource,
) -> float:
    """Return all-sample novelty as a percentage in [0, 100].

    novelty = generated samples absent from the reference database
              / all generated samples.

    Generated duplicates are deliberately retained in this denominator.
    Empty generated input returns 0.0.
    """
    cache: dict[Representation, tuple] = {}
    reference_keys = set(_canonical_keys(reference_graphs, "reference", cache))
    generated_keys = _canonical_keys(generated_graphs, "generated", cache)
    if not generated_keys:
        return 0.0
    novel_count = sum(key not in reference_keys for key in generated_keys)
    return 100.0 * novel_count / len(generated_keys)
