from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


NET_OR_PORT_TYPE_MAP = {
    "net": 0,
    "Vdd": 1,
    "Vss": 2,
    "Vout": 3,
    "VINN": 4,
    "VINP": 5,
    "IBIASP": 6,
    "IBIASN": 7,
    "IBIASPP": 8,
    "IBIASNN": 9,
}

NODE_TYPE_TO_ID = {
    "nch": 0,
    "pch": 1,
    "current_mirror_pmos": 2,
    "current_mirror_nmos": 3,
    "diff_pair_nmos": 4,
    "diff_pair_pmos": 5,
    "bias_pair_nmos": 6,
    "bias_pair_pmos": 7,
    "resistor": 8,
    "capacitor": 9,
    "Vdd": 10,
    "Vss": 11,
    "Vout": 12,
    "VINN": 13,
    "VINP": 14,
    "IBIASP": 15,
    "IBIASN": 16,
    "IBIASPP": 17,
    "IBIASNN": 18,
}

ID_TO_NODE_TYPE = {value: key for key, value in NODE_TYPE_TO_ID.items()}

MOS_NODE_TYPES = {
    "nch",
    "pch",
    "current_mirror_pmos",
    "current_mirror_nmos",
    "diff_pair_nmos",
    "diff_pair_pmos",
    "bias_pair_nmos",
    "bias_pair_pmos",
}

ID_TO_PIN = {
    0: ["D", "G", "S", "Vss"],
    1: ["D", "G", "S", "Vdd"],
    2: ["In", "Out", "Sources", "Vdd"],
    3: ["In", "Out", "Sources", "Vss"],
    4: ["InP", "InN", "OutP", "OutN", "Sources", "Vss"],
    5: ["InP", "InN", "OutP", "OutN", "Sources", "Vdd"],
    6: ["Bias", "Out1", "Out2", "Out3", "Out4", "Vss"],
    7: ["Bias", "Out1", "Out2", "Out3", "Out4", "Vdd"],
    8: ["Pos", "Neg"],
    9: ["Pos", "Neg"],
}

ENCODED_PIN_COUNT = {
    0: 3,
    1: 3,
    2: 3,
    3: 3,
    4: 5,
    5: 5,
    6: 5,
    7: 5,
    8: 2,
    9: 2,
}

EDGE_LABELS_TOKEN_TO_ID = {
    0: 0,
    "D": 1,
    "G": 2,
    "S": 3,
    "In": 4,
    "Out": 5,
    "CMSources": 6,
    "DPSources": 7,
    "InP": 8,
    "InN": 9,
    "OutP": 10,
    "OutN": 11,
    "Bias": 12,
    "Out1": 13,
    "Out2": 14,
    "Out3": 15,
    "Out4": 16,
    "Pos": 17,
    "Neg": 18,
    "D,G": 19,
    "Out3,Out4": 20,
}

NO_EDGE_LABEL_ID = EDGE_LABELS_TOKEN_TO_ID[0]
DEFAULT_LOAD_CAPACITANCE = 1e-12

SOURCE_EDGE_TOKEN_NODE_TYPES = {
    "CMSources": {NODE_TYPE_TO_ID["current_mirror_pmos"], NODE_TYPE_TO_ID["current_mirror_nmos"]},
    "DPSources": {NODE_TYPE_TO_ID["diff_pair_nmos"], NODE_TYPE_TO_ID["diff_pair_pmos"]},
}


def _invert_edge_labels(edge_labels_token_to_id: dict[Any, int]) -> dict[int, Any]:
    id_to_token: dict[int, Any] = {}
    for token, label_id in edge_labels_token_to_id.items():
        label_id = int(label_id)
        if label_id in id_to_token and id_to_token[label_id] != token:
            raise ValueError(f"Duplicate edge label id {label_id}: {id_to_token[label_id]!r} and {token!r}.")
        id_to_token[label_id] = token
    return id_to_token


def _as_1d_int_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be a 1-D array, got shape {array.shape}.")
    return array


def _valid_indices(values: np.ndarray, mask: Any | None, axis_size: int, name: str) -> np.ndarray:
    if mask is None:
        return np.arange(min(len(values), axis_size), dtype=np.int64)

    mask_array = np.asarray(mask).astype(bool).reshape(-1)
    indices = np.flatnonzero(mask_array)
    indices = indices[(indices < len(values)) & (indices < axis_size)]
    return indices.astype(np.int64)


def _extract_graph_arrays(g: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    U = _as_1d_int_array("U", g["U"])
    V = _as_1d_int_array("V", g["V"])
    E = np.asarray(g["E"], dtype=np.int64)
    if E.ndim != 2:
        raise ValueError(f"`E` must be a multiclass [U, V] matrix, got shape {E.shape}.")

    u_indices = _valid_indices(U, g.get("U_mask"), E.shape[0], "U")
    v_indices = _valid_indices(V, g.get("V_mask"), E.shape[1], "V")
    return U, V, E, u_indices, v_indices


def _load_template(template_dir: str | Path, template_name: str) -> str:
    template_path = Path(template_dir) / template_name
    return template_path.read_text(encoding="utf-8").strip()


def _render_testbench_tail(template_dir: str | Path, template_name: str, load_capacitance: float) -> str:
    tail_text = _load_template(template_dir, template_name)
    c_load_line = f"C_Load (Vout 0) capacitor c={_spectre_value(load_capacitance)}"
    return re.sub(
        r"^C_Load\s+\(Vout 0\)\s+capacitor\s+c=.*$",
        c_load_line,
        tail_text,
        count=1,
        flags=re.MULTILINE,
    )


def _v_type_id_to_name(v_type_id: int, net_counter: list[int]) -> str:
    if v_type_id == NET_OR_PORT_TYPE_MAP["net"]:
        name = f"net{net_counter[0]}"
        net_counter[0] += 1
        return name

    for name, mapped_type_id in NET_OR_PORT_TYPE_MAP.items():
        if name == "net" or mapped_type_id != v_type_id:
            continue
        if name == "Vss":
            return "0"
        return name

    raise ValueError(f"Unknown V node type id: {v_type_id}.")


def _build_net_names_from_v(V: np.ndarray, v_indices: np.ndarray) -> dict[int, str]:
    net_counter = [1]
    return {int(v_idx): _v_type_id_to_name(int(V[int(v_idx)]), net_counter) for v_idx in v_indices}


def _spectre_value(value: float | int) -> str:
    return f"{float(np.asarray(value).item()):.12g}"


def _instance_statement(instance_name: str, node_type_name: str, pins: list[str], subckt_name: str | None = None) -> str:
    pin_text = " ".join(pins)

    if node_type_name in MOS_NODE_TYPES:
        target_name = subckt_name or node_type_name
        if subckt_name is None and node_type_name in {"nch", "pch"}:
            return f"{instance_name} ({pin_text}) {target_name} w=1 l=1"
        return f"{instance_name} ({pin_text}) {target_name} W=1 L=1"

    if node_type_name == "resistor":
        return f"{instance_name} ({pin_text}) resistor r=1"

    if node_type_name == "capacitor":
        return f"{instance_name} ({pin_text}) capacitor c=1"

    return f"{instance_name} ({pin_text}) {subckt_name or node_type_name}"


def _edge_token_to_pin_labels(token: str, node_type_id: int) -> tuple[str, ...]:
    if node_type_id not in ID_TO_PIN:
        raise ValueError(f"Node type id {node_type_id} cannot own an edge label.")

    encoded_pins = ID_TO_PIN[node_type_id][: ENCODED_PIN_COUNT[node_type_id]]
    pin_labels: list[str] = []

    for token_part in token.split(","):
        token_part = token_part.strip()
        if not token_part:
            continue

        if token_part in SOURCE_EDGE_TOKEN_NODE_TYPES:
            if node_type_id not in SOURCE_EDGE_TOKEN_NODE_TYPES[token_part]:
                node_type_name = ID_TO_NODE_TYPE.get(node_type_id, str(node_type_id))
                raise ValueError(f"Edge token `{token_part}` is not valid for node type `{node_type_name}`.")
            pin_label = "Sources"
        else:
            pin_label = token_part

        if pin_label not in encoded_pins:
            node_type_name = ID_TO_NODE_TYPE.get(node_type_id, str(node_type_id))
            raise ValueError(f"Edge token `{token}` is not a pin of node type `{node_type_name}`.")
        if pin_label in pin_labels:
            raise ValueError(f"Edge token `{token}` maps to duplicated pin `{pin_label}`.")
        pin_labels.append(pin_label)

    if not pin_labels:
        raise ValueError("Empty edge token is not allowed.")
    return tuple(pin_labels)


def _edge_label_id_to_pin_labels(
    label_id: int,
    node_type_id: int,
    edge_labels_id_to_token: dict[int, Any],
    no_edge_id: int,
) -> tuple[str, ...]:
    label_id = int(label_id)
    if label_id == no_edge_id:
        return ()
    if label_id not in edge_labels_id_to_token:
        raise ValueError(f"Unknown edge label id: {label_id}.")

    token = edge_labels_id_to_token[label_id]
    if not isinstance(token, str):
        raise ValueError(f"Edge label id {label_id} maps to non-pin token {token!r}.")
    return _edge_token_to_pin_labels(token, node_type_id)


def _default_pin_net(pin_name: str, device_idx: int) -> str:
    if pin_name == "Vss":
        return "0"
    if pin_name == "Vdd":
        return "Vdd"
    raise ValueError(f"Device {device_idx} pin `{pin_name}` has no E connection and no default net.")


def _device_pin_nets(
    U: np.ndarray,
    E: np.ndarray,
    device_idx: int,
    v_indices: np.ndarray,
    net_names: dict[int, str],
    edge_labels_id_to_token: dict[int, Any],
    no_edge_id: int,
) -> list[str]:
    node_type_id = int(U[device_idx])
    if node_type_id not in ID_TO_PIN:
        raise ValueError(f"U[{device_idx}] has unsupported device node type id {node_type_id}.")

    assigned_nets: dict[str, str] = {}
    for v_idx in v_indices:
        label_id = int(E[device_idx, int(v_idx)])
        if label_id == no_edge_id:
            continue

        pin_labels = _edge_label_id_to_pin_labels(label_id, node_type_id, edge_labels_id_to_token, no_edge_id)
        for pin_label in pin_labels:
            net_name = net_names[int(v_idx)]
            existing_net_name = assigned_nets.get(pin_label)
            if existing_net_name is not None and existing_net_name != net_name:
                raise ValueError(
                    f"Device {device_idx} pin `{pin_label}` connects to both "
                    f"`{existing_net_name}` and `{net_name}`."
                )
            assigned_nets[pin_label] = net_name

    pin_names: list[str] = []
    pin_label_list = ID_TO_PIN[node_type_id]
    encoded_pin_count = ENCODED_PIN_COUNT[node_type_id]
    for pin_idx, pin_label in enumerate(pin_label_list):
        if pin_idx >= encoded_pin_count:
            pin_names.append(_default_pin_net(pin_label, device_idx))
            continue

        if pin_label not in assigned_nets:
            node_type_name = ID_TO_NODE_TYPE[node_type_id]
            raise ValueError(f"Device {device_idx} pin `{pin_label}` ({node_type_name}) is not connected.")
        pin_names.append(assigned_nets[pin_label])

    return pin_names


def _append_bias_templates(lines: list[str], V: np.ndarray, v_indices: np.ndarray, template_dir: Path) -> None:
    bias_templates = {
        NET_OR_PORT_TYPE_MAP["IBIASP"]: "IBIASP_1",
        NET_OR_PORT_TYPE_MAP["IBIASN"]: "IBIASN_1",
        NET_OR_PORT_TYPE_MAP["IBIASPP"]: "IBIASP_2",
        NET_OR_PORT_TYPE_MAP["IBIASNN"]: "IBIASN_2",
    }
    present_v_types = {int(V[int(v_idx)]) for v_idx in v_indices}
    for v_type_id, template_name in bias_templates.items():
        if v_type_id in present_v_types:
            lines.append(_load_template(template_dir, template_name))
            lines.append("")


def _append_subckt_templates(
    lines: list[str],
    U: np.ndarray,
    u_indices: np.ndarray,
    template_dir: Path,
) -> dict[int, str]:
    subckt_instance_names: dict[int, str] = {}
    subckt_counts: dict[str, int] = {}

    for device_idx in u_indices:
        device_idx = int(device_idx)
        node_type_id = int(U[device_idx])
        if node_type_id < NODE_TYPE_TO_ID["current_mirror_pmos"] or node_type_id > NODE_TYPE_TO_ID["bias_pair_pmos"]:
            continue

        node_type_name = ID_TO_NODE_TYPE[node_type_id]
        subckt_counts[node_type_name] = subckt_counts.get(node_type_name, 0) + 1
        instance_subckt_name = f"{node_type_name}_{subckt_counts[node_type_name]}"
        subckt_instance_names[device_idx] = instance_subckt_name

        template_text = _load_template(template_dir, node_type_name)
        template_text = template_text.replace(f"subckt {node_type_name}", f"subckt {instance_subckt_name}", 1)
        template_text = template_text.replace(f"ends {node_type_name}", f"ends {instance_subckt_name}", 1)
        lines.append(template_text)
        lines.append("")

    return subckt_instance_names


def build_multiclass_bipartite_netlist_lines(
    g: dict[str, Any],
    template_dir: str | Path,
    *,
    edge_labels_token_to_id: dict[Any, int] | None = None,
    no_edge_id: int = NO_EDGE_LABEL_ID,
) -> list[str]:
    """Build the shared Spectre circuit body from one multiclass bipartite graph.

    The input sample must contain `U`, `V`, and multiclass `E` with shape
    `[num_U, num_V]`. Padded `E` is also supported when `U_mask` and `V_mask`
    are present.
    """
    edge_labels_id_to_token = _invert_edge_labels(edge_labels_token_to_id or EDGE_LABELS_TOKEN_TO_ID)
    U, V, E, u_indices, v_indices = _extract_graph_arrays(g)
    net_names = _build_net_names_from_v(V, v_indices)
    template_dir = Path(template_dir)

    lines: list[str] = []
    _append_bias_templates(lines, V, v_indices, template_dir)
    subckt_instance_names = _append_subckt_templates(lines, U, u_indices, template_dir)

    for device_idx in u_indices:
        device_idx = int(device_idx)
        node_type_id = int(U[device_idx])
        node_type_name = ID_TO_NODE_TYPE[node_type_id]
        pin_names = _device_pin_nets(
            U=U,
            E=E,
            device_idx=device_idx,
            v_indices=v_indices,
            net_names=net_names,
            edge_labels_id_to_token=edge_labels_id_to_token,
            no_edge_id=no_edge_id,
        )
        lines.append(
            _instance_statement(
                instance_name=f"I{device_idx}",
                node_type_name=node_type_name,
                pins=pin_names,
                subckt_name=subckt_instance_names.get(device_idx),
            )
        )

    return lines


def MulticlassBipartite2Netlist(
    g: dict[str, Any],
    netlist_index: int,
    output_dir: str | Path | None = Path("netlists"),
    testbench_dir=Path("testbenches/DiffCkt"),
    subckt_template_dir=Path("netlist_templates/DiffCkt"),
    *,
    edge_labels_token_to_id: dict[Any, int] | None = None,
    no_edge_id: int = NO_EDGE_LABEL_ID,
    load_capacitance: float = DEFAULT_LOAD_CAPACITANCE,
) -> dict[str, Path] | bool:
    """Convert one multiclass bipartite graph sample to AC and STB netlists.

    Set `output_dir=None` to validate that the graph can be rendered without
    writing netlist files. In that mode the function returns True on success.
    """
    output_path = None if output_dir is None else Path(output_dir)
    testbench_dir = Path(testbench_dir)
    subckt_template_dir = Path(subckt_template_dir)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    shared_lines = build_multiclass_bipartite_netlist_lines(
        g,
        template_dir=subckt_template_dir,
        edge_labels_token_to_id=edge_labels_token_to_id,
        no_edge_id=no_edge_id,
    )

    saved_paths: dict[str, Path] = {}
    for analysis_type in ("ac", "stb"):
        lines = [
            _load_template(testbench_dir, f"{analysis_type}_header"),
            "",
            *shared_lines,
            "",
            _render_testbench_tail(testbench_dir, f"{analysis_type}_tail", load_capacitance),
        ]
        netlist_text = "\n".join(lines).rstrip() + "\n"
        if output_path is None:
            continue

        save_path = output_path / f"netlist_{netlist_index}_{analysis_type}.scs"
        save_path.write_text(netlist_text, encoding="utf-8")
        saved_paths[analysis_type] = save_path

    return True if output_path is None else saved_paths


def Bipartite2Netlist(*args: Any, **kwargs: Any) -> dict[str, Path] | bool:
    """Backward-compatible name for `MulticlassBipartite2Netlist`."""
    return MulticlassBipartite2Netlist(*args, **kwargs)


def _load_dataset(dataset: str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(dataset, (str, Path)):
        with Path(dataset).open("rb") as dataset_file:
            return pickle.load(dataset_file)
    return dataset


def MulticlassDataset2Netlists(
    dataset: str | Path | list[dict[str, Any]],
    output_dir: str | Path | None = Path("netlists"),
    testbench_dir=Path("testbenches/DiffCkt"),
    subckt_template_dir=Path("netlist_templates/DiffCkt"),
    *,
    limit: int | None = None,
    start_index: int = 0,
    use_sample_id: bool = False,
    edge_labels_token_to_id: dict[Any, int] | None = None,
    no_edge_id: int = NO_EDGE_LABEL_ID,
    load_capacitance: float = DEFAULT_LOAD_CAPACITANCE,
) -> list[dict[str, Path] | bool]:
    """Convert a loaded dataset list or pickle path into AC/STB netlists.

    Set `output_dir=None` to validate every selected sample without writing
    netlist files.
    """
    records = _load_dataset(dataset)
    saved_paths: list[dict[str, Path] | bool] = []

    for offset, sample in enumerate(records):
        if limit is not None and offset >= limit:
            break
        netlist_index = int(sample["id"]) if use_sample_id and "id" in sample else start_index + offset
        saved_paths.append(
            MulticlassBipartite2Netlist(
                sample,
                netlist_index=netlist_index,
                output_dir=output_dir,
                testbench_dir=testbench_dir,
                subckt_template_dir=subckt_template_dir,
                edge_labels_token_to_id=edge_labels_token_to_id,
                no_edge_id=no_edge_id,
                load_capacitance=load_capacitance,
            )
        )

    return saved_paths

if __name__ == "__main__":
    data_path = "generated_data/BiDiffCkt_Diffckt_multiclass_generated_100_results.pkl"
    with open(data_path, 'rb') as f:
        bipartite_graph = pickle.load(f)
    G = bipartite_graph['G']
    for i, g in enumerate(G):
        try:
            netlist = MulticlassDataset2Netlists(g, i, output_dir="netlists")
        except ValueError as e:
            print(f"Error in graph {i}: {e}")
