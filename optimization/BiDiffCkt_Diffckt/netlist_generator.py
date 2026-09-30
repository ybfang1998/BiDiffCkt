import torch
import numpy as np
import os
import re
from pathlib import Path
import pickle

MAX_PIN_COUNT = 5
SPEC_UNNORMALIZE_FACTORS = np.array(
    [
        1e-3,   # Power Consumption (Pdiss)
        100,    # DC Gain (Gain_DC)
        10e6,   # Gain-Bandwidth Product (GBW)
        180,    # Phase Margin (PM)
        10e6,   # Positive Slew Rate (SRP)
        10e6,   # Negative Slew Rate (SRN)
        1.2,    # Output Voltage Swing Low (VOL)
        1.2,    # Output Voltage Swing High (VOH)
        100,    # Common-Mode Rejection Ratio (CMRR)
        100,    # Power Supply Rejection Ratio (PSRR)
        1e-6,   # Input Equivalent Noise @ 1 kHz (Noise@1kHz)
        1e-7,   # Input Equivalent Noise @ 1 GHz (Noise@1GHz)
        10e-12, # Load Capacitance (CL)
    ],
    dtype=np.float32,
)

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

FIXED_NODE_VALUES = {
    "Vdd": {"voltage": 1.2},
    "Vss": {"voltage": 0.0},
    "VINP": {"voltage": 0.6},
    "IBIASP": {"current": 1e-6},
    "IBIASN": {"current": 1e-6},
    "IBIASPP": {"current": 5e-6},
    "IBIASNN": {"current": 5e-6},
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
    9: ["Pos", "Neg"]
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

DEFAULT_LOAD_CAPACITANCE = 1e-12

def _pin_count(node_type_id):
    return ENCODED_PIN_COUNT.get(node_type_id, 1)

def unnormalize_specs(normalized_specs):
    """Convert normalized Y values back to physical units using Appendix A.3 scales.

    Args:
        normalized_specs: A length-13 vector or a batch of vectors whose last
            dimension is 13. Supports list, numpy.ndarray, and torch.Tensor.

    Returns:
        The un-normalized values in the same container family:
        - torch.Tensor in, torch.Tensor out
        - anything else returns numpy.ndarray
    """
    expected_dim = len(SPEC_UNNORMALIZE_FACTORS)

    if isinstance(normalized_specs, torch.Tensor):
        if normalized_specs.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected the last dimension to be {expected_dim}, got {normalized_specs.shape[-1]}."
            )
        scale = torch.as_tensor(
            SPEC_UNNORMALIZE_FACTORS,
            dtype=normalized_specs.dtype,
            device=normalized_specs.device,
        )
        return normalized_specs * scale

    normalized_specs = np.asarray(normalized_specs, dtype=np.float32)
    if normalized_specs.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected the last dimension to be {expected_dim}, got {normalized_specs.shape[-1]}."
        )
    return normalized_specs * SPEC_UNNORMALIZE_FACTORS

def _load_template(template_dir, template_name):
    template_path = os.path.join(template_dir, template_name)
    with open(template_path, "r") as template_file:
        return template_file.read().strip()
    
def _render_testbench_tail(template_dir, template_name, load_capacitance):
    """Render a testbench tail template with the un-normalized output load capacitance."""
    tail_text = _load_template(template_dir, template_name)
    c_load_line = f"C_Load (Vout 0) capacitor c={_spectre_value(load_capacitance)}"
    return re.sub(
        r"^C_Load\s+\(Vout 0\)\s+capacitor\s+c=.*$",
        c_load_line,
        tail_text,
        count=1,
        flags=re.MULTILINE,
    )

def _V_type_id_to_name(V_type_id: int, net_counter: list[int]) -> str:
    """Recover the concrete net/port name from one V type id."""
    if V_type_id == 0:
        name = f"net{net_counter[0]}"
        net_counter[0] += 1
        return name

    for name, mapped_type_id in NET_OR_PORT_TYPE_MAP.items():
        if name != "net" and mapped_type_id == V_type_id:
            if name == "Vss":
                return "0"
            return name

def _build_net_names_from_V(V: np.ndarray) -> list[str]:
    """Assign stable concrete net names to the V nodes."""
    net_counter = [1]
    return [_V_type_id_to_name(int(V_type_id), net_counter) for V_type_id in V]

def _resolve_node_type(node_type):
    if isinstance(node_type, str):
        if node_type not in NODE_TYPE_TO_ID:
            raise ValueError(f"Unknown node type: {node_type}")
        return node_type
    if isinstance(node_type, (int, np.integer)):
        if int(node_type) not in ID_TO_NODE_TYPE:
            raise ValueError(f"Unknown node type id: {node_type}")
        return ID_TO_NODE_TYPE[int(node_type)]
    raise TypeError("node_type must be a node type name or integer id.")

def unnormalize_device_parameters(node_type):
    """Return the default physical parameters for one device node type."""
    node_type_name = _resolve_node_type(node_type)

    if node_type_name in FIXED_NODE_VALUES:
        return dict(FIXED_NODE_VALUES[node_type_name])
    if node_type_name in MOS_NODE_TYPES:
        return {"L": 1.0, "W": 1.0}

    if node_type_name == "resistor":
        return {"R": 1.0}

    if node_type_name == "capacitor":
        return {"C": 1.0}

    return {}


def _spectre_value(value):
    """Format a numeric value for spectre netlists."""
    return f"{float(np.asarray(value).item()):.12g}"

def _instance_statement(instance_name, node_type_name, pins, subckt_name=None):
    pin_text = " ".join(pins)
    physical_params = unnormalize_device_parameters(node_type_name)

    if node_type_name in MOS_NODE_TYPES:
        width = _spectre_value(physical_params["W"])
        length = _spectre_value(physical_params["L"])
        target_name = subckt_name or node_type_name
        if subckt_name is None and node_type_name in {"nch", "pch"}:
            return f"{instance_name} ({pin_text}) {target_name} w={width} l={length}"
        return f"{instance_name} ({pin_text}) {target_name} W={width} L={length}"

    if node_type_name == "resistor":
        return f"{instance_name} ({pin_text}) resistor r={_spectre_value(physical_params['R'])}"

    if node_type_name == "capacitor":
        return f"{instance_name} ({pin_text}) capacitor c={_spectre_value(physical_params['C'])}"

    return f"{instance_name} ({pin_text}) {subckt_name or node_type_name}"

def _build_bipartite_netlist_lines(
    U: np.ndarray,
    V: np.ndarray,
    E_bipartite: np.ndarray,
    template_dir,
) -> list[str]:
    """Build the shared circuit body from the bipartite representation."""
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    E_bipartite = np.asarray(E_bipartite, dtype=np.float64)

    if E_bipartite.shape != (len(U), len(V), MAX_PIN_COUNT):
        raise ValueError(
            f"Expected E_bipartite shape {(len(U), len(V), MAX_PIN_COUNT)}, "
            f"got {E_bipartite.shape}."
        )

    net_names = _build_net_names_from_V(V)

    lines = []
    bias_templates = {
        NET_OR_PORT_TYPE_MAP["IBIASP"]: "IBIASP_1",
        NET_OR_PORT_TYPE_MAP["IBIASN"]: "IBIASN_1",
        NET_OR_PORT_TYPE_MAP["IBIASPP"]: "IBIASP_2",
        NET_OR_PORT_TYPE_MAP["IBIASNN"]: "IBIASN_2",
    }
    present_V_types = {int(v) for v in V.tolist()}
    for V_type_id, template_name in bias_templates.items():
        if V_type_id in present_V_types:
            lines.append(_load_template(template_dir, template_name))
            lines.append("")

    subckt_instance_names = {}
    subckt_counts = {}
    for device_idx, node_type_id in enumerate(U.tolist()):
        node_type_name = ID_TO_NODE_TYPE[int(node_type_id)]
        if node_type_id < 2 or node_type_id > 7:
            continue

        subckt_counts[node_type_name] = subckt_counts.get(node_type_name, 0) + 1
        instance_subckt_name = f"{node_type_name}_{subckt_counts[node_type_name]}"
        subckt_instance_names[device_idx] = instance_subckt_name

        template_text = _load_template(template_dir, node_type_name)
        template_text = template_text.replace(
            f"subckt {node_type_name}",
            f"subckt {instance_subckt_name}",
            1,
        )
        template_text = template_text.replace(
            f"ends {node_type_name}",
            f"ends {instance_subckt_name}",
            1,
        )
        lines.append(template_text)
        lines.append("")

    for device_idx, node_type_id in enumerate(U.tolist()):
        node_type_name = ID_TO_NODE_TYPE[int(node_type_id)]
        pin_names = []
        pin_label_list = ID_TO_PIN[int(node_type_id)]
        encoded_pin_count = _pin_count(int(node_type_id))

        for pin_idx, default_pin_name in enumerate(pin_label_list):
            if pin_idx >= encoded_pin_count:
                if default_pin_name == "Vss":
                    pin_names.append("0")
                    continue
                if default_pin_name == "Vdd":
                    pin_names.append("Vdd")
                    continue
                raise ValueError(
                    f"Unsupported unencoded default pin `{default_pin_name}` for device {device_idx}."
                )

            connected_V_indices = np.flatnonzero(E_bipartite[device_idx, :, pin_idx] > 0.5)
            if len(connected_V_indices) == 0:
                if default_pin_name == "Vss":
                    pin_names.append("0")
                    continue
                if default_pin_name == "Vdd":
                    pin_names.append("Vdd")
                    continue
                raise ValueError(
                    f"Device {device_idx} pin {pin_idx} ({node_type_name}) is not connected."
                )
            if len(connected_V_indices) > 1:
                raise ValueError(
                    f"Device {device_idx} pin {pin_idx} connects to multiple V nodes: "
                    f"{connected_V_indices.tolist()}"
                )

            pin_names.append(net_names[int(connected_V_indices[0])])

        instance_line = _instance_statement(
            instance_name=f"I{device_idx}",
            node_type_name=node_type_name,
            pins=pin_names,
            subckt_name=subckt_instance_names.get(device_idx),
        )
        lines.append(instance_line)

    return lines

def Bipartite2Netlist(
    g,
    netlist_index,
    output_dir=Path("netlists"),
    testbench_dir=Path("testbenches/DiffCkt"),
    subckt_template_dir=Path("netlist_templates/DiffCkt")
):
    """Convert one DiffCkt sample to Spectre netlists through its bipartite form.

    If output_dir is None, only validate that the graph can be rendered as
    netlists and skip writing files.
    """

    U, V, E = g['U'], g['V'], g['E']
    testbench_dir = Path(testbench_dir)
    subckt_template_dir = Path(subckt_template_dir)
    output_path = None if output_dir is None else Path(output_dir)

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    shared_lines = _build_bipartite_netlist_lines(
        U=U,
        V=V,
        E_bipartite=E,
        template_dir=subckt_template_dir,
    )

    saved_paths = {}
    for analysis_type in ("ac", "stb"):
        lines = [
            _load_template(testbench_dir, f"{analysis_type}_header"),
            "",
            *shared_lines,
            "",
            _render_testbench_tail(testbench_dir, f"{analysis_type}_tail", DEFAULT_LOAD_CAPACITANCE),
        ]
        netlist_text = "\n".join(lines).rstrip() + "\n"
        if output_path is None:
            continue

        save_path = output_path / f"netlist_{netlist_index}_{analysis_type}.scs"
        with open(save_path, "w", encoding="utf-8") as netlist_file:
            netlist_file.write(netlist_text)
        saved_paths[analysis_type] = save_path

    if output_path is None:
        return True

    return saved_paths

if __name__ == "__main__":
    data_path = "generated_data/BiDiffCkt_Diffckt_generated_100_results.pkl"
    with open(data_path, 'rb') as f:
        bipartite_graph = pickle.load(f)
    G = bipartite_graph['G']
    for i, g in enumerate(G):
        try:
            netlist = Bipartite2Netlist(g, i, output_dir="netlists")
        except ValueError as e:
            print(f"Error in graph {i}: {e}")