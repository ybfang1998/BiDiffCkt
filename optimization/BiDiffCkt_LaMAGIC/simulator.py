"""Simulate netlists from datasets/345comp_base.json using ngspice.

    result = simulate_record(record)
    result = simulate_netlist(record["netlist"], record["duty_cycle"])

Use simulate_record to include nondefault component values and operating settings.
Units follow the dataset: vout in volts, eff as a fraction, freq in MHz,
capacitances in uF, inductances in uH, and resistances in ohms. The legacy
switch polarity, 4 ms transient, last-50-cycle averaging and efficiency formula
Pout / (Pin + 0.01) are preserved for comparison with the stored labels.
"""
import csv
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import pickle
import pickle
import re
import shutil
import tempfile
import random

import numpy as np
from tqdm import tqdm
import subprocess
import sys
import os
dir_path = os.getcwd()
sys.path.append(dir_path)

simulate_param = {"Duty_Cycle":[0.1, 0.3, 0.5, 0.7, 0.9],
    "Frequency":[1],
    "Rout": [50],
    "Vin": [100],
    "Cout": [10],
    "Ra": [100000],
    "Rb": [1],
    "Rin": [0.1],
    "C": [10],
    "L": [100]
}

root = Path(__file__).resolve().parents[1]
# Sample HSPICE Path
SIMULATOR_PATH = root / "Spice64" / "bin" / "ngspice_con.exe"

def convert_netlist_cki(path, netlist, duty_cycle, L=None, *, params=None, component_values=None):
    """Write a deck, optionally using per-call settings and individual C/L values."""
    params = simulate_param if params is None else params
    component_values = {} if component_values is None else component_values
    file = open(path, 'w')

    # if 'Ra0' in dn:
    #     Ron = pv[dn['Ra0']]
    #     Roff = pv[dn['Rb0']]
    # else:
    Ron = params["Ra"][0]
    Roff = params["Rb"][0]

    prefix = [
        ".title buck.cki",
        ".model MOSN NMOS level=8 version=3.3.0",
        ".model MOSP PMOS level=8 version=3.3.0",
        ".model MySwitch SW (Ron=%s Roff=%s vt=%s)" % (Ron, Roff, params["Vin"][0] / 2),
        ".PARAM vin=%s rin=%s rout=%s cout=%su freq=%sMeg D=%s" % (
            params["Vin"][0], params["Rin"][0], params["Rout"][0], params["Cout"][0], params["Frequency"][0], duty_cycle),
        "\n",
        "*input*",
        "Vclock1 gate_a 0 PULSE (0 {vin} 0 1n 1n %su %su)" % (
            1 / params["Frequency"][0] * duty_cycle, 1 / params["Frequency"][0]),
        "Vclock2 gate_b 0 PULSE ({vin} 0 0 1n 1n %su %su)" % (
            1 / params["Frequency"][0] * duty_cycle, 1 / params["Frequency"][0]),

        "Vin IN_exact 0 dc {vin} ac 1",
        "Rin IN_exact IN {rin}",
        "Rout OUT 0 {rout}",
        "Cout OUT 0 {cout}"
        "\n"]

    sufix = ["\n",
             ".save all",
             # ".save i(vind)",
             ".control",
             # "tran %su 4000u" %(1/simulate_param["Frequency"][0]/10),
             # "tran 1n 2000u",
             "tran 10n 4000u",
             "print V(OUT)",
             "print V(IN_exact,IN)",
             ".endc",
             ".end",
             ]

    file.write("\n".join(prefix) + '\n')
    file.write("*topology*" + '\n')

    line = ''
    print(netlist)
    for x in netlist:
        if 'S' == x[0]:
            if 'a' == x[1]:
                line = x + ' ' + netlist[x][0] + ' ' + netlist[x][1] + ' gate_a gate_b MySwitch'
            elif 'b' == x[1]:
                line = x + ' ' + netlist[x][0] + ' ' + netlist[x][1] + ' gate_b gate_a MySwitch'
        elif x[0] == 'C':
            line = x + ' ' + netlist[x][0] + ' ' + netlist[x][1] + ' ' + str(component_values.get(x, params["C"][0])) + 'u'
        elif x[0] == 'L':
            if L is not None:
                line = x + ' ' + netlist[x][0] + ' ' + netlist[x][1] + ' ' + str(L) + 'u'
            else:
                line = x + ' ' + netlist[x][0] + ' ' + netlist[x][1] + ' ' + str(component_values.get(x, params["L"][0])) + 'u'
        elif x[0] == 'R':
            if 'a' == x[1]:
                line = x +' '+ netlist[x][0] +' '+ netlist[x][1] +' '+ str(params["Ra"][0])
            elif 'b' == x[1]:
                line = x +' '+ netlist[x][0] +' '+ netlist[x][1] +' '+ str(params["Rb"][0])
        else:
            return 0

        line = line + '\n'
        file.write(line)

    file.write("\n".join(sufix) + '\n')
    file.close()
    return

def simulate(path):
    circuit = Path(path).resolve()
    output = circuit.with_suffix(".simu")

    with output.open("w") as log:
        result = subprocess.run(
            [str(SIMULATOR_PATH), "-b", str(circuit)],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=500,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"ngspice failed with exit code {result.returncode}; see log: {output}"
        )

    print('simulation finish')
    return False
        

def calculate_efficiency(path, killed, *, params=None):
    """Read simulation output using the same operating settings as the deck."""
    params = simulate_param if params is None else params
    simu_file = path[:-3] + 'simu'
    input_voltage = params["Vin"][0]
    freq = params["Frequency"][0] * 1000000
    rin = params["Rin"][0]
    rout = params["Rout"][0]
    V_in = input_voltage
    V_out = []
    I_in = []
    I_out = []
    time = []

    stable_ratio = 0.01

    cycle = 1 / freq
    # count = 0

    read_V_out, read_I_out, read_I_in = False, False, False
    with open(simu_file, 'r') as file:
        for line in file:
            # print(line)
            if "Transient solution failed" in line:
                return {'result_valid': False,
                        'efficiency': -1,
                        'Vout': -500,
                        'Iin': -1,
                        'error_msg': 'transient_simulation_failure'}
            if "Index   time            v(out)" in line and not read_V_out:
                read_V_out = True
                read_I_in = False
                continue
            elif "Index   time            v(in_exact,in)" in line and not read_I_in:
                read_V_out = False
                read_I_in = True
                continue

            tokens = line.split()

            # print(tokens)
            if len(tokens) == 3 and tokens[0] != "Index":
                if read_V_out:
                    time.append(float(tokens[1]))
                    try:
                        V_out.append(float(tokens[2]))
                        I_out.append(float(tokens[2]) / rout)
                    except:
                        print('Vout token error')
                elif read_I_in:
                    try:
                        I_in.append(float(tokens[2]) / rin)
                    except:
                        print('Iin token error')

    print(len(V_out), len(I_in), len(I_out))

    # print(len(V_out),len(I_out),len(I_in),len(time))
    if len(V_out) == len(I_in) == len(I_out) == len(time):
        pass
    else:
        print("don't match")
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -500,
                'Iin': -1,
                'error_msg': 'output_is_not_aligned'}

    if not V_out or not I_in or not I_out:
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -500,
                'Iin': -1,
                'error_msg': 'missing_output_type'}

    # print(I_out, I_in)
    end = len(V_out) - 1
    start = len(V_out) - 1
    print(cycle, start)
    while start >= 0:
        if time[end] - time[start] >= 50 * cycle:
            break
        start -= 1

    if start == -1:
        print("duration less than one cycle")
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -500,
                'Iin': -1,
                'error_msg': 'less_than_one_cycle'}
    mid = int((start + end) / 2)

    # print(start, end,time[end] - time[start])
    P_in = sum([(I_in[x] + I_in[x + 1]) / 2 * (V_in + V_in) / 2 *
                (time[x + 1] - time[x])
                for x in range(start, end)]) / (time[end] - time[start])

    P_out = sum([(I_out[x] + I_out[x + 1]) / 2 * (V_out[x] + V_out[x + 1]) / 2 *
                 (time[x + 1] - time[x])
                 for x in range(start, end)]) / (time[end] - time[start])

    V_out_ave = sum([(V_out[x] + V_out[x + 1]) / 2 * (time[x + 1] - time[x])
                     for x in range(start, end)]) / (time[end] - time[start])

    V_out_ave_1 = np.average(V_out[start:mid])

    V_out_ave_2 = np.average(V_out[mid:end - 1])

    I_in_ave = sum([(I_out[x] + I_out[x + 1]) / 2 * (time[x + 1] - time[x])
                    for x in range(start, end)]) / (time[end] - time[start])

    V_std = np.std(V_out[start:end - 1])

    # print('P_out, P_in', P_out, P_in)
    # if P_in == 0:
    if P_in < 0.001 and P_in > -0.001:
        P_in = 0
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -500,
                'Iin': -1,
                'error_msg': 'power_in_is_zero'}
    if P_out < 0.001 and P_out > -0.001:
        P_out = 0

    stable_flag = (abs(V_out_ave_1 - V_out_ave_2) <= max(abs(V_out_ave * stable_ratio), V_in / 200))

    # stable_flag = 1;

    eff = P_out / (P_in + 0.01)
    Vout = V_out_ave;
    Iin = I_in_ave;

    result = {'result_valid': (0 <= eff <= 1) and stable_flag,
              'efficiency': eff,
              'Vout': Vout,
              'Iin': Iin,
              'error_msg': 'None'}

    flag_candidate = 0

    if stable_flag == 0:
        result['error_msg'] = 'output_has_not_settled'

    elif eff < 0:
        result['error_msg'] = 'efficiency_is_less_than_zero'

    elif eff > 1:
        result['error_msg'] = 'efficiency_is_greater_than_one'
    elif (V_out_ave < 0.7 * input_voltage or V_out_ave > 1.2 * input_voltage) and eff > 0.7:
        flag_candidate = 1
        print('Promising candidates')

    return result


DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets/345comp_base.json"
# Five-component, inverting, nondefault settings, four- and three-component cases.
DEFAULT_INDICES = (0, 13, 67, 495, 118441, 130586)


_SPICE_NUMBER = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)\s*$"
)
_SPICE_MULTIPLIERS = {
    "": 1.0, "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3,
    "mil": 25.4e-6, "m": 1e-3, "u": 1e-6, "n": 1e-9,
    "p": 1e-12, "f": 1e-15,
}


def _positive(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return number


def normalize_netlist(netlist):
    """Convert [[device, terminal1, terminal2], ...] to the legacy dictionary.

    Also accepts an existing {device: [terminal1, terminal2]} dictionary.
    Reject malformed components instead of silently discarding terminals.
    """
    rows = ([name, *terminals] for name, terminals in netlist.items()) if isinstance(netlist, dict) else netlist
    normalized = {}
    seen = set()
    aliases = {"VIN": "IN", "VOUT": "OUT", "GND": "0"}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"Expected [device, terminal1, terminal2], got {row!r}")
        name, *terminals = row
        if not isinstance(name, str) or not re.fullmatch(r"(?:Sa|Sb|C|L|Ra|Rb)\d+", name):
            raise ValueError(f"Unsupported component name: {name!r}")
        if name.lower() in seen:
            raise ValueError(f"Duplicate component: {name}")
        seen.add(name.lower())
        nodes = [aliases.get(str(node).upper(), str(node)) for node in terminals]
        if any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", node) for node in nodes):
            raise ValueError(f"Invalid node name in {row!r}")
        if nodes[0].lower() == nodes[1].lower():
            raise ValueError(f"Component {name} has identical terminals")
        normalized[name] = nodes
    if not normalized:
        raise ValueError("netlist must contain at least one component")
    return normalized


def simulate_netlist(netlist, duty_cycle, *, edge_attr0=None, rout=50, cout=10,
                     freq=1, vin=100, rin=0.1, path=None):
    """Run a dataset netlist and return vout, eff, result_valid and error_msg.

    netlist and duty_cycle are sufficient for the default dataset parameters.
    edge_attr0 supplies physical component values (not normalized edge_attr).
    It uses slots 0/1 for the shared Sa/Sb switch resistances, 2 for C and 3
    for L. When supplied, it must include every netlist device.

    path optionally retains the .cki and .simu files for inspection. Otherwise
    each call uses a temporary directory and removes its potentially large log.
    Launch errors, nonzero exit codes and timeouts propagate as exceptions;
    invalid electrical results are returned with result_valid=False.
    """
    normalized, duty_cycle, params, component_values = _prepare_simulation(
        netlist, duty_cycle, edge_attr0=edge_attr0, rout=rout, cout=cout,
        freq=freq, vin=vin, rin=rin,
    )

    def run(circuit):
        convert_netlist_cki(str(circuit), normalized, duty_cycle, params=params,
                            component_values=component_values)
        killed = simulate(str(circuit))
        result = calculate_efficiency(str(circuit), killed, params=params)
        return {"vout": float(result["Vout"]), "eff": float(result["efficiency"]),
                "result_valid": bool(result["result_valid"]),
                "error_msg": result["error_msg"]}

    if path is not None:
        circuit = _circuit_path(path)
        return run(circuit)
    with tempfile.TemporaryDirectory(prefix="lamagic_sim_") as folder:
        return run(Path(folder) / "circuit.cki")


def _prepare_simulation(netlist, duty_cycle, *, edge_attr0=None, rout=50, cout=10,
                        freq=1, vin=100, rin=0.1):
    """Validate and prepare settings shared by simulation and export-only calls."""
    normalized = normalize_netlist(netlist)
    duty_cycle = float(duty_cycle)
    if not math.isfinite(duty_cycle) or not 0 < duty_cycle < 1:
        raise ValueError("duty_cycle must be between 0 and 1 (exclusive)")

    # Per-call parameters keep later simulations independent of this record.
    params = deepcopy(simulate_param)
    for key, value in {"Rout": rout, "Cout": cout, "Frequency": freq,
                       "Vin": vin, "Rin": rin}.items():
        params[key] = [_positive(value, key)]
    component_values = {}
    if edge_attr0 is not None:
        missing = normalized.keys() - edge_attr0.keys()
        if missing:
            raise ValueError(f"edge_attr0 is missing components: {sorted(missing)}")
        for name, attr in edge_attr0.items():
            if not isinstance(attr, (list, tuple)) or len(attr) != 6:
                raise ValueError(f"edge_attr0[{name!r}] must have six values")
        for prefix, slot, parameter in [("Sa", 0, "Ra"), ("Sb", 1, "Rb")]:
            values = {_positive(attr[slot], name) for name, attr in edge_attr0.items()
                      if name.startswith(prefix)}
            if len(values) > 1:
                raise ValueError(f"The shared switch model requires a single {prefix} resistance")
            if values:
                params[parameter] = [values.pop()]
        for name in normalized:
            if name.startswith(("C", "L")):
                slot = 2 if name.startswith("C") else 3
                component_values[name] = _positive(edge_attr0[name][slot], name)

    return normalized, duty_cycle, params, component_values


def _circuit_path(path):
    circuit = Path(path).resolve()
    if circuit.suffix != ".cki":
        raise ValueError("path must end in .cki")
    circuit.parent.mkdir(parents=True, exist_ok=True)
    return circuit


def write_netlist(netlist, duty_cycle, path, **settings):
    """Write a complete ngspice .cki deck without running ngspice; return its path.

    settings accepts the same component/operating parameters as simulate_netlist.
    """
    normalized, duty_cycle, params, values = _prepare_simulation(
        netlist, duty_cycle, **settings,
    )
    circuit = _circuit_path(path)
    with redirect_stdout(io.StringIO()):
        convert_netlist_cki(str(circuit), normalized, duty_cycle, params=params,
                            component_values=values)
    return circuit


class GraphConversionError(ValueError):
    """A graph cannot be represented by the supported two-terminal netlist."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def bipartite_arrays(graph):
    """Read U/V type-ID vectors and a binary E[U_index, V_index] matrix.

    Accept lists, NumPy arrays and CPU/GPU torch tensors. No isolated nodes are
    stripped and no edge thresholding or graph repair is performed.
    """
    if not isinstance(graph, dict) or not {"U", "V", "E"} <= graph.keys():
        raise GraphConversionError("invalid_format", "Each graph must contain U, V and E")
    arrays = []
    for key in ("U", "V", "E"):
        value = graph[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        try:
            arrays.append(np.asarray(value))
        except (TypeError, ValueError) as exc:
            raise GraphConversionError("invalid_format", f"Invalid {key} array") from exc
    u, v, edges = arrays
    if u.ndim != 1 or v.ndim != 1 or edges.shape != (len(u), len(v)):
        raise GraphConversionError("invalid_shape", "Expected U[n], V[m], E[n, m]")
    for key, labels in (("U", u), ("V", v)):
        if (labels.dtype.kind not in "iuf" or not np.all(np.isfinite(labels))
                or not np.all(labels == np.floor(labels))):
            raise GraphConversionError("invalid_type_id", f"{key} must contain integer type IDs")
    if not np.all((edges == 0) | (edges == 1)):
        raise GraphConversionError("invalid_edges", "E must contain only 0 and 1")
    return u, v, edges.astype(bool)


def graph_to_netlist(graph, *, u_to_id, v_to_id):
    """Convert one U/V/E graph into [[component, node1, node2], ...].

    Requirements: known type IDs, exactly one ground/IN/OUT V node, and exactly
    two distinct V neighbors per U device. Connectivity is a separate metric:
    disconnected circuits may be representable, and unused V junctions produce
    no component rows. Conversion does not establish ngspice convergence.
    """
    u, v, edges = bipartite_arrays(graph)
    u_types = {value: name for name, value in u_to_id.items()}
    v_types = {value: name for name, value in v_to_id.items()}
    if len(u_types) != len(u_to_id) or len(v_types) != len(v_to_id):
        raise ValueError("Type mappings must use unique IDs")
    if not {"0", "IN", "OUT", "net"} <= v_to_id.keys():
        raise ValueError("v_to_id must define 0, IN, OUT and net")
    if not len(u):
        raise GraphConversionError("empty_netlist", "Graph has no U components")
    unknown_u = sorted(set(u.tolist()) - u_types.keys())
    unknown_v = sorted(set(v.tolist()) - v_types.keys())
    if unknown_u or unknown_v:
        raise GraphConversionError("unknown_type", f"Unknown type IDs: U={unknown_u}, V={unknown_v}")
    for terminal in ("0", "IN", "OUT"):
        count = int(np.count_nonzero(v == v_to_id[terminal]))
        if count != 1:
            raise GraphConversionError("special_node_count", f"Expected one {terminal} V node, found {count}")
    degrees = edges.sum(axis=1)
    bad_degrees = {i: int(d) for i, d in enumerate(degrees) if d != 2}
    if bad_degrees:
        raise GraphConversionError("device_degree_not_two", f"U nodes must have two terminals: {bad_degrees}")

    nodes = [f"net_{i}" if v_types[label] == "net" else v_types[label]
             for i, label in enumerate(v)]
    counters = {}
    rows = []
    for i, label in enumerate(u):
        kind = u_types[label]
        component_index = counters.get(kind, 0)
        counters[kind] = component_index + 1
        a, b = np.flatnonzero(edges[i])
        rows.append([f"{kind}{component_index}", nodes[a], nodes[b]])
    try:
        normalize_netlist(rows)
    except ValueError as exc:
        raise GraphConversionError("invalid_netlist", str(exc)) from exc
    return rows


def convert_graphs_to_netlists(graphs, *, u_to_id, v_to_id, generate_netlist=False,
                               output_dir=None, duty_cycle=0.5):
    """Return how many graphs can be converted; optionally export their .cki files.

    With generate_netlist=False no directory or file is created and ngspice is
    never launched. With True, output_dir is required, and convertible graphs
    are saved as graph_<original zero-based index>.cki. Existing same-name files
    are replaced. Both modes return the same integer count.

    Generated U/V/E graphs have no duty-cycle prediction. The export default
    is explicitly 0.5; callers can supply another value. Connectivity is not
    used as a conversion filter (see graph_to_netlist).
    """
    if generate_netlist:
        if output_dir is None:
            raise ValueError("output_dir is required when generate_netlist=True")
        output_dir = Path(output_dir)
    count = 0
    for index, graph in enumerate(graphs):
        try:
            netlist = graph_to_netlist(graph, u_to_id=u_to_id, v_to_id=v_to_id)
        except GraphConversionError:
            continue
        if generate_netlist:
            duty_cycle = graph["duty_cycle"] if "duty_cycle" in graph else duty_cycle
            duty_cycle = float(duty_cycle)
            if not math.isfinite(duty_cycle) or not 0 < duty_cycle < 1:
                raise ValueError("duty_cycle must be between 0 and 1 (exclusive)")
            write_netlist(netlist, duty_cycle, output_dir / f"graph_{index:04d}.cki")
        count += 1
    return count


def simulate_record(record, *, path=None):
    """Simulate one JSON record; vout/eff labels are never read by this method."""
    return simulate_netlist(
        record["netlist"], record["duty_cycle"],
        edge_attr0=record.get("edge_attr0"), rout=record.get("rout", 50),
        cout=record.get("cout", 10), freq=record.get("freq", 1), path=path,
    )


def _parse_spice_number(value):
    """Convert a SPICE number such as 1Meg or 10u to a float."""
    match = _SPICE_NUMBER.fullmatch(value)
    if not match:
        raise ValueError(f"Unsupported SPICE numeric value: {value!r}")
    number, suffix = match.groups()
    suffix = suffix.lower()
    if suffix.startswith("meg"):
        multiplier = _SPICE_MULTIPLIERS["meg"]
    elif suffix.startswith("mil"):
        multiplier = _SPICE_MULTIPLIERS["mil"]
    else:
        multiplier = _SPICE_MULTIPLIERS.get(suffix[:1])
    if multiplier is None:
        raise ValueError(f"Unsupported SPICE suffix in {value!r}")
    return float(number) * multiplier


def _simulation_params_from_cki(path):
    """Read operating values needed by calculate_efficiency from a .cki deck."""
    params = deepcopy(simulate_param)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    assignments = {key.lower(): value for key, value in re.findall(
        r"(?i)\b(vin|rin|rout|freq)\s*=\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[A-Za-z]*)",
        text,
    )}
    key_map = {"vin": "Vin", "rin": "Rin", "rout": "Rout"}
    for source_key, target_key in key_map.items():
        if source_key in assignments:
            params[target_key] = [_positive(
                _parse_spice_number(assignments[source_key]), target_key,
            )]
    if "freq" in assignments:
        frequency_hz = _positive(_parse_spice_number(assignments["freq"]), "Frequency")
        params["Frequency"] = [frequency_hz / 1e6]
    return params


def simulate_netlist_folder(netlist_dir, csv_path, *, pattern="*.cki", verbose=True):
    """Simulate matching netlists in a folder and write their results to CSV.

    The CSV columns are netlist_id, voltage_conversion_ratio and efficiency.
    netlist_id is the input filename without its extension. Voltage conversion
    ratio is Vout/Vin. Failed or electrically invalid simulations are recorded
    as NaN so one bad circuit does not stop a long batch.

    Each deck is copied to a temporary directory before simulation, preventing
    large .simu logs from accumulating beside the input netlists. Consequently,
    the input decks should be self-contained, as generated by write_netlist.
    """
    netlist_dir = Path(netlist_dir).resolve()
    if not netlist_dir.is_dir():
        raise NotADirectoryError(f"Netlist directory does not exist: {netlist_dir}")
    netlist_paths = sorted(
        (path for path in netlist_dir.glob(pattern) if path.is_file()),
        key=lambda path: path.name.lower(),
    )
    if not netlist_paths:
        raise ValueError(f"No netlists matching {pattern!r} found in {netlist_dir}")

    csv_path = Path(csv_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["netlist_id", "voltage_conversion_ratio", "efficiency"]
    rows = []
    with tempfile.TemporaryDirectory(prefix="lamagic_batch_sim_") as folder, \
            csv_path.open("w", newline="", encoding="utf-8") as output:
        scratch_cki = Path(folder) / "circuit.cki"
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for index, source_cki in tqdm(enumerate(netlist_paths, start=1), total=len(netlist_paths)):
            netlist_id = source_cki.stem
            try:
                params = _simulation_params_from_cki(source_cki)
                shutil.copyfile(source_cki, scratch_cki)
                with redirect_stdout(io.StringIO()):
                    killed = simulate(str(scratch_cki))
                    result = calculate_efficiency(str(scratch_cki), killed, params=params)
                if not result["result_valid"]:
                    raise RuntimeError(result["error_msg"])
                voltage_conversion_ratio = float(result["Vout"]) / params["Vin"][0]
                efficiency = float(result["efficiency"])
            except Exception as exc:
                voltage_conversion_ratio = math.nan
                efficiency = math.nan
                if verbose:
                    print(f"[{index}/{len(netlist_paths)}] {netlist_id}: FAILED ({exc})",
                          flush=True)
            else:
                if verbose:
                    print(
                        f"[{index}/{len(netlist_paths)}] {netlist_id}: "
                        f"VCR={voltage_conversion_ratio:.8g}, efficiency={efficiency:.8g}",
                        flush=True,
                    )
            row = {
                "netlist_id": netlist_id,
                "voltage_conversion_ratio": voltage_conversion_ratio,
                "efficiency": efficiency,
            }
            writer.writerow(row)
            output.flush()
            rows.append(row)
    return rows


def load_dataset(path=DEFAULT_DATASET):
    """Load the JSON array once, then reuse its records for multiple simulations."""
    with Path(path).open(encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list):
        raise ValueError("Expected a JSON array of circuit records")
    return records


def compare_record(record, *, path=None, vout_atol=0.1, eff_atol=0.002, rtol=0.001):
    """Compare actual outputs to vout/eff labels, allowing numerical variation.

    Each field passes when absolute error <= max(atol, rtol * max(|a|, |b|)).
    Default absolute tolerances are 0.1 V and 0.002 efficiency (0.2 percentage
    points); relative tolerance is 0.1%. A valid electrical result is required.
    The defaults are regression tolerances, not a guarantee for all records.
    """
    for name, value in [("vout_atol", vout_atol), ("eff_atol", eff_atol), ("rtol", rtol)]:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    expected = {key: float(record[key]) for key in ("vout", "eff")}
    actual = simulate_record(record, path=path)
    errors = {key: abs(actual[key] - expected[key]) for key in expected}
    matches = {key: math.isclose(actual[key], expected[key], rel_tol=rtol, abs_tol=atol)
               for key, atol in [("vout", vout_atol), ("eff", eff_atol)]}
    return {"expected": expected, "actual": actual, "absolute_error": errors,
            "tolerances": {"vout_atol": vout_atol, "eff_atol": eff_atol, "rtol": rtol},
            "matches": matches, "passed": actual["result_valid"] and all(matches.values())}


if __name__ == "__main__":
    Duty_Cycle = [0.1, 0.3, 0.5, 0.7, 0.9]
    with open("generated_data/BiDiffCkt_LaMAGIC_generated_1000_results.pkl", 'rb') as f:
        G = pickle.load(f)['G']
    # randomly allocate duty cycles to the graphs
    for graph in G:
        graph['duty_cycle'] = random.choice(Duty_Cycle)
    U_TO_ID = {"L": 0, "C": 1, "Sa": 2, "Sb": 3}
    V_TO_ID = {"0": 0, 'IN': 1, 'OUT': 2, 'net': 3}
    convert_graphs_to_netlists(G, u_to_id=U_TO_ID, v_to_id=V_TO_ID, generate_netlist=True, output_dir="netlists")
    simulate_netlist_folder("netlists", "netlists_1000_results.csv")
    # raise SystemExit(main())
