from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path


INSTANCE_LINE_RE = re.compile(r"^(I\d+)\s*\([^)]*\)\s+(\S+)", re.IGNORECASE)
INSTANCE_VALUE_RE = re.compile(r"^((I\d+)\s*\([^)]*\)\s+(\S+))(.*)$", re.IGNORECASE)
CAP_PARAM_RE = re.compile(r"\bc\s*=\s*([^\s]+)", re.IGNORECASE)
RES_PARAM_RE = re.compile(r"\br\s*=\s*([^\s]+)", re.IGNORECASE)
MOS_W_RE = re.compile(r"\bw\s*=\s*([^\s]+)", re.IGNORECASE)
MOS_L_RE = re.compile(r"\bl\s*=\s*([^\s]+)", re.IGNORECASE)
C_LOAD_RE = re.compile(r"^(C_Load\s*\(Vout\s+0\)\s+capacitor\s+)c\s*=\s*([^\s]+)", re.IGNORECASE)


def _classify_instance_target(target_name: str) -> str:
    """Map a Spectre instance target name to `mos`, `c`, or `r`."""
    normalized = target_name.strip().lower()

    if normalized == "capacitor":
        return "c"
    if normalized == "resistor":
        return "r"

    mos_keywords = (
        "nch",
        "pch",
        "nmos",
        "pmos",
        "diff_pair",
        "current_mirror",
        "bias_pair",
    )
    if any(keyword in normalized for keyword in mos_keywords):
        return "mos"

    raise ValueError(f"Unsupported device type `{target_name}`.")


def read_instance_device_types(netlist_path: str | Path) -> "OrderedDict[str, str]":
    """Read I0...I_N device categories from a testbench netlist.

    Returns:
        OrderedDict like {"I0": "mos", "I1": "mos", "I8": "c"}.
    """
    netlist_path = Path(netlist_path)
    device_types: "OrderedDict[str, str]" = OrderedDict()

    with netlist_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            match = INSTANCE_LINE_RE.match(line)
            if not match:
                continue

            instance_name, target_name = match.groups()
            device_types[instance_name] = _classify_instance_target(target_name)

    return device_types


def _format_spectre_value(value: float | int) -> str:
    """Format a numeric value into a compact Spectre-friendly string."""
    return f"{float(value):.12g}"


def _normalize_parameter_payload(raw_value) -> list[float]:
    """Normalize dict payloads like 1, [1], (L, W) into a float list."""
    if isinstance(raw_value, (list, tuple)):
        values = [float(v) for v in raw_value]
    else:
        values = [float(raw_value)]
    return values


def _extract_c_load_value(parameter_updates: dict) -> float | None:
    """Read the optional load capacitance from supported dict keys."""
    for key in ("C_Load", "C_loader", "c_load", "c_loader"):
        if key in parameter_updates:
            values = _normalize_parameter_payload(parameter_updates[key])
            if len(values) != 1:
                raise ValueError(f"`{key}` must provide exactly one capacitance value.")
            return values[0]
    return None


def _replace_named_param(param_re: re.Pattern[str], text: str, value: float | int) -> str:
    """Replace a named Spectre parameter while preserving the original key case."""
    def repl(match: re.Match[str]) -> str:
        matched_text = match.group(0)
        key = matched_text.split("=", 1)[0].strip()
        return f"{key}={_format_spectre_value(value)}"

    updated_text, count = param_re.subn(repl, text, count=1)
    if count == 0:
        raise ValueError(f"Could not find parameter pattern `{param_re.pattern}` in: {text}")
    return updated_text


def _update_instance_line(line: str, parameter_updates: dict[str, object]) -> str:
    """Update one netlist instance line if it appears in the provided dict."""
    match = INSTANCE_VALUE_RE.match(line)
    if not match:
        return line

    prefix, instance_name, target_name, suffix = match.groups()
    if instance_name not in parameter_updates:
        return line

    device_type = _classify_instance_target(target_name)
    values = _normalize_parameter_payload(parameter_updates[instance_name])

    if device_type == "mos":
        if len(values) != 2:
            raise ValueError(f"`{instance_name}` expects [L, W] for MOS updates.")
        length, width = values
        updated_suffix = _replace_named_param(MOS_L_RE, suffix, length)
        updated_suffix = _replace_named_param(MOS_W_RE, updated_suffix, width)
        return f"{prefix}{updated_suffix}"

    if device_type == "c":
        if len(values) != 1:
            raise ValueError(f"`{instance_name}` expects a single capacitance value.")
        updated_suffix = _replace_named_param(CAP_PARAM_RE, suffix, values[0])
        return f"{prefix}{updated_suffix}"

    if device_type == "r":
        if len(values) != 1:
            raise ValueError(f"`{instance_name}` expects a single resistance value.")
        updated_suffix = _replace_named_param(RES_PARAM_RE, suffix, values[0])
        return f"{prefix}{updated_suffix}"

    return line


def update_netlist_parameters(netlist_path: str | Path, parameter_updates: dict[str, object]) -> Path:
    """Update instance parameters and optional C_Load in one netlist file.

    Parameter format:
        {"I0": [L, W], "I8": [C], "I9": [R], "C_Load": [load_cap]}
    """
    netlist_path = Path(netlist_path)
    c_load_value = _extract_c_load_value(parameter_updates)
    updated_lines: list[str] = []

    with netlist_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            updated_line = _update_instance_line(line, parameter_updates)

            if c_load_value is not None:
                c_load_match = C_LOAD_RE.match(updated_line.strip())
                if c_load_match:
                    prefix = c_load_match.group(1)
                    updated_line = f"{prefix}c={_format_spectre_value(c_load_value)}"

            updated_lines.append(updated_line)

    netlist_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return netlist_path


def update_ac_stb_netlists(
    parameter_updates: dict[str, object],
    ac_netlist_path: str | Path,
    stb_netlist_path: str | Path,
) -> tuple[Path, Path]:
    """Apply the same parameter update dict to both AC and STB netlists."""
    ac_path = update_netlist_parameters(ac_netlist_path, parameter_updates)
    stb_path = update_netlist_parameters(stb_netlist_path, parameter_updates)
    return ac_path, stb_path

def fom(gbw, c_load, pwr):
    return (gbw/1e6) * (c_load*1e12) / (pwr*1e3)


# if __name__ == "__main__":
#     sample_netlist = Path("netlists/netlist_2_ac.scs")
#     print(read_instance_device_types(sample_netlist))
    # update_ac_stb_netlists(
    # {
    #     "I0": [2e-6, 8e-6],
    #     "I2": [1.5e-6, 3e-6],
    #     "I8": [4e-12],
    #     "C_loader": [2.2e-12],
    # },
    # "simulation/netlists/netlist_1_ac.scs",
    # "simulation/netlists/netlist_1_stb.scs",
    # )
