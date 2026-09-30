import math
import re
import numpy as np
import cdspythonsrr
import cdspythonsrr.core.ocean as ocean

from pathlib import Path
import subprocess

SIMULATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = SIMULATION_DIR / "simulation_cache"
# TEST_BENCH_PATH = SIMULATION_DIR / "netlists" / "netlist_0.scs"
OUTPUT_PATH = REPO_ROOT / "Output"

MDL_PATH = SIMULATION_DIR / "testbench_templates" / "ac.mdl"
 
def run_spectre(test_bench_path: Path, output_dir: Path, mdl_path:Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "spectre.out"
    return subprocess.run(
        [
            "spectre",
            str(test_bench_path),
            "=mdle",
            str(mdl_path),
            "=log",
            str(log_path),
            "-format",
            "psfbin",
            "-raw",
            str(output_dir),
            "-outdir",
            str(output_dir),
        ],
        check=True,
    )


def psf_ac_readers(output_dir: Path):
    # Get results from ac simulation
    metrics = {
        'gain': None,
        'pm': None,
        'gbw': None,
    }
    pattern = re.compile(r'\b(gain|pm|gbw)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\b')
    with open(output_dir / "ac.measure", 'r', encoding='utf-8') as f:
        content = f.read()
        for match in pattern.finditer(content):
            var_name = match.group(1)
            var_value = float(match.group(2))
            if var_name == "pm":
                if var_value < -90:
                    metrics[var_name] = var_value + 180
                else:
                    metrics[var_name] = var_value
            else:
                metrics[var_name] = var_value
    return metrics


# if __name__ == "__main__":
#     # Test
#     spectre_process = run_spectre(TEST_BENCH_PATH, OUTPUT_PATH, MDL_PATH)
#     if spectre_process.returncode == 0:
#         res = psf_ac_readers(OUTPUT_PATH)
#         print(res)