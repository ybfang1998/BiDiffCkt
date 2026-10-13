import math
import re
import numpy as np
import cdspythonsrr
import cdspythonsrr.core.ocean as ocean

from pathlib import Path
import subprocess

SIMULATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = SIMULATION_DIR / "simulation_cache"
# TEST_BENCH_PATH = REPO_ROOT / "Netlists" / "test_netlist_0.scs"
TEST_BENCH_AC_PATH = SIMULATION_DIR / "netlists" / "netlist_351983_ac.scs"
TEST_BENCH_STB_PATH = SIMULATION_DIR / "netlists" / "netlist_351983_stb.scs"

STB_MDL = SIMULATION_DIR / "testbench_templates" / "stb.mdl"
AC_MDL = SIMULATION_DIR / "testbench_templates" / "ac.mdl"

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

def psf_dc_readers(output_dir: Path, VDD=1.2):
    # Get Power
    supply_current = abs(ocean.getData(signal="V1:p", resultsName="dcOp-dc", resultsDir=output_dir))
    power = VDD * supply_current
    return power

def psf_ac_readers(output_dir: Path):
    # Get CMRR from ac simulation
    cmrr = ocean.getData(signal="Vout", resultsName="ac-ac", resultsDir=output_dir)
    freq = cmrr['x']
    mag_db = 20 * (np.log10(1/np.abs(cmrr['y'])))
    cmrr_100 = float(np.interp(100, freq, mag_db))
    return cmrr_100

def psf_xf_readers(output_dir: Path):
    # Get PSRR from xf simulation
    psrr = ocean.getData(signal="V1", resultsName="xf-xf", resultsDir=output_dir)
    freq = psrr['x']
    mag_db = 20 * (np.log10(1/np.abs(psrr['y'])))
    psrr_100 = float(np.interp(100, freq, mag_db))
    return psrr_100

def psf_stb_readers(output_dir: Path):
    o_waveform = ocean.getData(signal="loopGain", resultsName="stb-stb", resultsDir=output_dir)
    loop_gain_mag = np.abs(o_waveform['y'])
    freq = o_waveform['x']
    # Get Loop gain
    dc_gain_mag = float(np.interp(1, freq, loop_gain_mag))
    if dc_gain_mag == 0:
        dc_gain = 0.0
    else:
        dc_gain = 20 * np.log10(dc_gain_mag)
    # loop_gain = 20 * np.log10(loop_gain_mag)
    # dc_gain = float(np.interp(1, freq, loop_gain))

    # Get Phase Margin
    phase_margin = ocean.getData(signal="phaseMargin", resultsName="stb-margin.stb", resultsDir=output_dir)

    # Get GBW
    pattern = re.compile(r'\bgbw\s*=\s*([0-9eE\+\-\.]+)')
    gbw = None
    with open(output_dir / "stb.measure", 'r', encoding='utf-8') as f:
        content = f.read()
        match = pattern.search(content)
        if match:
            gbw = float(match.group(1))

    return dc_gain, gbw, phase_margin
