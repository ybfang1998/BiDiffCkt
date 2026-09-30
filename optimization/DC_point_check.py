"""Check all CMOS in the saturation region (Spectre region == 2)."""

from pathlib import Path
import json

from cdspythonsrr.core.pysrr import srrDataBase
from simulator import REPO_ROOT, SIMULATION_DIR

OUTPUT_STB_PATH = REPO_ROOT / "Output" / "STB"


def check_dc_operating_points(results_dir=OUTPUT_STB_PATH):
    results_dir = Path(results_dir).resolve()
    if not (results_dir / "dcOpInfo.info").is_file():
        raise FileNotFoundError("no dcOpInfo.info found, run dcOpInfo in MDL")

    database = srrDataBase(str(results_dir))
    if "dcOpInfo-info" not in database.dataSetNameList():
        raise ValueError("PSF logFile could not find dcOpInfo-info")
    dataset = database.getDataSet("dcOpInfo-info")
    signals = dataset.evalSignalAll()
    names = {signal.getName() for signal in signals}
    mos_names = {name.rsplit(":", 1)[0] for name in names if name.endswith(":vgs")}
    regions = {
        signal.getName().rsplit(":", 1)[0]: signal.getValue()
        for signal in signals
        if signal.getName().endswith(":region")
        and signal.getName().rsplit(":", 1)[0] in mos_names
    }

    if not mos_names or set(regions) != mos_names:
        raise ValueError("no CMOS data")
    if any(value not in (0, 1, 2, 3, 4) for value in regions.values()):
        raise ValueError(f"invalid region: {regions}")

    regions = {name: int(value) for name, value in sorted(regions.items())}
    not_saturation = {name: region for name, region in regions.items() if region != 2}
    return {
        "results_dir": str(results_dir),
        "all_in_saturation": all(region == 2 for region in regions.values()),
        "mos_count": len(regions),
        "saturation_count": len(regions) - len(not_saturation),
        "regions": regions,
        "not_saturation": not_saturation,
    }


def all_mos_in_saturation(results_dir=OUTPUT_STB_PATH):
    return check_dc_operating_points(results_dir)["all_in_saturation"]