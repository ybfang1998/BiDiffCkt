from __future__ import annotations
import csv
import json
import multiprocessing
import os
import pickle
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import argparse
import math
from pathlib import Path

import matplotlib
import optuna

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization.ocb_sim_fun import fom as calculate_fom
from optimization.ocb_sim_fun import pwr as calculate_pwr
from optimization.ocb_sim_fun import OCB_bipartite_graph_to_netlist, init_attr
from simulator import (
    MDL_PATH,
    psf_ac_readers,
    run_spectre,
)


SIMULATION_DIR = Path(__file__).resolve().parent
NETLIST_DIR = SIMULATION_DIR / "netlists"
OUTPUT_DIR = SIMULATION_DIR / "output"
FOM_PLOT_PATH = OUTPUT_DIR / "bo_fom_curve.png"
SIMULATION_CACHE_DIR = SIMULATION_DIR / "simulation_cache"
INVALID_FOM = -1.0

# [min, max, step]
r_range = [0.01, 1.01, 0.02] # M ohm
c_range = [0.1, 10.1, 0.2] # pF
gm_Range=[0.01,1.01,0.02] # mS
Gain_A_Range = [40, 80, 1]

constrain = {
    "gain": 85,
    "pm": 55,
    "gbw": 7e5,
    "pwr": 0.25
}


def objective(gain:float, gbw: float, pm:float, pwr: float) -> float:
    """Return the FOM value to maximize."""
    values = (gain, gbw, pm, pwr)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return INVALID_FOM
    if gain < constrain["gain"] or pm < constrain["pm"] or gbw < constrain["gbw"] or pwr > constrain["pwr"]:
        return INVALID_FOM

    return float(calculate_fom(gbw, pwr))


def plot_fom_curve(study: optuna.Study, output_path: str | Path = FOM_PLOT_PATH) -> Path:
    """Save the current best FOM curve as a PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = []
    ys = []
    best = -float("inf")
    for trial in study.trials:
        if trial.value is None:
            continue
        value = float(trial.value)
        if not math.isfinite(value):
            continue
        xs.append(trial.number)
        best = max(best, value)
        ys.append(best)

    plt.figure(figsize=(8, 4))
    if xs:
        plt.plot(xs, ys, color="tab:blue", linewidth=2)
        plt.scatter(xs[-1], ys[-1], color="tab:red", s=20)
    plt.xlabel("Trial")
    plt.ylabel("Best FOM")
    plt.title("Optuna BO FOM Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _quantize(value: float, bounds: list[float]) -> float:
    low, high, step = bounds
    steps = round((value - low) / step)
    snapped = low + steps * step
    return float(min(high, max(low, snapped)))


def _sample_parameter(trial: optuna.Trial, name: str, bounds: list[float], kind: str) -> float:
    low, high, step = bounds
    return trial.suggest_float(name, low, high, step=step)


def _parameter_name(device_idx: int, attr_name: str) -> str:
    return f"node_{device_idx}_{attr_name}"


def _build_parameter_updates(
    trial: optuna.Trial,
    attributes: list[dict[str, float]],
) -> list[dict[str, float]]:
    parameter_ranges = {
        "r": r_range,
        "c": c_range,
        "gm": gm_Range,
        "Gain_A": Gain_A_Range,
    }
    parameter_updates: list[dict[str, float]] = []

    for device_idx, device_attrs in enumerate(attributes):
        sampled_attrs: dict[str, float] = {}
        for attr_name in device_attrs:
            if attr_name in parameter_ranges:
                sampled_attrs[attr_name] = _sample_parameter(
                    trial,
                    _parameter_name(device_idx, attr_name),
                    parameter_ranges[attr_name],
                    attr_name,
                )
            else:
                raise ValueError(f"Unsupported attribute for BO sampling: {attr_name}")
        parameter_updates.append(sampled_attrs)

    return parameter_updates


def _netlist_paths(netlist_id: int) -> Path:
    return NETLIST_DIR / f"netlist_{netlist_id}.scs"


def _best_parameter_updates(
    attributes: list[dict[str, float]],
    best_params: dict[str, float],
) -> list[dict[str, float]]:
    parameter_updates: list[dict[str, float]] = []

    for device_idx, device_attrs in enumerate(attributes):
        best_attrs: dict[str, float] = {}
        for attr_name in device_attrs:
            best_attrs[attr_name] = float(best_params[_parameter_name(device_idx, attr_name)])
        parameter_updates.append(best_attrs)

    return parameter_updates


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def _create_run_cache(netlist_id: int, cache_root: str | Path) -> Path:
    """Keep separate netlists and repeated BO runs from sharing any outputs."""
    netlist_cache = Path(cache_root).resolve() / f"netlist_{netlist_id}"
    netlist_cache.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run_", dir=netlist_cache))


def _simulate_parameters(
    g,
    netlist_id: int,
    parameter_updates: list[dict[str, float]],
    output_dir: Path,
) -> dict[str, float]:
    """Archive the input, raw outputs, and metrics in a fresh simulation folder."""
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "netlist_id": netlist_id,
        "parameters": parameter_updates,
        "fom": INVALID_FOM,
        "status": "failed",
    }

    try:
        gm_list = OCB_bipartite_graph_to_netlist(
            g, netlist_id, output_dir=output_dir, attributes=parameter_updates
        )
        netlist_path = output_dir / f"netlist_{netlist_id}.scs"
        mdl_path = output_dir / MDL_PATH.name
        shutil.copy2(MDL_PATH, mdl_path)

        spectre_process = run_spectre(netlist_path, output_dir, mdl_path)
        if spectre_process.returncode != 0:
            raise RuntimeError(f"AC simulation failed with exit code {spectre_process.returncode}.")
        # Read exactly the directory passed to this simulation, never a shared path.
        metrics = {
            key: float(value) if value is not None and math.isfinite(float(value)) else None
            for key, value in psf_ac_readers(output_dir).items()
        }
        metrics["pwr"] = float(calculate_pwr(gm_list))
        metrics["fom"] = objective(**metrics)
        result.update(metrics)
        result["status"] = "completed"
        return metrics
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_json(output_dir / "result.json", result)


def rerun_best_parameters(
    g,
    netlist_id: int,
    best_params: dict[str, float],
    cache_dir: str | Path | None = None,
) -> dict[str, float]:
    """Rerun AC with the best sizing and write the verified netlist back."""
    if cache_dir is None:
        cache_dir = _create_run_cache(netlist_id, SIMULATION_CACHE_DIR) / "best"
    cache_dir = Path(cache_dir).resolve()
    parameter_updates = _best_parameter_updates(init_attr(g), best_params)
    metrics = _simulate_parameters(g, netlist_id, parameter_updates, cache_dir)
    NETLIST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_dir / f"netlist_{netlist_id}.scs", _netlist_paths(netlist_id))
    return metrics


def optimize_netlist(
    g,
    netlist_id: int,
    n_trials: int = 1000,
    study_name: str = "ocb_bo_fom",
    enable_plot: bool = True,
    cache_root: str | Path = SIMULATION_CACHE_DIR,
):
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    # Read the sizing attr
    attributes = init_attr(g)
    run_cache = _create_run_cache(netlist_id, cache_root)
    plot_path = run_cache / "bo_fom_curve.png"

    study = optuna.create_study(
        direction="maximize", study_name=f"{study_name}_netlist_{netlist_id}"
    )
    study.set_user_attr("netlist_id", netlist_id)
    study.set_user_attr("cache_dir", str(run_cache))

    def trial_objective(trial: optuna.Trial) -> float:
        parameter_updates = _build_parameter_updates(trial, attributes)
        trial_cache = run_cache / f"trial_{trial.number:05d}"
        trial.set_user_attr("cache_dir", str(trial_cache))
        try:
            metrics = _simulate_parameters(g, netlist_id, parameter_updates, trial_cache)
            for name, value in metrics.items():
                trial.set_user_attr(name, value)
            return metrics["fom"]
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            trial.set_user_attr("fom", INVALID_FOM)
            return INVALID_FOM

    def _plot_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if not enable_plot:
            return
        plot_fom_curve(study, plot_path)

    # Parallelism is across netlists; each netlist's BO trials remain sequential.
    study.optimize(trial_objective, n_trials=n_trials, n_jobs=1, callbacks=[_plot_callback])

    if enable_plot:
        plot_fom_curve(study, plot_path)

    _write_json(run_cache / "study.json", {
        "netlist_id": netlist_id,
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_fom": study.best_value,
        "best_params": study.best_params,
        "trials": [
            {"number": trial.number, "value": trial.value,
             "params": trial.params, "user_attrs": trial.user_attrs}
            for trial in study.trials
        ],
    })
    best_parameters = rerun_best_parameters(
        g, netlist_id, study.best_params, cache_dir=run_cache / "best"
    )
    print(f"Netlist {netlist_id}: best rerun FOM={best_parameters['fom']}; cache={run_cache}", flush=True)
    return study, best_parameters


def _optimize_worker(g, netlist_id: int, n_trials: int, enable_plot: bool, cache_root: Path):
    """Return only the results needed by the parent process for aggregation."""
    study, best_parameters = optimize_netlist(
        g, netlist_id, n_trials=n_trials, enable_plot=enable_plot, cache_root=cache_root
    )
    best_parameters["id"] = netlist_id
    return best_parameters, study.best_params


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-name", type=str, default="BiDiffckt_OCB_fom")
    parser.add_argument("--n-trials", type=_positive_int, default=1000)
    parser.add_argument("--n-jobs", type=_positive_int, default=50,
                        help="Number of netlists to optimize in parallel (default: 50)")
    parser.add_argument("--n-netlists", type=_positive_int, default=50,
                        help="Optimize the first N dataset entries (default: 50)")
    parser.add_argument("--cache-dir", type=Path, default=SIMULATION_CACHE_DIR)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    generated_circuit_path = SIMULATION_DIR / "data/BiDiffCkt_OCB101_generated_50_results.pkl"
    with generated_circuit_path.open("rb") as f:
        graphs = pickle.load(f)["G"]

    tasks = []
    for netlist_id, g in enumerate(graphs):
        if netlist_id >= args.n_netlists:
            break
        if not _netlist_paths(netlist_id).exists():
            print(f"Skipping netlist {netlist_id}: {_netlist_paths(netlist_id)} does not exist.")
            continue
        # Netlist generation only needs array operations; avoid importing PyTorch
        # and transferring tensor storage in every spawned worker.
        worker_graph = {
            key: value.detach().cpu().numpy() if hasattr(value, "detach") else value
            for key, value in g.items()
        }
        tasks.append((netlist_id, worker_graph))
    if not tasks:
        parser.error("No existing netlists found to optimize.")

    result_file_name = Path(args.test_name + "_results.csv")
    result_para_path = Path(args.test_name + "best_results_para.json")
    result_file_name.parent.mkdir(parents=True, exist_ok=True)
    result_para_path.parent.mkdir(parents=True, exist_ok=True)
    result_para_dict = {}
    failed_ids = []
    n_jobs = min(args.n_jobs, len(tasks))
    cache_root = args.cache_dir.resolve()
    print(f"Optimizing {len(tasks)} netlists with {n_jobs} workers; cache={cache_root}", flush=True)
    # Prevent each spawned worker from starting a full machine's BLAS threads.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    with ProcessPoolExecutor(
        max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {
            executor.submit(_optimize_worker, g, netlist_id, args.n_trials, args.plot, cache_root): netlist_id
            for netlist_id, g in tasks
        }
        for future in as_completed(futures):
            netlist_id = futures[future]
            try:
                best_parameters, best_sizing = future.result()
            except Exception as exc:
                failed_ids.append(netlist_id)
                print(f"Netlist {netlist_id} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            # Only the parent writes shared summary files.
            write_header = not result_file_name.exists() or result_file_name.stat().st_size == 0
            with result_file_name.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["gain", "gbw", "pm", "pwr", "fom", "id"])
                if write_header:
                    writer.writeheader()
                writer.writerow(best_parameters)
            result_para_dict[str(netlist_id)] = best_sizing
            _write_json(result_para_path, result_para_dict)

    if failed_ids:
        print(f"Failed netlists: {sorted(failed_ids)}. See each netlist's simulation cache for details.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
