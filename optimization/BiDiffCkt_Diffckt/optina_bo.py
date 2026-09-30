from __future__ import annotations
import csv
import json
import os
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import math
import numpy as np
from pathlib import Path

import optuna
import optunahub
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bo_utils import read_instance_device_types, update_ac_stb_netlists
from bo_utils import fom as calculate_fom
from optimization.DC_point_check import all_mos_in_saturation
from simulator import (
    AC_MDL,
    STB_MDL,
    REPO_ROOT,
    psf_ac_readers,
    psf_stb_readers,
    psf_xf_readers,
    psf_dc_readers,
    run_spectre,
)

OUTPUT_AC_PATH = REPO_ROOT / "Output" / "AC"
OUTPUT_STB_PATH = REPO_ROOT / "Output" / "STB"


SIMULATION_DIR = Path(__file__).resolve().parent
NETLIST_DIR = SIMULATION_DIR / "netlists"
OUTPUT_DIR = SIMULATION_DIR / "output"
LOSS_PLOT_PATH = OUTPUT_DIR / "bo_loss_curve.png"
METRIC_NAMES = ("gain", "GBW", "PM", "CMRR", "PSRR", "PWR")
TARGET_SATISFIED_TOLERANCE = 1e-12

# [min, max, step]
mos_W_Range = [1e-6, 1e-4, 1e-6]
mos_L_Range = [1e-6, 1e-5, 1e-6]
r_Range = [1e3, 1e5, 1e3]
c_Range = [1e-12, 10e-12, 1e-12]

SPEC_UNNORMALIZE_FACTORS = {
    "gain": 100,
    "GBW": 10e6,
    "PM": 180,
    "CMRR": 100,
    "PSRR": 100,
    "PWR": 1e-3
}

normalised_targets_external = {
    "gain": 0.8,
    "GBW": 1.0,
    "PM": 0.31,
    "CMRR": 0.6,
    "PSRR": 0.7,
    "PWR": 0.2,
    "C_Load": 10e-12
}

normalised_targets_high = {
    "gain": 0.6,
    "GBW": 0.7,
    "PM": 0.31,
    "CMRR": 0.53,
    "PSRR": 0.53,
    "PWR": 0.35,
    "C_Load": 8e-12
}

normalised_targets_med = {
    "gain": 0.53,
    "GBW": 0.4,
    "PM": 0.28,
    "CMRR": 0.37,
    "PSRR": 0.37,
    "PWR": 0.65,
    "C_Load": 5e-12
}

normalised_targets_low = {
    "gain": 0.4,
    "GBW": 0.1,
    "PM": 0.25,
    "CMRR": 0.2,
    "PSRR": 0.2,
    "PWR": 1.0,
    "C_Load": 2e-12
}

INVALID_LOSS = 10.0
INVALID_FOM = -1.0

def target_satisfied(loss: float) -> bool:
    return math.isfinite(float(loss)) and float(loss) <= TARGET_SATISFIED_TOLERANCE


def _save_progress_checkpoint(
    study: optuna.Study,
    trial_count: int,
    output_checkpoint_file_path: str | Path,
) -> None:
    """Append the current best trial FOM to the progress CSV."""
    checkpoint_path = Path(output_checkpoint_file_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0

    with checkpoint_path.open("a", newline="", encoding="utf-8") as checkpoint_file:
        writer = csv.writer(checkpoint_file)
        if write_header:
            writer.writerow(["trial_count", "best_fom"])
        best_fom = study.best_trial.user_attrs.get("fom", INVALID_FOM)
        writer.writerow([trial_count, float(best_fom)])


def objective(gain, gbw, pm, cmrr, psrr, pwr, c_load, normalised_targets):
    values = (gain, gbw, pm, cmrr, psrr, pwr)
    if any((not math.isfinite(float(value))) or float(value) <= 0 for value in values):
        return INVALID_LOSS, INVALID_FOM

    gain_penalty = max(0.0, normalised_targets["gain"] - gain / SPEC_UNNORMALIZE_FACTORS["gain"])
    gbw_penalty = max(0.0, normalised_targets["GBW"] - gbw / SPEC_UNNORMALIZE_FACTORS["GBW"])
    pm_penalty = max(0.0, normalised_targets["PM"] - pm / SPEC_UNNORMALIZE_FACTORS["PM"])
    cmrr_penalty = max(0.0, normalised_targets["CMRR"] - cmrr / SPEC_UNNORMALIZE_FACTORS["CMRR"])
    psrr_penalty = max(0.0, normalised_targets["PSRR"] - psrr / SPEC_UNNORMALIZE_FACTORS["PSRR"])
    pwr_penalty = max(0.0, pwr / SPEC_UNNORMALIZE_FACTORS["PWR"] - normalised_targets["PWR"])

    loss = gain_penalty + gbw_penalty + pm_penalty + cmrr_penalty + psrr_penalty + pwr_penalty

    if gain_penalty + gbw_penalty + pm_penalty + cmrr_penalty + psrr_penalty + pwr_penalty <= TARGET_SATISFIED_TOLERANCE:
        fom = float(calculate_fom(gbw, c_load, pwr))
        loss -= fom/100
    else:
        fom = INVALID_FOM
    return loss, fom


def plot_loss_curve(study: optuna.Study, output_path: str | Path = LOSS_PLOT_PATH) -> Path:
    """Save the current best loss curve as a PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = []
    ys = []
    best = float("inf")
    for trial in study.trials:
        if trial.value is None:
            continue
        value = float(trial.value)
        if not math.isfinite(value):
            continue
        xs.append(trial.number)
        best = min(best, value)
        ys.append(best)

    plt.figure(figsize=(8, 4))
    if xs:
        plt.plot(xs, ys, color="tab:blue", linewidth=2)
        plt.scatter(xs[-1], ys[-1], color="tab:red", s=20)
    plt.xlabel("Trial")
    plt.ylabel("Best Loss")
    plt.title("Optuna BO Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_raw_metric_curves(
    study: optuna.Study,
    netlist_id: int,
    output_dir: str | Path = OUTPUT_DIR,
) -> list[Path]:
    """Save raw metric curves for each pre-normalized objective component."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for metric_name in METRIC_NAMES:
        xs = []
        ys = []
        for trial in study.trials:
            value = trial.user_attrs.get(metric_name)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            xs.append(trial.number)
            ys.append(value)

        fig_path = output_dir / f"netlist_{netlist_id}_{metric_name}_curve.png"
        plt.figure(figsize=(8, 4))
        if xs:
            plt.plot(xs, ys, linewidth=2)
        plt.xlabel("Trial")
        plt.ylabel(metric_name)
        plt.title(f"Netlist {netlist_id} {metric_name} Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        saved_paths.append(fig_path)

    return saved_paths


def _unnormalize(value: float, bounds: list[float], kind: str) -> float:
    low, high, step = bounds
    if kind not in ("mos_w", "mos_l", "r", "c"):
        raise ValueError(f"Unsupported parameter kind: {kind}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Normalized parameter must be in [0, 1]: {value}")

    unnormalized_value = 10 ** (np.log10(low) + value * (np.log10(high) - np.log10(low)))
    max_steps = math.floor((high - low) / step + 1e-9)
    steps = max(0, min(max_steps, round((unnormalized_value - low) / step)))
    return float(low + steps * step)


def _sample_parameter(trial: optuna.Trial, name: str) -> float:
    return trial.suggest_float(name, 0.0, 1.0)

def _build_parameter_updates(trial: optuna.Trial, device_types: dict[str, str]) -> dict[str, list[float]]:
    updates: dict[str, list[float]] = {}
    for instance_name, device_type in device_types.items():
        if device_type == "mos":
            l_value = _sample_parameter(trial, f"{instance_name}_L")
            w_value = _sample_parameter(trial, f"{instance_name}_W")
            updates[instance_name] = [_unnormalize(l_value, mos_L_Range, "mos_l"), _unnormalize(w_value, mos_W_Range, "mos_w")]
        elif device_type == "c":
            c_value = _sample_parameter(trial, f"{instance_name}_C")
            updates[instance_name] = [_unnormalize(c_value, c_Range, "c")]
        elif device_type == "r":
            r_value = _sample_parameter(trial, f"{instance_name}_R")
            updates[instance_name] = [_unnormalize(r_value, r_Range, "r")]
    return updates


def _netlist_paths(netlist_id: int) -> tuple[Path, Path]:
    return (
        NETLIST_DIR / f"netlist_{netlist_id}_ac.scs",
        NETLIST_DIR / f"netlist_{netlist_id}_stb.scs",
    )


def rerun_best_parameters(netlist_id: int, best_params: dict[str, float], c_load, output_ac_path, output_stb_path, normalised_targets) -> tuple[float, float, float, float, float]:
    """Write best parameters back to netlists and rerun AC/STB/XF once."""
    ac_path, stb_path = _netlist_paths(netlist_id)
    device_types = read_instance_device_types(ac_path)
    parameter_updates: dict[str, list[float]] = {}

    for instance_name, device_type in device_types.items():
        if device_type == "mos":
            mos_l = _unnormalize(float(best_params[f"{instance_name}_L"]), mos_L_Range, "mos_l")
            mos_w = _unnormalize(float(best_params[f"{instance_name}_W"]), mos_W_Range, "mos_w")
            parameter_updates[instance_name] = [mos_l, mos_w]
        elif device_type == "c":
            c_value = _unnormalize(float(best_params[f"{instance_name}_C"]), c_Range, "c")
            parameter_updates[instance_name] = [c_value]
        elif device_type == "r":
            r_value = _unnormalize(float(best_params[f"{instance_name}_R"]), r_Range, "r")
            parameter_updates[instance_name] = [r_value]
    parameter_updates["C_Load"] = [c_load]

    ac_original = ac_path.read_text(encoding="utf-8")
    stb_original = stb_path.read_text(encoding="utf-8")
    try:
        update_ac_stb_netlists(parameter_updates, ac_path, stb_path)

        ac_proc = run_spectre(ac_path, output_ac_path, AC_MDL)
        if ac_proc.returncode != 0:
            raise RuntimeError("AC rerun failed.")
        cmrr = psf_ac_readers(output_ac_path)

        stb_proc = run_spectre(stb_path, output_stb_path, STB_MDL)
        if stb_proc.returncode != 0:
            raise RuntimeError("STB rerun failed.")
        gain, gbw, pm = psf_stb_readers(output_stb_path)
        psrr = psf_xf_readers(output_stb_path)
        pwr = psf_dc_readers(output_stb_path)
        _, fom = objective(gain, gbw, pm, cmrr, psrr, pwr, c_load, normalised_targets)
        if not all_mos_in_saturation(output_stb_path):
            fom = INVALID_FOM

        return gain, gbw, pm, cmrr, psrr, pwr, fom

    finally:
        pass


def optimize_netlist(
    netlist_id: int,
    n_trials: int = 200,
    study_name: str = "diffckt_bo",
    enable_plot: bool = False,
    normalised_targets = normalised_targets_high,
    early_stop: bool = False,
    track_progress = False,
    output_ac_path = OUTPUT_AC_PATH,
    output_stb_path = OUTPUT_STB_PATH,
    output_checkpoint_path = None
):
    ac_path, stb_path = _netlist_paths(netlist_id)
    device_types = read_instance_device_types(ac_path)

    #sampler = optunahub.load_module(package="samplers/turbo").TuRBOSampler(n_startup_trials=6, warn_independent_sampling=False)
    study = optuna.create_study(direction="minimize", study_name=study_name,sampler=TPESampler(n_startup_trials=250))
    c_load = normalised_targets["C_Load"]

    def trial_objective(trial: optuna.Trial) -> float:
        parameter_updates = _build_parameter_updates(trial, device_types)
        trial.set_user_attr("fom", INVALID_FOM)
        parameter_updates["C_Load"] = [c_load]
        ac_original = ac_path.read_text(encoding="utf-8")
        stb_original = stb_path.read_text(encoding="utf-8")
        try:
            update_ac_stb_netlists(parameter_updates, ac_path, stb_path)

            ac_proc = run_spectre(ac_path, output_ac_path, AC_MDL)
            if ac_proc.returncode == 0:
                cmrr = psf_ac_readers(output_ac_path)
            else:
                return INVALID_LOSS

            stb_proc = run_spectre(stb_path, output_stb_path, STB_MDL)
            if stb_proc.returncode == 0:
                if not all_mos_in_saturation(output_stb_path):
                    return INVALID_LOSS
                gain, gbw, pm = psf_stb_readers(output_stb_path)
                psrr = psf_xf_readers(output_stb_path)
                pwr = psf_dc_readers(output_stb_path)
            else:
                return INVALID_LOSS

            trial.set_user_attr("gain", float(gain))
            trial.set_user_attr("GBW", float(gbw))
            trial.set_user_attr("PM", float(pm))
            trial.set_user_attr("CMRR", float(cmrr))
            trial.set_user_attr("PSRR", float(psrr))
            trial.set_user_attr("PWR", float(pwr))
            loss, fom = objective(gain, gbw, pm, cmrr, psrr, pwr, c_load, normalised_targets)
            trial.set_user_attr("fom", fom)

            return loss

        except FloatingPointError as e:
            return INVALID_LOSS
        except Exception as e:
            return INVALID_LOSS

        finally:
            ac_path.write_text(ac_original, encoding="utf-8")
            stb_path.write_text(stb_original, encoding="utf-8")

    def _study_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if enable_plot:
            plot_loss_curve(study, LOSS_PLOT_PATH)
            plot_raw_metric_curves(study, netlist_id, OUTPUT_DIR)

        if early_stop and target_satisfied(study.best_value):
            print(
                f"Netlist {netlist_id}: target satisfied at trial {trial.number} "
                f"with best loss {study.best_value}. Stopping early."
            )
            study.stop()

        if track_progress:
            if output_checkpoint_path is None:
                raise ValueError("output_checkpoint_path is required when track_progress is enabled")
            trial_count = trial.number + 1
            if trial_count % 50 == 0:
                output_checkpoint_file_path = (
                    Path(output_checkpoint_path) / "checkpoint_results.csv"
                )
                _save_progress_checkpoint(
                    study, trial_count, output_checkpoint_file_path
                )

    study.optimize(trial_objective, n_trials=n_trials, callbacks=[_study_callback])

    if enable_plot:
        plot_loss_curve(study, LOSS_PLOT_PATH)
        plot_raw_metric_curves(study, netlist_id, OUTPUT_DIR)

    best_gain, best_gbw, best_pm, best_cmrr, best_psrr, best_pwr, best_fom = rerun_best_parameters(netlist_id, study.best_params, c_load, output_ac_path, output_stb_path, normalised_targets)
    best_parameters = {
        "pwr": best_pwr,
        "gain": best_gain,
        "gbw": best_gbw,
        "pm": best_pm,
        "cmrr": best_cmrr,
        "psrr": best_psrr,
        "fom": best_fom
    }
    return study, best_parameters


def _optimize_netlist_worker(
    netlist_id: int,
    n_trials: int,
    target_level: str,
    normalised_targets: dict[str, float],
    early_stop: bool,
    track_progress: bool,
) -> tuple[int, dict[str, float], dict[str, float]]:
    """Optimize one netlist in a child process and return serializable results."""
    output_ac_path = REPO_ROOT / target_level / str(netlist_id) / "AC"
    output_stb_path = REPO_ROOT / target_level / str(netlist_id) / "STB"
    output_checkpoint_path = REPO_ROOT / target_level / str(netlist_id) / "checkpoints"

    print(f"[netlist {netlist_id}] optimization started (pid={os.getpid()})", flush=True)
    study, best_parameters = optimize_netlist(
        netlist_id=netlist_id,
        n_trials=n_trials,
        study_name=f"diffckt_bo_netlist_{netlist_id}",
        enable_plot=False,
        normalised_targets=normalised_targets,
        early_stop=early_stop,
        track_progress = track_progress,
        output_ac_path=output_ac_path,
        output_stb_path=output_stb_path,
        output_checkpoint_path = output_checkpoint_path,
    )
    best_parameters["id"] = netlist_id
    print(f"[netlist {netlist_id}] optimization finished", flush=True)
    return netlist_id, best_parameters, dict(study.best_params)


def _append_result_row(result_file_name: str, best_parameters: dict[str, float]) -> None:
    write_header = not os.path.exists(result_file_name) or os.path.getsize(result_file_name) == 0
    with open(result_file_name, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=best_parameters.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(best_parameters)


def _save_parameter_results(result_path: str, result_para_dict: dict[str, dict[str, float]]) -> None:
    """Atomically save all results completed so far."""
    result_path_obj = Path(result_path)
    temporary_path = result_path_obj.with_suffix(result_path_obj.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(result_para_dict, f, indent=4, ensure_ascii=False)
    os.replace(temporary_path, result_path_obj)


def main() -> None:

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-name", type=str, default="BiDiffckt")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=50,
        help="maximum number of netlists optimized concurrently (default: 50)",
    )
    parser.add_argument("--target-level", type=str, choices=['external', 'high', 'med', 'low'], default="high", help="select the target level from: external, high, med, low")
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="stop the current netlist once all normalized targets are satisfied, then save and continue",
    )
    parser.add_argument(
        "--track-progress",
        action="store_true",
    )
    args = parser.parse_args()

    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    if args.target_level == "high":
        normalised_targets = normalised_targets_high
    elif args.target_level == "med":
        normalised_targets = normalised_targets_med
    elif args.target_level == "external":
        normalised_targets = normalised_targets_external
    else:
        normalised_targets = normalised_targets_low

    # remove the previous run cache
    REPO_ROOT / args.target_level
    shutil.rmtree(REPO_ROOT / args.target_level, ignore_errors=True)

    result_file_name = args.test_name + "_level_" + args.target_level + "_results.csv"
    result_para_path = args.test_name + "_level_" + args.target_level + "best_results_para.json"
    result_para_dict = {}
    max_netlist_id = 99
    netlist_ids = []
    for i in range(max_netlist_id + 1):
        ac_path, stb_path = _netlist_paths(i)
        missing_paths = [path for path in (ac_path, stb_path) if not path.exists()]
        if missing_paths:
            missing_names = ", ".join(path.name for path in missing_paths)
            print(
                f"[netlist {i}] skipped because files are missing: {missing_names}", flush=True
            )
            continue
        netlist_ids.append(i)

    if not netlist_ids:
        parser.error("no complete AC/STB netlist pairs were found")

    worker_count = min(args.max_workers, len(netlist_ids))
    print(
        f"Starting {len(netlist_ids)} netlist optimizations with "
        f"{worker_count} worker processes.",
        flush=True,
    )

    failures: dict[int, str] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_netlist_id = {
            executor.submit(
                _optimize_netlist_worker,
                netlist_id,
                args.n_trials,
                args.target_level,
                normalised_targets,
                args.early_stop,
                args.track_progress
            ): netlist_id
            for netlist_id in netlist_ids
        }

        for future in as_completed(future_to_netlist_id):
            netlist_id = future_to_netlist_id[future]
            try:
                completed_id, best_parameters, best_sizing = future.result()
            except Exception as exc:
                failures[netlist_id] = f"{type(exc).__name__}: {exc}"
                print(f"[netlist {netlist_id}] failed: {failures[netlist_id]}", flush=True)
                continue

            _append_result_row(result_file_name, best_parameters)
            result_para_dict[str(completed_id)] = best_sizing
            _save_parameter_results(result_para_path, result_para_dict)

    if failures:
        failed_ids = ", ".join(str(netlist_id) for netlist_id in sorted(failures))
        raise RuntimeError(f"optimization failed for netlist(s): {failed_ids}")

    print(f"All {len(netlist_ids)} netlist optimizations completed.", flush=True)


if __name__ == "__main__":
    main()
