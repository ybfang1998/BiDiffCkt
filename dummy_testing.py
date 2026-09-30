import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from hydra.utils import to_absolute_path
from torch.optim import Adam

from modules.diffusion.discrete_diffusion import DiscreteDiffusion

import utils

class DummyTester(nn.Module):
    def __init__(self, cfg, denoiser):
        super().__init__()
        self.denoiser = denoiser
        self.cfg = cfg

        self.hidden_dims = cfg.model.GraphTransformer.hidden_dims['dx']
        denoiser_hidden_dims = self.denoiser.tf_layers[0].self_attn.dx
        if self.hidden_dims != denoiser_hidden_dims:
            raise ValueError(
                "Probe/config hidden dimension does not match the checkpoint: "
                f"cfg={self.hidden_dims}, checkpoint={denoiser_hidden_dims}."
            )

        # Freeze the denoiser parameters
        for param in self.denoiser.parameters():
            param.requires_grad = False

        self.params_layer = nn.Sequential(
            nn.Linear(2, int(self.hidden_dims / 4)),
            nn.ReLU(),
        )  # Linear layer to fetch U_parameters features

        self.U_layer = nn.Sequential(
            nn.Linear(self.hidden_dims + int(self.hidden_dims / 4), self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.hidden_dims),
        )

        self.V_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.hidden_dims),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dims * 2, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.hidden_dims),
            nn.Linear(self.hidden_dims, 4)  # Final output layer
        )

    def train(self, mode=True):
        super().train(mode)
        self.denoiser.eval()  # Ensure the denoiser is always in evaluation mode
        return self
    
    def forward(self, U, V, E, U_params, U_mask, V_mask, t_float, probe_t):

        with torch.no_grad():  # Ensure no gradients are computed for the denoiser
            # probe_t is exposed as a human-readable, 1-based layer number. The
            # transformer itself indexes its ModuleList from zero.
            node_features, net_features, _ = self.denoiser(
                U, V, E, t_float, U_mask, V_mask, probe_t=probe_t - 1
            )
        
        param_features = self.params_layer(U_params)  # Process U_params through the linear layer
        node_features = torch.cat((node_features, param_features), dim=-1)

        U_features = self.U_layer(node_features)
        V_features = self.V_layer(net_features)

        # mean pooling
        U_features = U_features * U_mask.unsqueeze(-1)  # Apply the mask to U_features
        U_features = U_features.sum(dim=1) / U_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)  # Mean pooling for U_features
        V_features = V_features * V_mask.unsqueeze(-1)  # Apply the mask to V_features
        V_features = V_features.sum(dim=1) / V_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)  # Mean pooling for V_features

        x = torch.cat((U_features, V_features), dim=-1)  # Concatenate U and V features
        out = self.output_layer(x)  # Final output layer

        return out


def _regression_metrics(predictions, targets):
    """Return MSE, MAE and uniform-average R-squared for four targets.

    The probe targets are continuous, so classification accuracy is not
    applicable. ``accuracy`` in the result CSV is the mean R-squared across
    target dimensions with non-zero variance.
    """
    predictions = predictions.float()
    targets = targets.float()

    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    residual_sum_squares = torch.sum((targets - predictions) ** 2, dim=0)
    target_mean = torch.mean(targets, dim=0)
    total_sum_squares = torch.sum((targets - target_mean) ** 2, dim=0)
    valid_targets = total_sum_squares > torch.finfo(targets.dtype).eps

    per_target_r2 = torch.full_like(total_sum_squares, torch.nan)
    per_target_r2[valid_targets] = (
        1.0
        - residual_sum_squares[valid_targets]
        / total_sum_squares[valid_targets]
    )
    if valid_targets.any():
        accuracy = per_target_r2[valid_targets].mean().item()
    else:
        accuracy = float("nan")

    return mse, mae, accuracy, per_target_r2.tolist()


def _save_probe_results(results, results_path):
    """Persist completed probe results so partial HPC runs remain useful."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = results_path.with_suffix(f"{results_path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["probe_t", "accuracy", "mse", "mae"])
        writer.writeheader()
        for result in results:
            writer.writerow({
                "probe_t": result["probe_t"],
                "accuracy": result["accuracy"],
                "mse": result["test_mse"],
                "mae": result["test_mae"],
            })
    temporary_path.replace(results_path)


def dummy_train(cfg, model_path, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = str(Path(model_path).resolve())
    discrete_model = DiscreteDiffusion.load_from_checkpoint(
        model_path, map_location=device, weights_only=False
    )
    denoiser = discrete_model.model
    edge_loss_fn = discrete_model.edge_loss_fn
    denoiser.eval()
    epochs = int(cfg.experiments.dummy_test.epochs)
    learning_rate = float(cfg.experiments.dummy_test.get("lr", 1e-3))

    t = int(cfg.experiments.dummy_test.t)
    T = discrete_model.T
    if not 0 <= t <= T:
        raise ValueError(f"Diffusion timestep t must be in [0, {T}], got {t}.")
    probe_t_list = [int(probe_t) for probe_t in cfg.experiments.dummy_test.probe_t]
    n_transformer_layers = len(denoiser.tf_layers)
    invalid_probe_layers = [
        probe_t for probe_t in probe_t_list
        if not 1 <= probe_t <= n_transformer_layers
    ]
    if invalid_probe_layers:
        raise ValueError(
            "probe_t uses 1-based transformer layer numbers. "
            f"Expected values in [1, {n_transformer_layers}], got "
            f"{invalid_probe_layers}."
        )
    if not probe_t_list:
        raise ValueError("probe_t must contain at least one transformer layer.")
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}.")

    results_path_config = cfg.experiments.dummy_test.get(
        "results_path", "outputs/Probe_Results/probe_results.csv"
    )
    results_path = Path(to_absolute_path(str(results_path_config)))
    if results_path.suffix.lower() != ".csv":
        results_path = results_path / "probe_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    log_every_steps = max(1, int(cfg.train.log_every_steps))
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    train_samples = len(train_loader.dataset)
    test_samples = len(test_loader.dataset)

    criterion = nn.MSELoss()
    results = []
    global_train_step = 0
    run_succeeded = False
    run_before_setup = wandb.run
    if run_before_setup is None:
        utils.setup_wandb(cfg)
    wandb_started_here = run_before_setup is None and wandb.run is not None

    device_description = str(device)
    if device.type == "cuda":
        device_description += f" ({torch.cuda.get_device_name(device)})"
    print("=" * 88, flush=True)
    print("Starting transformer probe training", flush=True)
    print(f"Checkpoint       : {model_path}", flush=True)
    print(f"Device           : {device_description}", flush=True)
    print(f"Probe layers     : {probe_t_list} (1-based)", flush=True)
    print(f"Diffusion t/T    : {t}/{T} = {t / T:.6f}", flush=True)
    print(f"Epochs / LR      : {epochs} / {learning_rate:g}", flush=True)
    print(
        f"Train / test     : {train_samples} samples in {train_batches} batches / "
        f"{test_samples} samples in {test_batches} batches",
        flush=True,
    )
    print(f"Results CSV      : {results_path}", flush=True)
    print("Accuracy metric  : uniform-average R^2 over y[:, :4]", flush=True)
    if wandb.run:
        print(
            f"W&B              : {cfg.experiments.general.wandb} | "
            f"run={wandb.run.name} | dir={wandb.run.dir}",
            flush=True,
        )
    else:
        print("W&B              : disabled", flush=True)
    print("=" * 88, flush=True)

    try:
        for probe_index, probe_t in enumerate(probe_t_list, start=1):
            # Give each layer an independent, reproducible probe head and Adam
            # state. Reusing them would leak training from earlier layers.
            utils.set_seed(cfg.train.seed)
            dummy_tester = DummyTester(cfg, denoiser).to(device)
            trainable_params = [
                param for param in dummy_tester.parameters() if param.requires_grad
            ]
            optimizer = Adam(trainable_params, lr=learning_rate)
            trainable_parameter_count = sum(
                param.numel() for param in trainable_params
            )

            print(
                f"[Probe {probe_index}/{len(probe_t_list)}] Layer {probe_t} | "
                f"trainable parameters={trainable_parameter_count:,}",
                flush=True,
            )
            probe_start_time = time.perf_counter()

            for epoch in range(1, epochs + 1):
                dummy_tester.train()
                epoch_start_time = time.perf_counter()
                epoch_loss_sum = 0.0
                epoch_sample_count = 0

                for batch_index, batch in enumerate(train_loader, start=1):
                    if edge_loss_fn == 'CE':
                        batch = {k: utils.to_tf_device(v, device, dtype=torch.float32) for k, v in batch.items()}
                    elif edge_loss_fn == 'BCE':
                        batch = {k: utils.to_tf_device(v, device, dtype=torch.float32) if k != 'E'
                                else utils.to_tf_device(v, device, dtype=torch.long) for k, v in batch.items()}
                    U_mask, V_mask = batch['U_mask'].to(dtype=torch.bool), batch['V_mask'].to(dtype=torch.bool)
                    targets = batch['y'][:, :4]
                    if targets.ndim != 2 or targets.shape[1] != 4:
                        raise ValueError(
                            "The probe expects y to contain at least four "
                            f"continuous targets, got shape {tuple(batch['y'].shape)}."
                        )

                    optimizer.zero_grad(set_to_none=True)

                    # Apply Noise
                    t_int = torch.tensor([t], device=device).float()
                    t_int = t_int.expand(batch['U'].shape[0], 1)  # Expand t_int to match batch size
                    noisy_data = discrete_model.apply_noise(batch['U'], batch['V'], batch['E'], U_mask, V_mask, batch['y'], t_int=t_int)
                    U_t, V_t, E_t, t_float = noisy_data['U_t'], noisy_data['V_t'], noisy_data['E_t'], noisy_data['t']
                    U_params = batch['U_params']

                    # Forward pass through the dummy tester
                    predictions = dummy_tester(U_t, V_t, E_t, U_params, U_mask, V_mask, t_float, probe_t)
                    loss = criterion(predictions, targets)
                    loss.backward()
                    optimizer.step()

                    batch_size = targets.shape[0]
                    epoch_loss_sum += loss.item() * batch_size
                    epoch_sample_count += batch_size
                    global_train_step += 1

                    should_log_batch = (
                        batch_index == 1
                        or batch_index % log_every_steps == 0
                        or batch_index == train_batches
                    )
                    if wandb.run and should_log_batch:
                        wandb.log({
                            "probe_t": probe_t,
                            "train/probe_epoch": epoch,
                            "train/global_step": global_train_step,
                            "train/batch_mse": loss.item(),
                        })

                if epoch_sample_count == 0:
                    raise RuntimeError("The training DataLoader produced no samples.")
                epoch_mse = epoch_loss_sum / epoch_sample_count
                epoch_seconds = time.perf_counter() - epoch_start_time
                print(
                    f"  Layer {probe_t} | epoch {epoch:03d}/{epochs:03d} complete | "
                    f"train_mse={epoch_mse:.6f} | {epoch_seconds:.2f}s",
                    flush=True,
                )
                if wandb.run:
                    wandb.log({
                        "probe_t": probe_t,
                        "train/probe_epoch": epoch,
                        "train/global_step": global_train_step,
                        "train/epoch_mse": epoch_mse,
                        "train/epoch_seconds": epoch_seconds,
                        f"probe_{probe_t}/train_epoch_mse": epoch_mse,
                    })

            dummy_tester.eval()
            all_predictions = []
            all_targets = []
            test_start_time = time.perf_counter()
            with torch.no_grad():
                for batch in test_loader:
                    if edge_loss_fn == 'CE':
                        batch = {k: utils.to_tf_device(v, device, dtype=torch.float32) for k, v in batch.items()}
                    elif edge_loss_fn == 'BCE':
                        batch = {k: utils.to_tf_device(v, device, dtype=torch.float32) if k != 'E'
                                else utils.to_tf_device(v, device, dtype=torch.long) for k, v in batch.items()}
                    U_mask, V_mask = batch['U_mask'].to(dtype=torch.bool), batch['V_mask'].to(dtype=torch.bool)
                    targets = batch['y'][:, :4]
                    if targets.ndim != 2 or targets.shape[1] != 4:
                        raise ValueError(
                            "The probe expects y to contain at least four "
                            f"continuous targets, got shape {tuple(batch['y'].shape)}."
                        )
                    
                    # Apply Noise
                    t_int = torch.tensor([t], device=device).float()
                    t_int = t_int.expand(batch['U'].shape[0], 1)  # Expand t_int to match batch size
                    noisy_data = discrete_model.apply_noise(batch['U'], batch['V'], batch['E'], U_mask, V_mask, batch['y'], t_int=t_int)
                    U_t, V_t, E_t, t_float = noisy_data['U_t'], noisy_data['V_t'], noisy_data['E_t'], noisy_data['t']
                    U_params = batch['U_params']

                    # Forward pass through the dummy tester
                    predictions = dummy_tester(U_t, V_t, E_t, U_params, U_mask, V_mask, t_float, probe_t)
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())

            if not all_targets:
                raise RuntimeError("The test DataLoader produced no samples.")
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            test_mse, test_mae, accuracy, per_target_r2 = _regression_metrics(
                predictions, targets
            )
            test_seconds = time.perf_counter() - test_start_time

            result = {
                "probe_t": probe_t,
                "accuracy": accuracy,
                "test_mse": test_mse,
                "test_mae": test_mae,
                "per_target_r2": per_target_r2,
            }
            results.append(result)
            _save_probe_results(results, results_path)

            r2_text = ", ".join(
                "nan" if not math.isfinite(value) else f"{value:.6f}"
                for value in per_target_r2
            )
            print(
                f"[Probe {probe_index}/{len(probe_t_list)}] Layer {probe_t} test "
                f"complete | accuracy(mean R^2)={accuracy:.6f} | "
                f"mse={test_mse:.6f} | mae={test_mae:.6f} | "
                f"R^2 per target=[{r2_text}] | {test_seconds:.2f}s",
                flush=True,
            )
            print(
                f"[Probe {probe_index}/{len(probe_t_list)}] Layer {probe_t} "
                f"total time={time.perf_counter() - probe_start_time:.2f}s | "
                f"CSV updated: {results_path}",
                flush=True,
            )

            if wandb.run:
                test_log = {
                    "probe_t": probe_t,
                    "test/accuracy": accuracy,
                    "test/mse": test_mse,
                    "test/mae": test_mae,
                    "test/seconds": test_seconds,
                }
                for target_index, r2_value in enumerate(per_target_r2):
                    if math.isfinite(r2_value):
                        test_log[f"test/r2_target_{target_index}"] = r2_value
                wandb.log(test_log)
                wandb.run.summary[f"probe_{probe_t}/accuracy"] = accuracy
                wandb.run.summary[f"probe_{probe_t}/test_mse"] = test_mse

            del dummy_tester, optimizer

        print("=" * 88, flush=True)
        print("All transformer probes completed.", flush=True)
        for result in results:
            print(
                f"  layer={result['probe_t']:>2d} | "
                f"accuracy(mean R^2)={result['accuracy']:.6f} | "
                f"test_mse={result['test_mse']:.6f}",
                flush=True,
            )
        print(f"Final results saved to: {results_path}", flush=True)
        print("=" * 88, flush=True)

        if wandb.run:
            results_table = wandb.Table(
                columns=["probe_t", "accuracy", "test_mse", "test_mae"],
                data=[
                    [
                        result["probe_t"],
                        result["accuracy"],
                        result["test_mse"],
                        result["test_mae"],
                    ]
                    for result in results
                ],
            )
            wandb.log({"probe/results": results_table})
            wandb.run.summary["probe_results_csv"] = str(results_path)
            artifact_name = (
                f"{cfg.experiments.general.project_name}-probe-results"
                .replace("/", "-")
                .replace(" ", "-")
            )
            results_artifact = wandb.Artifact(
                artifact_name, type="probe-results"
            )
            results_artifact.add_file(str(results_path))
            wandb.log_artifact(results_artifact)

        run_succeeded = True
        return results
    finally:
        if wandb_started_here and wandb.run is not None:
            wandb.finish(exit_code=0 if run_succeeded else 1)
