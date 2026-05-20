import json
import logging
import os
import random
import argparse
import concurrent.futures
import csv
import multiprocessing as mp
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from network import Network
from utils import reshape_image


@dataclass
class ExperimentConfig:
    # Data and optimization
    batch_size: int = 512
    lr: float = 1e-3
    epochs_stage1: int = 300
    epochs_stage2: int = 300
    epochs_lstm: int = 300
    epochs_rnn: int = 300
    n_runs: int = 5

    # Model sizes
    gfr_stage2_target_params: int = 7434
    gfr_hidden_dim: int = 86
    rnn_hidden_dim: int = 68
    num_classes: int = 10

    # GFR settings
    bio_units: bool = False

    # Reproducibility
    base_seed: int = 1234

    # Runtime
    output_root: str = "runs"
    parallel_runs: bool = True
    gpu_ids: str = ""


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)
        self._state = None

    def reset(self, batch_size: int) -> None:
        device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        self._state = (h0, c0)

    def zero_input(self, batch_size: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.input_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept both [B, in_dim] (single recurrent step) and [B, T, in_dim].
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, self._state = self.lstm(x, self._state)
        last = torch.relu(torch.tanh(out[:, -1, :]))
        return self.head(last)


class RNNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)
        self._hidden = None

    def reset(self, batch_size: int) -> None:
        device = next(self.parameters()).device
        self._hidden = torch.zeros(batch_size, self.hidden_size, device=device)

    def zero_input(self, batch_size: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.input_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._hidden = torch.relu(torch.tanh(self.input_to_hidden(x) + self.hidden_to_hidden(self._hidden)))
        return self.head(self._hidden)


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("l_mnist_pipeline")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def setup_run_logging(log_path: Path, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(processName)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lmnist_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
        ]
    )
    train_set = torchvision.datasets.MNIST("data/mnist/train", download=True, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST("data/mnist/test", download=True, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def effective_trainable_parameters(model: nn.Module) -> int:
    return int(getattr(model, "_effective_trainable_params", count_parameters(model, trainable_only=True)))


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def make_gfr_model(hidden_dim: int, freeze_neurons: bool, freeze_activation: bool, device: torch.device, bio_units: bool) -> Network:
    try:
        model = Network(
            in_dim=28,
            hidden_dim=hidden_dim,
            out_dim=10,
            freeze_neurons=freeze_neurons,
            freeze_g=freeze_activation,
            device=device,
            bio_units=bio_units,
        ).to(device)
    except TypeError:
        model = Network(
            in_dim=28,
            hidden_dim=hidden_dim,
            out_dim=10,
            freeze_neurons=freeze_neurons,
            freeze_g=freeze_activation,
            device=device,
        ).to(device)

    if hasattr(model, "hidden_neurons") and hasattr(model.hidden_neurons, "bio_units"):
        model.hidden_neurons.bio_units = bio_units

    return model


def forward_gfr_sequence(model: Network, x: torch.Tensor) -> torch.Tensor:
    model.reset(x.shape[0])
    for t in range(x.shape[1]):
        model(x[:, t, :])
    return model(model.zero_input(x.shape[0]))


def evaluate_sequence_classifier(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = reshape_image(x, variant="l").to(device)
            y = y.to(device)

            model.reset(x.shape[0])
            for t in range(x.shape[1]):
                model(x[:, t, :])

            logits = model(model.zero_input(x.shape[0]))
            vote_scores = torch.softmax(logits, dim=1)
            for _ in range(4):
                vote_scores += torch.softmax(model(model.zero_input(x.shape[0])), dim=1)
            pred = torch.argmax(vote_scores, dim=1)

            target = nn.functional.one_hot(y, num_classes=10).to(torch.float32).to(device)
            total_loss += criterion(logits, target).item()
            correct += (pred == y).sum().item()
            total += y.numel()

    return total_loss, correct / max(total, 1)


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    is_gfr: bool,
    device: torch.device,
    logger: logging.Logger,
    run_name: str,
    progress_path: Optional[Path] = None,
) -> Dict[str, object]:
    # Matches train_network loss style in train.py used by network_pipeline.py
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = reshape_image(x, variant="l").to(device)
            y = y.to(device)

            # Match train_network in train.py for both GFR and LSTM:
            # process sequence step-by-step, then one zero-input prediction.
            model.reset(x.shape[0])
            for t in range(x.shape[1]):
                model(x[:, t, :])
            logits = model(model.zero_input(x.shape[0]))

            target = nn.functional.one_hot(y, num_classes=10).to(torch.float32).to(device)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=False)
            optimizer.step()

            total_train_loss += loss.item()
            pred = torch.argmax(logits.detach(), dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.numel()

        model.eval()
        total_test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = reshape_image(x, variant="l").to(device)
                y = y.to(device)

                # Match evaluate.accuracy behavior for both models:
                # sequential processing + 5 zero-input softmax votes.
                model.reset(x.shape[0])
                for t in range(x.shape[1]):
                    model(x[:, t, :])

                logits = model(model.zero_input(x.shape[0]))
                vote_scores = torch.softmax(logits, dim=1)
                for _ in range(4):
                    vote_scores += torch.softmax(model(model.zero_input(x.shape[0])), dim=1)
                pred = torch.argmax(vote_scores, dim=1)

                target = nn.functional.one_hot(y, num_classes=10).to(torch.float32).to(device)
                loss = criterion(logits, target)
                total_test_loss += loss.item()

                correct += (pred == y).sum().item()
                total += y.numel()

        history["train_loss"].append(total_train_loss)
        history["train_acc"].append(train_correct / max(train_total, 1))
        history["test_loss"].append(total_test_loss)
        history["test_acc"].append(correct / max(total, 1))

        logger.info(
            "%s | Epoch %03d | train_loss=%.4f | train_acc=%.4f | test_loss=%.4f | test_acc=%.4f",
            run_name,
            epoch + 1,
            history["train_loss"][-1],
            history["train_acc"][-1],
            history["test_loss"][-1],
            history["test_acc"][-1],
        )
        save_progress(
            progress_path,
            {
                "run_name": run_name,
                "status": "running",
                "epoch": epoch + 1,
                "epochs": epochs,
                "history": history,
            },
        )

    eval_train_loss, eval_train_acc = evaluate_sequence_classifier(model, train_loader, device)
    history["eval_train_loss"] = eval_train_loss
    history["eval_train_acc"] = eval_train_acc
    logger.info(
        "%s | Final train_eval_loss=%.4f | train_eval_acc=%.4f",
        run_name,
        eval_train_loss,
        eval_train_acc,
    )
    save_progress(
        progress_path,
        {
            "run_name": run_name,
            "status": "complete",
            "epoch": epochs,
            "epochs": epochs,
            "history": history,
        },
    )

    return history


def copy_and_freeze_gfr_neurons(target_model: Network, source_model: Network) -> None:
    with torch.no_grad():
        target_model.hidden_neurons.load_state_dict(source_model.hidden_neurons.state_dict())
    target_model.hidden_neurons.freeze_parameters()


def set_trainable_recurrent_only(model: Network, target_params: Optional[int] = None) -> None:
    freeze_all(model)
    model.fc2.weight.requires_grad = True
    effective_params = model.fc2.weight.numel()

    if model.fc2.bias is not None:
        if target_params is None:
            bias_trainable = model.fc2.bias.numel()
        else:
            bias_trainable = target_params - effective_params
            if bias_trainable < 0 or bias_trainable > model.fc2.bias.numel():
                raise ValueError(
                    f"Cannot realize target_params={target_params} with hidden_dim={model.hidden_dim}; "
                    f"recurrent weight contributes {effective_params} params and bias has {model.fc2.bias.numel()} entries."
                )

        if bias_trainable > 0:
            model.fc2.bias.requires_grad = True
            if bias_trainable < model.fc2.bias.numel():
                mask = torch.zeros_like(model.fc2.bias)
                mask[:bias_trainable] = 1
                model.fc2.bias.register_hook(lambda grad, mask=mask: grad * mask)
            effective_params += bias_trainable

    model._effective_trainable_params = effective_params


def aggregate_histories(histories: List[Dict[str, object]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array([h[key] for h in histories], dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_mean_std_curve(mean_curve: np.ndarray, std_curve: np.ndarray, title: str, ylabel: str, save_path: Path) -> None:
    epochs = np.arange(1, len(mean_curve) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, mean_curve, label="mean", linewidth=2)
    plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.25, label="std")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_log_train_loss_comparison(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    save_path: Path,
) -> None:
    plt.figure(figsize=(7.5, 4.5))
    for label, (mean_curve, std_curve) in curves.items():
        epochs = np.arange(1, len(mean_curve) + 1)
        lower = np.maximum(mean_curve - std_curve, 1e-12)
        upper = np.maximum(mean_curve + std_curve, 1e-12)
        plt.plot(epochs, mean_curve, label=label, linewidth=2)
        plt.fill_between(epochs, lower, upper, alpha=0.2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss (log scale)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def choose_lstm_hidden_size(target_params: int, low: int = 2, high: int = 256) -> Tuple[int, int]:
    best_h = low
    best_gap = 10**18
    for h in range(low, high + 1):
        model = LSTMClassifier(input_size=28, hidden_size=h, num_classes=10)
        n = count_parameters(model, trainable_only=True)
        gap = abs(n - target_params)
        if gap < best_gap:
            best_gap = gap
            best_h = h
    final_model = LSTMClassifier(input_size=28, hidden_size=best_h, num_classes=10)
    return best_h, count_parameters(final_model, trainable_only=True)


def rnn_trainable_param_count(hidden_size: int, input_size: int = 28, num_classes: int = 10) -> int:
    model = RNNClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    return count_parameters(model, trainable_only=True)


def parse_gpu_ids(raw_gpu_ids: str) -> List[int]:
    if raw_gpu_ids.strip():
        return [int(item) for item in raw_gpu_ids.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [int(item) for item in visible.split(",") if item.strip()]
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def load_frozen_gfr_neurons_from_checkpoint(model: Network, checkpoint_path: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["state_dict"]
    hidden_state = {
        name.removeprefix("hidden_neurons."): value
        for name, value in state_dict.items()
        if name.startswith("hidden_neurons.")
    }
    model.hidden_neurons.load_state_dict(hidden_state)
    model.hidden_neurons.freeze_parameters()


def summarize_run_group(histories: List[Dict[str, object]]) -> Dict[str, Any]:
    train_loss_mean, train_loss_std = aggregate_histories(histories, key="train_loss")
    test_loss_mean, test_loss_std = aggregate_histories(histories, key="test_loss")
    final_train_accs = [float(h["eval_train_acc"]) for h in histories]
    final_test_accs = [float(h["test_acc"][-1]) for h in histories]
    return {
        "final_train_accs": final_train_accs,
        "final_test_accs": final_test_accs,
        "train_acc_mean": float(np.mean(final_train_accs)),
        "train_acc_std": float(np.std(final_train_accs)),
        "acc_mean": float(np.mean(final_test_accs)),
        "acc_std": float(np.std(final_test_accs)),
        "train_loss_mean": train_loss_mean,
        "train_loss_std": train_loss_std,
        "test_loss_mean": test_loss_mean,
        "test_loss_std": test_loss_std,
        "histories": histories,
    }


def train_parallel_run(task: Dict[str, Any]) -> Dict[str, Any]:
    cfg = ExperimentConfig(**task["config"])
    model_name = task["model_name"]
    run_idx = int(task["run_idx"])
    seed = int(task["seed"])
    gpu_id = int(task["gpu_id"])
    output_dir = Path(task["output_dir"])

    set_seed(seed)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    log_dir = output_dir / "run_logs"
    progress_dir = output_dir / "progress"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_name}_run{run_idx}"
    logger = setup_run_logging(log_dir / f"{run_name}.log", f"l_mnist_pipeline.{run_name}")
    logger.info("%s | assigned_gpu=%d | seed=%d", run_name, gpu_id, seed)

    train_loader, test_loader = get_lmnist_data_loaders(cfg.batch_size)

    if model_name == "stage2":
        model = make_gfr_model(
            hidden_dim=cfg.gfr_hidden_dim,
            freeze_neurons=False,
            freeze_activation=True,
            device=device,
            bio_units=cfg.bio_units,
        )
        load_frozen_gfr_neurons_from_checkpoint(model, Path(task["stage1_ckpt"]), device)
        set_trainable_recurrent_only(model, target_params=cfg.gfr_stage2_target_params)
        params = effective_trainable_parameters(model)
        epochs = cfg.epochs_stage2
    elif model_name == "lstm":
        hidden_size = int(task["hidden_size"])
        model = LSTMClassifier(input_size=28, hidden_size=hidden_size, num_classes=cfg.num_classes).to(device)
        params = count_parameters(model, trainable_only=True)
        epochs = cfg.epochs_lstm
    elif model_name == "rnn":
        model = RNNClassifier(input_size=28, hidden_size=cfg.rnn_hidden_dim, num_classes=cfg.num_classes).to(device)
        params = count_parameters(model, trainable_only=True)
        epochs = cfg.epochs_rnn
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    logger.info("%s | trainable_params=%d | epochs=%d", run_name, params, epochs)
    history = train_one_run(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=cfg.lr,
        is_gfr=(model_name == "stage2"),
        device=device,
        logger=logger,
        run_name=run_name,
        progress_path=progress_dir / f"{run_name}.json",
    )

    result = {
        "model_name": model_name,
        "run_idx": run_idx,
        "seed": seed,
        "gpu_id": gpu_id,
        "trainable_params": int(params),
        "history": history,
    }
    save_json(output_dir / "histories" / f"{run_name}.json", result)
    logger.info("%s | complete", run_name)
    return result


def save_json(path: Path, payload: Dict) -> None:
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_convert)


def save_progress(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    save_json(tmp_path, payload)
    tmp_path.replace(path)


def save_loss_artifacts(output_dir: Path, summary: Dict[str, Any]) -> None:
    curves_dir = output_dir / "loss_artifacts"
    curves_dir.mkdir(parents=True, exist_ok=True)
    model_keys = ["stage2", "rnn", "lstm"]

    npz_payload: Dict[str, np.ndarray] = {}
    for key in model_keys:
        for curve_key in ["train_loss_mean", "train_loss_std", "test_loss_mean", "test_loss_std"]:
            npz_payload[f"{key}_{curve_key}"] = np.asarray(summary[key][curve_key], dtype=np.float64)
    np.savez(curves_dir / "loss_curves.npz", **npz_payload)

    with open(curves_dir / "loss_curves.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "train_loss_mean", "train_loss_std", "test_loss_mean", "test_loss_std"])
        for key in model_keys:
            n_epochs = len(summary[key]["train_loss_mean"])
            for epoch in range(n_epochs):
                writer.writerow(
                    [
                        key,
                        epoch + 1,
                        summary[key]["train_loss_mean"][epoch],
                        summary[key]["train_loss_std"][epoch],
                        summary[key]["test_loss_mean"][epoch],
                        summary[key]["test_loss_std"][epoch],
                    ]
                )

    with open(curves_dir / "run_histories.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "run_idx", "epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        for key in model_keys:
            for run_idx, history in enumerate(summary[key]["histories"], start=1):
                n_epochs = len(history["train_loss"])
                for epoch in range(n_epochs):
                    writer.writerow(
                        [
                            key,
                            run_idx,
                            epoch + 1,
                            history["train_loss"][epoch],
                            history["train_acc"][epoch],
                            history["test_loss"][epoch],
                            history["test_acc"][epoch],
                        ]
                    )


def run_full_pipeline(cfg: ExperimentConfig, device: torch.device, output_dir: Path, logger: logging.Logger) -> Dict[str, object]:
    run_seeds = [cfg.base_seed + i for i in range(cfg.n_runs)]
    logger.info("Run seeds: %s", run_seeds)

    train_loader, test_loader = get_lmnist_data_loaders(cfg.batch_size)

    # Stage-1
    set_seed(cfg.base_seed)
    stage1_model = make_gfr_model(
        hidden_dim=cfg.gfr_hidden_dim,
        freeze_neurons=False,
        freeze_activation=True,
        device=device,
        bio_units=cfg.bio_units,
    )
    logger.info("Stage-1 model trainable params: %d", count_parameters(stage1_model, trainable_only=True))

    stage1_history = train_one_run(
        model=stage1_model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=cfg.epochs_stage1,
        lr=cfg.lr,
        is_gfr=True,
        device=device,
        logger=logger,
        run_name="stage1",
        progress_path=output_dir / "progress" / "stage1.json",
    )

    stage1_ckpt = output_dir / "gfr_lmnist_stage1_shared.pt"
    torch.save(
        {
            "state_dict": stage1_model.state_dict(),
            "config": asdict(cfg),
            "history": stage1_history,
        },
        stage1_ckpt,
    )
    logger.info("Saved stage-1 checkpoint: %s", stage1_ckpt)

    stage1_total_params = count_parameters(stage1_model, trainable_only=False)
    del stage1_model
    torch.cuda.empty_cache()

    stage2_trainable_params = cfg.gfr_stage2_target_params
    logger.info(
        "Stage-2 effective trainable params (recurrent-only target): %d",
        stage2_trainable_params,
    )

    lstm_hidden_size, lstm_params = choose_lstm_hidden_size(stage2_trainable_params)
    logger.info(
        "LSTM match | hidden_size=%d | trainable_params=%d | target_params=%d",
        lstm_hidden_size,
        lstm_params,
        stage2_trainable_params,
    )
    rnn_params = rnn_trainable_param_count(cfg.rnn_hidden_dim, num_classes=cfg.num_classes)
    logger.info(
        "RNN baseline | hidden_size=%d | trainable_params=%d",
        cfg.rnn_hidden_dim,
        rnn_params,
    )

    gpu_ids = parse_gpu_ids(cfg.gpu_ids)
    if not gpu_ids:
        gpu_ids = [device.index if device.index is not None else 0]
    logger.info("Parallel run GPU pool: %s", gpu_ids)

    tasks: List[Dict[str, Any]] = []
    cfg_payload = asdict(cfg)
    model_order = ["stage2", "rnn", "lstm"]
    for model_name in model_order:
        for idx, seed in enumerate(run_seeds, start=1):
            task = {
                "config": cfg_payload,
                "model_name": model_name,
                "run_idx": idx,
                "seed": seed,
                "gpu_id": gpu_ids[len(tasks) % len(gpu_ids)],
                "output_dir": str(output_dir),
                "stage1_ckpt": str(stage1_ckpt),
                "hidden_size": lstm_hidden_size,
            }
            tasks.append(task)

    logger.info(
        "Launching %d repeated model runs with %d worker(s). Progress JSON: %s",
        len(tasks),
        min(len(gpu_ids), len(tasks)) if cfg.parallel_runs else 1,
        output_dir / "progress",
    )

    run_results: List[Dict[str, Any]] = []
    if cfg.parallel_runs and len(gpu_ids) > 1:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx) as executor:
            future_to_task = {executor.submit(train_parallel_run, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()
                run_results.append(result)
                logger.info(
                    "Completed %s_run%d on GPU %s",
                    task["model_name"],
                    task["run_idx"],
                    task["gpu_id"],
                )
    else:
        for task in tasks:
            result = train_parallel_run(task)
            run_results.append(result)
            logger.info("Completed %s_run%d", task["model_name"], task["run_idx"])

    grouped: Dict[str, List[Dict[str, Any]]] = {name: [] for name in model_order}
    for result in run_results:
        grouped[result["model_name"]].append(result)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["run_idx"]))

    stage2_histories = [item["history"] for item in grouped["stage2"]]
    rnn_histories = [item["history"] for item in grouped["rnn"]]
    lstm_histories = [item["history"] for item in grouped["lstm"]]

    stage2_stats = summarize_run_group(stage2_histories)
    rnn_stats = summarize_run_group(rnn_histories)
    lstm_stats = summarize_run_group(lstm_histories)

    stage2_train_loss_mean = stage2_stats["train_loss_mean"]
    stage2_train_loss_std = stage2_stats["train_loss_std"]
    stage2_test_loss_mean = stage2_stats["test_loss_mean"]
    stage2_test_loss_std = stage2_stats["test_loss_std"]
    stage2_final_train_accs = stage2_stats["final_train_accs"]
    stage2_final_accs = stage2_stats["final_test_accs"]
    stage2_train_acc_mean = stage2_stats["train_acc_mean"]
    stage2_train_acc_std = stage2_stats["train_acc_std"]
    stage2_acc_mean = stage2_stats["acc_mean"]
    stage2_acc_std = stage2_stats["acc_std"]

    rnn_train_loss_mean = rnn_stats["train_loss_mean"]
    rnn_train_loss_std = rnn_stats["train_loss_std"]
    rnn_test_loss_mean = rnn_stats["test_loss_mean"]
    rnn_test_loss_std = rnn_stats["test_loss_std"]
    rnn_final_train_accs = rnn_stats["final_train_accs"]
    rnn_final_accs = rnn_stats["final_test_accs"]
    rnn_train_acc_mean = rnn_stats["train_acc_mean"]
    rnn_train_acc_std = rnn_stats["train_acc_std"]
    rnn_acc_mean = rnn_stats["acc_mean"]
    rnn_acc_std = rnn_stats["acc_std"]

    lstm_train_loss_mean = lstm_stats["train_loss_mean"]
    lstm_train_loss_std = lstm_stats["train_loss_std"]
    lstm_test_loss_mean = lstm_stats["test_loss_mean"]
    lstm_test_loss_std = lstm_stats["test_loss_std"]
    lstm_final_train_accs = lstm_stats["final_train_accs"]
    lstm_final_accs = lstm_stats["final_test_accs"]
    lstm_train_acc_mean = lstm_stats["train_acc_mean"]
    lstm_train_acc_std = lstm_stats["train_acc_std"]
    lstm_acc_mean = lstm_stats["acc_mean"]
    lstm_acc_std = lstm_stats["acc_std"]

    # Save plots
    plot_mean_std_curve(
        stage2_train_loss_mean,
        stage2_train_loss_std,
        title="GFR-RNN Stage-2 Train Loss (mean +/- std)",
        ylabel="Train Loss",
        save_path=output_dir / "stage2_train_loss.png",
    )
    plot_mean_std_curve(
        stage2_test_loss_mean,
        stage2_test_loss_std,
        title="GFR-RNN Stage-2 Test Loss (mean +/- std)",
        ylabel="Test Loss",
        save_path=output_dir / "stage2_test_loss.png",
    )
    plot_mean_std_curve(
        lstm_train_loss_mean,
        lstm_train_loss_std,
        title="LSTM Train Loss (mean +/- std)",
        ylabel="Train Loss",
        save_path=output_dir / "lstm_train_loss.png",
    )
    plot_mean_std_curve(
        lstm_test_loss_mean,
        lstm_test_loss_std,
        title="LSTM Test Loss (mean +/- std)",
        ylabel="Test Loss",
        save_path=output_dir / "lstm_test_loss.png",
    )
    plot_mean_std_curve(
        rnn_train_loss_mean,
        rnn_train_loss_std,
        title="RNN Train Loss (mean +/- std)",
        ylabel="Train Loss",
        save_path=output_dir / "rnn_train_loss.png",
    )
    plot_mean_std_curve(
        rnn_test_loss_mean,
        rnn_test_loss_std,
        title="RNN Test Loss (mean +/- std)",
        ylabel="Test Loss",
        save_path=output_dir / "rnn_test_loss.png",
    )
    plot_log_train_loss_comparison(
        {
            "RNN": (rnn_train_loss_mean, rnn_train_loss_std),
            "GFR-RNN Stage-2": (stage2_train_loss_mean, stage2_train_loss_std),
        },
        title="L-MNIST Log Train Loss: RNN vs GFR-RNN",
        save_path=output_dir / "rnn_gfr_log_train_loss.png",
    )
    plot_log_train_loss_comparison(
        {
            "RNN": (rnn_train_loss_mean, rnn_train_loss_std),
            "GFR-RNN Stage-2": (stage2_train_loss_mean, stage2_train_loss_std),
            "LSTM": (lstm_train_loss_mean, lstm_train_loss_std),
        },
        title="L-MNIST Log Train Loss: RNN, GFR-RNN, LSTM",
        save_path=output_dir / "all_models_log_train_loss.png",
    )

    summary = {
        "config": asdict(cfg),
        "device": str(device),
        "run_seeds": run_seeds,
        "stage1": {
            "checkpoint": str(stage1_ckpt),
            "history": stage1_history,
        },
        "stage2": {
            "trainable_params": int(stage2_trainable_params),
            "total_params": int(stage1_total_params),
            "final_train_accs": stage2_final_train_accs,
            "final_test_accs": stage2_final_accs,
            "train_acc_mean": stage2_train_acc_mean,
            "train_acc_std": stage2_train_acc_std,
            "acc_mean": stage2_acc_mean,
            "acc_std": stage2_acc_std,
            "train_loss_mean": stage2_train_loss_mean,
            "train_loss_std": stage2_train_loss_std,
            "test_loss_mean": stage2_test_loss_mean,
            "test_loss_std": stage2_test_loss_std,
            "histories": stage2_histories,
        },
        "lstm": {
            "hidden_size": int(lstm_hidden_size),
            "matched_trainable_params": int(lstm_params),
            "final_train_accs": lstm_final_train_accs,
            "final_test_accs": lstm_final_accs,
            "train_acc_mean": lstm_train_acc_mean,
            "train_acc_std": lstm_train_acc_std,
            "acc_mean": lstm_acc_mean,
            "acc_std": lstm_acc_std,
            "train_loss_mean": lstm_train_loss_mean,
            "train_loss_std": lstm_train_loss_std,
            "test_loss_mean": lstm_test_loss_mean,
            "test_loss_std": lstm_test_loss_std,
            "histories": lstm_histories,
        },
        "rnn": {
            "hidden_size": int(cfg.rnn_hidden_dim),
            "trainable_params": int(rnn_params),
            "final_train_accs": rnn_final_train_accs,
            "final_test_accs": rnn_final_accs,
            "train_acc_mean": rnn_train_acc_mean,
            "train_acc_std": rnn_train_acc_std,
            "acc_mean": rnn_acc_mean,
            "acc_std": rnn_acc_std,
            "train_loss_mean": rnn_train_loss_mean,
            "train_loss_std": rnn_train_loss_std,
            "test_loss_mean": rnn_test_loss_mean,
            "test_loss_std": rnn_test_loss_std,
            "histories": rnn_histories,
        },
    }

    save_loss_artifacts(output_dir, summary)
    save_json(output_dir / "results.json", summary)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Model,Trainable Params,Train Acc Mean,Train Acc Std,Test Acc Mean,Test Acc Std\n")
        f.write(
            f"GFR-RNN (Stage-2),{stage2_trainable_params},{stage2_train_acc_mean:.6f},"
            f"{stage2_train_acc_std:.6f},{stage2_acc_mean:.6f},{stage2_acc_std:.6f}\n"
        )
        f.write(
            f"RNN,{rnn_params},{rnn_train_acc_mean:.6f},{rnn_train_acc_std:.6f},"
            f"{rnn_acc_mean:.6f},{rnn_acc_std:.6f}\n"
        )
        f.write(
            f"LSTM (matched),{lstm_params},{lstm_train_acc_mean:.6f},{lstm_train_acc_std:.6f},"
            f"{lstm_acc_mean:.6f},{lstm_acc_std:.6f}\n"
        )

    logger.info("Saved results to %s", output_dir)
    logger.info(
        "Final | Stage-2 test acc mean/std: %.4f / %.4f | RNN: %.4f / %.4f | LSTM: %.4f / %.4f",
        stage2_acc_mean,
        stage2_acc_std,
        rnn_acc_mean,
        rnn_acc_std,
        lstm_acc_mean,
        lstm_acc_std,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the L-MNIST GFR-RNN/RNN/LSTM pipeline.")
    parser.add_argument("--output-root", default=None, help="Root directory for timestamped runs.")
    parser.add_argument("--output-dir", default=None, help="Exact output directory to use.")
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated CUDA device ids to use for parallel runs.")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel repeated runs.")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.output_root is not None:
        cfg.output_root = args.output_root
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    if args.no_parallel:
        cfg.parallel_runs = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. Please run on a CUDA-enabled environment.")
    gpu_ids = parse_gpu_ids(cfg.gpu_ids)
    if not gpu_ids:
        raise RuntimeError("No CUDA GPUs are visible.")
    device = torch.device(f"cuda:{gpu_ids[0]}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(cfg.output_root) / f"l_mnist_pipeline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("Starting training pipeline")
    logger.info("Output directory: %s", output_dir)
    logger.info("Config: %s", asdict(cfg))
    logger.info("CUDA device: %s", device)
    logger.info("GPU ids: %s", gpu_ids)

    run_full_pipeline(cfg, device, output_dir, logger)
    logger.info("Training pipeline complete")


if __name__ == "__main__":
    main()
