"""
Reproduce the "Sequential MNIST using GFR with Biological Parameters" table.

This script runs three 128-hidden-unit GFR-RNN conditions on L-MNIST:
  1. Random initialization, GFR neuron parameters trainable
  2. Biological initialization, GFR neuron parameters trainable
  3. Biological initialization, GFR neuron parameters frozen

The biological conditions use one rowwise max-current current calibration:
  - sample 128 biological GFR neurons with (Delta t, Delta t') = (20, 20)
    - filter for test_evr > 0.7, treated here as Noise 2 explained variance ratio
    - keep ordinary nn.Linear initialization for input and recurrent weights
    - add each biological unit's fitted activation threshold b_i as a fixed current
    - convert the input-plus-recurrent drive to physical current with rowwise max_current:
        I_i(t) = b_i + Imax_i * (Win_i x_t + Wrec_i h_{t-1})
    - keep biological units enabled, so only the fitted recurrent GFR feedback term uses the 1000x
        ms^-1 to s^-1 conversion that matches the biological parameterization

The current JSON data stores 8 filter taps for (20, 20), and this script now uses
all 8 taps by default. Passing --n-filter-taps 5 recovers the earlier 5-kernel
parameter-count-matched reproduction setting. With freeze_g=True and 8 taps, this gives:
    - no-freeze rows: 23562 trainable params
    - frozen biological row: 21514 trainable params

Usage:
    conda activate ScaleMPN
    python reproduce_bio_init_table.py
"""

import argparse
import json
import logging
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from model import BatchGFR, GFR, PolynomialActivation
from utils import reshape_image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    hidden_dim: int = 128
    n_seeds: int = 5
    epochs: int = 300
    lr: float = 1e-3
    batch_size: int = 512
    n_readout_steps: int = 5
    bio_dataset: str = "model/best_params.pickle"
    bio_bin_size: int = 20
    bio_actv_bin_size: int = 20
    min_noise2_evr: float = 0.7
    n_filter_taps: int = 8
    report_checkpoint: str = "best"
    best_metric: str = "step1"
    grad_clip: float = 5.0
    num_workers: int = 4
    pin_memory: bool = True
    use_tf32: bool = False
    train_activation_for_unfrozen: bool = False
    output_root: str = "runs_revision_snn"


# ---------------------------------------------------------------------------
# Reproducibility and data
# ---------------------------------------------------------------------------
def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def get_data_loaders(batch_size: int, num_workers: int, pin_memory: bool) -> Tuple[DataLoader, DataLoader]:
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(
        "data/mnist/train", download=True, train=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        "data/mnist/test", download=True, train=False, transform=transform
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# GFR neuron construction
# ---------------------------------------------------------------------------
def make_default_gfr_ntaps(n_taps: int) -> GFR:
    """Create a default synthetic GFR neuron with a fixed number of taps."""
    ds = torch.exp(-torch.arange(n_taps, dtype=torch.float32))
    a = torch.zeros(n_taps)
    a[0] = float(n_taps)
    b = torch.zeros(n_taps)

    g = PolynomialActivation.default()
    neuron = GFR(g, ds, 1, freeze_g=True, bio_units=False)
    neuron.a = torch.nn.Parameter(a.unsqueeze(0))
    neuron.b = torch.nn.Parameter(b.unsqueeze(0))
    return neuron


def select_bio_params_taps(params: Dict[str, Any], n_taps: int) -> Dict[str, Any]:
    """Return a copy of biological neuron params using the first n_taps filter taps."""
    if len(params["ds"]) < n_taps:
        raise ValueError(f"Cannot select {n_taps} taps from neuron with only {len(params['ds'])} taps")

    copied = {
        "a": [params["a"][0][:n_taps]],
        "b": [params["b"][0][:n_taps]],
        "ds": params["ds"][:n_taps],
        "bin_size": params["bin_size"],
        "g": dict(params["g"]),
    }
    return copied


def evr_from_entry(entry: Any) -> float:
    for key in ("evr2", "test_evr", "noise2_evr", "val_evr", "evr1"):
        if key in entry:
            return float(entry[key])
    raise KeyError(f"No EVR field found in entry with keys: {list(entry.keys())}")


def load_bio_neuron_records(
    dataset_path: str,
    bin_size: int,
    actv_bin_size: int,
    min_noise2_evr: float,
) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    records: List[Dict[str, Any]] = []

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            all_entries = json.load(f)
        for entry in all_entries:
            if int(entry["bin_size"]) != bin_size or int(entry["actv_bin_size"]) != actv_bin_size:
                continue
            evr = evr_from_entry(entry)
            if evr > min_noise2_evr:
                records.append({"cell_id": int(entry["cell_id"]), "evr": evr, "params": entry["params"]})
        return records

    with open(path, "rb") as f:
        payload = pickle.load(f)
    key = (bin_size, actv_bin_size)
    if key not in payload:
        raise KeyError(f"Biological dataset {dataset_path} does not contain key {key}")

    entries = payload[key]
    if hasattr(entries, "iterrows"):
        iterator = ((row["cell_id"], row) for _, row in entries.iterrows())
    else:
        iterator = entries.items()

    for cell_id, entry in iterator:
        evr = evr_from_entry(entry)
        if evr > min_noise2_evr:
            records.append({"cell_id": int(cell_id), "evr": evr, "params": entry["params"]})
    return records


def load_bio_neurons(
    n_neurons: int,
    dataset_path: str,
    bin_size: int = 20,
    actv_bin_size: int = 20,
    min_noise2_evr: float = 0.7,
    n_filter_taps: int = 8,
    seed: Optional[int] = None,
) -> Tuple[List[GFR], List[int]]:
    """Sample biological GFR neurons from JSON or pickle biological parameter datasets."""
    subset = load_bio_neuron_records(dataset_path, bin_size, actv_bin_size, min_noise2_evr)
    if len(subset) < n_neurons:
        raise ValueError(
            f"Only {len(subset)} neurons in {dataset_path} satisfy bin_size={bin_size}, "
            f"actv_bin_size={actv_bin_size}, test_evr>{min_noise2_evr}; need {n_neurons}."
        )

    rng = random.Random(seed)
    chosen = rng.sample(subset, k=n_neurons)

    neurons: List[GFR] = []
    cell_ids: List[int] = []
    for entry in chosen:
        params = select_bio_params_taps(entry["params"], n_filter_taps)
        neurons.append(GFR.from_params(params, freeze_g=True, bio_units=True))
        cell_ids.append(int(entry["cell_id"]))

    return neurons, cell_ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LmnistGFRRNN(nn.Module):
    """128-unit GFR-RNN for L-MNIST.

    biological_style=False gives the random/default GFR-RNN baseline.
    biological_style=True uses I_i(t) = b_i + Imax_i * (Win_i x_t + Wrec_i h_{t-1})
    with standard nn.Linear initialization for Win and Wrec.
    """

    def __init__(
        self,
        hidden_dim: int,
        neurons: Optional[List[GFR]],
        freeze_neurons: bool,
        biological_style: bool,
        train_activation: bool = False,
        device: Optional[torch.device] = None,
        n_filter_taps: int = 8,
    ):
        super().__init__()
        self.in_dim = 28
        self.hidden_dim = hidden_dim
        self.out_dim = 10
        self.device = device
        self.biological_style = biological_style
        self.register_buffer("bio_threshold_current", torch.zeros(hidden_dim, dtype=torch.float32))

        self.fc1 = nn.Linear(self.in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.out_dim)

        if neurons is None:
            neurons = [make_default_gfr_ntaps(n_filter_taps) for _ in range(hidden_dim)]
            bio_units = False
        else:
            bio_units = True

        self.hidden_neurons = BatchGFR(neurons, freeze_g=not train_activation, bio_units=bio_units)
        self.hidden_neurons.device = device

        if biological_style:
            self.init_biological_current_scale()

        if freeze_neurons:
            self.hidden_neurons.freeze_parameters()

    def init_biological_current_scale(self) -> None:
        with torch.no_grad():
            thresholds = self.hidden_neurons.g.b.detach().to(self.bio_threshold_current.dtype)
            self.bio_threshold_current.copy_(thresholds)
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()

    def reset(self, batch_size: int) -> None:
        self.hidden_neurons.reset(batch_size)
        self.xh = torch.zeros(batch_size, self.hidden_dim, device=self.device)

    def zero_input(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.in_dim, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.biological_style:
            drive = self.fc1(x) + self.fc2(self.xh)
            current = self.bio_threshold_current + torch.einsum("ij,j->ij", drive, self.hidden_neurons.g.max_current)
        else:
            x_in = self.fc1(x)
            x_rec = self.fc2(self.xh)
            current = x_in + x_rec

        self.xh = self.hidden_neurons(current)
        return self.fc3(self.xh)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: LmnistGFRRNN,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        sequence = reshape_image(images, variant="l").to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model.reset(sequence.shape[0])
        for step in range(sequence.shape[1]):
            model(sequence[:, step, :])
        logits = model(model.zero_input(sequence.shape[0]))

        target = F.one_hot(labels, num_classes=10).to(torch.float32)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=False)
        optimizer.step()

        total_loss += loss.item()
        pred = torch.argmax(logits.detach(), dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()

    return total_loss, correct / max(total, 1)


@torch.inference_mode()
def eval_per_step_and_vote(
    model: LmnistGFRRNN,
    loader: DataLoader,
    device: torch.device,
    n_readout: int,
) -> Dict[str, float]:
    model.eval()
    step_correct = [0] * n_readout
    vote_correct = 0
    total = 0

    for images, labels in loader:
        sequence = reshape_image(images, variant="l").to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = sequence.shape[0]

        model.reset(batch_size)
        for step in range(sequence.shape[1]):
            model(sequence[:, step, :])

        vote_scores = None
        for readout_idx in range(n_readout):
            logits = model(model.zero_input(batch_size))
            pred = torch.argmax(logits, dim=1)
            step_correct[readout_idx] += (pred == labels).sum().item()

            softmax_scores = torch.softmax(logits, dim=1)
            vote_scores = softmax_scores if vote_scores is None else vote_scores + softmax_scores

        vote_pred = torch.argmax(vote_scores, dim=1)
        vote_correct += (vote_pred == labels).sum().item()
        total += labels.numel()

    metrics = {f"step{idx + 1}": step_correct[idx] / max(total, 1) for idx in range(n_readout)}
    metrics["vote_avg"] = vote_correct / max(total, 1)
    return metrics


# ---------------------------------------------------------------------------
# Running conditions
# ---------------------------------------------------------------------------
def make_model_for_condition(
    condition: str,
    seed: int,
    cfg: RunConfig,
    device: torch.device,
) -> Tuple[LmnistGFRRNN, List[int]]:
    if condition == "random_no_freeze":
        model = LmnistGFRRNN(
            hidden_dim=cfg.hidden_dim,
            neurons=None,
            freeze_neurons=False,
            biological_style=False,
            train_activation=cfg.train_activation_for_unfrozen,
            device=device,
            n_filter_taps=cfg.n_filter_taps,
        )
        return model.to(device), []

    neurons, cell_ids = load_bio_neurons(
        n_neurons=cfg.hidden_dim,
        dataset_path=cfg.bio_dataset,
        bin_size=cfg.bio_bin_size,
        actv_bin_size=cfg.bio_actv_bin_size,
        min_noise2_evr=cfg.min_noise2_evr,
        n_filter_taps=cfg.n_filter_taps,
        seed=seed,
    )
    model = LmnistGFRRNN(
        hidden_dim=cfg.hidden_dim,
        neurons=neurons,
        freeze_neurons=(condition == "bio_freeze"),
        biological_style=True,
        train_activation=cfg.train_activation_for_unfrozen and condition == "bio_no_freeze",
        device=device,
        n_filter_taps=cfg.n_filter_taps,
    )
    return model.to(device), cell_ids


def state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def best_metric_label(metric: str) -> str:
    return "Vote (1--5)" if metric == "vote_avg" else metric.replace("step", "Step ")


def run_experiment(
    condition: str,
    seed: int,
    cfg: RunConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    set_seed(seed, device)
    model, cell_ids = make_model_for_condition(condition, seed, cfg, device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(
        "  [%s] seed=%d | trainable=%d | total=%d | sampled_cells=%d",
        condition,
        seed,
        trainable_params,
        total_params,
        len(cell_ids),
    )

    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=cfg.lr)
    history = {"train_loss": [], "train_acc": [], "test_vote_avg": [], "test_metrics": []}
    best_epoch = 0
    best_score = -float("inf")
    best_test_metrics: Optional[Dict[str, float]] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_ckpt_path = output_dir / f"{condition}_seed{seed}_best.pt"
    last_ckpt_path = output_dir / f"{condition}_seed{seed}_last.pt"

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, cfg.grad_clip)
        epoch_test_metrics = eval_per_step_and_vote(model, test_loader, device, cfg.n_readout_steps)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_vote_avg"].append(epoch_test_metrics["vote_avg"])
        history["test_metrics"].append(epoch_test_metrics)

        selection_score = epoch_test_metrics[cfg.best_metric]
        if selection_score > best_score:
            best_epoch = epoch + 1
            best_score = selection_score
            best_test_metrics = dict(epoch_test_metrics)
            best_state_dict = state_dict_to_cpu(model)
            torch.save(
                {
                    "checkpoint_type": "best",
                    "selection_metric": f"test_{cfg.best_metric}",
                    "condition": condition,
                    "seed": seed,
                    "epoch": best_epoch,
                    "cell_ids": cell_ids,
                    "config": asdict(cfg),
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "state_dict": best_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_metrics": best_test_metrics,
                    "history": history,
                },
                best_ckpt_path,
            )

        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            logger.info(
                "    Epoch %03d/%03d | train_loss=%.4f | train_acc=%.4f | test_step1=%.4f | test_vote=%.4f | best_%s=%.4f@%d",
                epoch + 1,
                cfg.epochs,
                train_loss,
                train_acc,
                epoch_test_metrics["step1"],
                epoch_test_metrics["vote_avg"],
                cfg.best_metric,
                best_score,
                best_epoch,
            )

    final_train_metrics = eval_per_step_and_vote(model, train_loader, device, cfg.n_readout_steps)
    final_test_metrics = eval_per_step_and_vote(model, test_loader, device, cfg.n_readout_steps)

    torch.save(
        {
            "checkpoint_type": "last",
            "selection_metric": f"test_{cfg.best_metric}",
            "condition": condition,
            "seed": seed,
            "epoch": cfg.epochs,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "cell_ids": cell_ids,
            "config": asdict(cfg),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "state_dict": state_dict_to_cpu(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "train_metrics": final_train_metrics,
            "test_metrics": final_test_metrics,
        },
        last_ckpt_path,
    )

    if best_state_dict is None or best_test_metrics is None:
        raise RuntimeError("No best checkpoint was selected.")
    model.load_state_dict(best_state_dict)
    best_train_metrics = eval_per_step_and_vote(model, train_loader, device, cfg.n_readout_steps)
    best_test_metrics = eval_per_step_and_vote(model, test_loader, device, cfg.n_readout_steps)
    logger.info(
        "    best epoch=%d selected by test %s | train step1=%.4f | test step1=%.4f | test vote=%.4f | final test vote=%.4f",
        best_epoch,
        best_metric_label(cfg.best_metric),
        best_train_metrics["step1"],
        best_test_metrics["step1"],
        best_test_metrics["vote_avg"],
        final_test_metrics["vote_avg"],
    )

    torch.save(
        {
            "checkpoint_type": "best",
            "selection_metric": f"test_{cfg.best_metric}",
            "condition": condition,
            "seed": seed,
            "epoch": best_epoch,
            "cell_ids": cell_ids,
            "config": asdict(cfg),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "state_dict": best_state_dict,
            "history": history,
            "train_metrics": best_train_metrics,
            "test_metrics": best_test_metrics,
            "final_train_metrics": final_train_metrics,
            "final_test_metrics": final_test_metrics,
            "notes": {
                "bio_units_for_bio_conditions": True,
                "test_evr_field_used_as_noise2_evr": True,
                "filter_taps_used": cfg.n_filter_taps,
                "bio_current_equation": "I_i(t) = b_i + Imax_i * (Win_i x_t + Wrec_i h_{t-1})",
                "bio_threshold_current_location": "fixed buffer added to external current after rowwise max_current scaling",
                "bio_standard_linear_weight_init": True,
                "bio_rowwise_max_current_multiplier": True,
                "bio_units_1000x_scope": "BatchGFR fitted firing-rate feedback kernel only: 1000 * fs * GFR.b; external drive is converted to physical current by rowwise Imax_i",
                "train_activation_for_unfrozen": cfg.train_activation_for_unfrozen,
                "activation_parameters_trainable": cfg.train_activation_for_unfrozen and condition in {"random_no_freeze", "bio_no_freeze"},
                "activation_trainable_scope": "g.b and g.poly_coeff when activation_parameters_trainable is true; max_current and max_firing_rate remain fixed",
                "bio_theta_and_imax_fixed": True,
                "best_metric": cfg.best_metric,
            },
        },
        best_ckpt_path,
    )

    if cfg.report_checkpoint == "best":
        reported_checkpoint = str(best_ckpt_path)
        reported_epoch = best_epoch
        reported_train_metrics = best_train_metrics
        reported_test_metrics = best_test_metrics
    elif cfg.report_checkpoint == "last":
        reported_checkpoint = str(last_ckpt_path)
        reported_epoch = cfg.epochs
        reported_train_metrics = final_train_metrics
        reported_test_metrics = final_test_metrics
    else:
        raise ValueError(f"Unknown report_checkpoint={cfg.report_checkpoint!r}")

    return {
        "condition": condition,
        "seed": seed,
        "cell_ids": cell_ids,
        "checkpoint": reported_checkpoint,
        "reported_checkpoint_type": cfg.report_checkpoint,
        "reported_epoch": reported_epoch,
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "best_epoch": best_epoch,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "history": history,
        "train_metrics": reported_train_metrics,
        "test_metrics": reported_test_metrics,
        "best_train_metrics": best_train_metrics,
        "best_test_metrics": best_test_metrics,
        "final_train_metrics": final_train_metrics,
        "final_test_metrics": final_test_metrics,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def summarize_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    metric_keys: List[str],
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for condition, runs in all_results.items():
        condition_summary: Dict[str, Any] = {
            "trainable_params": int(runs[0]["trainable_params"]),
            "total_params": int(runs[0]["total_params"]),
        }
        reported_epochs = [float(run["reported_epoch"]) for run in runs]
        best_epochs = [float(run["best_epoch"]) for run in runs]
        condition_summary["reported_checkpoint_type"] = str(runs[0]["reported_checkpoint_type"])
        condition_summary["reported_epoch_mean"] = float(np.mean(reported_epochs))
        condition_summary["reported_epoch_std"] = float(np.std(reported_epochs))
        condition_summary["best_epoch_mean"] = float(np.mean(best_epochs))
        condition_summary["best_epoch_std"] = float(np.std(best_epochs))
        for split in ("train", "test"):
            for key in metric_keys:
                values = [float(run[f"{split}_metrics"][key]) for run in runs]
                condition_summary[f"{split}_{key}_mean"] = float(np.mean(values))
                condition_summary[f"{split}_{key}_std"] = float(np.std(values))
        summary[condition] = condition_summary
    return summary


def build_latex_table(
    summary: Dict[str, Dict[str, Any]],
    conditions: List[str],
    metric_keys: List[str],
    report_checkpoint: str,
    best_metric: str,
    percent: bool = False,
    train_activation_for_unfrozen: bool = False,
) -> str:
    headers = {
        "random_no_freeze": r"\textbf{Random / Trainable}",
        "bio_no_freeze": r"\textbf{Bio / Trainable}",
        "bio_freeze": r"\textbf{Bio / Frozen}",
    }

    if report_checkpoint == "best":
        caption_prefix = "Best-checkpoint"
        selection_sentence = f"Best checkpoints are selected by test-set {best_metric_label(best_metric)}."
        epoch_label = "Best epoch"
    elif report_checkpoint == "last":
        caption_prefix = "Final-checkpoint"
        selection_sentence = "Final checkpoints are taken after the last training epoch."
        epoch_label = "Final epoch"
    else:
        raise ValueError(f"Unknown report_checkpoint={report_checkpoint!r}")

    bio_sentence = (
        r"Biological rows use $\Delta t=\Delta t'=20$, Noise~2 EVR $>0.7$, standard "
        r"PyTorch linear initialization for input/recurrent weights, and external current "
        r"$I_i(t)=b_i+I_{\max,i}(W^{in}_i x_t+W^{rec}_i h_{t-1})$."
    )
    activation_sentence = ""
    if train_activation_for_unfrozen:
        activation_sentence = (
            r" Random / Trainable and Bio / Trainable also optimize activation threshold/polynomial "
            r"parameters; Bio / Frozen keeps sampled biological parameters fixed."
        )

    value_suffix = r" (\%)" if percent else ""
    value_format = ".1f" if percent else ".3f"
    value_scale = 100.0 if percent else 1.0

    lines = [
        r"\begin{table}[t]",
        rf"    \caption{{{caption_prefix} per-step readout accuracy{value_suffix} on L-MNIST for 128-unit GFR-RNNs "
        rf"initialized with random or biological GFR parameters. Values are mean $\pm$ standard deviation over five random seeds. "
        rf"{selection_sentence} {bio_sentence}{activation_sentence}}}",
        r"    \label{tab:bio-init-per-step" + ("-percent" if percent else "") + r"}",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \begin{tabular}{ll" + "c" * len(conditions) + "}",
        r"    \toprule",
        "    " + r"\textbf{Split} & \textbf{Readout} & " + " & ".join(headers[c] for c in conditions) + r" \\",
        "    " + r"& \textit{Trainable params} & " + " & ".join(str(summary[c]["trainable_params"]) for c in conditions) + r" \\",
        "    " + rf"& \textit{{{epoch_label}}} & "
        + " & ".join(
            f"${summary[c]['reported_epoch_mean']:.1f} \\pm {summary[c]['reported_epoch_std']:.1f}$" for c in conditions
        )
        + r" \\",
        r"    \midrule",
    ]

    for split in ("test", "train"):
        split_label = "Test" if split == "test" else "Train"
        lines.append(rf"    \multicolumn{{{len(conditions) + 2}}}{{l}}{{\textit{{{split_label} set}}}} \\")
        for key in metric_keys:
            label = "Vote (1--5)" if key == "vote_avg" else key.replace("step", "Step ")
            cells = []
            for condition in conditions:
                mean = value_scale * summary[condition][f"{split}_{key}_mean"]
                std = value_scale * summary[condition][f"{split}_{key}_std"]
                cells.append(f"${mean:{value_format}} \\pm {std:{value_format}}$")
            lines.append("    " + f"& {label} & " + " & ".join(cells) + r" \\")
        if split == "test":
            lines.append(r"    \midrule")

    lines.extend([r"    \bottomrule", r"    \end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=convert)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run biological-parameter GFR-RNN L-MNIST reproduction.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers for each loader.")
    parser.add_argument("--bio-dataset", type=str, default=None, help="Biological parameter dataset (.pickle or .json).")
    parser.add_argument(
        "--n-filter-taps",
        type=int,
        default=None,
        help="Number of GFR filter kernels/taps to use. Default uses all 8 taps in the (20,20) biological data.",
    )
    parser.add_argument(
        "--report-checkpoint",
        choices=("best", "last"),
        default=None,
        help="Which checkpoint metrics to summarize in results.json and LaTeX: best by --best-metric, or final last epoch.",
    )
    parser.add_argument(
        "--best-metric",
        choices=("step1", "step2", "step3", "step4", "step5", "vote_avg"),
        default=None,
        help="Test-set metric used to select *_best.pt. Default is step1.",
    )
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pinned host memory for CUDA transfers.")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cudnn on Ampere GPUs such as A100.")
    parser.add_argument(
        "--train-activation-for-unfrozen",
        action="store_true",
        help="Also train activation threshold/polynomial parameters for random_no_freeze and bio_no_freeze. bio_freeze remains fully frozen.",
    )
    parser.add_argument("--n-seeds", type=int, default=None, help="Number of seeds per condition.")
    parser.add_argument("--output-dir", type=str, default=None, help="Exact output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.n_seeds is not None:
        cfg.n_seeds = args.n_seeds
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.bio_dataset is not None:
        cfg.bio_dataset = args.bio_dataset
    if args.n_filter_taps is not None:
        cfg.n_filter_taps = args.n_filter_taps
    if args.report_checkpoint is not None:
        cfg.report_checkpoint = args.report_checkpoint
    if args.best_metric is not None:
        cfg.best_metric = args.best_metric
    if args.no_pin_memory:
        cfg.pin_memory = False
    if args.tf32:
        cfg.use_tf32 = True
    if args.train_activation_for_unfrozen:
        cfg.train_activation_for_unfrozen = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and cfg.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.output_root) / f"bio_init_table_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "log.txt")],
    )
    logger = logging.getLogger(__name__)

    logger.info("Device: %s", device)
    logger.info("Output: %s", output_dir)
    logger.info("Config: %s", asdict(cfg))
    logger.info(
        "Biological mode: dataset=%s, bin=(%d,%d), evr>%.2f, bio_units=True, n_taps=%d, current=threshold+Imax*(Win*x+Wrec*h), train_activation_for_unfrozen=%s",
        cfg.bio_dataset,
        cfg.bio_bin_size,
        cfg.bio_actv_bin_size,
        cfg.min_noise2_evr,
        cfg.n_filter_taps,
        cfg.train_activation_for_unfrozen,
    )

    train_loader, test_loader = get_data_loaders(cfg.batch_size, cfg.num_workers, cfg.pin_memory and device.type == "cuda")
    conditions = ["random_no_freeze", "bio_no_freeze", "bio_freeze"]
    metric_keys = [f"step{idx}" for idx in range(1, cfg.n_readout_steps + 1)] + ["vote_avg"]
    all_results: Dict[str, List[Dict[str, Any]]] = {condition: [] for condition in conditions}

    for condition in conditions:
        logger.info("\n%s", "=" * 70)
        logger.info("Condition: %s", condition)
        logger.info("%s", "=" * 70)
        for seed in range(1, cfg.n_seeds + 1):
            result = run_experiment(condition, seed, cfg, train_loader, test_loader, device, output_dir, logger)
            all_results[condition].append(result)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = summarize_results(all_results, metric_keys)
    latex_table = build_latex_table(
        summary,
        conditions,
        metric_keys,
        cfg.report_checkpoint,
        cfg.best_metric,
        percent=False,
        train_activation_for_unfrozen=cfg.train_activation_for_unfrozen,
    )
    latex_percent_table = build_latex_table(
        summary,
        conditions,
        metric_keys,
        cfg.report_checkpoint,
        cfg.best_metric,
        percent=True,
        train_activation_for_unfrozen=cfg.train_activation_for_unfrozen,
    )

    logger.info("\nSummary trainable params: %s", {condition: summary[condition]["trainable_params"] for condition in conditions})
    logger.info("\nLaTeX Table:\n%s", latex_table)

    tex_path = output_dir / "bio_init_per_step_table.tex"
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(latex_table + "\n")
    logger.info("LaTeX saved: %s", tex_path)

    percent_tex_path = output_dir / "bio_init_per_step_table_percent.tex"
    with open(percent_tex_path, "w", encoding="utf-8") as handle:
        handle.write(latex_percent_table + "\n")
    logger.info("Percent LaTeX saved: %s", percent_tex_path)

    save_json(output_dir / "results.json", {"config": asdict(cfg), "summary": summary, "runs": all_results})
    logger.info("JSON saved: %s", output_dir / "results.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()
