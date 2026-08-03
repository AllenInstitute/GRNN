import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing as mp
import pickle
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from model import BatchGFR, GFR
from network import Network
from utils import reshape_image


@dataclass
class BioTableConfig:
    hidden_dim: int = 128
    num_classes: int = 10
    input_dim: int = 28
    batch_size: int = 512
    epochs: int = 300
    lr: float = 1e-3
    runs: int = 5
    base_seed: int = 1234
    bin_size: int = 20
    activation_bin_size: int = 20
    evr_threshold: float = 0.7
    bio_dataset: str = "model/best_params.pickle"
    output_root: str = "runs"
    normal_init_layer: str = "fc1"
    normal_init_mean: float = 1.5
    normal_init_std: float = 3.0
    bio_units: bool = True
    train_activation: bool = False
    num_workers: int = 2
    bio_current_scaling: str = "old_no_div"


class BiologicalInitGFRNetwork(nn.Module):
    """GFR-RNN whose hidden layer is sampled from fitted biological GFR neurons."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        neurons: Sequence[GFR],
        freeze_neurons: bool,
        freeze_g: bool = True,
        bio_units: bool = True,
        normal_init_layer: str = "fc1",
        normal_init_mean: float = 1.5,
        normal_init_std: float = 3.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if len(neurons) != hidden_dim:
            raise ValueError(f"Expected {hidden_dim} neurons, got {len(neurons)}")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        with torch.no_grad():
            if normal_init_layer in {"fc1", "both"}:
                self.fc1.weight.normal_(normal_init_mean, normal_init_std)
            if normal_init_layer in {"fc2", "both"}:
                self.fc2.weight.normal_(normal_init_mean, normal_init_std)

        self.hidden_neurons = BatchGFR(neurons, freeze_g=freeze_g, bio_units=bio_units)
        self.hidden_neurons.device = device
        if freeze_neurons:
            self.hidden_neurons.freeze_parameters()

    def reset(self, batch_size: int) -> None:
        self.hidden_neurons.reset(batch_size)
        self.xh = torch.zeros(batch_size, self.hidden_dim, device=self.device)

    def zero_input(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.in_dim, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_current = self.hidden_neurons.g.max_current
        x_in = torch.einsum("ij,j->ij", self.fc1(x), max_current)
        x_rec = self.fc2(self.xh)
        self.xh = self.hidden_neurons(x_in + x_rec)
        return self.fc3(self.xh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        raise TypeError(f"Cannot serialize {type(obj)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=convert)


def move_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_cpu(item) for item in value)
    return value


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, Any],
    metadata: Dict[str, Any],
    checkpoint_type: str,
    best_epoch: int,
    best_test_acc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": checkpoint_type,
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_test_acc": float(best_test_acc),
            "model_state_dict": move_to_cpu(model.state_dict()),
            "optimizer_state_dict": move_to_cpu(optimizer.state_dict()),
            "history": tensor_to_jsonable(history),
            "metadata": tensor_to_jsonable(metadata),
        },
        path,
    )


def tensor_to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: tensor_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_jsonable(item) for item in value]
    return value


def get_lmnist_data_loaders(
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
        ]
    )
    train_set = torchvision.datasets.MNIST("data/mnist/train", download=True, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST("data/mnist/test", download=True, train=False, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def _evr_from_payload(payload: Dict[str, Any], evr_key: str) -> float:
    if evr_key == "auto":
        for key in ("evr2", "test_evr", "noise2_evr", "val_evr", "evr1"):
            if key in payload:
                return float(payload[key])
    if evr_key in payload:
        return float(payload[evr_key])
    raise KeyError(f"EVR key {evr_key!r} not found in payload keys: {sorted(payload.keys())}")


def _records_from_best_params(
    payload: Dict[Any, Any],
    bin_size: int,
    activation_bin_size: int,
    evr_threshold: float,
    evr_key: str,
) -> List[Dict[str, Any]]:
    key = (bin_size, activation_bin_size)
    if key not in payload:
        raise KeyError(f"Dataset does not contain key {key}")
    entries = payload[key]

    records: List[Dict[str, Any]] = []
    if hasattr(entries, "iterrows"):
        for _, row in entries.iterrows():
            evr = _evr_from_payload(row, evr_key)
            if evr > evr_threshold:
                records.append(
                    {
                        "cell_id": int(row["cell_id"]),
                        "evr": evr,
                        "params": tensor_to_jsonable(row["params"]),
                    }
                )
        return records

    for cell_id, item in entries.items():
        evr = _evr_from_payload(item, evr_key)
        if evr > evr_threshold:
            records.append(
                {
                    "cell_id": int(cell_id),
                    "evr": evr,
                    "params": tensor_to_jsonable(item["params"]),
                }
            )
    return records


def _records_from_json_list(
    payload: List[Dict[str, Any]],
    bin_size: int,
    activation_bin_size: int,
    evr_threshold: float,
    evr_key: str,
) -> List[Dict[str, Any]]:
    records = []
    for item in payload:
        if int(item["bin_size"]) != bin_size or int(item["actv_bin_size"]) != activation_bin_size:
            continue
        evr = _evr_from_payload(item, evr_key)
        if evr > evr_threshold:
            records.append(
                {
                    "cell_id": int(item["cell_id"]),
                    "evr": evr,
                    "params": item["params"],
                }
            )
    return records


def load_biological_neuron_pool(
    dataset_path: Path,
    bin_size: int,
    activation_bin_size: int,
    evr_threshold: float,
    evr_key: str = "auto",
) -> List[Dict[str, Any]]:
    if dataset_path.suffix == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise TypeError(f"Expected JSON list in {dataset_path}")
        records = _records_from_json_list(payload, bin_size, activation_bin_size, evr_threshold, evr_key)
    else:
        with open(dataset_path, "rb") as f:
            payload = pickle.load(f)
        records = _records_from_best_params(payload, bin_size, activation_bin_size, evr_threshold, evr_key)

    if not records:
        raise RuntimeError(
            f"No biological neurons found in {dataset_path} for "
            f"bin_size={bin_size}, activation_bin_size={activation_bin_size}, evr>{evr_threshold}"
        )
    return records


def sample_biological_neurons(
    pool: Sequence[Dict[str, Any]],
    hidden_dim: int,
    seed: int,
) -> Tuple[List[GFR], List[int]]:
    if len(pool) < hidden_dim:
        raise RuntimeError(f"Need {hidden_dim} biological neurons, but only {len(pool)} passed the filter")

    rng = random.Random(seed)
    selected = rng.sample(list(pool), hidden_dim)
    neurons = [GFR.from_params(item["params"], freeze_g=True, bio_units=True) for item in selected]
    cell_ids = [int(item["cell_id"]) for item in selected]
    return neurons, cell_ids


def make_model(
    row_name: str,
    cfg: BioTableConfig,
    device: torch.device,
    seed: int,
    biological_pool: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[nn.Module, List[int]]:
    if row_name == "random":
        model = Network(
            in_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.num_classes,
            freeze_neurons=False,
            freeze_g=not cfg.train_activation,
            device=device,
            bio_units=False,
        ).to(device)
        return model, []

    if biological_pool is None:
        raise ValueError("biological_pool is required for biological rows")
    freeze_neurons = row_name == "bio_frozen"
    neurons, cell_ids = sample_biological_neurons(biological_pool, cfg.hidden_dim, seed)
    model = BiologicalInitGFRNetwork(
        in_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.num_classes,
        neurons=neurons,
        freeze_neurons=freeze_neurons,
        freeze_g=not cfg.train_activation,
        bio_units=cfg.bio_units,
        normal_init_layer=cfg.normal_init_layer,
        normal_init_mean=cfg.normal_init_mean,
        normal_init_std=cfg.normal_init_std,
        device=device,
    ).to(device)
    return model, cell_ids


def forward_sequence(model: nn.Module, sequence: torch.Tensor) -> torch.Tensor:
    model.reset(sequence.shape[0])
    for step in range(sequence.shape[1]):
        model(sequence[:, step, :])
    return model(model.zero_input(sequence.shape[0]))


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            sequence = reshape_image(images, variant="l").to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = forward_sequence(model, sequence)
            total_loss += criterion(logits, labels).item()

            vote_scores = F.softmax(logits, dim=1)
            for _ in range(4):
                vote_scores += F.softmax(model(model.zero_input(sequence.shape[0])), dim=1)
            predictions = torch.argmax(vote_scores, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.numel()

    return total_loss, correct / max(total, 1)


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: BioTableConfig,
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=cfg.lr)
    history: Dict[str, Any] = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    checkpoint_metadata = checkpoint_metadata or {}
    best_test_acc = -float("inf")
    best_epoch = 0
    best_checkpoint_path: Optional[Path] = None
    last_checkpoint_path: Optional[Path] = None

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            sequence = reshape_image(images, variant="l").to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = forward_sequence(model, sequence)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=False)
            optimizer.step()

            total_train_loss += loss.item()
            predictions = torch.argmax(logits.detach(), dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.numel()

        test_loss, test_acc = evaluate(model, test_loader, device)
        history["train_loss"].append(total_train_loss)
        history["train_acc"].append(train_correct / max(train_total, 1))
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if checkpoint_dir is not None and test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_checkpoint_path = checkpoint_dir / f"{checkpoint_metadata['row']}_run{checkpoint_metadata['run_idx']}_best.pt"
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch + 1,
                history,
                checkpoint_metadata,
                checkpoint_type="best",
                best_epoch=best_epoch,
                best_test_acc=best_test_acc,
            )

        print(
            f"Epoch {epoch + 1:03d}/{cfg.epochs} | "
            f"train_loss={total_train_loss:.4f} | train_acc={history['train_acc'][-1]:.4f} | "
            f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}",
            flush=True,
        )

    train_eval_loss, train_eval_acc = evaluate(model, train_loader, device)
    history["eval_train_loss"] = train_eval_loss
    history["eval_train_acc"] = train_eval_acc
    history["best_epoch"] = best_epoch
    history["best_test_acc"] = best_test_acc

    if checkpoint_dir is not None:
        last_checkpoint_path = checkpoint_dir / f"{checkpoint_metadata['row']}_run{checkpoint_metadata['run_idx']}_last.pt"
        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            cfg.epochs,
            history,
            checkpoint_metadata,
            checkpoint_type="last",
            best_epoch=best_epoch,
            best_test_acc=best_test_acc,
        )
        history["checkpoints"] = {
            "best": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
            "last": str(last_checkpoint_path),
        }
    return history


def summarize_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    test_accs = [float(run["history"]["test_acc"][-1]) for run in runs]
    train_accs = [float(run["history"]["eval_train_acc"]) for run in runs]
    trainable_params = [int(run["trainable_params"]) for run in runs]
    total_params = [int(run["total_params"]) for run in runs]
    return {
        "trainable_params": trainable_params,
        "trainable_params_mean": float(np.mean(trainable_params)),
        "trainable_params_std": float(np.std(trainable_params)),
        "total_params": total_params,
        "test_accs": test_accs,
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "train_accs": train_accs,
        "train_acc_mean": float(np.mean(train_accs)),
        "train_acc_std": float(np.std(train_accs)),
    }


def row_label(row_name: str) -> Tuple[str, str]:
    if row_name == "random":
        return "Random", "No"
    if row_name == "bio_trainable":
        return "Biological", "No"
    if row_name == "bio_frozen":
        return "Biological", "Yes"
    raise ValueError(f"Unknown row: {row_name}")


def write_summary_files(output_dir: Path, summary: Dict[str, Any]) -> None:
    rows = [row for row in ["random", "bio_trainable", "bio_frozen"] if row in summary["rows"]]

    with open(output_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row",
                "init",
                "freeze_params",
                "trainable_params_mean",
                "trainable_params_std",
                "test_acc_mean",
                "test_acc_std",
                "test_accs",
            ]
        )
        for row in rows:
            payload = summary["rows"][row]["summary"]
            init, freeze = row_label(row)
            writer.writerow(
                [
                    row,
                    init,
                    freeze,
                    payload["trainable_params_mean"],
                    payload["trainable_params_std"],
                    payload["test_acc_mean"],
                    payload["test_acc_std"],
                    " ".join(f"{acc:.6f}" for acc in payload["test_accs"]),
                ]
            )

    tex_lines = [
        "\\begin{table}[htp]",
        "    \\caption{Test accuracy on L-MNIST for GFR-RNNs with 128 hidden units. Results report mean $\\pm$ standard deviation over five random seeds.}",
        "    \\label{tab:bio-lmnist-reproduced}",
        "    \\centering",
        "    \\begin{tabular}{llll}",
        "    \\toprule",
        "    \\textbf{Init.} & \\textbf{Freeze Params.} & \\textbf{Trainable Params.} & \\textbf{Test Acc.} \\\\",
        "    \\midrule",
    ]
    for row in rows:
        payload = summary["rows"][row]["summary"]
        init, freeze = row_label(row)
        param_mean = payload["trainable_params_mean"]
        param_std = payload["trainable_params_std"]
        params = f"{param_mean:.0f}" if math.isclose(param_std, 0.0) else f"{param_mean:.0f} $\\pm$ {param_std:.0f}"
        tex_lines.append(
            f"    {init} & {freeze} & {params} & "
            f"${payload['test_acc_mean']:.3f} \\pm {payload['test_acc_std']:.3f}$ \\\\"
        )
    tex_lines.extend(
        [
            "    \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    with open(output_dir / "bio_lmnist_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))


def parse_rows(raw_rows: str) -> List[str]:
    valid = {"random", "bio_trainable", "bio_frozen"}
    rows = [row.strip() for row in raw_rows.split(",") if row.strip()]
    unknown = [row for row in rows if row not in valid]
    if unknown:
        raise ValueError(f"Unknown row(s): {unknown}. Valid rows: {sorted(valid)}")
    return rows


def parse_gpu_ids(raw_gpu_ids: str) -> List[int]:
    if raw_gpu_ids.strip():
        return [int(item) for item in raw_gpu_ids.split(",") if item.strip()]
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def run_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
    cfg = BioTableConfig(**task["config"])
    row = str(task["row"])
    run_idx = int(task["run_idx"])
    seed = int(task["seed"])
    device = torch.device(str(task["device"]))
    output_dir = Path(task["output_dir"])
    evr_key = str(task["evr_key"])
    dry_run = bool(task["dry_run"])

    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    biological_pool = None
    if row.startswith("bio_"):
        biological_pool = load_biological_neuron_pool(
            dataset_path=Path(cfg.bio_dataset),
            bin_size=cfg.bin_size,
            activation_bin_size=cfg.activation_bin_size,
            evr_threshold=cfg.evr_threshold,
            evr_key=evr_key,
        )

    model, cell_ids = make_model(row, cfg, device, seed, biological_pool)
    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)

    print(
        f"Started {row} run {run_idx} | seed={seed} | device={device} | "
        f"trainable_params={trainable_params} | total_params={total_params}",
        flush=True,
    )

    if dry_run:
        history = {"test_acc": [float("nan")], "eval_train_acc": float("nan")}
    else:
        train_loader, test_loader = get_lmnist_data_loaders(
            batch_size=cfg.batch_size,
            seed=seed,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        checkpoint_metadata = {
            "row": row,
            "run_idx": run_idx,
            "seed": seed,
            "device": str(device),
            "cell_ids": cell_ids,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "config": asdict(cfg),
        }
        history = train_one_run(
            model,
            train_loader,
            test_loader,
            cfg,
            device,
            checkpoint_dir=output_dir / "checkpoints",
            checkpoint_metadata=checkpoint_metadata,
        )

    result = {
        "row": row,
        "run_idx": run_idx,
        "seed": seed,
        "device": str(device),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "cell_ids": cell_ids,
        "history": history,
    }
    save_json(output_dir / "histories" / f"{row}_run{run_idx}.json", result)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_tasks_parallel(
    tasks: Sequence[Dict[str, Any]],
    max_workers: int,
) -> List[Dict[str, Any]]:
    ctx = mp.get_context("spawn")
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        future_to_task = {executor.submit(run_single_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results.append(result)
            print(
                f"Completed {task['row']} run {task['run_idx']} on {task['device']} | "
                f"test_acc={result['history']['test_acc'][-1]}",
                flush=True,
            )
    results.sort(key=lambda item: (str(item["row"]), int(item["run_idx"])))
    return results


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = BioTableConfig(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        runs=args.runs,
        base_seed=args.base_seed,
        bin_size=args.bin_size,
        activation_bin_size=args.activation_bin_size,
        evr_threshold=args.evr_threshold,
        bio_dataset=str(args.bio_dataset),
        output_root=str(args.output_root),
        normal_init_layer=args.normal_init_layer,
        normal_init_mean=args.normal_init_mean,
        normal_init_std=args.normal_init_std,
        bio_units=not args.no_bio_units,
        train_activation=args.train_activation,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(cfg.output_root) / f"l_mnist_bio_table_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "histories").mkdir(exist_ok=True)

    biological_pool: Optional[List[Dict[str, Any]]] = None
    if any(row.startswith("bio_") for row in args.rows):
        biological_pool = load_biological_neuron_pool(
            dataset_path=args.bio_dataset,
            bin_size=cfg.bin_size,
            activation_bin_size=cfg.activation_bin_size,
            evr_threshold=cfg.evr_threshold,
            evr_key=args.evr_key,
        )
        print(
            f"Loaded {len(biological_pool)} biological neurons from {args.bio_dataset} "
            f"(bin_size={cfg.bin_size}, activation_bin_size={cfg.activation_bin_size}, evr>{cfg.evr_threshold}).",
            flush=True,
        )

    summary: Dict[str, Any] = {
        "config": asdict(cfg),
        "device": str(device),
        "rows": {},
    }

    if args.parallel:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if device.type == "cuda" and not gpu_ids:
            raise RuntimeError("No CUDA GPUs found. Pass --gpu-ids or use --device cpu.")
        devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids] if device.type == "cuda" else [str(device)]
        tasks = []
        cfg_payload = asdict(cfg)
        for row in args.rows:
            for run_idx in range(1, cfg.runs + 1):
                task_idx = len(tasks)
                tasks.append(
                    {
                        "config": cfg_payload,
                        "row": row,
                        "run_idx": run_idx,
                        "seed": cfg.base_seed + run_idx - 1,
                        "device": devices[task_idx % len(devices)],
                        "output_dir": str(output_dir),
                        "evr_key": args.evr_key,
                        "dry_run": args.dry_run,
                    }
                )

        max_workers = min(len(devices), len(tasks))
        print(f"Launching {len(tasks)} task(s) with {max_workers} worker(s): {devices}", flush=True)
        parallel_results = run_tasks_parallel(tasks, max_workers=max_workers)
        for row in args.rows:
            row_runs = [result for result in parallel_results if result["row"] == row]
            row_runs.sort(key=lambda item: int(item["run_idx"]))
            summary["rows"][row] = {
                "runs": row_runs,
                "summary": summarize_runs(row_runs),
            }

        save_json(output_dir / "results.json", summary)
        write_summary_files(output_dir, summary)
        print(f"\nSaved results to {output_dir}", flush=True)
        return summary

    for row in args.rows:
        row_runs = []
        print(f"\n=== Row: {row} ===", flush=True)
        for run_idx in range(1, cfg.runs + 1):
            seed = cfg.base_seed + run_idx - 1
            set_seed(seed)
            model, cell_ids = make_model(row, cfg, device, seed, biological_pool)
            trainable_params = count_parameters(model, trainable_only=True)
            total_params = count_parameters(model, trainable_only=False)
            print(
                f"Run {run_idx}/{cfg.runs} | seed={seed} | "
                f"trainable_params={trainable_params} | total_params={total_params}",
                flush=True,
            )

            if args.dry_run:
                history = {"test_acc": [float("nan")], "eval_train_acc": float("nan")}
            else:
                train_loader, test_loader = get_lmnist_data_loaders(
                    batch_size=cfg.batch_size,
                    seed=seed,
                    num_workers=cfg.num_workers,
                    pin_memory=pin_memory,
                )
                checkpoint_metadata = {
                    "row": row,
                    "run_idx": run_idx,
                    "seed": seed,
                    "device": str(device),
                    "cell_ids": cell_ids,
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "config": asdict(cfg),
                }
                history = train_one_run(
                    model,
                    train_loader,
                    test_loader,
                    cfg,
                    device,
                    checkpoint_dir=output_dir / "checkpoints",
                    checkpoint_metadata=checkpoint_metadata,
                )

            result = {
                "row": row,
                "run_idx": run_idx,
                "seed": seed,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "cell_ids": cell_ids,
                "history": history,
            }
            row_runs.append(result)
            save_json(output_dir / "histories" / f"{row}_run{run_idx}.json", result)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary["rows"][row] = {
            "runs": row_runs,
            "summary": summarize_runs(row_runs),
        }

    save_json(output_dir / "results.json", summary)
    write_summary_files(output_dir, summary)
    print(f"\nSaved results to {output_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the L-MNIST biological-initialization table for 128-hidden-unit "
            "GFR-RNNs, with five runs for random and biological rows."
        )
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu. Defaults to CUDA if visible.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parallel", action="store_true", help="Run row/seed tasks in parallel.")
    parser.add_argument("--gpu-ids", default="", help="Comma-separated CUDA ids for --parallel. Defaults to all visible GPUs.")
    parser.add_argument(
        "--rows",
        type=parse_rows,
        default=parse_rows("random,bio_trainable,bio_frozen"),
        help="Comma-separated subset of: random,bio_trainable,bio_frozen",
    )

    parser.add_argument("--bio-dataset", type=Path, default=Path("model/best_params.pickle"))
    parser.add_argument("--bin-size", type=int, default=20)
    parser.add_argument("--activation-bin-size", type=int, default=20)
    parser.add_argument("--evr-threshold", type=float, default=0.7)
    parser.add_argument(
        "--evr-key",
        default="auto",
        help="EVR field to threshold. Use auto for evr2/test_evr depending on the dataset format.",
    )
    parser.add_argument(
        "--normal-init-layer",
        choices=("none", "fc1", "fc2", "both"),
        default="fc1",
        help=(
            "Layer whose weights are sampled from N(mean, std) for biological rows. "
            "fc1 matches the existing BiologicalGFRNetwork implementation; fc2 follows the manuscript wording literally."
        ),
    )
    parser.add_argument("--normal-init-mean", type=float, default=1.5)
    parser.add_argument("--normal-init-std", type=float, default=3.0)
    parser.add_argument("--no-bio-units", action="store_true", help="Disable biological-unit recurrence scaling for biological rows.")
    parser.add_argument("--train-activation", action="store_true", help="Also train the polynomial activation parameters g.")
    parser.add_argument("--dry-run", action="store_true", help="Build models and record parameter counts without training.")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
