"""
Unified comparison: GFR-RNN vs SNN (LIF) vs SNN (Synaptic) on sequential MNIST.

Ensures identical train/test split, architecture size, epochs, optimizer, and
evaluation protocol across all three models.

Usage:
    conda run -n ScaleMPN python compare_models.py [--hidden_dim 256] [--epochs 30] [--lr 1e-3] [--batch_size 128] [--variant l] [--seed 42]
"""

import torch
import torch.nn.functional as F
import torchvision
import argparse
import os
import json
import numpy as np

# ── Local imports (model/network don't depend on allensdk) ──────────────
from network import Network                        # GFR-RNN
from snn_network import SNNNetwork, SNNNetworkSynaptic  # SNN baselines

# ── Reuse reshape_image from utils.py ──────────────────────────────────
from utils import reshape_image


# ═══════════════════════════════════════════════════════════════════════════
#  Data  (self-contained – avoids importing data.py which pulls allensdk)
# ═══════════════════════════════════════════════════════════════════════════
def get_mnist_loaders(batch_size, variant="l", seed=42):
    """Return train/test DataLoaders with a fixed random seed for the split."""
    size = 28 if variant == "l" else 24
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor(),
    ])
    train_set = torchvision.datasets.MNIST(
        "data/mnist/train", download=True, train=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        "data/mnist/test", download=True, train=False, transform=transform
    )
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=g,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
    )
    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════
#  Training loop  (identical for all models)
# ═══════════════════════════════════════════════════════════════════════════
def train_network(model, train_loader, epochs, lr, variant, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for x, label in train_loader:
            x = reshape_image(x, variant=variant).to(device)

            model.reset(x.shape[0])
            for i in range(x.shape[1]):
                model(x[:, i, :])

            pred_y = model(model.zero_input(x.shape[0]))
            loss = criterion(
                pred_y,
                F.one_hot(label, num_classes=10).to(torch.float32).to(device),
            )

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=False)
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")
        losses.append(total_loss)

    return losses


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation  (identical for all models — 5 zero-input readout steps)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(model, data_loader, variant, device):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for x, label in data_loader:
            x = reshape_image(x, variant=variant).to(device)
            label = label.to(device)

            model.reset(x.shape[0])
            for i in range(x.shape[1]):
                model(x[:, i, :])

            total_pred = torch.zeros(x.shape[0], 10).to(device)
            for _ in range(5):
                pred_y = model(model.zero_input(x.shape[0]))
                total_pred += F.softmax(pred_y, dim=1)

            correct += torch.sum(torch.argmax(total_pred, dim=1) == label)
            total += x.shape[0]
    model.train()
    return (correct / total).item()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="GFR-RNN vs SNN comparison on seq-MNIST")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--variant",    type=str, default="l", choices=["l", "p"])
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--beta",       type=float, default=0.95, help="LIF decay")
    parser.add_argument("--alpha",      type=float, default=0.9,  help="Synaptic current decay")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = 1 if args.variant == "p" else 28
    out_dim = 10

    print(f"Device: {device}")
    print(f"Config: hidden_dim={args.hidden_dim}, epochs={args.epochs}, lr={args.lr}, "
          f"batch_size={args.batch_size}, variant={args.variant}, seed={args.seed}\n")

    # ── Same data for all models (seeded shuffle) ──────────────────────
    train_loader, test_loader = get_mnist_loaders(
        args.batch_size, variant=args.variant, seed=args.seed
    )

    # ── Define all models ──────────────────────────────────────────────
    models = {
        "GFR-RNN": Network(
            in_dim, args.hidden_dim, out_dim,
            freeze_neurons=False, freeze_g=True, device=device,
        ),
        "SNN-LIF": SNNNetwork(
            in_dim, args.hidden_dim, out_dim,
            beta=args.beta, device=device,
        ),
        "SNN-Synaptic": SNNNetworkSynaptic(
            in_dim, args.hidden_dim, out_dim,
            alpha=args.alpha, beta=args.beta, device=device,
        ),
    }

    results = {}
    save_dir = "model/network_params"
    os.makedirs(save_dir, exist_ok=True)

    for name, model in models.items():
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'='*60}")
        print(f"Training {name}  ({n_params} trainable params)")
        print(f"{'='*60}")

        # Fix random seed before each training run for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Re-create loader to get same shuffle order for each model
        train_loader, _ = get_mnist_loaders(
            args.batch_size, variant=args.variant, seed=args.seed
        )

        losses = train_network(
            model, train_loader,
            epochs=args.epochs, lr=args.lr,
            variant=args.variant, device=device,
        )

        train_acc = evaluate(model, train_loader, args.variant, device)
        test_acc  = evaluate(model, test_loader,  args.variant, device)
        print(f"  ➜ Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}\n")

        results[name] = {
            "trainable_params": n_params,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "losses": losses,
        }

        # Save individual checkpoint
        tag = name.lower().replace("-", "_")
        ckpt_path = os.path.join(
            save_dir, f"compare_{tag}_{args.variant}_{args.hidden_dim}.pt"
        )
        torch.save({
            "model_state_dict": model.cpu().state_dict(),
            **results[name],
            "config": vars(args),
        }, ckpt_path)
        model.to(device)  # move back for any further use

    # ── Print comparison table ─────────────────────────────────────────
    print("\n" + "="*72)
    print(f"  COMPARISON TABLE  (variant={args.variant}, hidden={args.hidden_dim}, "
          f"epochs={args.epochs}, lr={args.lr})")
    print("="*72)
    print(f"{'Model':<18} {'Params':>8} {'Train Acc':>11} {'Test Acc':>11}")
    print("-"*50)
    for name, r in results.items():
        print(f"{name:<18} {r['trainable_params']:>8} "
              f"{r['train_accuracy']:>10.4f} {r['test_accuracy']:>10.4f}")
    print("-"*50)

    # ── Save full results as JSON ──────────────────────────────────────
    json_path = os.path.join(
        save_dir, f"comparison_{args.variant}_{args.hidden_dim}.json"
    )
    # Convert non-serializable types
    results_serializable = {}
    for name, r in results.items():
        results_serializable[name] = {
            "trainable_params": r["trainable_params"],
            "train_accuracy": r["train_accuracy"],
            "test_accuracy": r["test_accuracy"],
            "losses": r["losses"],
        }
    with open(json_path, "w") as f:
        json.dump({"config": vars(args), "results": results_serializable}, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
