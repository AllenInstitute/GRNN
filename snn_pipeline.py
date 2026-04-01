import torch
import torch.nn.functional as F
import argparse
import numpy as np
import os

from snn_network import SNNNetwork, SNNNetworkSynaptic
from utils import reshape_image

import torchvision


def get_MNIST_data_loaders(batch_size, variant="l"):
    size = 28 if variant == "l" else 24
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor()
    ])
    train_set = torchvision.datasets.MNIST('data/mnist/train', download=True, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST('data/mnist/test', download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training  (mirrors train.train_network but works with SNN reset/forward)
# ---------------------------------------------------------------------------
def train_snn(model, train_loader, epochs=30, lr=1e-3, variant="l", device=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for x, label in train_loader:
            x = reshape_image(x, variant=variant).to(device)

            # sequentially send input into network  — same loop as GFR-RNN
            model.reset(x.shape[0])
            for i in range(x.shape[1]):
                model(x[:, i, :])

            # one extra readout step with zero input (same as GFR-RNN)
            pred_y = model(model.zero_input(x.shape[0]))
            loss = criterion(
                pred_y,
                F.one_hot(label, num_classes=10).to(torch.float32).to(device),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=False)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
        losses.append(total_loss)

    return losses


# ---------------------------------------------------------------------------
# Evaluation  (mirrors evaluate.accuracy — 5 zero-input readout steps)
# ---------------------------------------------------------------------------
def accuracy_snn(model, data_loader, variant="l", device=None):
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
        acc = correct / total
    return acc.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train SNN baseline on sequential MNIST")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--hidden_dim", type=int, default=256, help="Number of hidden neurons")
parser.add_argument("--beta", type=float, default=0.95, help="LIF membrane decay rate")
parser.add_argument("--variant", type=str, default="l",
                    choices=["l", "p"],
                    help="Image presentation: l=line-by-line (28 steps), p=pixel-by-pixel")
parser.add_argument("--neuron", type=str, default="leaky",
                    choices=["leaky", "synaptic"],
                    help="Neuron type: leaky (1st-order LIF) or synaptic (2nd-order)")
parser.add_argument("--alpha", type=float, default=0.9,
                    help="Synaptic current decay (only used when --neuron=synaptic)")

args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}, "
          f"hidden_dim={args.hidden_dim}, beta={args.beta}, variant={args.variant}, "
          f"neuron={args.neuron}")

    in_dim = 1 if args.variant == "p" else 28
    out_dim = 10

    train_loader, test_loader = get_MNIST_data_loaders(args.batch_size, variant=args.variant)

    if args.neuron == "leaky":
        model = SNNNetwork(
            in_dim, args.hidden_dim, out_dim,
            beta=args.beta, device=device,
        ).to(device)
    else:
        model = SNNNetworkSynaptic(
            in_dim, args.hidden_dim, out_dim,
            alpha=args.alpha, beta=args.beta, device=device,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    losses = train_snn(
        model, train_loader,
        epochs=args.epochs, lr=args.lr,
        variant=args.variant, device=device,
    )

    train_acc = accuracy_snn(model, train_loader, variant=args.variant, device=device)
    test_acc = accuracy_snn(model, test_loader, variant=args.variant, device=device)
    print(f"Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}")

    # ---- save ----
    save_dir = "model/network_params/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"snn_{args.neuron}_{args.variant}_{args.hidden_dim}_{args.beta}.pt",
    )

    torch.save(
        {
            "model_state_dict": model.to(torch.device("cpu")).state_dict(),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "lr": args.lr,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "variant": args.variant,
            "beta": args.beta,
            "neuron": args.neuron,
            "losses": losses,
        },
        save_path,
    )
    print(f"Saved to {save_path}")
