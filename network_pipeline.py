import torch
import argparse
import numpy as np

from network import Network
from data import get_MNIST_data_loaders
from train import train_network
from evaluate import accuracy

parser = argparse.ArgumentParser()
parser.add_argument("lr", type=float, help="Learing rate")
parser.add_argument("epochs", type=int, help="Number of training epochs")
parser.add_argument("batch_size", type=int, help="Batch size")
parser.add_argument("n_nodes", type=int, help="Number of recurrent nodes")
parser.add_argument("variant", type=str, help="MNIST variant (p or l)")
parser.add_argument("freeze_neurons", type=str, help="Freeze neuron weights")
parser.add_argument("freeze_activations", type=str, help="Freeze activation weights")
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
hidden_dim = args.n_nodes
variant = args.variant
freeze_neurons = eval(args.freeze_neurons)
freeze_activations = eval(args.freeze_activations)

def load_network(fname):
    state = torch.load(fname)
    model = Network(
        1 if state["variant"] == "p" else 28, 
        state["hidden_dim"], 
        10,
        freeze_neurons=state["freeze_neurons"], 
        freeze_g=state["freeze_activations"]
    )
    model.load_state_dict(state["model_state_dict"])
    return model

def permuted_network(fname, variant):
    model = load_network(fname)

    in_dim = model.in_dim
    hidden_dim = model.hidden_dim
    out_dim = model.out_dim
    
    new_model = Network(
        in_dim, 
        hidden_dim, 
        out_dim,
        freeze_neurons=True, 
        freeze_g=True
    )

    idxs = (np.random.rand(256) * 256).astype(int)
    new_model.hidden_neurons.g.max_current = torch.nn.Parameter(model.hidden_neurons.g.max_current.detach().clone()[idxs], requires_grad=False)
    new_model.hidden_neurons.g.max_firing_rate = torch.nn.Parameter(model.hidden_neurons.g.max_firing_rate.detach().clone()[idxs], requires_grad=False)
    new_model.hidden_neurons.g.max_firing_rate = torch.nn.Parameter(model.hidden_neurons.g.max_firing_rate.detach().clone()[idxs], requires_grad=False)
    new_model.hidden_neurons.g.b = torch.nn.Parameter(model.hidden_neurons.g.b.detach().clone()[idxs], requires_grad=False)
    new_model.hidden_neurons.g.poly_coeff = torch.nn.Parameter(model.hidden_neurons.g.poly_coeff.detach().clone()[idxs], requires_grad=False)
    new_model.hidden_neurons.a = torch.nn.Parameter(model.hidden_neurons.a.detach().clone()[idxs, :], requires_grad=False)
    new_model.hidden_neurons.b = torch.nn.Parameter(model.hidden_neurons.b.detach().clone()[idxs, :], requires_grad=False)

    return new_model

if __name__ == "__main__":
    print(f"{lr=}\n{epochs=}\n{batch_size=}\n{hidden_dim=}\n{variant=}\n{freeze_neurons=}\n{freeze_activations=}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    in_dim = 1 if variant == "p" else 28
    out_dim = 10
    
    train_loader, test_loader = get_MNIST_data_loaders(batch_size, variant=variant)

    if freeze_neurons:
        model = permuted_network(f"model/network_params/{variant}_{hidden_dim}_False_True.pt", variant)
    else:
        model = Network(
            in_dim, 
            hidden_dim, 
            out_dim, 
            freeze_neurons=freeze_neurons, 
            freeze_g=freeze_activations,
            device=device
        ).to(device)

    train_network(
        model, 
        train_loader, 
        epochs=epochs, 
        lr=lr, 
        variant=variant,
        C=0,
        device=device
    )

    train_acc = accuracy(model, train_loader, variant=variant, device=device)
    test_acc = accuracy(model, test_loader, variant=variant, device=device)
    print(f"Train accuracy: {train_acc} | Test accuracy: {test_acc}")
    
    save_path = f"model/network_params/{variant}_{hidden_dim}_{freeze_neurons}_{freeze_activations}.pt"
    
    torch.save(
        {
            "model_state_dict": model.to(torch.device("cpu")).state_dict(),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "lr": lr,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "variant": variant,
            "freeze_neurons": freeze_neurons,
            "freeze_activations": freeze_activations
        },
        save_path
    )