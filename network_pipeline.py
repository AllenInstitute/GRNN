import torch
import argparse

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

if __name__ == "__main__":
    print(f"{lr=}\n{epochs=}\n{batch_size=}\n{hidden_dim=}\n{variant=}\n{freeze_neurons=}\n{freeze_activations=}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    in_dim = 1 if variant == "p" else 28
    out_dim = 10
    
    train_loader, test_loader = get_MNIST_data_loaders(batch_size, variant=variant)
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