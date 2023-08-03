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
parser.add_argument("neuron_type", type=str, help="Neuron type (gfr or ekfr)")
parser.add_argument("variant", type=str, help="MNIST variant (p or l)")
parser.add_argument("freeze_neurons", type=bool, help="Freeze neuron weights")
parser.add_argument("freeze_activations", type=bool, help="Freeze activation weights")
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
hidden_dim = args.n_nodes
neuron_type = args.neuron_type
variant = args.variant
freeze_neurons = args.freeze_neurons
freeze_activations = args.freeze_activations

if __name__ == "__main__":
    print(f"{lr=}\n{epochs=}\n{batch_size=}\n{hidden_dim=}\n{neuron_type=}\n{variant=}\n{freeze_neurons=}\n{freeze_activations=}")
    
    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    in_dim = 1 if variant == "p" else 28
    out_dim = 10
    
    train_loader, test_loader = get_MNIST_data_loaders(batch_size)
    model = Network(
        in_dim, 
        hidden_dim, 
        out_dim, 
        neuron_type=neuron_type, 
        freeze_neurons=freeze_neurons, 
        freeze_g=freeze_activations
    ).to(device) # just train on cpu for now

    train_network(
        model, 
        train_loader, 
        epochs=epochs, 
        lr=lr, 
        variant=variant
    )

    train_acc = accuracy(model, train_loader, variant=variant)
    test_acc = accuracy(model, test_loader, variant=variant)
    print(f"Train accuracy: {train_acc} | Test accuracy: {test_acc}")

    torch.save(
        {
            "model_state_dict": model.to(torch.device("cpu")).state_dict(),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "lr": lr,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "neuron_type": neuron_type,
            "variant": variant,
            "freeze_neurons": freeze_neurons,
            "freeze_activations": freeze_activations
        },
        f"model/network_params/{variant}_{neuron_type}_{hidden_dim}_{freeze_neurons}_{freeze_activations}.pt"
    )