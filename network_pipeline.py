import torch
import argparse

from network import Network
from data import get_MNIST_data_loaders
from train import train_network
from evaluate import accuracy

parser = argparse.ArgumentParser()
parser.add_argument("lr", type=int, help="Learing rate")
parser.add_argument("epochs", type=int, help="Number of training epochs")
parser.add_argument("batch_size", type=int, help="Batch size")
parser.add_argument("n_nodes", type=int, help="Number of recurrent nodes")
parser.add_argument("neuron_type", type=str, help="Neuron type (gfr or ekfr)")
parser.add_argument("variant", type=str, help="MNIST variant (p or l)")
parser.add_argument('-n', '--train_neuron', action='store_true')
parser.add_argument('-a', '--train_activation', action='store_true')
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
hidden_dim = args.n_nodes
variant = args.variant

if __name__ == "__main__":
    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    print(device)

    in_dim = 1 if variant == "p" else 28
    out_dim = 10
    
    train_loader, test_loader = get_MNIST_data_loaders(batch_size)
    model = Network(in_dim, hidden_dim, out_dim, neuron_type="ekfr", freeze_neurons=False)
    train_network(model, train_loader, epochs, lr=lr, variant=variant)
    train_acc = accuracy(model, train_loader, variant=variant)
    test_acc = accuracy(model, test_loader, variant=variant)
    print(f"Train accuracy: {train_acc} / Test accuracy: {test_acc}")