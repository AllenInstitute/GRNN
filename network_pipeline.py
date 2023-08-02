from network import Network
from data import get_MNIST_data_loaders
from train import train_network
from evaluate import accuracy

if __name__ == "__main__":
    batch_size = 256
    variant = "l"
    in_dim = 1 if variant == "p" else 28
    out_dim = 10
    hidden_dim = 128
    epochs = 30
    train_loader, test_loader = get_MNIST_data_loaders(batch_size)
    model = Network(in_dim, hidden_dim, out_dim, neuron_type="ekfr", freeze_neurons=False)
    train_network(model, train_loader, epochs, lr=0.005, variant=variant)
    train_acc = accuracy(model, train_loader, variant=variant)
    test_acc = accuracy(model, test_loader, variant=variant)
    print(f"Train accuracy: {train_acc} / Test accuracy: {test_acc}")