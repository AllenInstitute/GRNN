import torch
import pickle
import os
import random
import numpy as np
from model import load_model

def get_params(save_path="model/params/"):
    params = {}
    for fname in os.listdir(save_path):
        with open(f"{save_path}{fname}", "rb") as f:
            p = pickle.load(f)
            params[int(fname.split(".")[0])] = p
    return params

def get_random_neurons(n_neurons, save_path="model/params/"):
    params = get_params(save_path)
    cell_ids = []

    for cell_id in params:
        if params[cell_id]["evr"] >= 0.7:
            cell_ids.append(cell_id)

    chosen_ids = random.sample(cell_ids, k=n_neurons)
    neurons = {}
    for cell_id in chosen_ids:
        model = load_model(params[cell_id]["params"])
        neurons[cell_id] = model
    return neurons

class Network(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_neurons) -> None:
        super().__init__()
        
        # first in_dim neurons are input neurons
        # last out_dim neurons are output neurons
        assert n_neurons >= in_dim + out_dim
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_neurons = n_neurons
        self.neurons = get_random_neurons(n_neurons)
        
        ## connectivity matrix
        self.A = torch.nn.Parameter(torch.randn(n_neurons, n_neurons)) / np.sqrt(n_neurons)
        # readout weights
        self.w = torch.nn.Parameter(torch.randn(out_dim))
        
        self.neurons = torch.nn.ModuleList([neurons[cell_id] for cell_id in neurons])
        self.cell_ids = list(neurons.keys())
    
    def reset(self, batch_size):
        for neuron in self.neurons:
            neuron.reset(batch_size)
        self.fs = torch.zeros(batch_size, self.n_neurons)
    
    # x: [batch_size, in_dim]
    def forward(self, x):
        for i in range(self.in_dim):
            self.fs[i] = self.neurons[i](x[:, i] + torch.tensordot(self.fs, self.A[i, :], dims=1), self.fs[:, i])
        return torch.tensordot(self.fs[:, self.out_dim:], self.w, dims=1)


if __name__ == "__main__":
    neurons = get_random_neurons(10)
    for cell_id in neurons:
        print(cell_id, neurons[cell_id].get_params())