import torch
import pickle
import os
import random

from model import (
    BatchEKFR, BatchGFR, GeneralizedFiringRateModel, ExponentialKernelFiringRateModel
)

def get_params(save_path="model/params/"):
    params = {}
    for fname in os.listdir(save_path):
        if fname.endswith(".pickle"):
            with open(f"{save_path}{fname}", "rb") as f:
                p = pickle.load(f)
                params[int(fname.split(".")[0])] = p
    return params

def get_random_neurons(n_neurons, save_path="model/params/", threshold=0.7, neuron_type="ekfr"):
    params = get_params(save_path)
    cell_ids = []

    for cell_id in params:
        if params[cell_id]["evr2"] >= threshold:
            cell_ids.append(cell_id)

    chosen_ids = random.sample(cell_ids, k=n_neurons)
    neurons = []
    for cell_id in chosen_ids:
        neurons.append(ExponentialKernelFiringRateModel.from_params(params[cell_id]["params"]))
    if neuron_type == "gfr":
        neurons = [GeneralizedFiringRateModel.from_ekfr(model, 5, 5) for model in neurons]
        
    return neurons, chosen_ids

def get_neuron_layer(n_neurons, save_path="model/params/", threshold=0.7, neuron_type="ekfr", freeze_g=True):
    neurons = get_random_neurons(n_neurons, save_path=save_path, threshold=threshold, neuron_type=neuron_type)[0]
    cls = BatchEKFR if neuron_type == "ekfr" else BatchGFR
    return cls(neurons, freeze_g=freeze_g)

class Network(torch.nn.Module):
    def __init__(
            self, 
            in_dim, 
            hidden_dim, 
            out_dim, 
            neuron_type="ekfr", 
            freeze_neurons=True,
            freeze_g=True,
            device=None
        ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.neuron_type = neuron_type
        self.device = device
        
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        with torch.no_grad():
            self.fc1.weight.normal_(1, 1)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, out_dim)

        self.hidden_neurons = get_neuron_layer(hidden_dim, neuron_type=neuron_type, freeze_g=freeze_g)
        self.hidden_neurons.device = device
        if freeze_neurons:
            self.hidden_neurons.freeze_parameters()
    
    def reset(self, batch_size):
        self.hidden_neurons.reset(batch_size)
        self.xh = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        
    def reg(self):
        return self.hidden_neurons.smoothness_reg() if self.neuron_type == "gfr" else 0
        
    def zero_input(self, batch_size):
        return torch.zeros(batch_size, self.in_dim)
    
    # x: [batch_size, in_dim]
    def forward(self, x):
        x_in = torch.einsum("ij,j->ij", self.fc1(x), self.hidden_neurons.g.max_current)
        x_rec = self.fc2(self.xh)
        self.xh = self.hidden_neurons(x_in + x_rec)
        out = self.fc3(self.xh)
        return out