import torch
import pickle
import random

import utils

from model import BatchGFR, GFR

def get_random_neurons(n_neurons, save_path="model/gfr_dataset.pickle", bin_size=20, activation_bin_size=20):
    with open(save_path, "rb") as f:
        all_params = pickle.load(f)
    df = all_params[(bin_size, activation_bin_size)]
    cell_ids = df["cell_id"].tolist()

    chosen_ids = random.sample(cell_ids, k=n_neurons)
    neurons = []
    for cell_id in chosen_ids:
        neurons.append(utils.load_gfr_model(all_params, cell_id, bin_size, activation_bin_size))
        
    return neurons, chosen_ids

def get_neuron_layer(n_neurons, freeze_g=True, default=True):
    if default:
        neurons = [GFR.default() for _ in range(n_neurons)]
    else:
        neurons, _ = get_random_neurons(n_neurons)
    return BatchGFR(neurons, freeze_g=freeze_g)

# GFR-RNN with default parameters
class Network(torch.nn.Module):
    def __init__(
            self, 
            in_dim, 
            hidden_dim, 
            out_dim, 
            freeze_neurons=True,
            freeze_g=True,
            device=None
        ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device
        
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, out_dim)

        self.hidden_neurons = get_neuron_layer(hidden_dim, freeze_g=freeze_g)
        self.hidden_neurons.device = device
        if freeze_neurons:
            self.hidden_neurons.freeze_parameters()
    
    def reset(self, batch_size):
        self.hidden_neurons.reset(batch_size)
        self.xh = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        
    def zero_input(self, batch_size):
        return torch.zeros(batch_size, self.in_dim).to(self.device)
    
    # x: [batch_size, in_dim]
    def forward(self, x):
        x_in = self.fc1(x)
        x_rec = self.fc2(self.xh)
        self.xh = self.hidden_neurons(x_in + x_rec)
        out = self.fc3(self.xh)
        return out

# GFR-RNN with biological parameters
class BiologicalGFRNetwork(torch.nn.Module):
    def __init__(
            self, 
            in_dim, 
            hidden_dim, 
            out_dim, 
            freeze_neurons=True,
            freeze_g=True,
            device=None
        ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device
        
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        with torch.no_grad():
            self.fc1.weight.normal_(1.5, 3)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, out_dim)

        self.hidden_neurons = get_neuron_layer(hidden_dim, freeze_g=freeze_g, default=False)
        self.hidden_neurons.device = device
        if freeze_neurons:
            self.hidden_neurons.freeze_parameters()
    
    def reset(self, batch_size):
        self.hidden_neurons.reset(batch_size)
        self.xh = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        
    def zero_input(self, batch_size):
        return torch.zeros(batch_size, self.in_dim).to(self.device)
    
    # x: [batch_size, in_dim]
    def forward(self, x):
        x_in = torch.einsum("ij,j->ij", self.fc1(x), self.hidden_neurons.g.max_current) / self.in_dim
        x_rec = self.fc2(self.xh) / self.hidden_dim
        self.xh = self.hidden_neurons(x_in + x_rec)
        out = self.fc3(self.xh)
        return out