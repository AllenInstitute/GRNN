import torch
import os
import numpy as np

from model import PolynomialActivation
from train import fit_activation
from utils import get_max_firing_rate
from data import get_data, obtain_spike_time_and_current_and_voltage, preprocess_data

def get_activations(
    Is,
    fs,
    bin_size, 
    epochs=1000, 
    repeats=1,
    C=0,
    max_firing_rate=100,
    device=None
):  
    max_current = np.max(np.abs(Is.cpu().numpy()))
    degree = 1
    actvs = []
    losses = []

    best_actv = None
    best_loss = 1e10

    for i in range(repeats):
        actv = PolynomialActivation()
        actv.init_params(degree, max_current, max_firing_rate, Is, fs, C=C)
        actv.bin_size = bin_size
        actv = actv.to(device)
        criterion = torch.nn.PoissonNLLLoss(log_input=False)
        optimizer = torch.optim.Adam(actv.parameters(), lr=0.05)

        ls = fit_activation(
            actv,
            criterion,
            optimizer,
            Is,
            fs,
            epochs=epochs,
            C=C
        )

        loss = ls[-1]

        if loss < best_loss:
            best_actv, best_loss = actv, loss

    actvs.append(best_actv)
    losses.append(best_loss)
    best_actv = actvs[np.argmin(losses)]
    return best_actv

def activation_pipeline(cell_id, bin_size=20):
    raw_data = obtain_spike_time_and_current_and_voltage(cell_id)
    aligned_data = get_data(cell_id, aligned=True)
    Is, fs = preprocess_data(aligned_data)
    Is, fs = torch.tensor(Is).to(device), torch.tensor(fs).to(device)
    max_firing_rate = get_max_firing_rate(raw_data)
    
    # get best activation
    return get_activations(
        Is,
        fs,
        bin_size, 
        epochs=500, 
        repeats=3,
        C=0,
        max_firing_rate=max_firing_rate,
        device=device
    )

def append_to_file(fname, s):
    with open(fname, "a") as f:
        f.write(s)

def fit_all_activations(cell_ids, bin_size):
    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)}) Cell {cell_id}")
        if os.path.isfile(f"model/activation/bin_size_{bin_size}/{cell_id}.pickle"):
            print("Skipping")
        else:
            g = None
            try:
                g = activation_pipeline(cell_id, bin_size=bin_size)
            except:
                print("Missing")
                append_to_file("model/missing.txt", f"{cell_id}\n")
            if g is None:
                print("Error")
                append_to_file("model/error.txt", f"{cell_id}\n")
            else:
                g.save_params(f"model/activation/bin_size_{bin_size}/{cell_id}.pickle")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    cell_ids = [int(x) for x in np.genfromtxt('misc/valid_ids.csv', delimiter=',')]
    bin_size = 20
    fit_all_activations(cell_ids, bin_size)