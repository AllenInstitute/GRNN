import torch
import pickle
import numpy as np
import sklearn.utils
import argparse
import os
import json

from model import ExponentialKernelFiringRateModel, PolynomialActivation
from train import train_model, fit_activation
from evaluate import explained_variance_ratio
from data import get_data, get_train_test_data, preprocess_data

parser = argparse.ArgumentParser()
parser.add_argument("chunk_id", type=int)
parser.add_argument("bin_size", type=int)
parser.add_argument("activation_bin_size", type=int)
parser.add_argument("degree", type=int)
parser.add_argument("C", type=float)
parser.add_argument("save_path", type=str)
parser.add_argument("config_path", type=str)
args = parser.parse_args()

chunk_id = args.chunk_id
bin_size = args.bin_size
activation_bin_size = args.activation_bin_size
degree = args.degree
C = args.C
save_path = args.save_path

with open(args.config_path, "r") as f:
    config = json.load(f)

def get_activations(
    Is,
    fs,
    bin_size, 
    epochs=1000, 
    repeats=1,
    max_firing_rate=100,
    degree=1,
    device=None
):  
    max_current = np.max(np.abs(Is.cpu().numpy()))
    gs = []
    losses = []
    best_g = None
    best_loss = 1e10
    
    for i in range(repeats):
        g = PolynomialActivation.from_data(degree, max_current, max_firing_rate, bin_size, Is, fs).to(device)
        criterion = torch.nn.PoissonNLLLoss(log_input=False)
        optimizer = torch.optim.Adam(g.parameters(), lr=0.05)

        ls = fit_activation(
            g,
            criterion,
            optimizer,
            Is,
            fs,
            epochs=epochs,
        )
        loss = ls[-1]
        print(f"Repeat {i}, final loss {loss}")
        if loss < best_loss:
            best_g, best_loss = g, loss
        gs.append(best_g)
        losses.append(best_loss)
    return gs[np.argmin(losses)]

def train(
    Is_tr, 
    fs_tr, 
    Is_val, 
    fs_val, 
    Is_te, 
    fs_te,  
    g,
    ds,
    cell_id, 
    bin_size,  
    device = None,
    hparams=[{"lr": 1e-3, "epochs": 100}]
):
    best_model, best_evr1, best_losses, best_test_losses = None, -1e10, [], []
    
    for i, hs in enumerate(hparams):
        print(f"Run {i+1}/{len(hparams)}: {hs}")
        if config["finetune"] and os.path.isfile(f"{save_path}{cell_id}.pickle"):
            print("Loading previous save...")
            with open(f"{save_path}{cell_id}.pickle", "rb") as f:
                params = pickle.load(f)["params"]
            model = ExponentialKernelFiringRateModel.from_params(
                params, freeze_g=config["freeze_activation"], device=device
            ).to(device)
        else:
            model = ExponentialKernelFiringRateModel(
                g, ds, bin_size, freeze_g=config["freeze_activation"], device=device
            ).to(device)

        criterion = torch.nn.PoissonNLLLoss(log_input=False, reduction="none", eps=config["eps"])
        optimizer = torch.optim.Adam(model.parameters(), lr=hs["lr"])

        losses, test_losses = train_model(
            model, 
            criterion, 
            optimizer,
            Is_tr,
            fs_tr,
            Is_te,
            fs_te,
            epochs = hs["epochs"],
            print_every = 1,
            bin_size = bin_size,
            C = C
        )
        
        evr1 = explained_variance_ratio(model, Is_val[0], fs_val[0], bin_size)
        
        if evr1 > best_evr1:
            best_evr1 = evr1
            best_losses = losses
            best_test_losses = test_losses
            best_model = model
    
    best_evr2 = explained_variance_ratio(best_model, Is_te[0], fs_te[0], bin_size)
    
    return best_model, best_evr1, best_evr2, best_losses, best_test_losses

def fit_model(cell_id, bin_size, activation_bin_size, degree, max_firing_rate, device=None, g=None):
    print("Loading data for activation")
    aligned_data = get_data(cell_id, aligned=True)
    Is, fs = preprocess_data(aligned_data, activation_bin_size)
    Is, fs = torch.tensor(Is).to(device), torch.tensor(fs).to(device) # shape [seq_length]
    
    if g is None:
        print("Fitting activation")
        # get best activation
        g = get_activations(
            Is,
            fs,
            activation_bin_size, 
            epochs=config["activation_epochs"], 
            repeats=config["activation_repeats"],
            max_firing_rate=max_firing_rate,
            degree=degree,
            device=torch.device("cpu")
        )

    if g is None:
        print(f"Error: activation for cell {cell_id} is None")
        print(f"Sum of firing rates (for activation fit): {torch.sum(fs)}")
        raise Exception
    else:
        print("Loading data for model...")

        taus = torch.tensor([10, 20, 50, 100, 200, 500, 1000, 2000])
        taus = taus[taus >= bin_size]
        ds = 1 - torch.exp(-bin_size /  taus)
        ds = torch.cat([torch.ones(1), ds])
        ds = ds.to(torch.float32)
        print(f"Using timescales: {taus}")
        print(f"Corresponding decay rates: {ds}")

        data = get_data(cell_id, aligned=False)
        Is_tr, fs_tr, Is_val, fs_val, Is_te, fs_te, stims = get_train_test_data(data, bin_size, device=device)
        Is_tr, fs_tr, stims = sklearn.utils.shuffle(Is_tr, fs_tr, stims) # list of Tensors, each with shape [B, seq_len]
        
        if len(Is_te) == 0 or len(Is_val) == 0:
            print("No noise 1/2 data. Skipping.")
            raise Exception
        
        print("Start training model...")
        hparams = config["hparams"]
        model, evr1, evr2, losses, test_losses = train(
            Is_tr, fs_tr, Is_val, fs_val, Is_te, fs_te, g.to(device), ds, cell_id, bin_size, device=device, hparams=hparams
        )
        
        print(f"{evr1}, {evr2}")
        
        return model.get_params(), evr1, evr2, losses, test_losses

def model_pipeline(cell_id, bin_size, activation_bin_size, degree, max_firing_rate, device):        
    retrain = config["retrain_model"]
    params_old = None
    g = None

    if os.path.isfile(f"{save_path}{cell_id}.pickle"):
        print("Model already exists")
        with open(f"{save_path}{cell_id}.pickle", 'rb') as f:
            params_old = pickle.load(f)
            g = PolynomialActivation.from_params(params_old["params"]["g"])
    else:
        print("Model doesn't exist")
            
    if retrain or params_old is None or g is None:
        p, evr1, evr2, losses, test_losses = fit_model(
            cell_id, bin_size, activation_bin_size, degree, max_firing_rate, device=device, g=g
        )

        params = {
            "params": p,
            "evr1": evr1,
            "evr2": evr2,
            "train_losses": losses,
            "test_losses": test_losses,
            "bin_size": bin_size
        }
        
        # referesh parameters
        if params_old is not None:
            with open(f"{save_path}{cell_id}.pickle", 'rb') as f:
                params_old = pickle.load(f)
        
        if params_old is None or evr1 > params_old["evr1"]:
            print("Saving model params")
            with open(f'{save_path}{cell_id}.pickle', 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_ids = np.genfromtxt(f'misc/chunks/chunk{chunk_id}.csv', delimiter=',')
    cell_ids = [int(cell_ids)] if cell_ids.shape == () else list(map(int, cell_ids))

    print(f"{device=}\n{chunk_id=}\n{bin_size=}\n{activation_bin_size=}\n{degree=}\n{C=}")
    print(config)

    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)}) Cell {cell_id}")
        try:
            with open(f"model/max_firing_rates/{cell_id}.pickle", "rb") as f:
                max_firing_rate = pickle.load(f)
            model_pipeline(cell_id, bin_size, activation_bin_size, degree, max_firing_rate, device)
        except Exception as e:
            print(f"Skipping {cell_id} due to error: {e}")
