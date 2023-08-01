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
parser.add_argument("chunk_num", type=int, help="Chunk number")
parser.add_argument("config_path", type=str, help="Path to config")
args = parser.parse_args()

CHUNK_NUM = args.chunk_num
with open(args.config_path, "r") as f:
    CONFIG = json.load(f)

def get_activations(
    Is,
    fs,
    bin_size, 
    epochs=1000, 
    repeats=1,
    max_firing_rate=100,
    max_degree=1,
    device=None
):  
    max_current = np.max(np.abs(Is.cpu().numpy()))
    gs = []
    losses = []
    best_g = None
    best_loss = 1e10
    
    for degree in range(1, max_degree+1):
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
            print(f"degree {degree}, repeat {i}, final loss {loss}")
            if loss < best_loss:
                best_g, best_loss = g, loss

        gs.append(best_g)
        losses.append(best_loss)
    return gs[np.argmin(losses)]

def train(
    Is, 
    fs, 
    g,
    ds,
    cell_id, 
    bin_size,  
    device = None,
    hparams=[{"lr": 0.03, "gamma": 0.85, "step_size": 5, "epochs": 100}]
):
    best_model, best_losses = None, [0, 1e10]
    
    for i, hs in enumerate(hparams):
        print(f"\t\tRun {i}: {hs}")
        if CONFIG["use_previous_save"] and os.path.isfile(f"{CONFIG['save_path']}{cell_id}.pickle"):
            print("\t\tloading previous save")
            with open(f"{CONFIG['save_path']}{cell_id}.pickle", "rb") as f:
                params = pickle.load(f)["params"]
            if CONFIG["lift_degree"]:
                print("\t\tlifting degree")
                max_degree = CONFIG["activation_max_degree"]
                new_coeff = torch.zeros(max_degree + 1)
                old_coeff = params["g"]["poly_coeff"].detach().cpu()
                new_coeff[:len(old_coeff)] = old_coeff
                params["g"]["poly_coeff"] = new_coeff
            model = ExponentialKernelFiringRateModel.from_params(params, freeze_g=CONFIG["freeze_activation"]).to(device)
        else:
            model = ExponentialKernelFiringRateModel(
                g, ds, bin_size, freeze_g=CONFIG["freeze_activation"], device=device
            ).to(device)

        criterion = torch.nn.PoissonNLLLoss(log_input=False, reduction="none")
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hs["lr"], centered=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=hs["gamma"], step_size=hs["step_size"])

        losses = train_model(
            model, 
            criterion, 
            optimizer,
            Is,
            fs,
            epochs = hs["epochs"],
            print_every = 1,
            bin_size = bin_size,
            up_factor = CONFIG["up_factor"],
            scheduler = scheduler
        )
        
        if best_losses[-1] > losses[-1]:
            best_losses = losses
            best_model = model
    
    return best_model, best_losses

def fit_model(cell_id, bin_size, activation_bin_size, max_firing_rate, device=None, g=None):
    print("\tloading data for activation")
    aligned_data = get_data(cell_id, aligned=True)
    Is, fs = preprocess_data(aligned_data, activation_bin_size)
    Is, fs = torch.tensor(Is).to(device), torch.tensor(fs).to(device)
    
    if g is None:
        print("\tfitting activation")
        # get best activation
        g = get_activations(
            Is,
            fs,
            activation_bin_size, 
            epochs=CONFIG["activation_epochs"], 
            repeats=CONFIG["activation_repeats"],
            max_firing_rate=max_firing_rate,
            max_degree=CONFIG["activation_max_degree"],
            device=torch.device("cpu")
        )

    if g is None:
        print(f"\tError: activation for cell {cell_id} is None")
        print(f"Sum of firing rates (for activation fit): {torch.sum(fs)}")
        raise Exception
    else:
        print("\tloading data for model")
        ds = torch.tensor(CONFIG["decays"]).to(torch.float32)
        data = get_data(cell_id, aligned=False)
        Is_tr, fs_tr, Is_val, fs_val, Is_te, fs_te, stims = get_train_test_data(data, bin_size, device=device)
        Is_tr, fs_tr, stims = sklearn.utils.shuffle(Is_tr, fs_tr, stims)
        
        print("\tstart training model")
        hparams = CONFIG["hparams"]
        model, losses = train(Is_tr, fs_tr, g.to(device), ds, cell_id, bin_size, device=device, hparams=hparams)
        
        print("\tcomputing evr")
        evr1 = explained_variance_ratio(model, Is_val[0], fs_val[0], bin_size)
        evr2 = explained_variance_ratio(model, Is_te[0], fs_te[0], bin_size)
        print(f"\t{evr1}, {evr2}")
        
        return model.get_params(), evr1, evr2, losses

def model_pipeline(cell_id, bin_size, activation_bin_size, max_firing_rates, device, save_path="model/params/"):        
    retrain = CONFIG["retrain_model"]
    params_old = None
    g = None
    if os.path.isfile(f"{save_path}{cell_id}.pickle"):
        with open(f"{save_path}{cell_id}.pickle", 'rb') as f:
            print("model already exists")
            params_old = pickle.load(f)
        if not CONFIG["retrain_activation"]:
            g = PolynomialActivation.from_params(params_old["params"]["g"])
            
    if retrain or params_old is None:
        p, evr1, evr2, losses = fit_model(cell_id, bin_size, activation_bin_size, max_firing_rates[cell_id], device=device, g=g)
        params = {
            "params": p,
            "evr1": evr1,
            "evr2": evr2,
            "losses": losses,
            "bin_size": bin_size
        }
        
        if params_old is None or evr1 > params_old["evr1"]:
            print("saving model params")
            with open(f'{save_path}{cell_id}.pickle', 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_ids = np.genfromtxt(f'misc/chunks/chunk{CHUNK_NUM}.csv', delimiter=',')
    cell_ids = [int(cell_ids)] if cell_ids.shape == () else list(map(int, cell_ids))
    bin_size = CONFIG["bin_size"]
    activation_bin_size = CONFIG["activation_bin_size"]
    save_path = CONFIG["save_path"]
    
    with open("model/max_firing_rates.pickle", "rb") as f:
        max_firing_rates = pickle.load(f)

    print(CONFIG)
    print(f"Device: {device}")
    print(f"Chunk {CHUNK_NUM}")

    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)}) Cell {cell_id}")
        try:
            model_pipeline(cell_id, bin_size, activation_bin_size, max_firing_rates, device, save_path=save_path)
        except Exception as e:
            print(f"Skipping {cell_id} due to error: {e}")
