import torch
import pickle
import numpy as np
import sklearn.utils
import argparse
import os
import json

from model import FiringRateModel, PolynomialActivation
from train import train_model, fit_activation
from evaluate import explained_variance_ratio
from data import get_data, get_train_test_data, obtain_spike_time_and_current_and_voltage, preprocess_data

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
    actvs = []
    losses = []

    best_actv = None
    best_loss = 1e10
    
    for degree in range(1, max_degree+1):
        for i in range(repeats):
            actv = PolynomialActivation()
            actv.init_params(bin_size, degree, max_current, max_firing_rate, Is, fs)
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
            )

            loss = ls[-1]
            
            print(f"degree {degree}, repeat {i}, final loss {loss}")

            if loss < best_loss:
                best_actv, best_loss = actv, loss

    actvs.append(best_actv)
    losses.append(best_loss)
    best_actv = actvs[np.argmin(losses)]
    return best_actv

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
        model = FiringRateModel(
            g.to(device), ds, bin_size=bin_size, device=device
        ).to(device)

        criterion = torch.nn.PoissonNLLLoss(log_input=False, reduction="none")
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hs["lr"], centered=True)
        
        if CONFIG["scheduler"] == "step_lr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=hs["gamma"], step_size=hs["step_size"])
        elif CONFIG["scheduler"] == "cosine_annealing_lr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hs["T_max"], eta_min=hs["eta_min"])

        losses = train_model(
            model, 
            criterion, 
            optimizer,
            Is,
            fs,
            epochs = hs["epochs"],
            print_every = 1,
            bin_size = bin_size,
            up_factor = 1,
            scheduler = scheduler
        )
        
        if best_losses[-1] > losses[-1]:
            best_losses = losses
            best_model = model
    
    return best_model, best_losses

def model_pipeline(cell_id, bin_size, activation_bin_size, max_firing_rate, device=None, g=None):
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
        raise Exception
    else:
        print(g.poly_coeff, g.max_firing_rate)
        
        print("\tloading data for model")
        ds = CONFIG["decays"]
        data = get_data(cell_id, aligned=False)
        Is_tr, fs_tr, Is_te, fs_te, stims = get_train_test_data(data, bin_size, device=device)
        Is_tr, fs_tr, stims = sklearn.utils.shuffle(Is_tr, fs_tr, stims)
        
        print("\tstart training model")
        hparams = CONFIG["hparams"]
        model, losses = train(Is_tr, fs_tr, g.to(device), ds, cell_id, bin_size, device=device, hparams=hparams)
        
        print("\tcomputing evr")
        r = explained_variance_ratio(model, Is_te[0], fs_te[0], bin_size)
        print(f"\t{r}")
        
        return model.get_params(), r, losses

def fit_all_models(cell_ids, bin_size, activation_bin_size, max_firing_rates, device, save_path="model/params/"):
    print(f"Chunk {CHUNK_NUM}")
    
    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)}) Cell {cell_id}")
        
        retrain = False
        params_old = None
        g = None
        if os.path.isfile(f"{save_path}{cell_id}.pickle"):
            with open(f"{save_path}{cell_id}.pickle", 'rb') as f:
                print("model already exists")
                params_old = pickle.load(f)
            
            g = PolynomialActivation()
            g.init_from_params(params_old["params"]["g"])
            if len(params_old["losses"]) < 50:
                print("requirements satisfied. retraining model")
                retrain = True
                
        
        if retrain or params_old is None:
            try:
                p, r, losses = model_pipeline(cell_id, bin_size, activation_bin_size, max_firing_rates[cell_id], device=device, g=g)
                params = {
                    "params": p,
                    "evr": r,
                    "losses": losses
                }
                
                print("saving model params")
                if params_old is None:
                    with open(f'{save_path}{cell_id}.pickle', 'wb') as handle:
                        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    if losses[-1] < params_old["losses"][-1]:
                        print("overwriting old parameters")
                        with open(f'{save_path}{cell_id}.pickle', 'wb') as handle:
                            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as err:
                print(err)
        else:
            print("Skipping")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(CONFIG)
    print(device)
    cell_ids = list(map(int, np.genfromtxt(f'misc/chunks/chunk{CHUNK_NUM}.csv', delimiter=',')))
    bin_size = CONFIG["bin_size"]
    activation_bin_size = CONFIG["activation_bin_size"]
    
    with open("model/max_firing_rates.pickle", "rb") as f:
        max_firing_rates = pickle.load(f)
    
    save_path = CONFIG["save_path"]

    fit_all_models(cell_ids, bin_size, activation_bin_size, max_firing_rates, device, save_path=save_path)
