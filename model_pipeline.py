import torch
import pickle
import numpy as np
import sklearn.utils
import argparse
import os

from model import FiringRateModel, PolynomialActivation
from train import train_model, fit_activation
from evaluate import explained_variance_ratio
from data import get_data, get_train_test_data, obtain_spike_time_and_current_and_voltage, preprocess_data

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

    for _ in range(repeats):
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=hs["gamma"], step_size=hs["step_size"])

        losses = train_model(
            model, 
            criterion, 
            optimizer,
            Is,
            fs,
            epochs = hs["epochs"],
            print_every = 10,
            bin_size = bin_size,
            up_factor = 1,
            scheduler = scheduler
        )
        
        if best_losses[-1] > losses[-1]:
            best_losses = losses
            best_model = model
    
    return best_model, best_losses

def model_pipeline(cell_id, bin_size, max_firing_rate, device=None, g=None):
    print("\tloading data for activation")
    aligned_data = get_data(cell_id, aligned=True)
    Is, fs = preprocess_data(aligned_data)
    Is, fs = torch.tensor(Is).to(device), torch.tensor(fs).to(device)
    
    if g is None:
        print("\tfitting activation")
        # get best activation
        g = get_activations(
            Is,
            fs,
            bin_size, 
            epochs=500, 
            repeats=1,
            C=0,
            max_firing_rate=max_firing_rate,
            device=torch.device("cpu")
        )

    if g is None:
        print(f"\tError: activation for cell {cell_id} is None")
        raise Exception
    else:
        print(g.poly_coeff, g.max_firing_rate)
        
        print("\tloading data for model")
        ds = np.linspace(0.05, 1.0, 20)
        data = get_data(cell_id, aligned=False)
        Is_tr, fs_tr, Is_te, fs_te, stims = get_train_test_data(data, bin_size, device=device)
        Is_tr, fs_tr, stims = sklearn.utils.shuffle(Is_tr, fs_tr, stims)
        
        print("\tstart training model")
        hparams = [
            {"lr": 0.03, "gamma": 0.85, "step_size": 5, "epochs": 150},
            {"lr": 0.02, "gamma": 0.90, "step_size": 10, "epochs": 200},
            {"lr": 0.01, "gamma": 0.95, "step_size": 20, "epochs": 300}
        ]
        model, losses = train(Is_tr, fs_tr, g.to(device), ds, cell_id, bin_size, device=device, hparams=hparams)
        
        print("\tcomputing evr")
        r = explained_variance_ratio(model, Is_te[0], fs_te[0], bin_size)
        print(f"\t{r}")
        
        return model.get_params(), r, losses

def fit_all_models(cell_ids, bin_size, max_firing_rates, chunk_num, device):
    print(f"Chunk {chunk_num}")
    
    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)}) Cell {cell_id}")
        
        retrain = False
        params_old = None
        g = None
        if os.path.isfile(f"model/params/{cell_id}.pickle"):
            with open(f'model/params/{cell_id}.pickle', 'rb') as f:
                print("model already exists")
                params_old = pickle.load(f)
            
            g = PolynomialActivation()
            g.init_from_params(params_old["params"]["g"])
            if params_old["evr"] < 0.4 or params_old["losses"][-1] > 1000:
                print("requirements satisfied. retraining model")
                retrain = True
                
        
        if retrain:
            try:
                p, r, losses = model_pipeline(cell_id, bin_size, max_firing_rates[cell_id], device=device, g=g)
                params = {
                    "params": p,
                    "evr": r,
                    "losses": losses
                }
                
                print("saving model params")
                if params_old is None:
                    with open(f'model/params/{cell_id}.pickle', 'wb') as handle:
                        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    if losses[-1] < params_old["losses"][-1]:
                        print("overwriting old parameters")
                        with open(f'model/params/{cell_id}.pickle', 'wb') as handle:
                            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as err:
                print(err)
        else:
            print("Skipping")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_num", type=int, help="Chunk number")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_ids = [int(x) for x in np.genfromtxt(f'misc/chunks/chunk{args.chunk_num}.csv', delimiter=',')]
    bin_size = 20
    print(device)

    with open("model/max_firing_rates.pickle", "rb") as f:
        max_firing_rates = pickle.load(f)

    fit_all_models(cell_ids, bin_size, max_firing_rates, args.chunk_num, device)
