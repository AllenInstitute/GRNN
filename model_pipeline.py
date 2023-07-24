import torch
import pickle
import numpy as np
import sklearn.utils
import argparse

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

def train(Is, fs, g, cell_id, bin_size, ds=None, device=None, repeats=1):
    best_model, best_losses = None, [0, 1e10]
    
    for _ in range(repeats):
        model = FiringRateModel(
            g.to(device), ds, bin_size=bin_size, device=device
        ).to(device)

        criterion = torch.nn.PoissonNLLLoss(log_input=False, reduction="none")
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.03, centered=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.85, step_size=5)

        losses = train_model(
            model, 
            criterion, 
            optimizer,
            Is,
            fs,
            epochs = 150,
            print_every = 151,
            bin_size = bin_size,
            up_factor = 1,
            scheduler = scheduler
        )
        
        if best_losses[-1] > losses[-1]:
            best_losses = losses
            best_model = model
    
    return best_model, best_losses

def model_pipeline(cell_id, bin_size, max_firing_rates, device=None):
    aligned_data = get_data(cell_id, aligned=True)
    Is, fs = preprocess_data(aligned_data)
    Is, fs = torch.tensor(Is).to(device), torch.tensor(fs).to(device)
    max_firing_rate = max_firing_rates[cell_id]
    
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
        print(f"Error: activation for cell {cell_id} is None")
        raise Exception
    else:
        ds = np.linspace(0.05, 1.0, 20)
        data = get_data(cell_id, aligned=False)
        Is_tr, fs_tr, Is_te, fs_te, stims = get_train_test_data(data, bin_size, device=device)
        Is_tr, fs_tr, stims = sklearn.utils.shuffle(Is_tr, fs_tr, stims)
        model, losses = train(Is_tr, fs_tr, g.to(device), ds, cell_id, bin_size, device=device)
        
        r = explained_variance_ratio(model, Is_te[0], fs_te[0], bin_size)
        
        return model.get_params(), r, losses

def fit_all_models(cell_ids, bin_size, max_firing_rates, chunk_num, device):
    try:
        with open(f'model/params{chunk_num}.pickle', 'rb') as f:
            params = pickle.load(f)
    except:
        print(f"params{chunk_num}.pickle not found. Creating new one.")
        params = {}
    
    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)}) Cell {cell_id}")
        
        if cell_id in params:
            print("Skipping - model already exists")
        else:
            try:
                p, r, losses = model_pipeline(cell_id, bin_size, max_firing_rates, device=device)
                params[cell_id] = {
                    "params": p,
                    "evr": r,
                    "losses": losses
                }
                
                with open(f'model/params{chunk_num}.pickle', 'wb') as handle:
                    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as err:
                print(err)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_num", type=int, help="Chunk number")
    args = parser.parse_args()
    chunk_num = args.chunk_num

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #cell_ids = [int(x) for x in np.genfromtxt(f'misc/chunk{chunk_num}.csv', delimiter=',')]
    cell_ids = [605889373]
    bin_size = 20
    print(device)

    with open("model/max_firing_rates.pickle", "rb") as f:
        max_firing_rates = pickle.load(f)

    fit_all_models(cell_ids, bin_size, max_firing_rates, chunk_num, device)