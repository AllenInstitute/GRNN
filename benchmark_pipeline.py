import torch
import pickle
import numpy as np
import sklearn.utils
import argparse
import os
import json
import torch.nn.functional as F

from model import LSTM
from train import train_model
from evaluate import explained_variance_ratio
from data import get_data, get_train_test_data
from config import config as file_config

parser = argparse.ArgumentParser()
parser.add_argument("cell_ids", type=str)
parser.add_argument("--bin_size", type=int, default=20, choices=[10, 20, 50, 100])
parser.add_argument("--save_path", type=str, default="model/params/")
parser.add_argument("--config_path", type=str, default="configs/default.json")
parser.add_argument("--model_type", type=str, default="gfr")
parser.add_argument("--hidden_size", type=int, default=5)
args = parser.parse_args()

cell_id_path = args.cell_ids
bin_size = args.bin_size
model_type = args.model_type
hidden_size = args.hidden_size

save_path = args.save_path
if save_path[-1] != '/':
    save_path += '/'

with open(args.config_path, "r") as f:
    config = json.load(f)

def train(
    Is_tr, 
    fs_tr, 
    Is_val, 
    fs_val, 
    Is_te, 
    fs_te,  
    bin_size,
    hidden_size,
    hparams=[{"lr": 1e-3, "epochs": 100}]
):
    best_model, best_evr1, best_losses, best_test_losses = None, -1e10, [], []
    
    for i, hs in enumerate(hparams):
        model = LSTM(hidden_size)
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
            model_type='lstm'
        )
        
        fs_pred = model(Is_val[0]).detach().numpy()
        evr1 = explained_variance_ratio(fs_val[0], fs_pred, bin_size)
        
        if evr1 > best_evr1:
            best_evr1 = evr1
            best_losses = losses
            best_test_losses = test_losses
            best_model = model
    
    fs_pred = model(Is_te[0]).detach().numpy()
    best_evr2 = explained_variance_ratio(fs_te[0], fs_pred, bin_size)
    
    return best_model, best_evr1, best_evr2, best_losses, best_test_losses

def fit_model(cell_id, bin_size, hidden_size):
    data = get_data(cell_id)
    Is_tr, fs_tr, Is_val, fs_val, Is_te, fs_te, stims = get_train_test_data(data, bin_size)
    Is_tr, fs_tr, stims = sklearn.utils.shuffle(Is_tr, fs_tr, stims) # list of Tensors, each with shape [B, seq_len]
    
    if len(Is_te) == 0 or len(Is_val) == 0:
        print("No noise 1/2 data. Skipping.")
        raise Exception
    
    print("Start training model...")
    hparams = config["hparams"]
    model, evr1, evr2, losses, test_losses = train(
        Is_tr, fs_tr, Is_val, fs_val, Is_te, fs_te, bin_size, hidden_size, hparams=hparams
    )
        
    print(f"{evr1}, {evr2}")
    
    return model.state_dict(), evr1, evr2, losses, test_losses

def model_pipeline(cell_id, bin_size, hidden_size):   
    params_old = None

    p, evr1, evr2, losses, test_losses = fit_model(
        cell_id, bin_size, hidden_size
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
    cell_ids = np.genfromtxt(f'{cell_id_path}', delimiter=',')
    cell_ids = [int(cell_ids)] if cell_ids.shape == () else list(map(int, cell_ids))

    print(f"{device=}\n{bin_size=}\n{hidden_size=}")
    print(config)

    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)}) Cell {cell_id}")
        try:
            model_pipeline(cell_id, bin_size, hidden_size)
        except Exception as e:
            print(f"Skipping {cell_id} due to error: {e}")
