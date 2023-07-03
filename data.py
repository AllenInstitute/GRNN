import os
import pickle
import torch

def load_data(with_zero=False):
    # with_zero = load data with leading zeros not removed
    if with_zero:
        path = "data/processed_data_zero/"
    else:
        path = "data/processed_data/"

    data = {}
    for f in os.listdir(path):
        if os.path.isfile(path + f):
            if f.split(".")[1] == "pickle":
                cell_id = int(f.split(".")[0].split("_")[-1])
                with open(path + f, "rb") as file:
                    file_data = pickle.load(file)
                    data[cell_id] = file_data
                    
    for cell_id in data:
        with open(f"data/raw_data/raw_data_{cell_id}.pickle", "rb") as file:
            file_data = pickle.load(file)
            data[cell_id].append(file_data)
    return data

def preprocess_data(data, cell_id, bin_size=10):
    d = data[cell_id][:-1] # don't include last element; raw data
    
    # filter long squares
    is_long_square = lambda s: s["stimulus_name"] == "Long Square"
    sweeps = filter(is_long_square, d)
    
    Is = []
    fs = []
    for sweep in sweeps:
        Is.append(sweep["current"][bin_size][0])
        fs.append(sweep["firing_rate"][bin_size][0])
    return Is, fs

def get_train_test_data(data, cell_id, bin_size, device=None):
    Is_tr, fs_tr, Is_te, fs_te = tuple([] for _ in range(4))
    for sweep in data[cell_id][:-1]:
        stim_name = sweep["stimulus_name"]
        Is = torch.tensor(sweep["current"][bin_size], device=device)
        fs = torch.tensor(sweep["firing_rate"][bin_size], device=device)
        if stim_name == "Noise 2":
            Is_te.append(Is)
            fs_te.append(fs)
        elif stim_name != "Test":
            Is_tr.append(Is)
            fs_tr.append(fs)
    return Is_tr, fs_tr, Is_te, fs_te