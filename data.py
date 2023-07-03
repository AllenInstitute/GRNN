import os
import pickle

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