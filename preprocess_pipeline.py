import os
import pickle
import numpy as np
import argparse

from utils import get_max_firing_rate
from data import obtain_spike_time_and_current_and_voltage, obtain_firing_rate_and_current_given_time_bin
from config import config

parser = argparse.ArgumentParser()
parser.add_argument("chunk_num", type=int, help="Chunk number")
args = parser.parse_args()

if __name__ == "__main__":
    cell_ids = list(map(int, np.genfromtxt(f"misc/preprocess_chunks/chunk{args.chunk_num}.csv", delimiter=",")))

    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)} Cell {cell_id})")
        
        if not os.path.isfile(f"model/max_firing_rates/{cell_id}.pickle"):
            try:
                raw_data = obtain_spike_time_and_current_and_voltage(cell_id)
                max_firing_rate = get_max_firing_rate(raw_data)
                with open(f"model/max_firing_rates/{cell_id}.pickle", "wb") as f:
                    pickle.dump(max_firing_rate, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
        else:
            print("Max firing rate exists. Skipping.")

        fname = f"{config['data_path_aligned']}/processed_I_and_firing_rate_{cell_id}.pickle"
        if not os.path.isfile(fname):
            try:
                data = obtain_firing_rate_and_current_given_time_bin(cell_id=cell_id)
                with open(fname, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
        else:
            print("Preprocessed data already exists. Skipping.")
