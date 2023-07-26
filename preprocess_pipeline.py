import os
import pickle
import numpy as np
import argparse

from data import obtain_firing_rate_and_current_given_time_bin
from config import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_num", type=int, help="Chunk number")
    args = parser.parse_args()
    
    cell_ids = list(map(int, np.genfromtxt(f"misc/preprocess_chunks/chunk{args.chunk_num}.csv", delimiter=",")))

    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)} Cell {cell_id})")
        fname = f"{config['data_path_aligned']}/processed_I_and_firing_rate_{cell_id}.pickle"
        if not os.path.isfile(fname):
            try:
                data = obtain_firing_rate_and_current_given_time_bin(cell_id=cell_id)
                with open(fname, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
        else:
            print("File already exists. Skipping.")
