import os
import pickle
import numpy as np
import argparse

from utils import get_max_firing_rate
from data import obtain_spike_time_and_current_and_voltage, obtain_firing_rate_and_current_given_time_bin
from config import config
from allensdk.core.cell_types_cache import CellTypesCache

parser = argparse.ArgumentParser()
parser.add_argument("--cell_ids", type=str, help="CSV of cell ids", default=None)
args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(config["mfr_path"]):
        os.makedirs(config["mfr_path"])
    if not os.path.exists(config["data_path"]):
        os.makedirs(config["data_path"])

    if args.cell_ids is not None:
        cell_ids = list(map(int, np.genfromtxt(f"{args.cell_ids}", delimiter=",")))
    else:
        print("No cell ids given. Using all available cell ids by default.")
        ctc = CellTypesCache(manifest_file="cell_types/manifest.json")
        cell_ids = [x["id"] for x in ctc.get_cells()]
        cell_ids = cell_ids[:3]
        with open("data/cell_ids.csv", "w") as f:
            f.write(",".join(map(str, cell_ids)))

    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)} Cell {cell_id})")
        
        if not os.path.isfile(f"{config['mfr_path']}{cell_id}.pickle"):
            try:
                raw_data = obtain_spike_time_and_current_and_voltage(cell_id)
                max_firing_rate = get_max_firing_rate(raw_data)
                with open(f"{config['mfr_path']}{cell_id}.pickle", "wb") as f:
                    pickle.dump(max_firing_rate, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
        else:
            print("Max firing rate exists. Skipping.")

        fname = f"{config['data_path']}processed_I_and_firing_rate_{cell_id}.pickle"
        if not os.path.isfile(fname):
            try:
                data = obtain_firing_rate_and_current_given_time_bin(cell_id=cell_id)
                with open(fname, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
        else:
            print("Preprocessed data already exists. Skipping.")
