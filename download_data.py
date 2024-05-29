import os
import pickle
import data
import utils

from config import config
from tqdm import tqdm
from allensdk.core.cell_types_cache import CellTypesCache

if __name__ == "__main__":
    if not os.path.exists(config["mfr_path"]):
        os.makedirs(config["mfr_path"])
    if not os.path.exists(config["data_path"]):
        os.makedirs(config["data_path"])

    ctc = CellTypesCache(manifest_file="cell_types/manifest.json")
    cell_ids = [x["id"] for x in ctc.get_cells()]
    cell_ids = cell_ids[:3]
    with open("data/cell_ids.csv", "w") as f:
        f.write(",".join(map(str, cell_ids)))

    for cell_id in tqdm(cell_ids, desc="Downloading data"):
        dataset = data.obtain_spike_time_and_current_and_voltage(cell_id=cell_id)
        mfr = utils.get_max_firing_rate(dataset)
        with open(f"{config['mfr_path']}{cell_id}.pickle", 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        dataset = data.obtain_firing_rate_and_current_given_time_bin(cell_id=cell_id)
        fname = f"{config['data_path']}processed_I_and_firing_rate_{cell_id}.pickle"
        with open(fname, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)