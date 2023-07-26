import numpy as np
import pickle
import os

from utils import get_max_firing_rate
from data import obtain_spike_time_and_current_and_voltage

def compute_all_max_firing_rates(cell_ids):
    try:
        with open('model/max_firing_rates.pickle', 'rb') as f:
            max_firing_rates = pickle.load(f)
            print("File loaded")
    except:
        print("max_firing_rates.pickle not found. Creating new one.")
        max_firing_rates = {}
        
    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)}) Cell {cell_id}")
        if cell_id in max_firing_rates:
            print("Skipping")
        else:
            fname = f'/allen/programs/mindscope/workgroups/auto-model/zhixin.lu/clean_up_cell_type_mouse_data/Calvin_raw_data/raw_data_{cell_id}.pickle'
            if os.path.isfile(fname):
                with open(fname, 'rb') as f:
                    raw_data = pickle.load(f)
            else:
                try:
                    raw_data = obtain_spike_time_and_current_and_voltage(cell_id)
                    max_firing_rate = get_max_firing_rate(raw_data)
                    max_firing_rates[cell_id] = max_firing_rate
                    
                    if i > 500 and i % 10 == 0:
                        with open(f'model/max_firing_rates_backup_{i}.pickle', 'wb') as f:
                            pickle.dump(max_firing_rates, f, protocol=pickle.HIGHEST_PROTOCOL)
                    with open('model/max_firing_rates.pickle', 'wb') as f:
                        pickle.dump(max_firing_rates, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as err:
                    print(err)
            

if __name__ == "__main__":
    cell_ids = [int(x) for x in np.genfromtxt('misc/valid_ids.csv', delimiter=',')]
    compute_all_max_firing_rates(cell_ids)