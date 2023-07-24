import numpy as np
import pickle

from utils import get_max_firing_rate
from data import obtain_spike_time_and_current_and_voltage

def compute_all_max_firing_rates(cell_ids):
    try:
        with open('model/max_firing_rates.pickle', 'rb') as f:
            max_firing_rates = pickle.load(f)
    except:
        print("max_firing_rates.pickle not found. Creating new one.")
        max_firing_rates = {}
        
    for i, cell_id in enumerate(cell_ids):
        print(f"({i+1}/{len(cell_ids)}) Cell {cell_id}")
        if cell_id not in max_firing_rates:
            raw_data = obtain_spike_time_and_current_and_voltage(cell_id)
            max_firing_rate = get_max_firing_rate(raw_data)
            max_firing_rates[cell_id] = max_firing_rate

            with open('model/max_firing_rates.pickle', 'wb') as f:
                pickle.dump(max_firing_rates, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    cell_ids = [int(x) for x in np.genfromtxt('misc/valid_ids.csv', delimiter=',')]
    compute_all_max_firing_rates(cell_ids)