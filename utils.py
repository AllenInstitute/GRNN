import numpy as np

def get_max_firing_rate(data, cell_id):
    diffs = np.concatenate([np.diff(d["spike_times"]) for d in data[cell_id][-1]])
    return np.max(1 / diffs) / 1000 # return in ms^-1