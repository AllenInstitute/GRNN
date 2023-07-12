import torch
import numpy as np
from scipy.special import binom

def get_max_firing_rate(data, cell_id):
    diffs = np.concatenate([np.diff(d["spike_times"]) for d in data[cell_id][-1]])
    return np.max(1 / diffs) / 1000 # return in ms^-1

def compute_derivative(seq, n, dt):
    coeffs = torch.tensor([(-1) ** i * binom(n, i) for i in range(n+1)][::-1]).to(torch.float32)
    return seq[-n-1:] @ coeffs / (dt ** n)