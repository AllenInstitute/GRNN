import torch
import numpy as np
import matplotlib.pyplot as plt

from config import config

def get_max_firing_rate(data):
    diffs = np.concatenate([np.diff(d["spike_times"]) for d in data])
    return np.max(1 / diffs) / 1000 # return in ms^-1

def plot_predictions(model, Is, fs, cell_id, bin_size, evr=None, save=False, fname=None):    
    pred_fs, vs = model.predict(Is)
    ts = np.arange(len(Is)) * bin_size / 1000
    
    fig, axs = plt.subplots(3)
    if evr is not None:
        fig.suptitle(f"cell_id={cell_id}, bin_size={bin_size}, evr={evr[0]:.3f}/{evr[1]:.3f}")
    else:
        fig.suptitle(f"cell_id={cell_id}, bin_size={bin_size}")
        
    axs[0].plot(ts, fs, label="Actual")
    axs[0].plot(ts, pred_fs, label="Predicted")
    axs[1].plot(ts, vs)
    axs[2].plot(ts, Is)
    axs[0].legend()
    axs[0].set_ylabel("firing rate")
    axs[1].set_ylabel("v")
    axs[2].set_ylabel("current (pA)")
    axs[2].set_xlabel("time (s)")

    if save:
        plt.savefig(config["fig_save_path"] + f"{cell_id}/bin_size_{bin_size}/{fname}.png")
        plt.close()

def plot_kernel(model, cell_id, bin_size, save=False, fname=None):
    # kernel for currents
    def k(x, a, lamb, w):
        return torch.sum(w * a * torch.pow(lamb, x))
    
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    fig.suptitle(f"cell_id={cell_id}, bin_size={bin_size}")
    xs = torch.linspace(0, 6, 100)
    cs, ds = [], []
    a, b, d, w = model.a, model.b, model.ds, model.w
    with torch.no_grad():
        for x in xs:
            cs.append(k(x, a, d, w))
            ds.append(k(x, b, d, w))
    
    axs0 = subfigs[0].subplots(2)
    axs0[0].plot(xs,cs)
    axs0[1].plot(xs,ds)
    axs0[0].set_ylabel("$k_a(x)$")
    axs0[1].set_ylabel("$k_b(x)$")
    axs0[1].set_xlabel("$x$")
    
    axs1 = subfigs[1].subplots(3)
    axs1[0].bar(list(range(len(a))), a.detach().reshape(-1))
    axs1[1].bar(list(range(len(b))), b.detach().reshape(-1))
    axs1[2].bar(list(range(len(w))), w.detach().reshape(-1))
    axs1[0].set_ylabel("$a_i$")
    axs1[1].set_ylabel("$b_i$")
    axs1[2].set_ylabel("$w_i$")
    axs1[2].set_xlabel("$i$")
    
    fig.set_size_inches(8, 5, forward=True)

    if save:
        plt.savefig(config["fig_save_path"] + f"{cell_id}/bin_size_{bin_size}/{fname}.png")
        plt.close()