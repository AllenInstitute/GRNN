import torch
import numpy as np
import matplotlib.pyplot as plt

from config import config

def read_file(fname):
    arr = []
    with open(fname, "r") as f:
        for x in f:
            arr.append(int(x))
    return arr

def append_to_file(fname, s):
    with open(fname, "a") as f:
        f.write(s)

def get_max_firing_rate(data):
    diffs = np.concatenate([np.diff(d["spike_times"]) for d in data])
    return np.max(1 / diffs) / 1000 # return in ms^-1

def plot_predictions(model, Is, fs, cell_id, bin_size, evr=None, save=False, fname=None):
    pred_fs, vs = model.predict(Is)
    ts = np.arange(len(Is)) * bin_size / 1000
    
    if vs is None:
        fig, axs = plt.subplots(2)
        if evr is not None:
            fig.suptitle(f"cell_id={cell_id}, bin_size={bin_size}, evr={evr:.3f}")
        else:
            fig.suptitle(f"cell_id={cell_id}, bin_size={bin_size}")
            
        axs[0].plot(ts, fs, label="Actual")
        axs[0].plot(ts, pred_fs, label="Predicted")
        axs[1].plot(ts, Is)
        axs[0].legend()
        axs[0].set_ylabel("firing rate ($ms^{-1}$)")
        axs[1].set_ylabel("current ($pA$)")
        axs[1].set_xlabel("time ($s$)")
    else:
        fig, axs = plt.subplots(3)
        if evr is not None:
            fig.suptitle(f"cell_id={cell_id}, bin_size={bin_size}, evr={evr:.3f}")
        else:
            fig.suptitle(f"cell_id={cell_id}, bin_size={bin_size}")
            
        axs[0].plot(ts, fs, label="Actual")
        axs[0].plot(ts, pred_fs, label="Predicted")
        axs[1].plot(ts, vs)
        axs[2].plot(ts, Is)
        axs[0].legend()
        axs[0].set_ylabel("firing rate ($ms^{-1}$)")
        axs[1].set_ylabel("v")
        axs[2].set_ylabel("current ($pA$)")
        axs[2].set_xlabel("time ($s$)")

    if save:
        plt.savefig(config["fig_save_path"] + f"{cell_id}/bin_size_{bin_size}/{fname}.png")
        plt.close()

def plot_kernel(model, cell_id, bin_size, save=False, fname=None, xlim=10):
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    fig.suptitle(f"cell_id={cell_id}, bin_size={bin_size}")
    xs = torch.linspace(0, xlim, 100)
    cs, ds = [], []
    with torch.no_grad():
        for x in xs:
            cs.append(model.kernel(x, var="a"))
            ds.append(model.kernel(x, var="b"))
    
    axs0 = subfigs[0].subplots(2)
    axs0[0].plot(xs,cs)
    axs0[1].plot(xs,ds)
    axs0[0].set_ylabel("$k_a(x)$")
    axs0[1].set_ylabel("$k_b(x)$")
    axs0[1].set_xlabel("$x$")
    
    axs1 = subfigs[1].subplots(2)
    axs1[0].bar(list(range(model.a.shape[1])), model.a.detach().reshape(-1))
    axs1[1].bar(list(range(model.b.shape[1])), model.b.detach().reshape(-1))
    axs1[0].set_ylabel("$a_i$")
    axs1[1].set_ylabel("$b_i$")
    axs1[1].set_xlabel("$i$")
    
    fig.set_size_inches(8, 5, forward=True)

    if save:
        plt.savefig(config["fig_save_path"] + f"{cell_id}/bin_size_{bin_size}/{fname}.png")
        plt.close()
        
def get_activation_plot(actv, start=-100, end=270):
    currents = torch.linspace(start, end, steps=300).reshape(-1, 1)
    with torch.no_grad():
        fs = actv(currents)
    return currents.reshape(-1), fs.reshape(-1)

def plot_activation(Is, fs, actv):
    plt.figure()
    plt.title(f"bin_size={actv.bin_size}, degree={actv.degree}")
    plt.scatter(Is, fs)
    xs1, ys1 = get_activation_plot(actv, end=int(actv.max_current)+200)
    plt.plot(xs1, ys1)
    plt.xlabel("current (pA)")
    plt.ylabel("firing rate ($ms^{-1}$)")

# x: shape [batch_size, 28, 28]
# returns shape [batch_size, seq_length, in_dim]
def reshape_image(x, variant="p"):
    if variant == "p":
        x = x.reshape(x.shape[0], 24, 24)
        return x.reshape(x.shape[0], -1, 1)
    else:
        return x.reshape(x.shape[0], 28, 28)