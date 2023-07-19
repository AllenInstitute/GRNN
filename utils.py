import torch
import numpy as np
import matplotlib.pyplot as plt

def get_max_firing_rate(data, cell_id):
    diffs = np.concatenate([np.diff(d["spike_times"]) for d in data[cell_id][-1]])
    return np.max(1 / diffs) / 1000 # return in ms^-1

def plot_predictions(model, Is, fs, cell_id, bin_size, k, l, evr=None, closed=True, save=False, save_path=None):    
    pred_fs, vs = model.predict(Is, closed=closed, fs=fs)
    ts = np.arange(len(Is)) * bin_size / 1000
    k, l = model.k, model.l
    fig, axs = plt.subplots(3)
    if evr is not None:
        fig.suptitle(f"({'closed' if closed else 'open'}) cell_id={cell_id}, bin_size={bin_size}, k_l={k}_{l}, evr={evr[0]:.3f}/{evr[1]:.3f}")
    else:
        fig.suptitle(f"({'closed' if closed else 'open'}) cell_id={cell_id}, bin_size={bin_size}, k_l={k}_{l}")
        
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
        plt.savefig(save_path)
        plt.close()

def plot_kernel(model, cell_id, bin_size, k, l, save=False, save_path=None):
    # kernel for currents
    def c(x, a, lamb, w):
        _, k = a.shape
        L = torch.pow(torch.tile(lamb, dims=(k, 1)).T, x - torch.arange(1, k+1)).T
        return torch.diag(a @ L) @ w
    
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    fig.suptitle(f"cell_id={cell_id}, bin_size={bin_size}, k_l={k}_{l}")
    xs = torch.linspace(0, 6, 100)
    cs, ds = [], []
    a, b, d, w = model.a, model.b, model.ds, model.w
    with torch.no_grad():
        for x in xs:
            cs.append(c(x, a, d, w))
            ds.append(c(x, b, d, w))
    
    axs0 = subfigs[0].subplots(2)
    axs0[0].plot(xs,cs)
    axs0[1].plot(xs,ds)
    axs0[0].set_ylabel("$k_a(x)$")
    axs0[1].set_ylabel("$k_b(x)$")
    axs0[1].set_xlabel("$x$")
    
    axs1 = subfigs[1].subplots(3)
    axs1[0].bar(list(range(len(model.a))), model.a.detach().reshape(-1))
    axs1[1].bar(list(range(len(model.b))), model.b.detach().reshape(-1))
    axs1[2].bar([f"{x:.2f}" for x in model.ds.tolist()], model.w.tolist())
    axs1[0].set_ylabel("$a_i$")
    axs1[1].set_ylabel("$b_i$")
    axs1[2].set_ylabel("$w_i$")
    axs1[2].set_xlabel("$i$")
    
    fig.set_size_inches(8, 5, forward=True)

    if save:
        plt.savefig(save_path)
        plt.close()