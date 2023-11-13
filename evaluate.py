import torch
import torch.nn.functional as F
import numpy as np

from utils import reshape_image

def quantize_prediction(pred_fs, bin_size):
    return (pred_fs * bin_size + 0.5).astype('int32').astype('float32') / bin_size

def explained_variance(psth1, psth2):
    v1 = np.std(psth1) ** 2
    v2 = np.std(psth2) ** 2
    v3 = np.std(psth1 - psth2) ** 2
    return np.nan if (v1 + v2) == 0 else (v1 + v2 - v3) / (v1 + v2)

def explained_variance_ratio(model, Is_te, fs_te, bin_size, quantize=False):
    fs_te_np = fs_te.numpy()
    psth_d = np.mean(fs_te_np, axis=0)
    ev_d = np.mean([explained_variance(stpsth, psth_d) for stpsth in fs_te_np])
    psth_m = model.predict(Is_te[0])[0].squeeze().numpy()
    if quantize:
        psth_m = quantize_prediction(psth_m, bin_size)
    pwev_dm = np.mean([explained_variance(stpsth, psth_m) for stpsth in fs_te_np])
    return pwev_dm / ev_d

def accuracy(model, data_loader, variant="p", device=None):
    with torch.no_grad():
        correct, total = 0, 0
        for x, label in data_loader:
            x = reshape_image(x, variant=variant).to(device)
            label = label.to(device)

            # sequentially send input into network
            model.reset(x.shape[0])
            for i in range(x.shape[1]):
                model(x[:, i, :])

            pred_y = model(model.zero_input(x.shape[0]))
            pred = F.softmax(pred_y, dim=1) # add softmax
            correct += torch.sum(torch.argmax(pred, dim=1) == label)
            total += x.shape[0]
        acc = correct / total
    return acc.item()