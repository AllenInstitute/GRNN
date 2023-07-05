import numpy as np

def quantize_prediction(pred_fs, bin_size):
    return (pred_fs * bin_size + 0.5).astype('int32').astype('float32') / bin_size

def explained_variance(psth1, psth2):
    v1 = np.std(psth1) ** 2
    v2 = np.std(psth2) ** 2
    v3 = np.std(psth1 - psth2) ** 2
    return (v1 + v2 - v3) / (v1 + v2)

def explained_variance_ratio(model, Is_te, fs_te, bin_size, quantize=False):
    fs_te_np = np.array([fs.numpy() for fs in fs_te])
    psth_d = np.mean(fs_te_np, axis=0)
    ev_d = np.mean([explained_variance(stpsth, psth_d) for stpsth in fs_te_np])
    
    psth_m = model.predict(Is_te[0])
    if quantize:
        psth_m = quantize_prediction(psth_m, bin_size)
    pwev_dm = np.mean([explained_variance(stpsth, psth_m) for stpsth in fs_te_np])
    return pwev_dm / ev_d