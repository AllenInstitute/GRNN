import pickle
import torch
import torchvision
import numpy as np

#from allensdk.core.cell_types_cache import CellTypesCache

from config import config

def get_MNIST_data_loaders(batch_size, variant="l"):
    size = 28 if variant == "l" else 24
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor()
    ])
    train_set = torchvision.datasets.MNIST('data/mnist/train', download=True, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST('data/mnist/test', download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_data(cell_id, aligned=True, patch_seq=False):
    if patch_seq:
        path = config["patch_seq_path"] + f"processed_I_and_firing_rate_{cell_id}.pickle"
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        path = config["data_path_aligned" if aligned else "data_path"] + f"processed_I_and_firing_rate_{cell_id}.pickle"
        with open(path, "rb") as f:
            return pickle.load(f)
    
def get_raw_data(cell_id):
    with open(config["raw_data_path"] + f"raw_data_{cell_id}.pickle", "rb") as file:
        return pickle.load(file)

def obtain_spike_time_and_current_and_voltage(cell_id):
    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
    raw_data = ctc.get_ephys_data(cell_id)
    sweep_numbers = raw_data.get_sweep_numbers()
        
    data_set = []
    for sweep_id in sweep_numbers:
        sweep_data = {}
        
        stimulus_name = raw_data.get_sweep_metadata(sweep_id)['aibs_stimulus_name'].decode()
        sweep_data['stimulus_name'] = stimulus_name
        
        sampling_rate = raw_data.get_sweep(sweep_id)['sampling_rate']
        sweep_data['sampling_rate'] = sampling_rate
        
        # start/stop indices that exclude the experimental test pulse (if applicable)
        index_range = raw_data.get_sweep(sweep_id)['index_range']
        
        I = 1E12*raw_data.get_sweep(sweep_id)['stimulus']
        V = 1E3*raw_data.get_sweep(sweep_id)['response']
        spike_times = raw_data.get_spike_times(sweep_id)
        
        #make sure index_range[1] is corrected for ramp:
        if stimulus_name == 'Ramp':
            max_I_idx = np.argmax(I) if np.argmax(I) > index_range[0] else None
            voltage_dies_at = np.max(np.nonzero(V))
            index_range_1 = min(index_range[1],voltage_dies_at)
            if max_I_idx is not None:
                index_range_1 = min(index_range_1,max_I_idx)

            index_range_0 = index_range[0]
            index_range = (index_range_0,index_range_1)
        sweep_data['index_range'] = index_range
        
        sweep_data['current'] = I
        sweep_data['spike_times'] = spike_times
        sweep_data['voltage'] = V
        
        data_set.append(sweep_data)
    return data_set

def downsample_by_avg(x,n):
    '''this function downsample a given time series by using the average of chunks of time series that 
    are not overlaping. It preserves the integral of the downsampled time series.'''
    x=np.array(x)
    if n>1:
        interval_num = int(x.shape[0]/n)
        x = x[0:interval_num*n]
        return np.mean(x.reshape(-1, n), 1)
    else:
        return x

def obtain_firing_rate_and_current_given_time_bin(cell_id = 324257146,bin_durs = [10,20,50,100]):
    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
    raw_data = ctc.get_ephys_data(cell_id)
    sweep_numbers = raw_data.get_sweep_numbers()

    data_set = []
    for sweep_id in sweep_numbers:
        sweep_data = {}
        sweep_data['current'] = {}
        sweep_data['firing_rate'] = {}
        #print('processing sweep with sweep id = '+ str(sweep_id))
        stimulus_name = raw_data.get_sweep_metadata(sweep_id)['aibs_stimulus_name'].decode()
        sweep_data['stimulus_name'] = stimulus_name
        sampling_rate = raw_data.get_sweep(sweep_id)['sampling_rate']
        # start/stop indices that exclude the experimental test pulse (if applicable)
        index_range = raw_data.get_sweep(sweep_id)['index_range']
        I = 1E12*raw_data.get_sweep(sweep_id)['stimulus']
        V = 1E3*raw_data.get_sweep(sweep_id)['response']
        S = np.zeros_like(I)
        spike_times = raw_data.get_spike_times(sweep_id)
        spike_idxs = [int(spike_time*sampling_rate) for spike_time in spike_times]
        for spike_idx in spike_idxs:
            S[spike_idx]=1.0
        #make sure index_range[1] is corrected for ramp:
        if stimulus_name == 'Ramp':
            max_I_idx = np.argmax(I) if np.argmax(I) > index_range[0] else None
            voltage_dies_at = np.max(np.nonzero(V))
            index_range_1 = min(index_range[1],voltage_dies_at)
            if max_I_idx is not None:
                index_range_1 = min(index_range_1,max_I_idx)

            index_range_0 = index_range[0]
            index_range = (index_range_0,index_range_1)
            
        begin_idx = 0
        #find the begining of non-zero I within the I[index_range[0]:index_range[1]]:
        if stimulus_name!='Test':
            I_diff = np.diff(I[index_range[0]:index_range[1]])
            I_diff_nonzero_idxs = I_diff.nonzero()
            begin_idx = I_diff_nonzero_idxs[0][0]

        #save corrected index_range
        for bin_dur in bin_durs:
            bin_size = int(bin_dur*0.001*sampling_rate)
            mean_I_at_bin = downsample_by_avg(I[index_range[0]:index_range[1]][begin_idx:],bin_size)
            firing_rate_at_bin = downsample_by_avg(S[index_range[0]:index_range[1]][begin_idx:],bin_size)*bin_size/bin_dur
            sweep_data['current'][bin_dur] = mean_I_at_bin
            sweep_data['firing_rate'][bin_dur] = firing_rate_at_bin
        data_set.append(sweep_data)
    return data_set

def preprocess_data(data, bin_size):
    # filter long squares
    is_long_square = lambda s: s["stimulus_name"] == "Long Square"
    sweeps = filter(is_long_square, data)
    
    Is = []
    fs = []
    for sweep in sweeps:
        Is.append(sweep["current"][bin_size][0])
        fs.append(sweep["firing_rate"][bin_size][0])
    return Is, fs

# return train and test sets, batched by stimulus type
def get_train_test_data(data, bin_size, device=None, patch_seq=False):
    Is_tr, fs_tr, Is_val, fs_val, Is_te, fs_te = tuple([] for _ in range(6))
    Is, fs = {}, {}
    stims = []

    for sweep in data:
        s = sweep["stimulus_name"]
        if s not in Is:
            Is[s] = []
            fs[s] = []
        Is[s].append(torch.tensor(sweep["current"][bin_size], device=device))
        fs[s].append(torch.tensor(sweep["firing_rate"][bin_size], device=device))
    
    for s in Is:
        Is_padded = torch.nn.utils.rnn.pad_sequence(Is[s], batch_first=True)
        fs_padded = torch.nn.utils.rnn.pad_sequence(fs[s], batch_first=True)
        if s == "Noise 1":
            Is_val.append(Is_padded)
            fs_val.append(fs_padded)
        elif s == "Noise 2":
            Is_te.append(Is_padded)
            fs_te.append(fs_padded)
        elif patch_seq and s in ["Long Square", "Ramp"]:
            Is_tr.append(Is_padded)
            fs_tr.append(fs_padded)
            stims.append(s)
        elif s != "Test":
            Is_tr.append(Is_padded)
            fs_tr.append(fs_padded)
            stims.append(s)

    return Is_tr, fs_tr, Is_val, fs_val, Is_te, fs_te, stims