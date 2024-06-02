# Generalized Firing Rate Neurons
## Dataset
The dataset consists of trained GFR model parameters for different configurations of bin sizes and activation bin sizes. We only include models that pass certain criteria, namely:
- The data includes both noise 1 and noise 2 sweeps (used for validation and testing)
- The validation explained variance ratio is greater than 0.5
- The training loss (Poisson negative log likelihood) is less than 0.45

The table below lists the number of cells satisfying the above criteria:
| Bin size (ms) | Activation bin size (ms) | # cells |
|:-------------:|:------------------------:|---------|
|       10      |            20            | 1003    |
|       10      |            100           | 769     |
|       20      |            20            | 1124    |
|       20      |            100           | 1407    |
|       50      |            100           | 1524    |
|      100      |            100           | 1402    |

## Loading the dataset
To load the dataset, run
```
import pickle

with open("model/gfr_dataset.pickle", "rb") as f:
    dataset = pickle.load(f)
```

The dataset is a dictionary where keys are bin size, activation bin size pairs, and the values are dataframes. Each dataframe includes information about cell id, cre-line, validation and test explained variance ratio, train and test loss, and GFR model parameters.

To load a specific GFR model, use
```
load_gfr_model(dataset, cell_id, bin_size, activation_bin_size)
```
in `utils.py`.

## Reproducing Results
## Preprocessing data
To download and preprocess the data, run
```
python preprocess_pipeline.py --cell_ids [cell_ids]
```
where `[cell_ids]` is the path to a CSV file containing the cell ids of all the cells you want to preprocess. If not specified, all cell ids available will be used. A corresponding CSV with all cell ids will be saved in `data/cell_ids.csv`.

The preprocessed data will be saved as a pickle file in `data/processed_data/` as `processed_I_and_firing_rate_{cell_id}.pickle` for each cell id in the specified CSV file.

WARNING: this process will take a while.

## Training GFR neurons
To train the model, run
```
python model_pipeline.py [cell_ids] --bin_size [bin_size] --activation_bin_size [activation_bin_size] --degree [degree] --C [C] --save_path [save_path] --config_path [config_path]
```
where
- `[cell_ids]`: path to a CSV file containing the cell ids of all the cells you want to train models for.
- `[bin_size]`: time bin size used for discretizing the spike data for training the GFR model (not including activation function).
    - Default: 20
    - Allowed values: 10, 20, 50, 100
- `[activation_bin_size]`: time bin size used for discretizing the spike data for training the activation function.
    - Default: 20
    - Allowed values: 20, 100
- `[degree]`: degree of the polynomial in the activation function.
    - Default: 1
- `[C]`: constant for L1 regularization on the GFR model.
    - Default: 0
- `[save_path]`: path to save folder for models.
    - Default: `model/params/`
- `[config_path]`: path to config file specifying training parameters (see `configs/default.json` for an example).
    - Default: `configs/default.json`

## Training a network of GFR neurons for L-MNIST
To train the network, run
```
python network_pipeline.py [lr] [epochs] [batch_size][n_nodes] [freeze_neurons] [freeze_activations]
```
where
- `[lr]`: learning rate used for training the network.
- `[epochs]`: number of training epochs.
- `[batch_size]`: training batch size.
- `[n_nodes]`: number of recurrent nodes in the network.
- `[freeze_neurons]`: freeze neuron weights when training; only train recurrent connections and input/output weights.
- `[freeze_activation]`: freeze activation weights.