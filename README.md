# Generalized Firing Rate Neurons
## Preprocessing data
To preprocess the data, run
```
python preprocess_pipeline.py [cell_ids]
```
where `[cell_ids]` is the path to a CSV file containing the cell ids of all the cells you want to preprocess.

The preprocessed data will be saved as a pickle file in `data/processed_data/` as `processed_I_and_firing_rate_{cell_id}.pickle` for each cell id in the specified CSV file.

## Training GFR neurons
To train the model, run
```
python model_pipeline.py [cell_ids] [bin_size] [activation_bin_size] [degree] [C] [save_path] [config_path]
```
where
- `[cell_ids]`: path to a CSV file containing the cell ids of all the cells you want to train models for.
- `[bin_size]`: time bin size used for discretizing the spike data for training the GFR model (not including activation function).
- `[activation_bin_size]`: time bin size used for discretizing the spike data for training the activation function.
- `[degree]`: degree of the polynomial in the activation function.
- `[C]`: constant for L1 regularization on the GFR model.
- `[save_path]`: path to save folder for models.
- `[config_path]`: path to config file specifying training parameters (see `configs/default.json` for an example).

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