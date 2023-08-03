import subprocess

if __name__ == "__main__":
    
    sbatch_script = "scripts/run_network_pipeline.sh"
    lr = 0.005
    epochs = 100
    batch_size = 256
    neuron_type = "ekfr"
    freeze_activations = True

    for n_nodes in [16, 32, 64, 128, 256]:
        for variant in ["l", "p"]:
            for freeze_neurons in [True, False]:
                command = ["sbatch", sbatch_script, lr, epochs, batch_size, n_nodes, neuron_type, variant, freeze_neurons, freeze_activations]
                command = [str(component) for component in command]
                print(f"Running command: {' '.join(command)}")
                subprocess.run(command)