import subprocess

if __name__ == "__main__":
    
    sbatch_script = "scripts/run_network_pipeline.sh"
    lr = 0.005
    epochs = 50
    batch_size = 256
    neuron_type = "ekfr"
    train_neuron = "-n"
    train_activation = "-a"

    for n_nodes in [16, 32, 64, 128, 256]:
        for variant in ["l", "p"]:
            command = ["sbatch", sbatch_script, lr, epochs, batch_size, n_nodes, neuron_type, variant, train_neuron, train_activation]
            command = [str(component) for component in command]
            print(f"Running command: {command}")
            subprocess.run(command)