import subprocess

if __name__ == "__main__":
    
    sbatch_script = "scripts/run_network_pipeline.sh"
    lr = 1e-3
    epochs = 200
    batch_size = 256
    freeze_activations = True

    for n_nodes in [32, 64, 128]:
        for variant in ["l", "p"]:
            for freeze_neurons in [False]:
                command = ["sbatch", sbatch_script, lr, epochs, batch_size, n_nodes, variant, freeze_neurons, freeze_activations]
                command = [str(component) for component in command]
                print(f"Running command: {' '.join(command)}")
                subprocess.run(command)