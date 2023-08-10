import subprocess
import argparse
import numpy as np

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("cell_ids", type=str, help="CSV file containing cell ids to process")
parser.add_argument("config_path", type=str, help="Path to config file")
parser.add_argument("chunk_num", type=int, help="Number of jobs per configuration")
args = parser.parse_args()

bin_sizes = [10, 20, 50, 100]
activation_bin_sizes = [20, 100]
degrees = [1, 3]
n_kernels = [5, 7]

sbatch_script = "scripts/run_model_pipeline.sh"

def generate_chunks(cell_ids, chunk_num, save_path):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    chunk_size = int(len(cell_ids) / chunk_num + 0.5)
    for i in range(args.chunk_num):
        chunk = cell_ids[i*chunk_size:(i+1)*chunk_size]
        with open(f"{save_path}chunk{i}.csv", "w") as f:
            f.write(",".join(map(str, chunk)))

def run_jobs(i, bin_size, activation_bin_size, degree, n, save_path, config_path):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    command = ["sbatch", sbatch_script, i, bin_size, activation_bin_size, degree, n, save_path, config_path]
    command = [str(component) for component in command]
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)

if __name__ == "__main__":
    cell_ids = list(map(int, np.genfromtxt(args.cell_ids, delimiter=',')))
    chunk_num = args.chunk_num
    generate_chunks(cell_ids, chunk_num, "misc/chunks/")

    for bin_size in bin_sizes:
        for activation_bin_size, degree in zip(activation_bin_sizes, degrees):
            if activation_bin_size >= bin_size:
                for n in n_kernels:
                    save_path = f"model/params/{bin_size}_{activation_bin_size}_{n}/"
                    for i in range(chunk_num):
                        run_jobs(i, bin_size, activation_bin_size, degree, n, save_path, args.config_path)