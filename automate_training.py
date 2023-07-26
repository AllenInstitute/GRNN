import subprocess
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cell_ids", type=str, help="CSV file containing cell ids to process")
    parser.add_argument("chunk_num", type=int, help="Chunk number")
    parser.add_argument("config_path", type=str, help="Config file")
    args = parser.parse_args()
    
    cell_ids = list(map(int, np.genfromtxt(args.cell_ids, delimiter=',')))
    chunk_size = int(len(cell_ids) / args.chunk_num + 0.5)
    
    for i in range(args.chunk_num):
        chunk = cell_ids[i*chunk_size:(i+1)*chunk_size]
        with open(f"misc/chunks/chunk{i}.csv", "w") as f:
            f.write(",".join(map(str, chunk)))
    
    sbatch_script = "scripts/run_model_pipeline.sh"
    for i in range(args.chunk_num):
        command = ["sbatch", sbatch_script, i, args.config_path]
        command = [str(component) for component in command]
        print(f"Running command: {command}")
        subprocess.run(command)