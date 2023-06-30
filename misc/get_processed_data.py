import pandas as pd
import subprocess

valid_ids = pd.read_csv("valid_ids.csv", header=None)
valid_ids = valid_ids.to_numpy().flatten()

for cell_id in valid_ids[:10]:
    print(cell_id)