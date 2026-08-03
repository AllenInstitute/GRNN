# ============================================================
# Species analysis - extract per-cell GFR parameters
# Produces: data/human_vs_mouse_gfr_params.csv
#
# Provenance: AI-assisted analysis, author-reviewed (see README.md and the
# manuscript "Species Differences in GFR Parameters" section). Adapted from the
# co-author analysis repo (original file: code_for_human_vs_mouse_gfr_params.py)
# for portable, self-contained execution inside the GRNN repository.
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DATA = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(OUT_DATA, exist_ok=True)

# Load fitted GFR parameters and cell metadata from the repository
with open(os.path.join(REPO_ROOT, 'model', 'best_params.pickle'), 'rb') as f:
    all_params = pickle.load(f)
meta = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'metadata.csv'))

# Use (20, 20) config
config_key = (20, 20)
params = all_params[config_key]

# Build species lookup from metadata
species_lookup = dict(zip(meta['specimen__id'], meta['donor__species']))
dendrite_lookup = dict(zip(meta['specimen__id'], meta['tag__dendrite_type']))
line_lookup = dict(zip(meta['specimen__id'], meta['line_name']))
layer_lookup = dict(zip(meta['specimen__id'], meta['structure__layer']))

# Build dataset with species labels
records = []
for cell_id in params:
    species = species_lookup.get(cell_id, 'Unknown')
    dendrite = dendrite_lookup.get(cell_id, 'Unknown')
    line = line_lookup.get(cell_id, '')
    layer = layer_lookup.get(cell_id, '')
    
    p = params[cell_id]['params']
    evr1 = params[cell_id]['evr1']
    evr2 = params[cell_id]['evr2']
    
    a = np.asarray(p['a'], dtype=float).reshape(-1)
    b = np.asarray(p['b'], dtype=float).reshape(-1)
    pc = np.asarray(p['g']['poly_coeff'], dtype=float).reshape(-1)
    gb = np.asarray(p['g']['b'], dtype=float).reshape(-1)
    mc = np.asarray(p['g']['max_current'], dtype=float).reshape(-1)
    mfr = np.asarray(p['g']['max_firing_rate'], dtype=float).reshape(-1)
    
    # Derived activation function parameters
    c0 = (pc[0]**2 - gb[0]) / mc[0]
    c1 = pc[1]**2 / mc[0]
    gamma = mfr[0]
    
    record = {
        'cell_id': cell_id,
        'species': 'Human' if species == 'Homo Sapiens' else 'Mouse',
        'dendrite_type': dendrite,
        'line_name': str(line) if isinstance(line, str) else '',
        'layer': layer,
        'evr1': evr1,
        'evr2': evr2,
        'c0': c0,
        'c1': c1,
        'gamma': gamma,
    }
    
    for i in range(len(a)):
        record[f'alpha_{i+1}'] = a[i]
    for i in range(len(b)):
        record[f'beta_{i+1}'] = b[i]
    
    record['poly_coeff_0'] = pc[0]
    record['poly_coeff_1'] = pc[1]
    record['g_b'] = gb[0]
    record['max_current'] = mc[0]
    record['max_firing_rate'] = mfr[0]
    
    records.append(record)

df = pd.DataFrame(records)

# Filter to well-fit models (EVR2 > 0.5)
threshold = 0.5
df_analysis = df[df['evr2'] > threshold].copy()

# Add E/I classification
inhibitory_lines = ['Pvalb', 'Sst', 'Vip', 'Htr3a', 'Ndnf']
def classify_ei(row):
    if row['species'] == 'Human':
        if row['dendrite_type'] == 'aspiny':
            return 'Inhibitory (Human)'
        elif row['dendrite_type'] == 'spiny':
            return 'Excitatory (Human)'
        else:
            return 'Other (Human)'
    else:
        line = str(row['line_name']).split('-')[0] if isinstance(row['line_name'], str) and row['line_name'] else ''
        if line in inhibitory_lines:
            return 'Inhibitory (Mouse)'
        elif row['dendrite_type'] == 'spiny':
            return 'Excitatory (Mouse)'
        else:
            return 'Other (Mouse)'

df_analysis['cell_class'] = df_analysis.apply(classify_ei, axis=1)

def classify_simple(row):
    if row['dendrite_type'] == 'aspiny':
        return 'Inhibitory'
    elif row['dendrite_type'] == 'spiny':
        return 'Excitatory'
    else:
        return 'Sparsely Spiny'

df_analysis['ei_type'] = df_analysis.apply(classify_simple, axis=1)

# Save the analysis dataframe
out_csv = os.path.join(OUT_DATA, 'human_vs_mouse_gfr_params.csv')
df_analysis.to_csv(out_csv, index=False)
print(f"Wrote {out_csv} ({len(df_analysis)} well-fit cells: "
      f"{(df_analysis['species'] == 'Mouse').sum()} mouse, "
      f"{(df_analysis['species'] == 'Human').sum()} human)")