# ============================================================
# Species analysis - Mann-Whitney U tests (mouse vs human)
# Produces: data/statistical_tests_summary.csv
#
# Provenance: AI-assisted analysis, author-reviewed (see README.md and the
# manuscript "Species Differences in GFR Parameters" section). Adapted from the
# co-author analysis repo (original file: code_for_statistical_tests_summary.py)
# for portable, self-contained execution inside the GRNN repository.
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DATA = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(OUT_DATA, exist_ok=True)

# Load fitted GFR parameters and cell metadata from the repository
with open(os.path.join(REPO_ROOT, 'model', 'best_params.pickle'), 'rb') as f:
    all_params = pickle.load(f)
meta = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'metadata.csv'))

# Build lookups
species_lookup = dict(zip(meta['specimen__id'], meta['donor__species']))
dendrite_lookup = dict(zip(meta['specimen__id'], meta['tag__dendrite_type']))
line_lookup = dict(zip(meta['specimen__id'], meta['line_name']))
layer_lookup = dict(zip(meta['specimen__id'], meta['structure__layer']))

# Use (20, 20) config
config_key = (20, 20)
params = all_params[config_key]

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

# Filter well-fit cells
threshold = 0.5
df_analysis = df[df['evr2'] > threshold].copy()

# E/I classification
def classify_simple(row):
    if row['dendrite_type'] == 'aspiny':
        return 'Inhibitory'
    elif row['dendrite_type'] == 'spiny':
        return 'Excitatory'
    else:
        return 'Sparsely Spiny'

df_analysis['ei_type'] = df_analysis.apply(classify_simple, axis=1)
df_ei = df_analysis[df_analysis['ei_type'].isin(['Excitatory', 'Inhibitory'])].copy()

# Statistical tests summary
param_cols_nice = {
    'alpha_1': 'α₁', 'alpha_2': 'α₂', 'alpha_3': 'α₃', 'alpha_4': 'α₄',
    'alpha_5': 'α₅', 'alpha_6': 'α₆', 'alpha_7': 'α₇', 'alpha_8': 'α₈',
    'beta_1': 'β₁', 'beta_2': 'β₂', 'beta_3': 'β₃', 'beta_4': 'β₄',
    'beta_5': 'β₅', 'beta_6': 'β₆', 'beta_7': 'β₇', 'beta_8': 'β₈',
    'c0': 'c₀', 'c1': 'c₁', 'gamma': 'γ', 'evr2': 'EVR₂'
}

summary_stats = []
for param in list(param_cols_nice.keys()):
    for ei in ['All', 'Excitatory', 'Inhibitory']:
        if ei == 'All':
            m = df_analysis[df_analysis['species'] == 'Mouse'][param].dropna()
            h = df_analysis[df_analysis['species'] == 'Human'][param].dropna()
        else:
            m = df_ei[(df_ei['species'] == 'Mouse') & (df_ei['ei_type'] == ei)][param].dropna()
            h = df_ei[(df_ei['species'] == 'Human') & (df_ei['ei_type'] == ei)][param].dropna()

        if len(m) > 5 and len(h) > 5:
            u, p = stats.mannwhitneyu(m, h, alternative='two-sided')
            summary_stats.append({
                'Parameter': param_cols_nice[param],
                'Subgroup': ei,
                'Mouse_median': f"{m.median():.4f}",
                'Mouse_IQR': f"[{m.quantile(0.25):.4f}, {m.quantile(0.75):.4f}]",
                'Human_median': f"{h.median():.4f}",
                'Human_IQR': f"[{h.quantile(0.25):.4f}, {h.quantile(0.75):.4f}]",
                'p_value': f"{p:.2e}",
                'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            })

stats_df = pd.DataFrame(summary_stats)

# Save the statistical tests summary
out_csv = os.path.join(OUT_DATA, 'statistical_tests_summary.csv')
stats_df.to_csv(out_csv, index=False)
print(f"Wrote {out_csv} ({len(stats_df)} tests)")