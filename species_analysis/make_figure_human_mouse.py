# ============================================================
# Species analysis - human vs mouse GFR figure (manuscript Figure fig:human_mouse)
# Produces: figures/fig_human_mouse_v2.pdf and .png
#
# Provenance: AI-assisted analysis, author-reviewed (see README.md and the
# manuscript "Species Differences in GFR Parameters" section). Adapted from the
# co-author analysis repo (original file: code_for_fig_human_mouse_v2.py) for
# portable, self-contained execution inside the GRNN repository.
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_FIG = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUT_FIG, exist_ok=True)

# Load data
meta = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'metadata.csv'))
with open(os.path.join(REPO_ROOT, 'model', 'best_params.pickle'), 'rb') as f:
    all_params = pickle.load(f)

species_lookup = dict(zip(meta['specimen__id'], meta['donor__species']))
dendrite_lookup = dict(zip(meta['specimen__id'], meta['tag__dendrite_type']))

config_key = (20, 20)
params = all_params[config_key]

records = []
for cell_id in params:
    species = species_lookup.get(cell_id, 'Unknown')
    dendrite = dendrite_lookup.get(cell_id, 'Unknown')
    p = params[cell_id]['params']
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
    
    record = {'cell_id': cell_id, 'species': 'Human' if species == 'Homo Sapiens' else 'Mouse',
              'dendrite_type': dendrite, 'evr2': evr2, 'c0': c0, 'c1': c1, 'gamma': gamma}
    for i in range(len(a)):
        record[f'alpha_{i+1}'] = a[i]
    for i in range(len(b)):
        record[f'beta_{i+1}'] = b[i]
    records.append(record)

df = pd.DataFrame(records)
df_analysis = df[df['evr2'] > 0.5].copy()
df_analysis['ei_type'] = df_analysis['dendrite_type'].map(
    {'aspiny': 'Inhibitory', 'spiny': 'Excitatory', 'sparsely spiny': 'Sparsely Spiny'})
df_ei = df_analysis[df_analysis['ei_type'].isin(['Excitatory', 'Inhibitory'])].copy()

ds = torch.tensor([1.0000, 0.6321, 0.3935, 0.1813, 0.0952, 0.0392, 0.0198, 0.0100])

# ====== PUBLICATION FIGURE ======
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'font.family': 'sans-serif'})

fig = plt.figure(figsize=(7.2, 6.0))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32,
                       left=0.09, right=0.97, top=0.95, bottom=0.10)

MOUSE_COLOR = '#3B7DD8'
HUMAN_COLOR = '#E8833A'

# ---- Panel (a): Current kernel overlays ----
ax_a = fig.add_subplot(gs[0, 0])
ts_current = np.linspace(0, 10, 200)
for ei, ls in [('Excitatory', '-'), ('Inhibitory', '--')]:
    for sp, color in [('Mouse', MOUSE_COLOR), ('Human', HUMAN_COLOR)]:
        subset = df_ei[(df_ei['species'] == sp) & (df_ei['ei_type'] == ei)]
        weights = subset[[f'alpha_{i}' for i in range(1, 9)]].values
        vals = np.zeros((len(subset), len(ts_current)))
        for i, t in enumerate(ts_current):
            vals[:, i] = np.sum(weights * np.power(1 - ds.numpy(), t), axis=1)
        mean_v = np.mean(vals, axis=0)
        sem_v = np.std(vals, axis=0) / np.sqrt(len(subset))
        short_ei = 'Exc' if ei == 'Excitatory' else 'Inh'
        label = f'{sp} {short_ei} (n={len(subset)})'
        ax_a.plot(ts_current, mean_v, color=color, linewidth=1.5, linestyle=ls, label=label)
        ax_a.fill_between(ts_current, mean_v - sem_v*1.96, mean_v + sem_v*1.96, alpha=0.12, color=color)

ax_a.set_xlabel('Time (bins × 20 ms)')
ax_a.set_ylabel('Amplitude')
ax_a.set_title('Current kernel $k_I(t)$')
ax_a.legend(fontsize=6.5, frameon=True, fancybox=False, edgecolor='gray')
ax_a.set_ylim([-0.1, 1.8])
ax_a.text(-0.18, 1.05, 'a', transform=ax_a.transAxes, fontsize=14, fontweight='bold', va='top')

# ---- Panel (b): Firing rate kernel overlays ----
ax_b = fig.add_subplot(gs[0, 1])
ts_firing = np.linspace(0, 50, 200)
for ei, ls in [('Excitatory', '-'), ('Inhibitory', '--')]:
    for sp, color in [('Mouse', MOUSE_COLOR), ('Human', HUMAN_COLOR)]:
        subset = df_ei[(df_ei['species'] == sp) & (df_ei['ei_type'] == ei)]
        weights = subset[[f'beta_{i}' for i in range(1, 9)]].values
        vals = np.zeros((len(subset), len(ts_firing)))
        for i, t in enumerate(ts_firing):
            vals[:, i] = np.sum(weights * np.power(1 - ds.numpy(), t), axis=1)
        mean_v = np.mean(vals, axis=0)
        sem_v = np.std(vals, axis=0) / np.sqrt(len(subset))
        short_ei = 'Exc' if ei == 'Excitatory' else 'Inh'
        ax_b.plot(ts_firing, mean_v, color=color, linewidth=1.5, linestyle=ls, label=f'{sp} {short_ei}')
        ax_b.fill_between(ts_firing, mean_v - sem_v*1.96, mean_v + sem_v*1.96, alpha=0.12, color=color)

ax_b.set_xlabel('Time (bins × 20 ms)')
ax_b.set_ylabel('Amplitude')
ax_b.set_title('Firing-rate kernel $k_f(t)$')
ax_b.legend(fontsize=6.5, frameon=True, fancybox=False, edgecolor='gray')
ax_b.set_ylim([-4.0, 0.3])
ax_b.text(-0.18, 1.05, 'b', transform=ax_b.transAxes, fontsize=14, fontweight='bold', va='top')

# ---- Bottom row: 4 violin panels ----
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, :], wspace=0.50)

key_params = [
    ('alpha_8', r'$\alpha_8$' + '\n(slowest current)'),
    ('beta_5', r'$\beta_5$' + '\n(slow FR feedback)'),
    ('gamma', r'$\gamma$' + '\n(max firing rate)'),
    ('c1', r'$c_1$' + '\n(activation gain)'),
]

for pidx, (param, label) in enumerate(key_params):
    ax = fig.add_subplot(gs_bottom[0, pidx])
    
    data_groups = []
    xlabels = []
    for ei in ['Excitatory', 'Inhibitory']:
        for sp in ['Mouse', 'Human']:
            subset = df_ei[(df_ei['species'] == sp) & (df_ei['ei_type'] == ei)]
            data_groups.append(subset[param].values)
            xlabels.append(f'{"M" if sp == "Mouse" else "H"}\n{ei[:3]}')
    
    positions = [0, 1, 2.5, 3.5]
    parts = ax.violinplot(data_groups, positions=positions, showmeans=False, showmedians=True, widths=0.7)
    colors_v = [MOUSE_COLOR, HUMAN_COLOR, MOUSE_COLOR, HUMAN_COLOR]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_v[i])
        pc.set_alpha(0.55)
        pc.set_edgecolor('none')
    for partname in ['cmedians', 'cbars', 'cmins', 'cmaxes']:
        parts[partname].set_color('black')
        parts[partname].set_linewidth(0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=7, linespacing=0.9)
    ax.set_title(label, fontsize=8.5, linespacing=0.95)
    
    # Significance annotations
    for ei_idx, ei in enumerate(['Excitatory', 'Inhibitory']):
        m_vals = df_ei[(df_ei['species'] == 'Mouse') & (df_ei['ei_type'] == ei)][param]
        h_vals = df_ei[(df_ei['species'] == 'Human') & (df_ei['ei_type'] == ei)][param]
        _, p = stats.mannwhitneyu(m_vals, h_vals, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        x1, x2 = positions[ei_idx*2], positions[ei_idx*2+1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_pos = ax.get_ylim()[1] - y_range * 0.02
        
        # Draw bracket
        bracket_y = y_pos - y_range * 0.02
        ax.plot([x1, x1, x2, x2], [bracket_y - y_range*0.02, bracket_y, bracket_y, bracket_y - y_range*0.02],
                color='black', linewidth=0.7)
        ax.text((x1+x2)/2, bracket_y + y_range*0.01, sig, ha='center', fontsize=6.5, 
               color='#CC0000' if sig != 'n.s.' else '#666666', fontweight='bold')
    
    panel_letter = chr(ord('c') + pidx)
    ax.text(-0.22, 1.15, panel_letter, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

fig.savefig(os.path.join(OUT_FIG, 'fig_human_mouse_v2.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT_FIG, 'fig_human_mouse_v2.png'), dpi=300, bbox_inches='tight')
print(f"Wrote {os.path.join(OUT_FIG, 'fig_human_mouse_v2.pdf')} and .png")
plt.close()