# ============================================================
# Inhibitory subclass classification: train on mouse Cre-line
# ground truth, predict human subclass from electrophysiology
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats

meta = pd.read_csv('/tmp/GRNN/data/metadata.csv')

# Mouse inhibitory neurons with Cre-line ground truth
mouse_inhib = meta[(meta['donor__species'] == 'Mus musculus') & 
                   (meta['tag__dendrite_type'] == 'aspiny')].copy()

def get_subclass(line_name):
    if not isinstance(line_name, str): return 'Unknown'
    prefix = line_name.split('-')[0]
    return {'Pvalb':'Pvalb','Sst':'Sst','Vip':'Vip',
            'Htr3a':'Htr3a/Ndnf','Ndnf':'Htr3a/Ndnf'}.get(prefix, 'Other')

mouse_inhib['subclass'] = mouse_inhib['line_name'].apply(get_subclass)

# Key electrophysiological features for subclass discrimination
clf_features = ['ef__upstroke_downstroke_ratio_long_square',  # Pvalb: low (~1.3), Vip/Htr3a: high (~2.8-3.1)
                'ef__f_i_curve_slope',                        # Pvalb: high (~0.93), Sst: low (~0.36)
                'ef__tau',                                     # Pvalb: short (~7ms), Sst: long (~19ms)
                'ef__adaptation',                              # Pvalb: near 0, Sst/Vip: higher
                'ef__avg_firing_rate']                         # Pvalb: high (~80Hz), Sst: low (~25Hz)

# Show feature distributions by subclass
print("=== Key ephys features by subclass (mouse, ground truth) ===")
for feat in clf_features:
    print(f"\n{feat}:")
    for sc in ['Pvalb', 'Sst', 'Vip', 'Htr3a/Ndnf']:
        vals = mouse_inhib[mouse_inhib['subclass'] == sc][feat].dropna()
        if len(vals) > 0:
            print(f"  {sc:12s}: median={vals.median():.3f}, IQR=[{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}] (n={len(vals)})")

# Train classifier on mouse Pvalb/Sst/Vip
mouse_clf = mouse_inhib[mouse_inhib['subclass'].isin(['Pvalb', 'Sst', 'Vip'])].copy()
mouse_clf = mouse_clf.dropna(subset=clf_features)

X_mouse = mouse_clf[clf_features].values
y_mouse = mouse_clf['subclass'].values

scaler = StandardScaler()
X_mouse_scaled = scaler.fit_transform(X_mouse)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_mouse_scaled, y_mouse, cv=skf, scoring='balanced_accuracy')
print(f"\nMouse subclass classifier (Pvalb/Sst/Vip) 5-fold CV:")
print(f"  Balanced accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

# Train on all mouse data, predict human
rf.fit(X_mouse_scaled, y_mouse)

human_inhib = meta[(meta['donor__species'] == 'Homo Sapiens') & 
                   (meta['tag__dendrite_type'] == 'aspiny')].copy()
human_inhib = human_inhib.dropna(subset=clf_features)
X_human = human_inhib[clf_features].values
X_human_scaled = scaler.transform(X_human)

human_inhib['predicted_subclass'] = rf.predict(X_human_scaled)
human_inhib['predicted_proba_Pvalb'] = rf.predict_proba(X_human_scaled)[:, list(rf.classes_).index('Pvalb')]
human_inhib['predicted_proba_Sst'] = rf.predict_proba(X_human_scaled)[:, list(rf.classes_).index('Sst')]
human_inhib['predicted_proba_Vip'] = rf.predict_proba(X_human_scaled)[:, list(rf.classes_).index('Vip')]

print(f"\nHuman inhibitory neurons classified (n={len(human_inhib)}):")
print(human_inhib['predicted_subclass'].value_counts())

# Compare GFR parameters at subclass level
print("\n=== GFR parameter comparison by subclass ===")
for param in ['gamma', 'c1', 'alpha_8', 'beta_5']:
    print(f"\n{param}:")
    for sc in ['Pvalb', 'Sst', 'Vip', 'Excitatory']:
        m = df_sub[(df_sub['species'] == 'Mouse') & (df_sub['subclass'] == sc)][param].dropna()
        h = df_sub[(df_sub['species'] == 'Human') & (df_sub['subclass'] == sc)][param].dropna()
        if len(m) > 3 and len(h) > 3:
            _, p_val = stats.mannwhitneyu(m, h, alternative='two-sided')
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f"  {sc:12s}: Mouse={m.median():.4f} (n={len(m)}), Human={h.median():.4f} (n={len(h)}), p={p_val:.2e} {sig}")
