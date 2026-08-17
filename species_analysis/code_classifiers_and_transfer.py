# ============================================================
# Species classifier and cross-species E/I transfer analysis
# (Run in session but output was printed, not saved as artifact)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score

# Assumes df_analysis and df_ei are loaded from 02_build_dataset.py

feature_cols = [f'alpha_{i}' for i in range(1,9)] + [f'beta_{i}' for i in range(1,9)] + ['c0', 'c1', 'gamma']

df_clf = df_analysis[df_analysis['ei_type'].isin(['Excitatory', 'Inhibitory'])].copy()
X = df_clf[feature_cols].values
y_species = (df_clf['species'] == 'Human').astype(int).values
y_ei = (df_clf['ei_type'] == 'Inhibitory').astype(int).values
X_scaled = StandardScaler().fit_transform(X)

# ---- SPECIES CLASSIFIER ----
print("=" * 70)
print("CLASSIFIER 1: Species (Human vs Mouse)")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
                   ('Random Forest', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))]:
    scores = cross_val_score(clf, X_scaled, y_species, cv=skf, scoring='roc_auc')
    acc_scores = cross_val_score(clf, X_scaled, y_species, cv=skf, scoring='balanced_accuracy')
    print(f"\n{name}:")
    print(f"  ROC-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"  Balanced Accuracy: {acc_scores.mean():.3f} +/- {acc_scores.std():.3f}")

# Feature importances
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_scaled, y_species)
coefs = pd.DataFrame({'feature': feature_cols, 'coefficient': lr.coef_[0]})
coefs['abs_coef'] = np.abs(coefs['coefficient'])
coefs = coefs.sort_values('abs_coef', ascending=False)
print("\nLogistic Regression Feature Importances (species classifier):")
print(coefs.to_string(index=False))

# ---- E/I CLASSIFIER WITHIN EACH SPECIES ----
print("\n" + "=" * 70)
print("CLASSIFIER 2: E/I Type (within species)")
print("=" * 70)

for sp in ['Mouse', 'Human']:
    mask = df_clf['species'] == sp
    X_sp = X_scaled[mask]
    y_sp = y_ei[mask]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, clf in [('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
                       ('Random Forest', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))]:
        scores = cross_val_score(clf, X_sp, y_sp, cv=skf, scoring='roc_auc')
        acc_scores = cross_val_score(clf, X_sp, y_sp, cv=skf, scoring='balanced_accuracy')
        print(f"\n{sp} - {name}:")
        print(f"  ROC-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
        print(f"  Balanced Accuracy: {acc_scores.mean():.3f} +/- {acc_scores.std():.3f}")

# ---- CROSS-SPECIES E/I TRANSFER ----
print("\n" + "=" * 70)
print("CROSS-SPECIES GENERALIZATION: Train E/I on Mouse -> Test on Human")
print("=" * 70)

X_mouse = X_scaled[df_clf['species'].values == 'Mouse']
y_mouse_ei = y_ei[df_clf['species'].values == 'Mouse']
X_human = X_scaled[df_clf['species'].values == 'Human']
y_human_ei = y_ei[df_clf['species'].values == 'Human']

for name, clf in [('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
                   ('Random Forest', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))]:
    clf.fit(X_mouse, y_mouse_ei)
    y_pred_human = clf.predict(X_human)
    y_prob_human = clf.predict_proba(X_human)[:, 1]
    
    human_bacc = balanced_accuracy_score(y_human_ei, y_pred_human)
    human_auc = roc_auc_score(y_human_ei, y_prob_human)
    
    print(f"\n{name}:")
    print(f"  Human E/I (transfer): Balanced Acc = {human_bacc:.3f}, AUC = {human_auc:.3f}")
    print(classification_report(y_human_ei, y_pred_human, target_names=['Excitatory', 'Inhibitory']))

# ---- REVERSE DIRECTION ----
print("\nREVERSE: Train E/I on Human -> Test on Mouse")
for name, clf in [('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
                   ('Random Forest', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))]:
    clf.fit(X_human, y_human_ei)
    y_pred_mouse = clf.predict(X_mouse)
    y_prob_mouse = clf.predict_proba(X_mouse)[:, 1]
    mouse_bacc = balanced_accuracy_score(y_mouse_ei, y_pred_mouse)
    mouse_auc = roc_auc_score(y_mouse_ei, y_prob_mouse)
    print(f"  {name}: Balanced Acc = {mouse_bacc:.3f}, AUC = {mouse_auc:.3f}")
