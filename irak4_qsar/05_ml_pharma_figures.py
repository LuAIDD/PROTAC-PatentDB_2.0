"""
ML QSAR Pharmaceutical Interpretability Figures

Generates:
1. Permutation Feature Importance bar chart (Clean + Annotated)
2. Physicochemical SAR scatter grid: logP, linker flexibility, HBD vs pDC50
3. Combined inline figure for the manuscript

All SVG vector output with high-res PNG fallback.
"""
import numpy as np
import pandas as pd
import pickle, sys, os, warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

OUTDIR = 'figures'
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.linewidth': 1.0})

# ============================================================
# 1. Data loading and feature engineering (reproduce Part3 pipeline)
# ============================================================
print("[1/5] Loading and featurizing data...")

df = pd.read_excel('IRAK4_PROTAC_cleaned_v2.xlsx', sheet_name='Cleaned_Data', header=0)

# pDC50 target
sub = df[df['DC50_uM'].notna()].dropna(subset=['SMILES']).drop_duplicates('SMILES').reset_index(drop=True)
y = sub['pDC50'].values.astype(float)
tier = sub['Activity_tier'].values

# RDKit descriptors + Morgan fingerprint (same as Part3)
names = [d[0] for d in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)

def compute_features(smi):
    m = Chem.MolFromSmiles(smi)
    d = np.array(calc.CalcDescriptors(m), dtype=float)
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), dtype=float)
    return np.concatenate([d, fp])

X_raw = np.nan_to_num(np.array([compute_features(s) for s in sub['SMILES']]), nan=0, posinf=0, neginf=0)

# Add linker fragment descriptors
lk = sub[['Linker_heavyatoms', 'Linker_RotBonds', 'Linker_flexibility', 'Linker_PEGunits']].fillna(0).values
X_raw = np.concatenate([X_raw, lk], axis=1)
fnames = names + [f'MorganFP_{i}' for i in range(1024)] + ['linker_heavyatoms', 'linker_rotB', 'linker_flex', 'linker_PEG']

print(f"  {len(y)} compounds, {X_raw.shape[1]} raw features")

# Feature selection (same as Part3 pipeline)
idx = np.arange(len(y))
tr, tmp = train_test_split(idx, test_size=0.30, random_state=42)
va, te = train_test_split(tmp, test_size=0.50, random_state=42)

vt = VarianceThreshold(0.01).fit(X_raw[tr])
X1 = vt.transform(X_raw)
f1 = [f for f, k in zip(fnames, vt.get_support()) if k]

sc = StandardScaler().fit(X1[tr])
Xs = sc.transform(X1)

C = np.corrcoef(Xs[tr].T)
keep = np.ones(Xs.shape[1], bool)
for i in range(len(keep)):
    if not keep[i]: continue
    dup = np.where((np.abs(C[i]) > 0.9) & (np.arange(len(keep)) > i))[0]
    keep[dup] = False
Xs = Xs[:, keep]
f2 = [f for f, k in zip(f1, keep) if k]

rng = np.random.default_rng(0)
cnt = np.zeros(Xs.shape[1])
for i in range(100):
    bi = rng.choice(tr, size=int(0.8*len(tr)), replace=False)
    la = Lasso(alpha=0.02, max_iter=3000).fit(Xs[bi], y[bi])
    cnt += (np.abs(la.coef_) > 1e-6)
freq = cnt / 100
stable = np.where(freq >= 0.5)[0]
Xsel = Xs[:, stable]
fsel = [f2[i] for i in stable]
print(f"  {len(fsel)} selected features")

# ============================================================
# 2. Train Extra-Trees and compute permutation importance
# ============================================================
print("[2/5] Training Extra-Trees + permutation importance...")

trva = np.concatenate([tr, va])
et = ExtraTreesRegressor(n_estimators=600, max_depth=None, max_features='sqrt',
                         min_samples_leaf=1, random_state=1, n_jobs=-1)
et.fit(Xsel[trva], y[trva])
y_pred = et.predict(Xsel[te])
r2_test = r2_score(y[te], y_pred)
print(f"  Extra-Trees test R² = {r2_test:.3f}")

# Permutation importance on validation set
pi = permutation_importance(et, Xsel[te], y[te], n_repeats=20, random_state=2, n_jobs=-1)
imp_mean = pi.importances_mean
imp_std = pi.importances_std

# Sort and select top features
top_k = 30
top_idx = np.argsort(imp_mean)[-top_k:][::-1]
top_names = [fsel[i] for i in top_idx]
top_imp_mean = imp_mean[top_idx]
top_imp_std = imp_std[top_idx]

# Map MorganFP bit names to human-readable labels
def humanize(feature_name):
    if feature_name.startswith('MorganFP_'):
        bit = feature_name.split('_')[1]
        return f'FP bit {bit}'
    # Known descriptors → readable names
    name_map = {
        'AvgIpc': 'Avg information content',
        'PEOE_VSA14': 'PEOE VSA14 (polar area)',
        'RingCount': 'Ring count',
        'NumAromaticRings': 'Aromatic rings',
        'SlogP_VSA3': 'SlogP VSA3 (lipophilic)',
        'SlogP_VSA5': 'SlogP VSA5 (lipophilic area)',
        'SMR_VSA5': 'SMR VSA5 (molar refractivity)',
        'SlogP_VSA2': 'SlogP VSA2',
        'MolLogP': 'cLogP (whole-molecule)',
        'MolWt': 'Molecular weight',
        'NumRotatableBonds': 'Rotatable bonds',
        'NumHBD': 'H-bond donors',
        'NumHBA': 'H-bond acceptors',
        'TPSA': 'Polar surface area',
        'FractionCSP3': 'Fraction C(sp³)',
        'HeavyAtomCount': 'Heavy atom count',
        'linker_heavyatoms': 'Linker length (heavy atoms)',
        'linker_rotB': 'Linker rotatable bonds',
        'linker_flex': 'Linker flexibility (density)',
        'linker_PEG': 'Linker PEG units',
        'fr_Al_COO': 'Aliphatic ester count',
        'fr_Ar_N': 'Aromatic N count',
        'fr_Ar_COO': 'Aromatic ester count',
        'fr_C_O': 'C-O fragment count',
        'fr_C_O_noCOO': 'C-O (non-ester) count',
        'MaxAbsEStateIndex': 'Max E-State index',
        'MinAbsEStateIndex': 'Min E-State index',
        'MaxEStateIndex': 'Max E-State index',
        'MinEStateIndex': 'Min E-State index',
        'EState_VSA2': 'EState VSA2',
        'EState_VSA3': 'EState VSA3',
        'VSA_EState1': 'VSA EState1',
        'VSA_EState2': 'VSA EState2',
        'VSA_EState9': 'VSA EState9',
        'PEOE_VSA10': 'PEOE VSA10',
        'PEOE_VSA11': 'PEOE VSA11',
        'PEOE_VSA12': 'PEOE VSA12',
        'PEOE_VSA13': 'PEOE VSA13',
        'PEOE_VSA6': 'PEOE VSA6',
        'PEOE_VSA7': 'PEOE VSA7',
        'PEOE_VSA8': 'PEOE VSA8',
        'PEOE_VSA9': 'PEOE VSA9',
        'SMR_VSA1': 'SMR VSA1',
        'SMR_VSA2': 'SMR VSA2',
        'SMR_VSA3': 'SMR VSA3',
        'SMR_VSA4': 'SMR VSA4',
        'SMR_VSA6': 'SMR VSA6',
        'SlogP_VSA1': 'SlogP VSA1',
        'SlogP_VSA4': 'SlogP VSA4',
        'SlogP_VSA6': 'SlogP VSA6',
        'NumHAcceptors': 'H-bond acceptor count',
        'NumHDonors': 'H-bond donor count',
        'BertzCT': 'Bertz complexity index',
        'Chi0': 'Chi0 (connectivity)',
        'Chi1': 'Chi1 (connectivity)',
        'Chi0n': 'Chi0n (valence connectivity)',
        'Chi4n': 'Chi4n (valence path)',
        'HallKierAlpha': 'Hall-Kier alpha',
        'Kappa1': 'Kappa1 (shape)',
        'Kappa2': 'Kappa2 (shape)',
        'Kappa3': 'Kappa3 (shape)',
        'LabuteASA': 'Labute ASA',
        'MQNs': 'Molecular quantum numbers',
        'NHOHCount': 'NH+OH count',
        'NOCount': 'N+O count',
        'BalabanJ': 'Balaban J index',
    }
    if feature_name in name_map:
        return name_map[feature_name]
    return feature_name

top_labels = [humanize(n) for n in top_names]

# ============================================================
# 3. Feature Importance Bar Chart (Clean + Annotated)
# ============================================================
print("[3/5] Feature importance bar chart...")

def make_feature_importance_chart(out_prefix, title=None, show_labels=True):
    fig, ax = plt.subplots(figsize=(7.5, 8.5))

    colors = []
    for lbl in top_labels:
        if 'FP bit' in lbl:
            colors.append('#aec7e8')     # Morgan FP → blue
        elif 'linker' in lbl.lower():
            colors.append('#ff9896')     # linker descriptor → red
        else:
            colors.append('#98df8a')     # physicochemical descriptor → green

    bars = ax.barh(range(len(top_names)), top_imp_mean, xerr=top_imp_std,
                   height=0.7, color=colors, edgecolor='#555', linewidth=0.5,
                   capsize=2, error_kw={'lw': 1.0, 'capsize': 3})

    ax.set_yticks(range(len(top_names)))
    if show_labels:
        ax.set_yticklabels(top_labels, fontsize=8.5)
    else:
        ax.set_yticklabels([])

    ax.invert_yaxis()
    ax.set_xlabel('Permutation importance (ΔR²)', fontsize=11)
    ax.set_xlim(0, top_imp_mean[0] * 1.20)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    # Legend
    leg = [Patch(fc='#98df8a', label='Physicochemical descriptors'),
           Patch(fc='#aec7e8', label='Morgan fingerprints (bits)'),
           Patch(fc='#ff9896', label='Linker fragment descriptors')]
    if show_labels:
        ax.legend(handles=leg, fontsize=9, loc='lower right', framealpha=0.9,
                 edgecolor='#ccc', fancybox=False)

    ax.grid(axis='x', ls=':', alpha=0.35)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}.svg', facecolor='white', dpi=150)
    plt.savefig(f'{out_prefix}.png', facecolor='white', dpi=200)
    plt.close()
    print(f"    Saved {out_prefix}.svg + .png")

# Clean version
make_feature_importance_chart(f'{OUTDIR}/Fig_ML_feature_importance_clean',
                              title=None, show_labels=False)
# Annotated version
make_feature_importance_chart(f'{OUTDIR}/Fig_ML_feature_importance_annotated',
                              title=f'Extra-Trees Feature Importance — IRAK4 PROTAC Potency Prediction\nTest R² = {r2_test:.3f} · {len(y)} compounds · {len(fsel)} features',
                              show_labels=True)

# ============================================================
# 4. SAR Scatter Plots: Physical Chemistry vs Potency
# ============================================================
print("[4/5] Physicochemical SAR scatter plots...")

# Get key properties from the dataset
logP = sub['cLogP'].values
hbd = sub['HBD'].values
hba = sub['HBA'].values
mw = sub['MW'].values
rotB = sub['RotBonds'].values
tpsa = sub['TPSA'].values
linker_len = sub['Linker_heavyatoms'].values
linker_flex = sub['Linker_flexibility'].values
linker_rotB = sub['Linker_RotBonds'].values

# Tier colors
tier_colors = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#1f77b4'}
tier_markers = {'High': 'o', 'Medium': 's', 'Low': '^'}
tier_labels = {'High': 'High (≤10 nM)', 'Medium': 'Medium (10–100 nM)', 'Low': 'Low (>100 nM)'}

def make_sar_grid(out_prefix, annotation=True):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))

    panels = [
        (axes[0, 0], logP, 'cLogP', 'Lipophilicity', 'higher = more lipophilic'),
        (axes[0, 1], linker_flex, 'Linker flexibility\n(rotatable bond density)',
         'Linker conformational rigidity', 'higher = more flexible linker'),
        (axes[0, 2], hbd, 'H-bond donors', 'H-bond donor count', ''),
        (axes[1, 0], hba, 'H-bond acceptors', 'H-bond acceptor count', ''),
        (axes[1, 1], mw, 'Molecular weight (Da)', 'Molecular size', ''),
        (axes[1, 2], tpsa, 'TPSA (Å²)', 'Polar surface area', ''),
    ]

    for ax, xvals, xlabel, short_title, note in panels:
        valid = ~np.isnan(xvals)
        for t in ['High', 'Medium', 'Low']:
            mask = valid & (tier == t)
            ax.scatter(xvals[mask], y[mask], s=22, c=tier_colors[t],
                      marker=tier_markers[t], alpha=0.55, edgecolors='none',
                      label=tier_labels[t] if ax == axes[0, 0] else '',
                      rasterized=True)

        ax.set_xlabel(xlabel, fontsize=10)
        if annotation:
            ax.set_ylabel('pDC50', fontsize=10)
            # Spearman correlation
            mask2 = valid & ~np.isnan(y)
            from scipy.stats import spearmanr
            if mask2.sum() > 5:
                rho, pv = spearmanr(xvals[mask2], y[mask2])
                ax.text(0.95, 0.07, f'ρ = {rho:+.2f}', transform=ax.transAxes,
                       ha='right', fontsize=9.5, color='#555',
                       bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.7))
            if note:
                ax.text(0.05, 0.93, note, transform=ax.transAxes,
                       ha='left', fontsize=8, color='#888', fontstyle='italic')

        ax.grid(ls=':', alpha=0.3)
        ax.tick_params(labelsize=9)

    # Shared legend for first subplot only
    axes[0, 0].legend(fontsize=9, loc='upper left', framealpha=0.85,
                      markerscale=1.3, title='Potency tier',
                      title_fontsize=9.5, edgecolor='#ccc')

    if annotation:
        fig.suptitle('Physicochemical Determinants of IRAK4 PROTAC Degradation Potency',
                    fontsize=13.5, fontweight='bold', y=1.01)
        fig.text(0.5, -0.01,
                f'{len(y)} PROTACs · 16 patent families · Spearman ρ shown per panel\n'
                'Lower cLogP, fewer HBD, and more rigid linkers → higher potency',
                ha='center', fontsize=10, color='#555')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}.svg', facecolor='white', dpi=150, bbox_inches='tight')
    plt.savefig(f'{out_prefix}.png', facecolor='white', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved {out_prefix}.svg + .png")

make_sar_grid(f'{OUTDIR}/Fig_SAR_physicochemical_annotated', annotation=True)
make_sar_grid(f'{OUTDIR}/Fig_SAR_physicochemical_clean', annotation=False)

# ============================================================
# 5. Combined "pharmaceutical insight" multi-panel figure
# ============================================================
print("[5/5] Combined pharmaceutical insight figure...")

fig = plt.figure(figsize=(16, 12))

# Panel A: Feature importance (top 15 for compactness)
ax_a = fig.add_axes([0.05, 0.08, 0.28, 0.88])
top15_idx = np.argsort(imp_mean)[-15:][::-1]
top15_names = [humanize(fsel[i]) for i in top15_idx]
top15_imp = imp_mean[top15_idx]
top15_err = imp_std[top15_idx]
colors_a = []
for nm in top15_names:
    if 'FP bit' in nm: colors_a.append('#aec7e8')
    elif 'linker' in nm.lower() or 'rotat' in nm.lower() or 'flex' in nm.lower(): colors_a.append('#ff9896')
    else: colors_a.append('#98df8a')

ax_a.barh(range(15), top15_imp, xerr=top15_err, height=0.7, color=colors_a,
          edgecolor='#555', linewidth=0.5, capsize=2, error_kw={'lw': 1.0})
ax_a.set_yticks(range(15)); ax_a.set_yticklabels(top15_names, fontsize=7.8)
ax_a.invert_yaxis(); ax_a.set_xlabel('ΔR²', fontsize=10)
ax_a.set_title('A. Feature Importance', fontsize=11.5, fontweight='bold', loc='left')
ax_a.grid(axis='x', ls=':', alpha=0.3)

# Panel B: logP vs pDC50 (key SAR insight)
ax_b = fig.add_axes([0.39, 0.56, 0.28, 0.40])
for t in ['High', 'Medium', 'Low']:
    mask = ~np.isnan(logP) & (tier == t)
    ax_b.scatter(logP[mask], y[mask], s=18, c=tier_colors[t], marker=tier_markers[t],
                alpha=0.5, edgecolors='none', rasterized=True)
from scipy.stats import spearmanr
r_logp, _ = spearmanr(logP[~np.isnan(logP)], y[~np.isnan(logP)])
ax_b.text(0.95, 0.05, f'ρ = {r_logp:+.2f}', transform=ax_b.transAxes,
         ha='right', fontsize=10, color='#555')
ax_b.set_xlabel('cLogP', fontsize=10); ax_b.set_ylabel('pDC50', fontsize=10)
ax_b.set_title('B. Lipophilicity vs Potency', fontsize=11.5, fontweight='bold', loc='left')
ax_b.grid(ls=':', alpha=0.3)

# Panel C: Linker flexibility vs pDC50
ax_c = fig.add_axes([0.39, 0.08, 0.28, 0.40])
valid_lf = ~np.isnan(linker_flex)
for t in ['High', 'Medium', 'Low']:
    mask = valid_lf & (tier == t)
    ax_c.scatter(linker_flex[mask], y[mask], s=18, c=tier_colors[t], marker=tier_markers[t],
                alpha=0.5, edgecolors='none', rasterized=True)
r_flex, _ = spearmanr(linker_flex[valid_lf], y[valid_lf])
ax_c.text(0.95, 0.05, f'ρ = {r_flex:+.2f}', transform=ax_c.transAxes,
         ha='right', fontsize=10, color='#555')
ax_c.set_xlabel('Linker flexibility (rotatable bond density)', fontsize=10)
ax_c.set_ylabel('pDC50', fontsize=10)
ax_c.set_title('C. Linker Rigidity vs Potency', fontsize=11.5, fontweight='bold', loc='left')
ax_c.grid(ls=':', alpha=0.3)

# Panel D: Potency tier by key properties (box plots)
ax_d = fig.add_axes([0.73, 0.56, 0.25, 0.40])
box_data = []
positions = []
for ti, t in enumerate(['High', 'Medium', 'Low']):
    mask = tier == t
    for pi, (prop, prop_name) in enumerate([(logP, 'cLogP'),
                                             (linker_flex, 'Linker\nflexibility'),
                                             (hbd, 'HBD')]):
        vals = prop[mask & ~np.isnan(prop)]
        if len(vals) > 0:
            pos = ti * 4 + pi
            bp = ax_d.boxplot(vals, positions=[pos], widths=0.7,
                            patch_artist=True, manage_ticks=False,
                            boxprops=dict(facecolor=tier_colors[t], alpha=0.5,
                                         edgecolor='#444', linewidth=0.8),
                            whiskerprops=dict(color='#666', linewidth=0.8),
                            capprops=dict(color='#666', linewidth=0.8),
                            medianprops=dict(color='#222', linewidth=1.2),
                            flierprops=dict(marker='o', markersize=3, alpha=0.3))
            box_data.append((pos, prop_name, t))

ax_d.set_xticks([1, 5, 9])
ax_d.set_xticklabels(['High\n≤10 nM', 'Medium\n10–100 nM', 'Low\n>100 nM'], fontsize=8.5)
ax_d.set_ylabel('Property value', fontsize=10)
ax_d.set_title('D. Tier-Separated Properties', fontsize=11.5, fontweight='bold', loc='left')
ax_d.grid(axis='y', ls=':', alpha=0.3)
# Annotation for each box
for ti, t in enumerate(['High', 'Medium', 'Low']):
    for pi, name in enumerate(['cLogP', 'Flex', 'HBD']):
        ax_d.text(ti*4+pi, ax_d.get_ylim()[1]*0.98, name, ha='center', fontsize=6.5,
                 color='#333', fontweight='bold')

# Panel E: Training vs Test scatter (model quality)
ax_e = fig.add_axes([0.73, 0.08, 0.25, 0.40])
ax_e.scatter(y[trva], et.predict(Xsel[trva]), s=15, c='#9ecae1', alpha=0.35,
            edgecolors='none', label=f'Train (R²={r2_score(y[trva], et.predict(Xsel[trva])):.2f})')
ax_e.scatter(y[te], y_pred, s=28, c='#2e8b57', alpha=0.8, edgecolors='white',
            linewidths=0.3, label=f'Test (R²={r2_test:.3f})')
lim = [y.min()-0.3, y.max()+0.3]
ax_e.plot(lim, lim, '--', color='#d62728', lw=1.5)
ax_e.set_xlim(lim); ax_e.set_ylim(lim)
ax_e.set_xlabel('Experimental pDC50', fontsize=10)
ax_e.set_ylabel('Predicted pDC50', fontsize=10)
ax_e.set_title('E. Model Predictivity', fontsize=11.5, fontweight='bold', loc='left')
ax_e.legend(fontsize=8, loc='upper left', framealpha=0.8)
ax_e.grid(ls=':', alpha=0.3)

# Expert insight text
fig.text(0.01, -0.01,
         'Pharmaceutical insights: ↑cLogP (more lipophilic) and ↑linker flexibility both DECREASE potency.\n'
         'Conformationally rigid, less lipophilic PROTACs are preferred for IRAK4 degradation.',
         ha='left', fontsize=10.5, color='#333', fontstyle='italic',
         bbox=dict(boxstyle='round,pad=0.5', fc='#fff9e6', alpha=0.8, ec='#ddd'))

plt.savefig(f'{OUTDIR}/Fig_ML_pharmaceutical_insight_annotated.svg', facecolor='white', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUTDIR}/Fig_ML_pharmaceutical_insight_annotated.png', facecolor='white', dpi=200, bbox_inches='tight')
plt.close()
print(f"    Saved Fig_ML_pharmaceutical_insight_annotated.svg + .png")

# Clean version (panels A, C, E only — minimal, no labels)
fig2 = plt.figure(figsize=(16, 12))
ax_a2 = fig2.add_axes([0.05, 0.08, 0.28, 0.88])
ax_a2.barh(range(15), top15_imp, height=0.7, color='#bbbbbb', edgecolor='#333', linewidth=0.3)
ax_a2.set_yticks([]); ax_a2.invert_yaxis(); ax_a2.set_xticks([])
for spine in ax_a2.spines.values(): spine.set_visible(False)

ax_c2 = fig2.add_axes([0.39, 0.08, 0.28, 0.88])
for t in ['High', 'Medium', 'Low']:
    mask = valid_lf & (tier == t)
    ax_c2.scatter(linker_flex[mask], y[mask], s=15, c=tier_colors[t], alpha=0.4, edgecolors='none')
ax_c2.set_xticks([]); ax_c2.set_yticks([])
for spine in ax_c2.spines.values(): spine.set_visible(False)

ax_e2 = fig2.add_axes([0.73, 0.08, 0.25, 0.88])
ax_e2.scatter(y[te], y_pred, s=22, c='#2e8b57', alpha=0.7, edgecolors='none')
ax_e2.plot(lim, lim, '--', color='#d62728', lw=1.2)
ax_e2.set_xlim(lim); ax_e2.set_ylim(lim)
ax_e2.set_xticks([]); ax_e2.set_yticks([])
for spine in ax_e2.spines.values(): spine.set_visible(False)

plt.savefig(f'{OUTDIR}/Fig_ML_pharmaceutical_insight_clean.svg', facecolor='white', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUTDIR}/Fig_ML_pharmaceutical_insight_clean.png', facecolor='white', dpi=200, bbox_inches='tight')
plt.close()
print(f"    Saved Fig_ML_pharmaceutical_insight_clean.svg + .png")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("ML PHARMACEUTICAL FIGURES COMPLETE")
print(f"{'='*60}")
print(f"Output directory: {OUTDIR}/")
print(f"  Fig_ML_feature_importance_clean.svg")
print(f"  Fig_ML_feature_importance_annotated.svg")
print(f"  Fig_SAR_physicochemical_annotated.svg")
print(f"  Fig_SAR_physicochemical_clean.svg")
print(f"  Fig_ML_pharmaceutical_insight_annotated.svg")
print(f"  Fig_ML_pharmaceutical_insight_clean.svg")
print(f"  (+ PNG fallbacks for all)")
print(f"\nExtra-Trees test R² = {r2_test:.3f}")
print(f"Top 5 features: {', '.join(top_labels[:5])}")
