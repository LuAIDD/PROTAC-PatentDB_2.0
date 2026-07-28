"""
Step 1: Systematic series selection for 3D-QSAR
Identifies the most promising congeneric series for successful 3D-QSAR modeling.
Key criteria: common warhead, activity range > 1.5 log, n > 15, structural homogeneity.
"""
import pandas as pd
import numpy as np
import pickle, warnings
warnings.filterwarnings('ignore')

# Load
df = pd.read_excel('IRAK4_PROTAC_cleaned_v2.xlsx', sheet_name='Cleaned_Data', header=0)

# Accept ranges/dual as geometric means, exclude truly censored values
def is_acceptable(flag):
    if pd.isna(flag) or flag == '':
        return True
    for f in str(flag).split(';'):
        if f in ['<', '>', '<=', '>=']:
            return False
    return True

sub = df[df['DC50_uM'].notna() & df['DC50_flag'].apply(is_acceptable)].copy()
sub = sub.dropna(subset=['SMILES']).drop_duplicates('SMILES').reset_index(drop=True)
print(f"Usable compounds: {len(sub)}")

# ============================================================
# Analysis 1: Same-warhead series (>15 compounds, >1.5 log range)
# ============================================================
print("\n" + "="*80)
print("CANDIDATE SERIES for 3D-QSAR")
print("="*80)

candidates = []
for (pat, wh), g in sub.groupby(['Patent', 'Warhead_frag']):
    n = len(g)
    if n < 12:
        continue
    pdc50 = g['pDC50'].values
    log_range = np.log10(g['DC50_uM'].max() / g['DC50_uM'].min())
    n_e3 = g['E3ligand_frag'].nunique()
    linker_flex_median = g['Linker_flexibility'].median()
    candidates.append({
        'patent': pat, 'n': n, 'log_range': round(log_range, 2),
        'n_e3_types': n_e3, 'pDC50_min': round(pdc50.min(), 2),
        'pDC50_max': round(pdc50.max(), 2),
        'pDC50_std': round(pdc50.std(), 2),
        'linker_flex_median': round(linker_flex_median, 3),
        'warhead_smiles': wh
    })

cand_df = pd.DataFrame(candidates).sort_values('n', ascending=False)
print(cand_df.to_string())

# Score each series for 3D-QSAR suitability
# Score = n * log_range / (1 + n_e3_types^0.5)  ~ rewards size & range, penalizes E3 diversity
cand_df['score'] = cand_df['n'] * cand_df['log_range'] / (1 + np.sqrt(cand_df['n_e3_types']))
best = cand_df.sort_values('score', ascending=False)
print("\n=== RANKED by 3D-QSAR suitability score ===")
print(best[['patent','n','log_range','n_e3_types','pDC50_std','score']].to_string())

# ============================================================
# Analysis 2: Truly congeneric subsets (same warhead + same E3)
# ============================================================
print("\n" + "="*80)
print("TRULY CONGENERIC SUBSETS (same warhead + same E3, n>=8)")
print("="*80)
sub['wh_e3'] = sub['Warhead_frag'].astype(str) + '|||' + sub['E3ligand_frag'].astype(str)
congeneric = []
for (pat, whe3), g in sub.groupby(['Patent', 'wh_e3']):
    n = len(g)
    if n < 8:
        continue
    pdc50 = g['pDC50'].values
    log_range = np.log10(g['DC50_uM'].max() / g['DC50_uM'].min())
    congeneric.append({
        'patent': pat, 'n': n, 'log_range': round(log_range, 2),
        'pDC50_min': round(pdc50.min(), 2), 'pDC50_max': round(pdc50.max(), 2),
        'pDC50_std': round(pdc50.std(), 2)
    })

cong_df = pd.DataFrame(congeneric).sort_values('log_range', ascending=False)
print(cong_df.to_string())

# ============================================================
# Select final candidate series
# ============================================================
print("\n" + "="*80)
print("SELECTED SERIES FOR MODELING")
print("="*80)

# Series 1: Best overall (same warhead, large n, wide range)
s1 = best.iloc[0]
print(f"\n[SERIES 1 - Primary] {s1['patent']}, same warhead")
print(f"  n={s1['n']}, log_range={s1['log_range']}, E3 types={s1['n_e3_types']}")

# Series 2: Best truly congeneric
s2 = cong_df.iloc[0]
print(f"\n[SERIES 2 - Truly Congeneric] {s2['patent']}, same warhead+E3")
print(f"  n={s2['n']}, log_range={s2['log_range']}")

# Series 3: Original study's series (for comparison)
s3_match = cand_df[(cand_df['patent'] == 'US20230142629A1') & (cand_df['n'] == 51)]
if len(s3_match) > 0:
    s3 = s3_match.iloc[0]
    print(f"\n[SERIES 3 - Original Reference] {s3['patent']}, same warhead")
    print(f"  n={s3['n']}, log_range={s3['log_range']}, E3 types={s3['n_e3_types']}")

# Save selections
selections = {
    'series1': {'patent': s1['patent'], 'warhead': s1['warhead_smiles'], 'n': int(s1['n'])},
    'series2': {'patent': s2['patent'], 'congeneric': True, 'n': int(s2['n'])},
    'all_candidates': cand_df,
    'all_congeneric': cong_df
}

# Build the actual data subsets
pat1, wh1 = s1['patent'], s1['warhead_smiles']
series1_data = sub[(sub['Patent'] == pat1) & (sub['Warhead_frag'] == wh1)].copy()
series1_data = series1_data.drop_duplicates('SMILES').reset_index(drop=True)
series1_data['pDC50'] = -np.log10(series1_data['DC50_uM'] * 1e-6)
print(f"\nSeries 1 final: {len(series1_data)} compounds, pDC50 [{series1_data['pDC50'].min():.2f}, {series1_data['pDC50'].max():.2f}]")

pat2, whe3_key = s2['patent'], cong_df.iloc[0]
# Get the actual wh_e3 value
wh2, e3_2 = sub[(sub['Patent'] == pat2)].groupby('wh_e3').size().sort_values(ascending=False).index[0].split('|||')
series2_data = sub[(sub['Patent'] == pat2) & (sub['Warhead_frag'] == wh2) & (sub['E3ligand_frag'] == e3_2)].copy()
series2_data = series2_data.drop_duplicates('SMILES').reset_index(drop=True)
series2_data['pDC50'] = -np.log10(series2_data['DC50_uM'] * 1e-6)
print(f"Series 2 final: {len(series2_data)} compounds, pDC50 [{series2_data['pDC50'].min():.2f}, {series2_data['pDC50'].max():.2f}]")

# Also build the original reference series
series3_data = sub[(sub['Patent'] == 'US20230142629A1') &
                   (sub['Warhead_frag'] == s3_match.iloc[0]['warhead_smiles'])].copy()
series3_data = series3_data.drop_duplicates('SMILES').reset_index(drop=True)
series3_data['pDC50'] = -np.log10(series3_data['DC50_uM'] * 1e-6)
print(f"Series 3 (ref) final: {len(series3_data)} compounds, pDC50 [{series3_data['pDC50'].min():.2f}, {series3_data['pDC50'].max():.2f}]")

pickle.dump({
    'series1': series1_data,
    'series2': series2_data,
    'series3': series3_data,
    'meta': selections
}, open('selected_series.pkl', 'wb'))
print("\nSaved selected_series.pkl")
