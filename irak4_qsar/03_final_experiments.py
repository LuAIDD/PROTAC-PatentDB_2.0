"""
Final optimized 3D-QSAR experiments:
1. CoMFA on WO2023237049A1 (n=68, largest same-warhead, range ~2.0)
2. CoMFA on US20230142629A1 with 1.5Å grid (finer resolution)
3. Truly congeneric subset (same wh+E3, n=9, range 1.67) - cleanest test
4. WO2022266258A1 same-warhead series (n=40)
"""
import numpy as np
import pandas as pd
import pickle, sys
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdFMCS, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
import warnings
warnings.filterwarnings('ignore')

BONDI = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98}
DONOR_SM = Chem.MolFromSmarts('[$([N;!H0]),$([O,N;!H0])]')
ACCEPTOR_SM = Chem.MolFromSmarts('[$([O,S;H0;v2]),$([N;v3;!$(N-*=[O,N,P,S])]),n]')

def is_acc(flag):
    if pd.isna(flag) or flag == '':
        return True
    for f in str(flag).split(';'):
        if f in ['<', '>', '<=', '>=']: return False
    return True

df = pd.read_excel('IRAK4_PROTAC_cleaned_v2.xlsx', sheet_name='Cleaned_Data', header=0)
sub = df[df['DC50_uM'].notna() & df['DC50_flag'].apply(is_acc)].copy()

# ============================================================
# 1. Build series datasets
# ============================================================
N_CONFS = 6  # fast but adequate

def gen_confs(smi, seed, n=N_CONFS):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3(); params.randomSeed = seed; params.numThreads = 1
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n, params=params))
    if not cids:
        cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n, params=AllChem.ETKDGv2()))
    for c in cids:
        ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m), confId=c)
        if ff: ff.Minimize(maxIts=500)
    return m, cids

def align_series(smiles, y, ref_i=None):
    if ref_i is None: ref_i = int(np.argmax(y))
    tmpl_m, tmpl_cids = gen_confs(smiles[ref_i], 7)
    tmpl_en = []
    for c in tmpl_cids:
        ff = AllChem.MMFFGetMoleculeForceField(tmpl_m, AllChem.MMFFGetMoleculeProperties(tmpl_m), confId=c)
        if ff: tmpl_en.append((ff.CalcEnergy(), c))
    T = Chem.Mol(tmpl_m); T.RemoveAllConformers()
    T.AddConformer(tmpl_m.GetConformer(min(tmpl_en)[1]), assignId=True)
    mols = []
    for i, smi in enumerate(smiles):
        if i == ref_i: mols.append(T); continue
        m, cids = gen_confs(smi, i+7); best_rms=1e9; best_xyz=None
        for c in cids:
            try: rms = rdMolAlign.GetO3A(m, T, prbCid=c, refCid=0).Align()
            except: continue
            if rms < best_rms: best_rms=rms; best_xyz=m.GetConformer(c).GetPositions().copy()
        if best_xyz is None: best_xyz=m.GetConformer(cids[0]).GetPositions().copy()
        om = Chem.Mol(m); om.RemoveAllConformers()
        cf = Chem.Conformer(om.GetNumAtoms())
        for a in range(om.GetNumAtoms()): cf.SetAtomPosition(a, tuple(map(float, best_xyz[a])))
        om.AddConformer(cf, assignId=True); mols.append(om)
    for m in mols: AllChem.ComputeGasteigerCharges(m)
    return mols

def atom_props(m):
    conf = m.GetConformer(); crippen = rdMolDescriptors._CalcCrippenContribs(m)
    don_s = set(x[0] for x in m.GetSubstructMatches(DONOR_SM))
    acc_s = set(x[0] for x in m.GetSubstructMatches(ACCEPTOR_SM))
    props = []
    for a in m.GetAtoms():
        i = a.GetIdx(); p = conf.GetAtomPosition(i)
        q = a.GetPropsAsDict().get('_GasteigerCharge', 0.0)
        if np.isnan(q): q = 0.0
        props.append({'xyz': np.array([p.x, p.y, p.z]),
            'r_vdw': BONDI.get(a.GetAtomicNum(), 1.7), 'charge': float(q),
            'logP': crippen[i][0] if i < len(crippen) else 0.0,
            'is_heavy': 1.0 if a.GetAtomicNum()>1 else 0.0, '_idx': i})
    return props

def build_comfa(mols_props, linker_masks=None, step=2.0, alpha=0.3):
    all_xyz = np.vstack([np.array([a['xyz'] for a in pl if a['is_heavy']]) for pl in mols_props])
    lo = all_xyz.min(0)-3.5; hi = all_xyz.max(0)+3.5
    axes = [np.arange(lo[k], hi[k]+step, step) for k in range(3)]
    grid = np.array([[x,yy,z] for x in axes[0] for yy in axes[1] for z in axes[2]])
    md = np.full(len(grid), 1e9)
    for pi, pl in enumerate(mols_props):
        if linker_masks is not None:
            atom_xyz = np.array([a['xyz'] for a in pl if a['is_heavy'] and a['_idx'] in linker_masks[pi]])
            if len(atom_xyz)==0: atom_xyz = np.array([a['xyz'] for a in pl if a['is_heavy']])
        else:
            atom_xyz = np.array([a['xyz'] for a in pl if a['is_heavy']])
        if len(atom_xyz)==0: continue
        d = np.sqrt(((grid[:,None,:]-atom_xyz[None,:,:])**2).sum(-1)).min(1)
        md = np.minimum(md, d)
    grid = grid[md<3.5]; G=len(grid)
    # CoMFA steric+electrostatic
    X = np.zeros((len(mols_props), G*2))
    for pi, pl in enumerate(mols_props):
        xyz = np.array([a['xyz'] for a in pl]); heavy = np.array([a['is_heavy'] for a in pl])
        r_vdw = np.array([a['r_vdw'] for a in pl]); charge = np.array([a['charge'] for a in pl])
        diff_sq = ((grid[:,None,:]-xyz[None,:,:])**2).sum(-1); dist = np.sqrt(np.maximum(diff_sq,1e-10))
        # Gaussian steric
        r0 = np.maximum(r_vdw[None,:], 1e-6)
        steric = (np.exp(-alpha*((dist-r0)/r0)**2)*heavy[None,:]).sum(1)
        # Damped Coulomb
        elec = (charge[None,:]/(dist+1e-6)*np.exp(-alpha*dist)).sum(1)
        X[pi] = np.concatenate([steric, elec])
    return X, grid, G

def run_pls(X, y, G, tag, vq=0.5):
    sd = X.std(0); keep = np.zeros(X.shape[1], bool)
    for bi in range(2):
        sl = slice(bi*G, (bi+1)*G); s = sd[sl]; pos = s>1e-6
        if pos.any(): keep[sl] = s > max(np.quantile(s[pos], vq), 1e-3)
    Xk = X[:,keep]; cb = (np.arange(X.shape[1])//G)[keep]
    mu = Xk.mean(0); sg = Xk.std(0)+1e-9; Xs = (Xk-mu)/sg
    best = {'nc':1,'q2_loo':-9,'q2_5cv':-9}
    for nc in range(1, min(5, len(y)-1)):
        yp_loo = cross_val_predict(PLSRegression(nc,scale=False), Xs,y,cv=LeaveOneOut()).ravel()
        q2_loo = 1-((y-yp_loo)**2).sum()/((y-y.mean())**2).sum()
        yp_5cv = cross_val_predict(PLSRegression(nc,scale=False), Xs,y,cv=KFold(min(5,len(y)-1),shuffle=True,random_state=nc)).ravel()
        q2_5cv = 1-((y-yp_5cv)**2).sum()/((y-y.mean())**2).sum()
        pls = PLSRegression(nc,scale=False).fit(Xs,y); r2 = pls.score(Xs,y)
        if q2_5cv > best['q2_5cv']: best = {'nc':nc,'q2_loo':q2_loo,'q2_5cv':q2_5cv,'r2':r2,'yp_loo':yp_loo,'pls':pls}
    rng=np.random.default_rng(1); yr=[]
    for _ in range(30):
        ys=rng.permutation(y); ypr=cross_val_predict(PLSRegression(best['nc'],scale=False),Xs,ys,cv=LeaveOneOut()).ravel()
        yr.append(1-((ys-ypr)**2).sum()/((ys-ys.mean())**2).sum())
    sig = best['q2_loo'] > np.percentile(yr,95)
    sdep = np.sqrt(((y-best['yp_loo'])**2).mean())
    coef=best['pls'].coef_.ravel(); contrib=coef*sg
    fp={'Steric':round(100*np.abs(contrib[cb==0]).sum()/max(np.abs(contrib).sum(),1e-9),1),
        'Electrostatic':round(100*np.abs(contrib[cb==1]).sum()/max(np.abs(contrib).sum(),1e-9),1)}
    print(f"  [{tag}] n={len(y)} nLV={best['nc']} r²={best['r2']:.4f} q²_LOO={best['q2_loo']:.4f} q²_5CV={best['q2_5cv']:.4f} SDEP={sdep:.3f} grid={G} sig={'YES' if sig else 'no'} fields={fp}")
    sys.stdout.flush()
    return {'tag':tag,'nc':best['nc'],'r2':round(best['r2'],4),'q2_loo':round(best['q2_loo'],4),
            'q2_5cv':round(best['q2_5cv'],4),'sdep':round(sdep,3),'signal':sig,'y_rand_95':round(np.percentile(yr,95),4),
            'field_pct':fp,'y':y,'yp':best['yp_loo'],'n':len(y)}

# ============================================================
# Run experiments
# ============================================================
all_results = {}

for exp_name, series_data, linker_flag, grid_step, alpha_val, vq in [
    # Experiment 1: WO2023237049A1, largest same-warhead group
    ('EXP1_WO2023237049A1_n68', None, 'full', 2.0, 0.3, 0.5),
    # Experiment 2: US20230142629A1 with 1.5Å finer grid
    ('EXP2_US20230142629A1_fine', None, 'full', 1.5, 0.3, 0.5),
    # Experiment 3: US20230142629A1 with alpha=0.4
    ('EXP3_US20230142629A1_a04', None, 'full', 2.0, 0.4, 0.5),
    # Experiment 4: WO2022266258A1 linker-only truly congeneric
    ('EXP4_WO2022266258A1', None, 'full', 2.0, 0.3, 0.5),
]:
    print(f"\n{'='*60}")
    print(f"{exp_name}")
    print(f"{'='*60}")

    # Build data
    if 'WO2023237049A1' in exp_name:
        wh = 'O=C1CCC(c2c(F)cc([*:1])cc2F)C(=O)N1'
        s = sub[(sub['Patent']=='WO2023237049A1') & (sub['Warhead_frag']==wh)].copy()
    elif 'US20230142629A1' in exp_name:
        wh = 'CC(C)Nc1cc(-c2ccc3cc(C#N)cnn23)ncc1-c1nnc([*:1])s1'
        s = sub[(sub['Patent']=='US20230142629A1') & (sub['Warhead_frag']==wh)].copy()
    elif 'WO2022266258A1' in exp_name:
        # For WO2022266258A1, try BOTH same-warhead AND same-wh+e3 (truly congeneric)
        wh = 'O=C(Nc1cc2cn([*:1])nc2cc1OCC1CC1)c1cnn2cccnc12'
        s = sub[(sub['Patent']=='WO2022266258A1') & (sub['Warhead_frag']==wh)].copy()

    s = s.dropna(subset=['SMILES']).drop_duplicates('SMILES').reset_index(drop=True)
    s['pDC50'] = -np.log10(s['DC50_uM']*1e-6)
    smiles = list(s['SMILES']); y = s['pDC50'].values.astype(float)

    print(f"  n={len(y)}, range={y.max()-y.min():.2f}, pDC50 [{y.min():.2f}, {y.max():.2f}]")
    sys.stdout.flush()

    # Align
    mols = align_series(smiles, y)
    all_props = [atom_props(m) for m in mols]

    # Build fields (full molecule)
    X, grid, G = build_comfa(all_props, linker_masks=None, step=grid_step, alpha=alpha_val)
    result = run_pls(X, y, G, exp_name, vq=vq)
    result['series_patent'] = s['Patent'].iloc[0]
    result['grid_step'] = grid_step
    result['alpha'] = alpha_val
    all_results[exp_name] = result

# ============================================================
# Special: Truly congeneric subset (US20230142629A1, same wh+E3)
# ============================================================
print(f"\n{'='*60}")
print("EXP5_TRULY_CONGENERIC")
print(f"{'='*60}")

# Find biggest truly congeneric subset
wh = 'CC(C)Nc1cc(-c2ccc3cc(C#N)cnn23)ncc1-c1nnc([*:1])s1'
s_main = sub[(sub['Patent']=='US20230142629A1') & (sub['Warhead_frag']==wh)].copy()
# Group by E3 ligand, take the biggest group
e3_groups = s_main.groupby('E3ligand_frag')
best_e3 = max([(len(g), e3, g) for e3, g in e3_groups if len(g) >= 5])
n_e3, e3_lig, s_cong = best_e3
s_cong = s_cong.dropna(subset=['SMILES']).drop_duplicates('SMILES').reset_index(drop=True)
s_cong['pDC50'] = -np.log10(s_cong['DC50_uM']*1e-6)
smiles_c = list(s_cong['SMILES']); y_c = s_cong['pDC50'].values.astype(float)

print(f"  n={len(y_c)}, range={y_c.max()-y_c.min():.2f}, pDC50 [{y_c.min():.2f}, {y_c.max():.2f}]")
sys.stdout.flush()

mols_c = align_series(smiles_c, y_c)
all_props_c = [atom_props(m) for m in mols_c]

# Get TRUE linker masks for this series
linker_smiles_c = list(s_cong['Linker_frag'])
true_linker_c = []
for mi, m in enumerate(mols_c):
    lsmi = linker_smiles_c[mi]
    if pd.isna(lsmi):
        true_linker_c.append(set(a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()>1)); continue
    lm = Chem.MolFromSmiles(str(lsmi))
    if lm is None: true_linker_c.append(set(a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()>1)); continue
    lmc = Chem.RWMol(lm)
    for a in sorted([at.GetIdx() for at in lm.GetAtoms() if at.GetAtomicNum()==0], reverse=True): lmc.RemoveAtom(a)
    lc = lmc.GetMol()
    if lc.GetNumAtoms()==0: true_linker_c.append(set(a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()>1)); continue
    match = m.GetSubstructMatch(lc)
    true_linker_c.append(set(match) if match else set(a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()>1))
avg_l = np.mean([len(t) for t in true_linker_c])
print(f"  Mean TRUE linker atoms: {avg_l:.1f}")
sys.stdout.flush()

# Full-molecule CoMFA
X_c, grid_c, G_c = build_comfa(all_props_c, linker_masks=None, step=2.0, alpha=0.3)
r_c = run_pls(X_c, y_c, G_c, 'EXP5a_CONGEN_FULL', vq=0.5)
all_results['EXP5a_CONGEN_FULL'] = r_c

# Linker-focused CoMFA (should work better here since warhead+E3 are CONSTANT!)
X_cl, grid_cl, G_cl = build_comfa(all_props_c, linker_masks=true_linker_c, step=2.0, alpha=0.3)
r_cl = run_pls(X_cl, y_c, G_cl, 'EXP5b_CONGEN_LINKER', vq=0.5)
all_results['EXP5b_CONGEN_LINKER'] = r_cl

# ============================================================
# Summary
# ============================================================
print("\n\n" + "="*90)
print("COMPREHENSIVE 3D-QSAR RESULTS")
print("="*90)
print(f"{'Experiment':<30s} {'n':>4s} {'range':>6s} {'nLV':>3s} {'r²':>7s} {'q²_LOO':>8s} {'q²_5CV':>8s} {'SDEP':>6s} {'Sig':>4s} {'Steric%':>7s}")
print("-"*90)
for name, r in all_results.items():
    print(f"{name:<30s} {r['n']:4d} {r['y'].max()-r['y'].min():6.2f} {r['nc']:3d} {r['r2']:7.4f} {r['q2_loo']:8.4f} {r['q2_5cv']:8.4f} {r['sdep']:6.3f} {'YES' if r['signal'] else 'no':>4s} {r['field_pct']['Steric']:7.1f}")

best = max(all_results.items(), key=lambda x: x[1]['q2_5cv'])
print(f"\nBEST MODEL: {best[0]} — q²_5CV = {best[1]['q2_5cv']:.4f}")

pickle.dump(all_results, open('final_experiments.pkl', 'wb'))
print("Saved final_experiments.pkl")
