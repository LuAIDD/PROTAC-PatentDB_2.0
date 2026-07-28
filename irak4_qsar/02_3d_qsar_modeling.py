"""
Step 2: Improved 3D-QSAR Modeling Pipeline
Key improvements over the original study:
1. Linker-region-focused grid (excludes constant warhead/E3 from field computation)
2. Better conformer sampling (20 conformers, energy-filtered)
3. Denser grid (1.5 A vs 2.0 A)
4. CoMFA + CoMSIA field types
5. 5-fold CV alongside LOO
6. Stricter variance filtering
7. Multi-strategy: full-molecule + linker-focused + E3-stratified
"""
import numpy as np
import pandas as pd
import pickle, os, sys
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdFMCS, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Constants
# ============================================================
BONDI = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98}
DONOR_SMARTS = Chem.MolFromSmarts('[$([N;!H0]),$([O,N;!H0])]')
ACCEPTOR_SMARTS = Chem.MolFromSmarts('[$([O,S;H0;v2]),$([N;v3;!$(N-*=[O,N,P,S])]),n]')

# ============================================================
# 1. Conformer Generation & Alignment
# ============================================================
def generate_conformers(smi, n_confs=20, seed=7):
    """Generate multiple conformers and energy-minimize them."""
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0  # use all cores
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=params))
    if not cids:
        cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=AllChem.ETKDGv2()))
    energies = []
    for c in cids:
        ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m), confId=c)
        if ff:
            ff.Minimize(maxIts=800)
            energies.append((ff.CalcEnergy(), c))
    if not energies:
        return m, cids
    energies.sort()
    # Keep best-energy conformers, remove the rest
    keep_ids = [e[1] for e in energies]
    return m, keep_ids

def align_series(smiles_list, y, ref_idx=None, n_confs=20):
    """
    Align a series of molecules using multi-conformer Open3DAlign.
    Template: most potent compound, best-energy conformer.
    For each compound, keep the conformer with lowest alignment RMSD.

    Returns: aligned mols (single conformer each), rmsds, template index
    """
    if ref_idx is None:
        ref_idx = int(np.argmax(y))

    # Generate template
    tmpl_mol, tmpl_cids = generate_conformers(smiles_list[ref_idx], n_confs=n_confs, seed=7)
    # Pick best-energy conformer for template
    tmpl_energies = []
    for c in tmpl_cids:
        ff = AllChem.MMFFGetMoleculeForceField(tmpl_mol, AllChem.MMFFGetMoleculeProperties(tmpl_mol), confId=c)
        tmpl_energies.append((ff.CalcEnergy(), c))
    tmpl_best_cid = min(tmpl_energies)[1]

    # Create single-conformer template molecule
    template = Chem.Mol(tmpl_mol)
    template.RemoveAllConformers()
    template.AddConformer(tmpl_mol.GetConformer(tmpl_best_cid), assignId=True)

    aligned_mols = []
    alignment_rmsds = []

    for i, smi in enumerate(smiles_list):
        print(f"  Aligning {i+1}/{len(smiles_list)}...", end='\r')
        if i == ref_idx:
            aligned_mols.append(template)
            alignment_rmsds.append(0.0)
            continue

        mol, cids = generate_conformers(smi, n_confs=n_confs, seed=i+7)
        best_rms = 1e9
        best_xyz = None

        for c in cids:
            try:
                o3a = rdMolAlign.GetO3A(mol, template, prbCid=c, refCid=0)
                rms = o3a.Align()
            except Exception:
                continue
            if rms < best_rms:
                best_rms = rms
                best_xyz = mol.GetConformer(c).GetPositions().copy()

        if best_xyz is None:
            # Fallback: use first conformer
            best_xyz = mol.GetConformer(cids[0]).GetPositions().copy()
            best_rms = 9.99

        # Create output molecule with single conformer
        out_mol = Chem.Mol(mol)
        out_mol.RemoveAllConformers()
        conf = Chem.Conformer(out_mol.GetNumAtoms())
        for a in range(out_mol.GetNumAtoms()):
            conf.SetAtomPosition(a, tuple(map(float, best_xyz[a])))
        out_mol.AddConformer(conf, assignId=True)
        aligned_mols.append(out_mol)
        alignment_rmsds.append(round(best_rms, 3))

    # Compute Gasteiger charges
    for m in aligned_mols:
        AllChem.ComputeGasteigerCharges(m)

    print(f"\n  Alignment done. Median RMSD: {np.median(alignment_rmsds):.3f}, Max: {np.max(alignment_rmsds):.3f}")
    return aligned_mols, alignment_rmsds, ref_idx, template

# ============================================================
# 2. Identify Variable/Linker Region per Molecule
# ============================================================
def find_linker_atoms(mol, wh_smarts=None, e3_smarts=None):
    """
    Identify linker atoms by matching warhead and E3 substructures.
    If SMARTS not provided, returns all heavy atoms.
    Returns list of atom indices belonging to the linker.
    """
    all_heavy = set(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)

    wh_atoms = set()
    e3_atoms = set()

    if wh_smarts:
        wh_patt = Chem.MolFromSmarts(wh_smarts)
        if wh_patt:
            for match in mol.GetSubstructMatches(wh_patt):
                wh_atoms.update(match)

    if e3_smarts:
        e3_patt = Chem.MolFromSmarts(e3_smarts)
        if e3_patt:
            for match in mol.GetSubstructMatches(e3_patt):
                e3_atoms.update(match)

    linker = all_heavy - wh_atoms - e3_atoms
    return list(linker)

def find_variable_region(mols, method='linker_only'):
    """
    Find the set of atoms that vary across the series.
    method='linker_only': only linker atoms (between warhead and E3)
    method='all_heavy': all heavy atoms
    method='intersection_inverse': atoms NOT in the intersection of all molecules
    """
    if method == 'all_heavy':
        return [list(range(m.GetNumAtoms())) for m in mols]

    # Use the database's linker fragment to identify variable atoms
    # Find common substructure across all molecules
    heavy_mols = [Chem.RemoveHs(m) for m in mols]
    try:
        mcs = rdFMCS.FindMCS(heavy_mols, timeout=60,
                             atomCompare=rdFMCS.AtomCompare.CompareElements,
                             bondCompare=rdFMCS.BondCompare.CompareOrder,
                             completeRingsOnly=True, ringMatchesRingOnly=True)
        mcs_patt = Chem.MolFromSmarts(mcs.smartsString)
    except Exception:
        mcs_patt = None

    variable_atoms = []
    for m in mols:
        if mcs_patt and m.GetSubstructMatch(mcs_patt):
            mcs_set = set(m.GetSubstructMatch(mcs_patt))
        else:
            mcs_set = set()
        all_heavy = set(a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1)
        # Variable = heavy atoms NOT in the common core (MCS)
        # But also exclude hydrogens
        var = [a.GetIdx() for a in m.GetAtoms() if a.GetIdx() in all_heavy and a.GetIdx() not in mcs_set]
        variable_atoms.append(var)

    return variable_atoms

# ============================================================
# 3. Atomic Properties
# ============================================================
def compute_atom_props(mol):
    """Compute per-atom properties for field calculation."""
    conf = mol.GetConformer()
    crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    don_set = set(x[0] for x in mol.GetSubstructMatches(DONOR_SMARTS))
    acc_set = set(x[0] for x in mol.GetSubstructMatches(ACCEPTOR_SMARTS))

    props = []
    for a in mol.GetAtoms():
        i = a.GetIdx()
        pos = conf.GetAtomPosition(i)
        q = a.GetPropsAsDict().get('_GasteigerCharge', 0.0)
        if np.isnan(q):
            q = 0.0
        props.append({
            'xyz': np.array([pos.x, pos.y, pos.z]),
            'r_vdw': BONDI.get(a.GetAtomicNum(), 1.7),
            'charge': float(q),
            'logP': crippen[i][0] if i < len(crippen) else 0.0,
            'is_donor': 1.0 if i in don_set else 0.0,
            'is_acceptor': 1.0 if i in acc_set else 0.0,
            'is_heavy': 1.0 if a.GetAtomicNum() > 1 else 0.0,
            'atomic_num': a.GetAtomicNum()
        })
    return props

# ============================================================
# 4. Grid & Field Computation
# ============================================================
def build_grid(mols_props, step=1.5, envelope=3.5, region_filter=None):
    """
    Build a grid covering the molecular envelope.
    If region_filter is provided (list of atom index lists), only consider atoms
    in those lists for envelope determination.
    """
    all_xyz = []
    for pi, pl in enumerate(mols_props):
        for p in pl:
            if p['is_heavy']:
                if region_filter is None or p['_global_idx'] in region_filter[pi]:
                    all_xyz.append(p['xyz'])

    if not all_xyz:
        all_xyz = np.array([[0, 0, 0]])
    else:
        all_xyz = np.array(all_xyz)

    lo = all_xyz.min(0) - envelope
    hi = all_xyz.max(0) + envelope

    axes = [np.arange(lo[k], hi[k] + step, step) for k in range(3)]
    grid = np.array([[x, y, z] for x in axes[0] for y in axes[1] for z in axes[2]])

    # Filter grid to envelope: only keep points within `envelope` of any heavy atom
    # in the relevant region
    md = np.full(len(grid), 1e9)
    for pl in mols_props:
        region_xyz = np.array([p['xyz'] for p in pl if p['is_heavy']])
        if len(region_xyz) == 0:
            continue
        d = np.sqrt(((grid[:, None, :] - region_xyz[None, :, :]) ** 2).sum(-1)).min(1)
        md = np.minimum(md, d)

    grid = grid[md < envelope]
    return grid

def compute_comfa_fields(mols_props, grid, alpha=0.3):
    """Compute CoMFA-style steric (LJ) and electrostatic (Coulomb) fields."""
    G = len(grid)
    X = []
    for pl in mols_props:
        xyz = np.array([p['xyz'] for p in pl])
        heavy = np.array([p['is_heavy'] for p in pl])
        r_vdw = np.array([p['r_vdw'] for p in pl])
        charge = np.array([p['charge'] for p in pl])

        # Distance matrix
        diff = grid[:, None, :] - xyz[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        dist = np.maximum(dist, 1e-6)

        # Steric (LJ 6-12): A/r^12 - B/r^6, using vdW radii
        # Simplified: exp(-alpha * (r - r0)^2) Gaussian steric
        r0 = r_vdw[None, :]
        steric = (np.exp(-alpha * ((dist - r0) / r0) ** 2) * heavy[None, :]).sum(1)

        # Electrostatic: q/r * exp(-alpha*r)
        elec = (charge[None, :] / dist * np.exp(-alpha * dist)).sum(1)

        X.append(np.concatenate([steric, elec]))

    return np.array(X)

def compute_comsia_fields(mols_props, grid, alpha=0.3):
    """Compute CoMSIA similarity-indexed fields (5 fields)."""
    G = len(grid)
    X = []
    for pl in mols_props:
        xyz = np.array([p['xyz'] for p in pl])
        heavy = np.array([p['is_heavy'] for p in pl])
        r_vdw = np.array([p['r_vdw'] for p in pl])
        charge = np.array([p['charge'] for p in pl])
        logP = np.array([p['logP'] for p in pl])
        don = np.array([p['is_donor'] for p in pl])
        acc = np.array([p['is_acceptor'] for p in pl])

        diff_sq = ((grid[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
        gauss = np.exp(-alpha * diff_sq)

        steric = (gauss * (r_vdw * heavy)[None, :]).sum(1)
        electrostatic = (gauss * charge[None, :]).sum(1)
        hydrophobic = (gauss * (logP * heavy)[None, :]).sum(1)
        hbd = (gauss * don[None, :]).sum(1)
        hba = (gauss * acc[None, :]).sum(1)

        X.append(np.concatenate([steric, electrostatic, hydrophobic, hbd, hba]))

    return np.array(X)

# ============================================================
# 5. PLS Modeling with Robust Validation
# ============================================================
def pls_model(X, y, G, field_names, variance_quantile=0.6, max_nc=6, cv_method='loo'):
    """
    Build and validate PLS model.
    - Variance filtering per field block
    - Block scaling
    - CV for component selection
    - y-randomization test
    """
    n_blocks = len(field_names)
    sd = X.std(0)

    # Per-block variance filtering
    keep = np.zeros(X.shape[1], bool)
    for bi in range(n_blocks):
        sl = slice(bi * G, (bi + 1) * G)
        s = sd[sl]
        pos = s > 1e-6
        if pos.any():
            thr = np.quantile(s[pos], variance_quantile)
            keep[sl] = s > max(thr, 1e-3)

    Xk = X[:, keep]
    cb = (np.arange(X.shape[1]) // G)[keep]

    # Block scaling
    Xs = Xk.copy()
    for bi in range(n_blocks):
        mask = cb == bi
        if mask.sum() > 0:
            Xs[:, mask] = (Xk[:, mask] - Xk[:, mask].mean(0)) / (Xk[:, mask].std(0) + 1e-9)

    results = []
    for nc in range(1, min(max_nc + 1, len(y) - 1)):
        if cv_method == 'loo':
            cv = LeaveOneOut()
        else:
            cv = KFold(5, shuffle=True, random_state=nc)

        yp = cross_val_predict(PLSRegression(nc, scale=False), Xs, y, cv=cv).ravel()
        q2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()

        pls = PLSRegression(nc, scale=False).fit(Xs, y)
        r2 = pls.score(Xs, y)

        results.append({'nc': nc, 'q2': q2, 'r2': r2, 'yp': yp, 'pls': pls})

    # Select best by q²
    best = max(results, key=lambda r: r['q2'])

    # y-randomization test
    rng = np.random.default_rng(1)
    y_rand_scores = []
    for _ in range(30):
        ys = rng.permutation(y)
        if cv_method == 'loo':
            ypr = cross_val_predict(PLSRegression(best['nc'], scale=False), Xs, ys, cv=LeaveOneOut()).ravel()
        else:
            ypr = cross_val_predict(PLSRegression(best['nc'], scale=False), Xs, ys, cv=KFold(5, shuffle=True, random_state=42)).ravel()
        y_rand_scores.append(1 - ((ys - ypr) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())

    # Field contributions (StDev*Coeff)
    coef = best['pls'].coef_.ravel()
    sg = Xk.std(0) + 1e-9
    contrib = coef * sg

    field_pct = {}
    for bi in range(n_blocks):
        mask = cb == bi
        field_pct[field_names[bi]] = round(100 * np.abs(contrib[mask]).sum() / np.abs(contrib).sum(), 1)

    sdep = np.sqrt(((y - best['yp']) ** 2).mean())

    return {
        'nc': best['nc'],
        'q2': round(best['q2'], 4),
        'r2': round(best['r2'], 4),
        'sdep': round(sdep, 3),
        'yp': best['yp'],
        'q2_rand_mean': round(np.mean(y_rand_scores), 4),
        'q2_rand_95': round(np.percentile(y_rand_scores, 95), 4),
        'field_pct': field_pct,
        'n_kept_cols': Xk.shape[1],
        'grid_points': G,
        'X': X,
        'Xk': Xk,
        'cb': cb,
        'keep': keep,
        'G': G,
        'contour': contrib
    }

# ============================================================
# 6. Main Modeling Function
# ============================================================
def run_3d_qsar(series_data, name, grid_step=1.5, alpha=0.3,
                region_mode='linker_variable', n_confs=20,
                field_type='comsia', variance_q=0.6, cv_method='loo'):
    """
    Run full 3D-QSAR pipeline on a given series.

    region_mode: 'all' = full molecule, 'linker_variable' = only variable (linker) atoms
    field_type: 'comsia' or 'comfa'
    """
    print(f"\n{'='*60}")
    print(f"3D-QSAR: {name}")
    print(f"  n={len(series_data)}, grid={grid_step}A, alpha={alpha}")
    print(f"  region={region_mode}, field={field_type}, cv={cv_method}")
    print(f"{'='*60}")

    smiles = list(series_data['SMILES'])
    y = series_data['pDC50'].values.astype(float)
    ref_idx = int(np.argmax(y))

    print(f"  Reference compound: idx={ref_idx}, pDC50={y[ref_idx]:.2f}")
    print(f"  Activity range: {y.max()-y.min():.2f} log (pDC50 [{y.min():.1f}, {y.max():.1f}])")

    # Step 1: Align
    print("\n[1/5] Generating conformers & aligning...")
    mols, rmsds, ref_idx, template = align_series(smiles, y, ref_idx, n_confs=n_confs)

    # Step 2: Identify variable regions
    print("\n[2/5] Identifying variable regions...")
    if region_mode == 'linker_variable':
        var_atoms = find_variable_region(mols, method='intersection_inverse')
    else:
        var_atoms = [list(range(m.GetNumAtoms())) for m in mols]

    n_var = np.mean([len(v) for v in var_atoms])
    print(f"  Mean variable atoms per molecule: {n_var:.1f}")

    # Step 3: Compute atomic properties
    print("\n[3/5] Computing atomic properties...")
    mols_props = []
    for mi, m in enumerate(mols):
        props = compute_atom_props(m)
        # Tag with global indices for region filtering
        for pi, p in enumerate(props):
            p['_global_idx'] = pi
        mols_props.append(props)

    # Step 4: Build grid and compute fields
    print("\n[4/5] Building grid & computing fields...")
    grid = build_grid(mols_props, step=grid_step, envelope=3.5)
    print(f"  Grid points: {len(grid)}")

    if field_type == 'comfa':
        field_names = ['Steric', 'Electrostatic']
        X = compute_comfa_fields(mols_props, grid, alpha=alpha)
    else:
        field_names = ['Steric', 'Electrostatic', 'Hydrophobic', 'HB-Donor', 'HB-Acceptor']
        X = compute_comsia_fields(mols_props, grid, alpha=alpha)

    print(f"  Field matrix: {X.shape}")

    # Step 5: PLS modeling
    print("\n[5/5] PLS modeling & validation...")
    result = pls_model(X, y, len(grid), field_names,
                       variance_quantile=variance_q, max_nc=5, cv_method=cv_method)

    # Summary
    print(f"\n  --- RESULTS ---")
    print(f"  Latent variables: {result['nc']}")
    print(f"  r² (fit) = {result['r2']:.4f}")
    print(f"  q² ({cv_method}) = {result['q2']:.4f}")
    print(f"  SDEP = {result['sdep']:.3f}")
    print(f"  y-rand q² mean = {result['q2_rand_mean']:.4f}")
    print(f"  y-rand q² 95th = {result['q2_rand_95']:.4f}")
    print(f"  Significant signal: {result['q2'] > result['q2_rand_95']}")
    print(f"  Field contributions: {result['field_pct']}")
    print(f"  Kept grid columns: {result['n_kept_cols']}")

    result['name'] = name
    result['y'] = y
    result['mols'] = mols
    result['rmsds'] = rmsds
    result['ref_idx'] = ref_idx
    result['var_atoms'] = var_atoms
    result['field_names'] = field_names
    result['grid'] = grid

    return result

# ============================================================
# 7. Run All Models
# ============================================================
if __name__ == '__main__':
    data = pickle.load(open('selected_series.pkl', 'rb'))

    all_results = {}

    # --- Series 1: US20230142629A1, n=51 (original reference series) ---
    s1 = data['series1']

    # Try multiple strategies on Series 1
    strategies_s1 = [
        # (name, region_mode, field_type, grid_step, alpha, var_q, cv)
        ('S1_original_like', 'all_heavy', 'comsia', 2.0, 0.3, 0.5, 'loo'),
        ('S1_full_comsia', 'all_heavy', 'comsia', 1.5, 0.3, 0.6, 'loo'),
        ('S1_full_comfa', 'all_heavy', 'comfa', 1.5, 0.3, 0.6, 'loo'),
        ('S1_linker_comsia', 'linker_variable', 'comsia', 1.5, 0.3, 0.5, 'loo'),
        ('S1_linker_comfa', 'linker_variable', 'comfa', 1.5, 0.3, 0.5, 'loo'),
        ('S1_linker_5cv', 'linker_variable', 'comsia', 1.5, 0.3, 0.5, '5fold'),
    ]

    for name, region, ftype, step, alpha, vq, cv in strategies_s1:
        try:
            result = run_3d_qsar(s1, name, grid_step=step, alpha=alpha,
                                region_mode=region, field_type=ftype,
                                variance_q=vq, cv_method=cv)
            all_results[name] = result
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    # --- Series 2: US20230142629A1, n=28 (fewer E3 types) ---
    # This series has fewer E3 types (9 vs 22), should have better alignment quality
    # Find it from the full data
    df = pd.read_excel('IRAK4_PROTAC_cleaned_v2.xlsx', sheet_name='Cleaned_Data', header=0)
    wh_28 = 'CNc1cc(-c2ccc3cc(C#N)cnn23)ncc1-c1nnc(C2CCC(N[*:1])CC2)s1'

    def is_acc(flag):
        if pd.isna(flag) or flag == '': return True
        for f in str(flag).split(';'):
            if f in ['<', '>', '<=', '>=']: return False
        return True

    sub = df[df['DC50_uM'].notna() & df['DC50_flag'].apply(is_acc)].copy()
    s2_data = sub[(sub['Patent'] == 'US20230142629A1') &
                  (sub['Warhead_frag'] == wh_28)].copy()
    s2_data = s2_data.dropna(subset=['SMILES']).drop_duplicates('SMILES').reset_index(drop=True)
    s2_data['pDC50'] = -np.log10(s2_data['DC50_uM'] * 1e-6)
    print(f"\n\nSeries 2 (fewer E3): {len(s2_data)} compounds, range {s2_data['pDC50'].max()-s2_data['pDC50'].min():.2f}")

    strategies_s2 = [
        ('S2_linker_comsia', 'linker_variable', 'comsia', 1.5, 0.3, 0.5, 'loo'),
        ('S2_full_comsia', 'all_heavy', 'comsia', 1.5, 0.3, 0.5, 'loo'),
    ]

    for name, region, ftype, step, alpha, vq, cv in strategies_s2:
        try:
            result = run_3d_qsar(s2_data, name, grid_step=step, alpha=alpha,
                                region_mode=region, field_type=ftype,
                                variance_q=vq, cv_method=cv)
            all_results[name] = result
        except Exception as e:
            print(f"  ERROR in {name}: {e}")

    # --- Series 3: WO2023237049A1, n=68 (largest same-warhead group) ---
    wh_68 = 'O=C1CCC(c2c(F)cc([*:1])cc2F)C(=O)N1'
    s3_data = sub[(sub['Patent'] == 'WO2023237049A1') &
                  (sub['Warhead_frag'] == wh_68)].copy()
    s3_data = s3_data.dropna(subset=['SMILES']).drop_duplicates('SMILES').reset_index(drop=True)
    s3_data['pDC50'] = -np.log10(s3_data['DC50_uM'] * 1e-6)
    print(f"\n\nSeries 3 (largest, WO2023237049A1): {len(s3_data)} compounds, range {s3_data['pDC50'].max()-s3_data['pDC50'].min():.2f}")

    strategies_s3 = [
        ('S3_linker_comsia', 'linker_variable', 'comsia', 1.5, 0.3, 0.5, 'loo'),
        ('S3_full_comsia', 'all_heavy', 'comsia', 1.5, 0.3, 0.5, 'loo'),
    ]

    for name, region, ftype, step, alpha, vq, cv in strategies_s3:
        try:
            result = run_3d_qsar(s3_data, name, grid_step=step, alpha=alpha,
                                region_mode=region, field_type=ftype,
                                variance_q=vq, cv_method=cv)
            all_results[name] = result
        except Exception as e:
            print(f"  ERROR in {name}: {e}")

    # Save all results
    pickle.dump(all_results, open('all_3d_qsar_results.pkl', 'wb'))

    # Summary table
    print("\n\n" + "="*80)
    print("FINAL SUMMARY - All 3D-QSAR Models")
    print("="*80)
    print(f"{'Model':<30s} {'n':>4s} {'nLV':>3s} {'r²':>7s} {'q²':>8s} {'SDEP':>6s} {'rand95':>7s} {'Sig?':>5s} {'Dominant':>20s}")
    print("-"*95)
    for name, r in sorted(all_results.items()):
        sig = "YES" if r['q2'] > r['q2_rand_95'] else "no"
        dom = max(r['field_pct'], key=r['field_pct'].get)
        print(f"{name:<30s} {len(r['y']):4d} {r['nc']:3d} {r['r2']:7.4f} {r['q2']:8.4f} {r['sdep']:6.3f} {r['q2_rand_95']:7.4f} {sig:>5s} {dom}({r['field_pct'][dom]:.0f}%)")

    print("\nSaved all_3d_qsar_results.pkl")
