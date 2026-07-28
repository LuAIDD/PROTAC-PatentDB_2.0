"""
Comprehensive CoMFA contour map generator.
Rebuilds best CoMFA models with full field data saved,
then renders steric contour isosurfaces using matplotlib 3D marching cubes.
Output: clean + annotated SVG/PNG for each model.
"""
import numpy as np
import pandas as pd
import pickle, sys, os
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage import measure
import matplotlib.colors as mc

# ============================================================
# 0. Settings
# ============================================================
OUTDIR = 'figures'
os.makedirs(OUTDIR, exist_ok=True)

BONDI = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98}
DONOR_SM = Chem.MolFromSmarts('[$([N;!H0]),$([O,N;!H0])]')
ACCEPTOR_SM = Chem.MolFromSmarts('[$([O,S;H0;v2]),$([N;v3;!$(N-*=[O,N,P,S])]),n]')

N_CONFS = 6

def is_acc(flag):
    if pd.isna(flag) or flag == '': return True
    for f in str(flag).split(';'):
        if f in ['<', '>', '<=', '>=']: return False
    return True

# Element colors
ECOL = {'C': '#33bb44', 'N': '#2b5bd7', 'O': '#e0322a', 'S': '#e2a400',
        'F': '#59c36a', 'Cl': '#33a02c', 'Br': '#b5651d', 'P': '#e89c00'}

# Contour colors
STERIC_FAV = '#17862f'      # green = bulk favored
STERIC_DIS = '#c39a12'      # gold = bulk disfavored

df = pd.read_excel('IRAK4_PROTAC_cleaned_v2.xlsx', sheet_name='Cleaned_Data', header=0)
sub = df[df['DC50_uM'].notna() & df['DC50_flag'].apply(is_acc)].copy()

# ============================================================
# 1. Conformer generation & alignment (same as optimized pipeline)
# ============================================================
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

def align_series(smiles, y):
    ref_i = int(np.argmax(y))
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
    return mols, ref_i

# ============================================================
# 2. Atom properties
# ============================================================
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
            'is_heavy': 1.0 if a.GetAtomicNum()>1 else 0.0, '_idx': i,
            'symbol': a.GetSymbol(), 'atomic_num': a.GetAtomicNum()})
    return props

# ============================================================
# 3. CoMFA field computation (returns everything needed for contouring)
# ============================================================
def build_comfa_full(mols_props, step=2.0, alpha=0.3):
    """Compute CoMFA fields, return X + grid + full data for contouring."""
    all_xyz = np.vstack([np.array([a['xyz'] for a in pl if a['is_heavy']]) for pl in mols_props])
    lo = all_xyz.min(0)-3.5; hi = all_xyz.max(0)+3.5
    axes = [np.arange(lo[k], hi[k]+step, step) for k in range(3)]
    grid = np.array([[x,yy,z] for x in axes[0] for yy in axes[1] for z in axes[2]])
    md = np.full(len(grid), 1e9)
    for pl in mols_props:
        atom_xyz = np.array([a['xyz'] for a in pl if a['is_heavy']])
        d = np.sqrt(((grid[:,None,:]-atom_xyz[None,:,:])**2).sum(-1)).min(1)
        md = np.minimum(md, d)
    grid = grid[md<3.5]; G=len(grid)

    X = np.zeros((len(mols_props), G*2))
    for pi, pl in enumerate(mols_props):
        xyz = np.array([a['xyz'] for a in pl]); heavy = np.array([a['is_heavy'] for a in pl])
        r_vdw = np.array([a['r_vdw'] for a in pl]); charge = np.array([a['charge'] for a in pl])
        diff_sq = ((grid[:,None,:]-xyz[None,:,:])**2).sum(-1); dist = np.sqrt(np.maximum(diff_sq,1e-10))
        r0 = np.maximum(r_vdw[None,:], 1e-6)
        steric = (np.exp(-alpha*((dist-r0)/r0)**2)*heavy[None,:]).sum(1)
        elec = (charge[None,:]/(dist+1e-6)*np.exp(-alpha*dist)).sum(1)
        X[pi] = np.concatenate([steric, elec])
    return X, grid, G

# ============================================================
# 4. PLS fit with full contribution data saved
# ============================================================
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

def fit_pls_full(X, y, grid_in, G_in, vq=0.5):
    """Fit PLS and compute full StDev*Coeff contour data for both fields."""
    G = G_in; grid = grid_in
    sd = X.std(0); keep = np.zeros(X.shape[1], bool)
    for bi in range(2):
        sl = slice(bi*G, (bi+1)*G); s = sd[sl]; pos = s>1e-6
        if pos.any(): keep[sl] = s > max(np.quantile(s[pos], vq), 1e-3)
    Xk = X[:,keep]; cb = (np.arange(X.shape[1])//G)[keep]
    mu = Xk.mean(0); sg = Xk.std(0)+1e-9; Xs = (Xk-mu)/sg

    nc = 2
    yp = cross_val_predict(PLSRegression(nc,scale=False), Xs,y,cv=LeaveOneOut()).ravel()
    q2 = 1-((y-yp)**2).sum()/((y-y.mean())**2).sum()
    pls = PLSRegression(nc,scale=False).fit(Xs,y); r2 = pls.score(Xs,y)

    coef = pls.coef_.ravel()
    contrib = coef * sg   # StDev * Coeff

    # Map contributions back to full grid per field
    contour = {}
    for bi in range(2):
        full = np.zeros(G)
        block_keep = keep[bi*G:(bi+1)*G]
        if block_keep.sum() > 0:
            full[block_keep] = contrib[cb==bi]
        contour[['Steric','Electrostatic'][bi]] = full

    sdep = np.sqrt(((y-yp)**2).mean())
    fp = {}
    for bi in range(2):
        mask = cb==bi
        fp[['Steric','Electrostatic'][bi]] = round(100*np.abs(contrib[mask]).sum()/max(np.abs(contrib).sum(),1e-9),1)

    print(f"    PLS: nc={nc} r²={r2:.4f} q²={q2:.4f} SDEP={sdep:.3f} fields={fp}")
    return {'contour': contour, 'grid': grid, 'G': G, 'keep': keep, 'cb': cb,
            'q2': q2, 'r2': r2, 'sdep': sdep, 'nc': nc, 'field_pct': fp,
            'yp': yp, 'pls': pls}

# ============================================================
# 5. 3D rendering with matplotlib marching cubes
# ============================================================
def make_volume(grid, contour_field, mols_props, ref_i,
                step=0.55, sigma=1.5, mag_pct=86, pad=3.0):
    """Smooth contour field to a volume via Gaussian interpolation."""
    v = contour_field.copy(); mag = np.abs(v)
    if mag.max()==0: return None, None
    thr = np.percentile(mag[mag>0], mag_pct)
    sel = mag >= thr
    pts = grid[sel]; w = v[sel]

    # Reference molecule positions
    ref_xyz = np.array([a['xyz'] for a in mols_props[ref_i] if a['is_heavy']])
    lo = ref_xyz.min(0)-pad; hi = ref_xyz.max(0)+pad
    ax = [np.arange(lo[k], hi[k]+step, step) for k in range(3)]
    Xg, Yg, Zg = np.meshgrid(ax[0], ax[1], ax[2], indexing='ij')
    vol = np.zeros(Xg.shape)
    if len(pts) == 0: return vol, ax
    inv = 1.0/(2*sigma**2)
    for p, wt in zip(pts, w):
        vol += wt*np.exp(-(((Xg-p[0])**2+(Yg-p[1])**2+(Zg-p[2])**2)*inv))
    return vol, ax

def iso_mesh(vol, ax, level):
    """Marching cubes to extract isosurface."""
    sp = (ax[0][1]-ax[0][0], ax[1][1]-ax[1][0], ax[2][1]-ax[2][0])
    if np.abs(vol).max() < 1e-10: return None
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=level, spacing=sp)
    except Exception:
        return None
    if len(verts) == 0: return None
    verts = verts + np.array([ax[0][0], ax[1][0], ax[2][0]])
    return verts, faces

def face_normals(v, f):
    t = v[f]; n = np.cross(t[:,1]-t[:,0], t[:,2]-t[:,0])
    l = np.linalg.norm(n, axis=1, keepdims=True); l[l==0] = 1
    return n/l

def shade_color(base, nrm, light=np.array([0.35, 0.45, 0.82])):
    light = light / np.linalg.norm(light)
    b = np.array(mc.to_rgb(base))
    return np.clip(b[None]*(0.55 + 0.45*np.clip(nrm@light, 0, 1))[:,None], 0, 1)

def render_3d_figure(mols_props, ref_i, contour_data, grid,
                     field_name, tag, q2_val, r2_val, sdep_val,
                     field_pct_val, n_compounds, out_prefix):
    """
    Render a 3D figure with:
    - Licorice stick representation of reference molecule
    - Steric field isosurfaces (favored green, disfavored gold)
    - Clean version: no text
    - Annotated version: title + legend + stats
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.linewidth': 1.0})

    # Get reference molecule atom data
    pl_ref = mols_props[ref_i]
    # Build bonds from the actual mol
    mol_ref_data = [m for m in [None] if False]  # placeholder, we use the mol directly

    # We need the actual mol for bond info. Let's get it from the mols we saved.
    # For now, build heavy-atom bonds from positions (approximate but works for rendering)
    pos_dict = {a['_idx']: a['xyz'] for a in pl_ref if a['is_heavy']}
    sym_dict = {a['_idx']: a['symbol'] for a in pl_ref if a['is_heavy']}

    # Generate isosurfaces
    # Positive (favored): steric bulk → higher potency
    # Negative (disfavored): steric bulk → lower potency
    vol, ax = make_volume(grid, contour_data[field_name], mols_props, ref_i,
                          step=0.55, sigma=1.5, mag_pct=86, pad=4.0)

    if vol is None or np.abs(vol).max() < 1e-10:
        print(f"    WARNING: {tag} {field_name} contour is empty, skipping")
        return

    pk = vol.max(); tr = -vol.min()
    pos_level = 0.28 * pk if pk > 0 else 0
    neg_level = -0.28 * tr if tr > 0 else 0

    neg_mesh = iso_mesh(vol, ax, neg_level) if neg_level != 0 else None
    pos_mesh = iso_mesh(vol, ax, pos_level) if pos_level != 0 else None

    surfaces = []
    if neg_mesh is not None:
        nv, nf = neg_mesh
        nr = face_normals(nv, nf)
        surfaces.append((nv, nf, shade_color(STERIC_DIS, nr), 0.55, STERIC_DIS, 'Disfavored\n(bulk ↓ potency)'))
    if pos_mesh is not None:
        pv, pf = pos_mesh
        pr = face_normals(pv, pf)
        surfaces.append((pv, pf, shade_color(STERIC_FAV, pr), 0.55, STERIC_FAV, 'Favored\n(bulk ↑ potency)'))

    if not surfaces:
        print(f"    WARNING: {tag} no isosurfaces extracted")
        return

    # We need actual molecular structure from aligned mols
    # Re-build: use the template molecule's conformer for bonds
    # For now we'll render from atom_props directly
    # Let's compute bonds based on covalent radii
    COV = {'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05, 'F': 0.57, 'Cl': 0.99, 'Br': 1.14, 'P': 1.07}
    heavy_atoms = [(a['_idx'], a['xyz'], a['symbol']) for a in pl_ref if a['is_heavy']]
    bonds = []
    for i in range(len(heavy_atoms)):
        for j in range(i+1, len(heavy_atoms)):
            ai, pi, si = heavy_atoms[i]; aj, pj, sj = heavy_atoms[j]
            d = np.linalg.norm(pi - pj)
            r = COV.get(si, 0.7) + COV.get(sj, 0.7)
            if d < r * 1.35:
                bonds.append((pi, pj, si, sj))

    def render_one(out_path, title=None, legend_entries=None, stats_text=None):
        fig = plt.figure(figsize=(14, 5.6))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection='3d')

        # Draw surfaces
        pts_list = []
        for (v, f, colors, alpha, _, _) in surfaces:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            ax.add_collection3d(Poly3DCollection(v[f], alpha=alpha,
                facecolors=colors, edgecolors='none'))
            pts_list.append(v)

        # Draw sticks
        for (pi, pj, si, sj) in bonds:
            mid = (pi+pj)/2
            ax.plot(*zip(pi, mid), color=ECOL.get(si, '#777'), lw=6.2, solid_capstyle='round', zorder=10)
            ax.plot(*zip(mid, pj), color=ECOL.get(sj, '#777'), lw=6.2, solid_capstyle='round', zorder=10)

        # Draw atom spheres
        for ai, p, s in heavy_atoms:
            ax.scatter(*p, s=130, color=ECOL.get(s, '#777'),
                      edgecolors='#111', linewidths=0.5, depthshade=True, zorder=11)

        # Set view
        allp = np.concatenate(pts_list + [np.array([a[1] for a in heavy_atoms])])
        lo = allp.min(0); hi = allp.max(0); m = 1.5
        ax.set_xlim(lo[0]-m, hi[0]+m); ax.set_ylim(lo[1]-m, hi[1]+m); ax.set_zlim(lo[2]-m, hi[2]+m)
        ax.set_box_aspect((hi-lo)+2*m)
        ax.set_axis_off()
        ax.view_init(elev=12, azim=-60)

        if title:
            ax.text2D(0.5, 0.78, title, transform=ax.transAxes, ha='center',
                     fontsize=14.5, fontweight='800', color='#1a1a1a')
        if legend_entries:
            ax.legend(handles=legend_entries, loc='center', bbox_to_anchor=(0.5, 0.35),
                     fontsize=11.5, frameon=False, ncol=2,
                     handletextpad=0.8, columnspacing=1.5)
        if stats_text:
            ax.text2D(0.5, 0.28, stats_text, transform=ax.transAxes, ha='center',
                     fontsize=10, color='#555')

        plt.savefig(f'{out_path}.png', facecolor='white', dpi=200, bbox_inches='tight')
        plt.savefig(f'{out_path}.svg', facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"    Saved {out_path}.svg + .png")

    stat = f'{n_compounds} compounds · {field_name} field ({field_pct_val}%) · r²={r2_val:.2f} · q²_LOO={q2_val:.2f} · SDEP={sdep_val:.2f}'
    legends = []
    for (_, _, _, _, color, label) in surfaces:
        legends.append(Patch(fc=color, label=label))

    title_str = f'CoMFA — IRAK4 PROTAC {tag}'

    # Clean version (no text)
    render_one(f'{out_prefix}_clean', title=None, legend_entries=None, stats_text=None)
    # Annotated version
    render_one(f'{out_prefix}_annotated', title=title_str, legend_entries=legends, stats_text=stat)

# ============================================================
# 6. MAIN: Run for best models
# ============================================================
print("="*70)
print("GENERATING CoMFA CONTOUR MAPS")
print("="*70)

all_contour_data = {}

# Model A: WO2022266258A1 — BEST q²
print("\n[A] WO2022266258A1 (best q²=0.24)")
wh_a = 'O=C(Nc1cc2cn([*:1])nc2cc1OCC1CC1)c1cnn2cccnc12'
s_a = sub[(sub['Patent']=='WO2022266258A1') & (sub['Warhead_frag']==wh_a)].copy()
s_a = s_a.dropna(subset=['SMILES']).drop_duplicates('SMILES').reset_index(drop=True)
s_a['pDC50'] = -np.log10(s_a['DC50_uM']*1e-6)
y_a = s_a['pDC50'].values.astype(float)
print(f"  n={len(y_a)}, range={y_a.max()-y_a.min():.2f}")
sys.stdout.flush()

mols_a, ref_a = align_series(list(s_a['SMILES']), y_a)
props_a = [atom_props(m) for m in mols_a]
X_a, grid_a, G_a = build_comfa_full(props_a, step=2.0, alpha=0.3)
fit_a = fit_pls_full(X_a, y_a, grid_a, G_a, vq=0.5)

render_3d_figure(props_a, ref_a, fit_a['contour'], grid_a,
    'Steric', 'WO2022266258A1', fit_a['q2'], fit_a['r2'], fit_a['sdep'],
    fit_a['field_pct']['Steric'], len(y_a),
    f'{OUTDIR}/CoMFA_WO2022266258A1_steric')

all_contour_data['WO2022266258A1'] = fit_a

# Model B: US20230142629A1 — 1.5 Å fine grid
print("\n[B] US20230142629A1 (1.5 Å fine grid)")
wh_b = 'CC(C)Nc1cc(-c2ccc3cc(C#N)cnn23)ncc1-c1nnc([*:1])s1'
s_b = sub[(sub['Patent']=='US20230142629A1') & (sub['Warhead_frag']==wh_b)].copy()
s_b = s_b.dropna(subset=['SMILES']).drop_duplicates('SMILES').reset_index(drop=True)
s_b['pDC50'] = -np.log10(s_b['DC50_uM']*1e-6)
y_b = s_b['pDC50'].values.astype(float)
print(f"  n={len(y_b)}, range={y_b.max()-y_b.min():.2f}")
sys.stdout.flush()

mols_b, ref_b = align_series(list(s_b['SMILES']), y_b)
props_b = [atom_props(m) for m in mols_b]
X_b, grid_b, G_b = build_comfa_full(props_b, step=1.5, alpha=0.3)
fit_b = fit_pls_full(X_b, y_b, grid_b, G_b, vq=0.5)

render_3d_figure(props_b, ref_b, fit_b['contour'], grid_b,
    'Steric', 'US20230142629A1 (1.5Å)', fit_b['q2'], fit_b['r2'], fit_b['sdep'],
    fit_b['field_pct']['Steric'], len(y_b),
    f'{OUTDIR}/CoMFA_US20230142629A1_steric')

all_contour_data['US20230142629A1_15A'] = fit_b

# Save contour data for future use
pickle.dump(all_contour_data, open(f'{OUTDIR}/comfa_contour_data.pkl', 'wb'))
print(f"\nSaved {OUTDIR}/comfa_contour_data.pkl")
print("CoMFA contour maps done!")
