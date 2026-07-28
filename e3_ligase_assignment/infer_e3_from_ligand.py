"""
Stage I of the four-stage E3 ligase assignment workflow
(see PROTAC-PatentDB 2.0 manuscript, Methods - "Assignment of E3 ligase identity").

Each candidate E3 ligand is matched against a curated reference library using
canonical SMILES, Bemis-Murcko scaffold and Morgan fingerprint Tanimoto
similarity; the structural evidence is combined with keyword signals extracted
from the corresponding patent text and propagated through a patent-level
majority backfill.

The script writes a new spreadsheet (with an added "E3 ligase" column) and an
audit CSV documenting the source, score and confidence tier of each decision.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.warning")
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

TEXT_RULES = {
	"CRBN": [
		"crbn",
		"cereblon",
		"lenalidomide",
		"pomalidomide",
		"thalidomide",
		"iberdomide",
		"来那度胺",
		"泊马度胺",
		"沙利度胺",
	],
	"VHL": ["vhl", "vh032", "vh101", "vh298", "von hippel", "羟脯氨酸"],
	"IAP": ["iap", "ciap", "xiap", "birc", "bestatin"],
	"MDM2": ["mdm2", "nutlin"],
	"DCAF15": ["dcaf15", "indisulam", "e7820", "d-caf"],
	"RNF114": ["rnf114", "nimbolide"],
	"KEAP1": ["keap1"],
}


def normalize_smiles(text: str) -> str:
	s = str(text).strip()
	s = re.sub(r"\[\*:\d+\]", "[*]", s)
	return s


def normalize_patent_id(text: str) -> str:
	s = str(text).strip().upper().replace(" ", "")
	s = s.replace("-", "")
	return s


def split_patent_tokens(text: str):
	for t in re.split(r"[;,|\s]+", str(text)):
		t = normalize_patent_id(t)
		if t:
			yield t


def mol_from_smiles(smi: str):
	if not smi or smi.lower() == "nan":
		return None
	return Chem.MolFromSmiles(normalize_smiles(smi))


def canonical_smiles(mol) -> str:
	return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""


def scaffold_smiles(mol) -> str:
	if mol is None:
		return ""
	try:
		return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
	except Exception:
		return ""


def fp(mol):
	return MORGAN_GEN.GetFingerprint(mol)


def tier_from_score(score: float, high: float, medium: float) -> str:
	if score >= high:
		return "high"
	if score >= medium:
		return "medium"
	return "low"


def build_reference(ref_csv: Path):
	ref = pd.read_csv(ref_csv)
	rows = []
	for _, r in ref.dropna(subset=["e3 ligand smiles", "E3 ligase"]).iterrows():
		m = mol_from_smiles(r["e3 ligand smiles"])
		if m is None:
			continue
		rows.append(
			{
				"e3": str(r["E3 ligase"]).strip(),
				"uniprot": str(r.get("E3 ligase Uniprot", "")).strip(),
				"canon": canonical_smiles(m),
				"scaffold": scaffold_smiles(m),
				"fp": fp(m),
			}
		)
	if not rows:
		raise ValueError("No valid reference rows.")

	canon_map = {}
	scaffold_map = {}
	for key, bucket in [("canon", canon_map), ("scaffold", scaffold_map)]:
		groups = {}
		for x in rows:
			k = x[key]
			if not k:
				continue
			groups.setdefault(k, []).append(x["e3"])
		for k, labels in groups.items():
			c = Counter(labels)
			top, top_n = c.most_common(1)[0]
			purity = top_n / len(labels)
			bucket[k] = {"e3": top, "purity": purity, "n": len(labels)}

	return rows, canon_map, scaffold_map


def build_patent_text_map(info_xlsx: Path):
	info = pd.read_excel(info_xlsx)
	text_cols = [
		"Title (English)",
		"Title",
		"Abstract (English)",
		"Abstract",
		"Claims",
		"First Claim (English)",
		"Abstract - DWPI",
	]
	use_cols = [c for c in text_cols if c in info.columns]
	mp = {}
	for _, r in info.iterrows():
		pid = normalize_patent_id(r.get("Publication Number", ""))
		if not pid:
			continue
		parts = []
		for c in use_cols:
			v = r.get(c, "")
			if pd.notna(v):
				parts.append(str(v))
		mp[pid] = "\n".join(parts).lower()
	return mp


def infer_from_text(text_blob: str):
	if not text_blob:
		return {"e3": "", "score": 0.0, "method": "text_none"}
	hits = Counter()
	for e3, kws in TEXT_RULES.items():
		for kw in kws:
			if kw.lower() in text_blob:
				hits[e3] += 1
	if not hits:
		m = re.search(r"new\s+e3\s+ligase", text_blob)
		if m:
			return {"e3": "novel_E3_candidate", "score": 0.65, "method": "text_novel_candidate"}
		return {"e3": "", "score": 0.0, "method": "text_no_hit"}
	e3, n = hits.most_common(1)[0]
	score = 0.9 if n >= 2 else 0.8
	return {"e3": e3, "score": score, "method": "text_keyword"}


def infer_from_structure(ligand_value: str, ref_rows, canon_map, scaffold_map):
	m = mol_from_smiles(ligand_value)
	if m is None:
		return {"e3": "", "score": 0.0, "method": "structure_parse_failed", "matched": ""}

	can = canonical_smiles(m)
	if can in canon_map:
		info = canon_map[can]
		score = 0.99 if info["purity"] >= 0.9 else 0.9
		return {"e3": info["e3"], "score": score, "method": "structure_exact_canonical", "matched": can}

	scf = scaffold_smiles(m)
	if scf in scaffold_map and scaffold_map[scf]["purity"] >= 0.7:
		info = scaffold_map[scf]
		score = 0.93 if info["purity"] >= 0.9 else 0.85
		return {"e3": info["e3"], "score": score, "method": "structure_scaffold", "matched": scf}

	qfp = fp(m)
	best_score = -1.0
	best = None
	for r in ref_rows:
		s = DataStructs.TanimotoSimilarity(qfp, r["fp"])
		if s > best_score:
			best_score = s
			best = r
	if best is None:
		return {"e3": "", "score": 0.0, "method": "structure_none", "matched": ""}
	return {
		"e3": best["e3"] if best_score >= 0.60 else "",
		"score": float(best_score),
		"method": "structure_tanimoto",
		"matched": best["canon"],
	}


def choose_patent_id(row, text_map):
	for col in ["Patent Number", "Source Patent Number"]:
		if col in row and pd.notna(row[col]):
			for t in split_patent_tokens(row[col]):
				if t in text_map:
					return t
	return ""


def combine_decision(struct_res, text_res, high: float, medium: float):
	se3, ss = struct_res["e3"], struct_res["score"]
	te3, ts = text_res["e3"], text_res["score"]

	if se3 and te3:
		if se3 == te3:
			score = min(1.0, max(ss, ts) + 0.08)
			method = f"{struct_res['method']}+{text_res['method']}"
			e3 = se3
		else:
			if abs(ss - ts) >= 0.15:
				e3, score, method = (se3, ss, struct_res["method"]) if ss > ts else (te3, ts, text_res["method"])
			else:
				e3, score, method = "unknown", max(ss, ts) * 0.7, "conflict"
	elif se3:
		e3, score, method = se3, ss, struct_res["method"]
	elif te3:
		e3, score, method = te3, ts, text_res["method"]
	else:
		e3, score, method = "unknown", 0.0, "no_signal"

	tier = tier_from_score(score, high, medium)
	if e3 == "unknown":
		tier = "low"
	return {"inferred_e3": e3, "inference_score": score, "inference_tier": tier, "method": method}


def process_file(xlsx_path: Path, ref_rows, canon_map, scaffold_map, text_map, high: float, medium: float):
	df = pd.read_excel(xlsx_path)
	if "E3 Ligase Ligand" not in df.columns:
		raise ValueError(f"Missing column E3 Ligase Ligand in {xlsx_path}")

	lig_unique = df["E3 Ligase Ligand"].astype(str).fillna("").unique().tolist()
	struct_cache = {v: infer_from_structure(v, ref_rows, canon_map, scaffold_map) for v in lig_unique}

	pat_cache = {}
	for pid in set(choose_patent_id(r, text_map) for _, r in df.iterrows()):
		pat_cache[pid] = infer_from_text(text_map.get(pid, "")) if pid else {"e3": "", "score": 0.0, "method": "text_none"}

	records = []
	for _, row in df.iterrows():
		lig = str(row.get("E3 Ligase Ligand", ""))
		sr = struct_cache.get(lig, {"e3": "", "score": 0.0, "method": "structure_none", "matched": ""})
		pid = choose_patent_id(row, text_map)
		tr = pat_cache.get(pid, {"e3": "", "score": 0.0, "method": "text_none"})
		cr = combine_decision(sr, tr, high, medium)
		records.append(
			{
				"patent_id_for_text": pid,
				"structure_e3": sr["e3"] or "",
				"structure_score": sr["score"],
				"structure_method": sr["method"],
				"structure_matched": sr.get("matched", ""),
				"text_e3": tr["e3"] or "",
				"text_score": tr["score"],
				"text_method": tr["method"],
				**cr,
			}
		)

	infer_df = pd.DataFrame(records)

	# Patent-level majority backfill:
	# If one patent has a dominant high-confidence E3, backfill its no-signal unknown rows as medium.
	if "patent_id_for_text" in infer_df.columns:
		for pid, g in infer_df.groupby("patent_id_for_text"):
			if not pid:
				continue
			confident = g[(g["inferred_e3"] != "unknown") & (g["inference_tier"] == "high")]
			if len(confident) < 5:
				continue
			top = confident["inferred_e3"].value_counts()
			top_e3 = top.index[0]
			top_ratio = top.iloc[0] / len(confident)
			if top_ratio < 0.80:
				continue
			idx = g[(g["inferred_e3"] == "unknown") & (g["method"] == "no_signal")].index
			if len(idx) == 0:
				continue
			infer_df.loc[idx, "inferred_e3"] = top_e3
			infer_df.loc[idx, "inference_score"] = 0.75
			infer_df.loc[idx, "inference_tier"] = "medium"
			infer_df.loc[idx, "method"] = "patent_majority_backfill"
	out_df = df.copy()
	out_df["E3 ligase"] = infer_df["inferred_e3"]

	stem = xlsx_path.stem
	suffix = stem.split()[-1]
	out_xlsx = xlsx_path.with_name(f"{stem}_with_E3.xlsx")
	out_audit = xlsx_path.with_name(f"E3_inference_audit_{suffix}.csv")

	out_df.to_excel(out_xlsx, index=False)
	audit = pd.concat([df[["E3 Ligase Ligand"]], infer_df], axis=1)
	audit.to_csv(out_audit, index=False)
	return out_xlsx, out_audit, audit["inference_tier"].value_counts().to_dict()


def main():
	parser = argparse.ArgumentParser(description="Infer E3 ligase using scaffold-first + patent text assistance.")
	parser.add_argument(
		"--e3-dir",
		required=True,
		help="Directory containing the input PROTAC-PatentDB 2.0 spreadsheets (PROTAC-PatentDB 2.0-1.xlsx, PROTAC-PatentDB 2.0-2.xlsx).",
	)
	parser.add_argument(
		"--reference-csv",
		required=True,
		help="Curated E3 ligand reference CSV with columns 'e3 ligand smiles' and 'E3 ligase' (plus optional 'E3 ligase Uniprot').",
	)
	parser.add_argument(
		"--patent-info",
		required=True,
		help="Patent metadata xlsx with 'Publication Number' and text columns (Title, Abstract, Claims, ...).",
	)
	parser.add_argument("--high-threshold", type=float, default=0.85)
	parser.add_argument("--medium-threshold", type=float, default=0.65)
	args = parser.parse_args()

	e3_dir = Path(args.e3_dir)
	ref_rows, canon_map, scaffold_map = build_reference(Path(args.reference_csv))
	text_map = build_patent_text_map(Path(args.patent_info))

	targets = [
		e3_dir / "PROTAC-PatentDB 2.0-1.xlsx",
		e3_dir / "PROTAC-PatentDB 2.0-2.xlsx",
	]
	print(f"Reference entries: {len(ref_rows)} | canonical keys: {len(canon_map)} | scaffold keys: {len(scaffold_map)}")
	print(f"Patent text entries: {len(text_map)}")

	for t in targets:
		if not t.exists():
			print(f"Skipping (not found): {t}")
			continue
		out_xlsx, out_audit, summary = process_file(
			t,
			ref_rows,
			canon_map,
			scaffold_map,
			text_map,
			args.high_threshold,
			args.medium_threshold,
		)
		print(f"Processed: {t.name}")
		print(f"  output: {out_xlsx}")
		print(f"  audit : {out_audit}")
		print(f"  tiers : {summary}")


if __name__ == "__main__":
	main()
