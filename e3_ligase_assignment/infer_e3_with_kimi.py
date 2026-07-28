#!/usr/bin/env python3
"""
Stage III of the four-stage E3 ligase assignment workflow
(see PROTAC-PatentDB 2.0 manuscript, Methods - "Assignment of E3 ligase identity").

For every row whose Stage I annotation fell into the "low" confidence tier,
the script re-queries a large language model (default: moonshot-v1-8k) with
both the E3 ligand SMILES and the relevant patent text. A patent + scaffold
consistency backfill propagates confident predictions to remaining unknown
rows that share the same patent and Bemis-Murcko scaffold.

The Moonshot API key must be supplied through the environment variable
KIMI_API_KEY. No key material is read from disk or hard-coded.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:
    Chem = None
    MurckoScaffold = None


def normalize_patent_id(text: str) -> str:
    s = str(text).strip().upper().replace(" ", "")
    s = s.replace("-", "")
    return s


def split_patent_tokens(text: str):
    for token in re.split(r"[;,|\s]+", str(text)):
        token = normalize_patent_id(token)
        if token and token.lower() != "nan":
            yield token


def normalize_ligase_name(name: str) -> str:
    t = str(name).strip().upper()
    aliases = {
        "CEREBLON": "CRBN",
        "CRBN": "CRBN",
        "VHL": "VHL",
        "IAP": "IAP",
        "MDM2": "MDM2",
        "DCAF15": "DCAF15",
        "RNF114": "RNF114",
        "KEAP1": "KEAP1",
    }
    return aliases.get(t, str(name).strip())


def parse_confidence(value) -> float:
    if isinstance(value, (int, float)):
        x = float(value)
        return max(0.0, min(1.0, x))
    s = str(value).strip().lower()
    try:
        x = float(s)
        return max(0.0, min(1.0, x))
    except Exception:
        pass
    if s in {"very_high", "very high", "high"}:
        return 0.85
    if s in {"medium", "mid", "moderate"}:
        return 0.65
    if s in {"low", "very_low", "very low"}:
        return 0.35
    return 0.0


def ligand_scaffold(smiles_text: str) -> str:
    if not Chem or not MurckoScaffold:
        return ""
    smi = str(smiles_text).strip()
    smi = re.sub(r"\[\*:\d+\]", "[*]", smi)
    if not smi or smi.lower() == "nan":
        return ""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def build_text_map(info_xlsx: Path):
    info = pd.read_excel(info_xlsx)
    cols = [
        "Publication Number",
        "Title (English)",
        "Title",
        "Abstract (English)",
        "Abstract",
        "Claims",
        "First Claim (English)",
    ]
    use = [c for c in cols if c in info.columns]
    mp = {}
    for _, r in info.iterrows():
        pid = normalize_patent_id(r.get("Publication Number", ""))
        if not pid:
            continue
        parts = []
        for c in use:
            if c == "Publication Number":
                continue
            v = r.get(c, "")
            if pd.notna(v):
                parts.append(str(v))
        mp[pid] = "\n".join(parts)
    return mp


def build_source_patent_columns(source_xlsx: Path):
    source = pd.read_excel(source_xlsx)
    for column in ["Patent Number", "Source Patent Number"]:
        if column not in source.columns:
            source[column] = ""
    return source[["Patent Number", "Source Patent Number"]].copy()


def resolve_patent_id(row, text_map):
    candidates = []
    for column in ["source_patent_number", "patent_number"]:
        value = row.get(column, "")
        if pd.notna(value):
            candidates.extend(list(split_patent_tokens(value)))
    for candidate in candidates:
        if candidate in text_map:
            return candidate
    return candidates[0] if candidates else ""


def call_kimi(api_key: str, base_url: str, model: str, prompt: str, timeout: int = 60):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in PROTAC E3 ligase inference. "
                    "Return only JSON with keys: e3_ligase, confidence, is_novel_e3, rationale."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(url, headers=headers, json=data, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    content = payload["choices"][0]["message"]["content"]

    m = re.search(r"\{.*\}", content, flags=re.S)
    text = m.group(0) if m else content
    try:
        obj = json.loads(text)
    except Exception:
        obj = {
            "e3_ligase": "unknown",
            "confidence": 0.0,
            "is_novel_e3": False,
            "rationale": f"raw_response: {content[:500]}",
        }
    return obj


def call_kimi_batch(api_key: str, base_url: str, model: str, items, timeout: int = 45):
    # items: list[dict] with keys idx, ligand, patent_id, patent_text
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload_items = []
    for it in items:
        payload_items.append(
            {
                "idx": int(it["idx"]),
                "patent_id": str(it.get("patent_id", "")),
                "ligand": str(it.get("ligand", "")),
                "patent_text": str(it.get("patent_text", ""))[:1200],
            }
        )

    user_prompt = (
        "For each item, infer the most likely E3 ligase using both ligand and patent text.\n"
        "Return strict JSON array only.\n"
        "Each result object MUST contain keys: idx, e3_ligase, confidence, is_novel_e3, rationale.\n\n"
        f"Items:\n{json.dumps(payload_items, ensure_ascii=False)}"
    )

    data = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in PROTAC E3 ligase inference. "
                    "Return only JSON array."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(url, headers=headers, json=data, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    m = re.search(r"\[.*\]", content, flags=re.S)
    text = m.group(0) if m else content
    try:
        arr = json.loads(text)
        if isinstance(arr, dict):
            arr = [arr]
        if not isinstance(arr, list):
            arr = []
    except Exception:
        arr = []
    return arr


def build_prompt(ligand: str, patent_id: str, text_blob: str):
    text_short = text_blob[:2500] if text_blob else ""
    return (
        "Task: Infer the most likely E3 ligase for the given PROTAC context.\n"
        "Use BOTH the ligand structure string and patent text context to judge.\n"
        "If evidence suggests a novel/other E3 ligase, set is_novel_e3=true and name it if possible.\n"
        "Output strict JSON only.\n\n"
        f"Patent ID: {patent_id}\n"
        f"E3 Ligase Ligand (SMILES/fragment): {ligand}\n"
        "Patent text (title/abstract/claims):\n"
        f"{text_short}\n"
    )


def main():
    ap = argparse.ArgumentParser(description="Use Kimi to infer E3 for low-confidence rows.")
    ap.add_argument("--audit-csv", required=True)
    ap.add_argument("--source-xlsx", required=True)
    ap.add_argument("--patent-info-xlsx", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--base-url", default="https://api.moonshot.cn/v1")
    ap.add_argument("--model", default="moonshot-v1-8k")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--request-timeout", type=int, default=45)
    ap.add_argument("--request-retries", type=int, default=2)
    ap.add_argument("--sleep-seconds", type=float, default=0.2)
    ap.add_argument("--backfill-min-confidence", type=float, default=0.7)
    ap.add_argument("--backfill-min-count", type=int, default=3)
    ap.add_argument("--backfill-majority-ratio", type=float, default=0.7)
    args = ap.parse_args()

    api_key = os.environ.get("KIMI_API_KEY", "")
    if not api_key:
        raise RuntimeError("KIMI_API_KEY not set in environment.")

    audit = pd.read_csv(args.audit_csv)
    source_patents = build_source_patent_columns(Path(args.source_xlsx))
    text_map = build_text_map(Path(args.patent_info_xlsx))

    if len(source_patents) != len(audit):
        raise ValueError(
            f"Source workbook row count ({len(source_patents)}) does not match audit row count ({len(audit)})."
        )

    low = audit[audit["inference_tier"] == "low"].copy()
    if low.empty:
        pd.DataFrame().to_csv(args.out_csv, index=False)
        print("No low-confidence rows found.")
        return

    if "E3 Ligase Ligand" not in low.columns:
        raise ValueError("Audit CSV missing E3 Ligase Ligand")

    low["source_patent_number"] = source_patents.loc[low.index, "Source Patent Number"].astype(str)
    low["patent_number"] = source_patents.loc[low.index, "Patent Number"].astype(str)
    low["resolved_patent_id"] = low.apply(lambda row: resolve_patent_id(row, text_map), axis=1)
    low["has_text_context"] = low["resolved_patent_id"].isin(set(text_map.keys()))
    low["query_patent_key"] = low.apply(
        lambda row: row["resolved_patent_id"] if row["has_text_context"] else "NO_TEXT",
        axis=1,
    )

    uniq = low[
        [
            "E3 Ligase Ligand",
            "query_patent_key",
            "resolved_patent_id",
            "source_patent_number",
            "patent_number",
        ]
    ].drop_duplicates(subset=["E3 Ligase Ligand", "query_patent_key"])
    if args.limit > 0:
        uniq = uniq.head(args.limit)

    rows = []
    total = len(uniq)
    records = uniq.to_dict("records")

    batch_size = max(1, int(args.batch_size))
    last_print = 0
    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]

        prepared = []
        for j, rec in enumerate(batch):
            ligand = str(rec["E3 Ligase Ligand"])
            key_pid = normalize_patent_id(str(rec["query_patent_key"]))
            pid = normalize_patent_id(str(rec["resolved_patent_id"]))
            src_pid = normalize_patent_id(str(rec["source_patent_number"]))
            patent_pid = normalize_patent_id(str(rec["patent_number"]))
            if key_pid == "NO_TEXT":
                pid = ""
            if not pid:
                pid = src_pid or patent_pid
            text_blob = text_map.get(pid, "")
            if not text_blob:
                text_blob = text_map.get(src_pid, "") or text_map.get(patent_pid, "")
            prepared.append(
                {
                    "idx": j,
                    "ligand": ligand,
                    "query_patent_key": key_pid,
                    "patent_id": pid,
                    "source_patent_number": src_pid,
                    "patent_number": patent_pid,
                    "patent_text": text_blob,
                }
            )

        try:
            answers = None
            last_err = None
            for _ in range(max(1, int(args.request_retries))):
                try:
                    answers = call_kimi_batch(
                        api_key,
                        args.base_url,
                        args.model,
                        prepared,
                        timeout=int(args.request_timeout),
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
            if answers is None:
                raise RuntimeError(str(last_err) if last_err else "batch_call_failed")

            ans_map = {}
            for obj in answers:
                try:
                    ans_map[int(obj.get("idx"))] = obj
                except Exception:
                    continue

            for item in prepared:
                obj = ans_map.get(item["idx"], {})
                rows.append(
                    {
                        "E3 Ligase Ligand": item["ligand"],
                        "query_patent_key": item["query_patent_key"],
                        "patent_id_for_text": item["patent_id"],
                        "source_patent_number": item["source_patent_number"],
                        "patent_number": item["patent_number"],
                        "kimi_e3_ligase": normalize_ligase_name(obj.get("e3_ligase", "unknown")),
                        "kimi_confidence": parse_confidence(obj.get("confidence", 0.0)),
                        "kimi_is_novel_e3": obj.get("is_novel_e3", False),
                        "kimi_rationale": str(obj.get("rationale", ""))[:1000],
                        "status": "ok" if obj else "error: empty_batch_item",
                    }
                )
        except Exception as e:
            for item in prepared:
                rows.append(
                    {
                        "E3 Ligase Ligand": item["ligand"],
                        "query_patent_key": item["query_patent_key"],
                        "patent_id_for_text": item["patent_id"],
                        "source_patent_number": item["source_patent_number"],
                        "patent_number": item["patent_number"],
                        "kimi_e3_ligase": "unknown",
                        "kimi_confidence": 0.0,
                        "kimi_is_novel_e3": False,
                        "kimi_rationale": "",
                        "status": f"error: {e}",
                    }
                )

        done = min(total, start + batch_size)
        if (done - last_print) >= 20 or done == total:
            print(f"Processed {done}/{total}")
            last_print = done
        time.sleep(args.sleep_seconds)

    out = pd.DataFrame(rows)

    # Normalize confidence and prediction strings.
    out["kimi_confidence"] = pd.to_numeric(out["kimi_confidence"], errors="coerce").fillna(0.0)
    out["kimi_e3_ligase"] = out["kimi_e3_ligase"].astype(str).str.strip().map(normalize_ligase_name)
    out["ligand_scaffold"] = out["E3 Ligase Ligand"].map(ligand_scaffold)

    # Final decision before backfill.
    out["final_e3_ligase"] = out["kimi_e3_ligase"]
    out["final_source"] = "kimi_direct"

    # Patent + scaffold consistency backfill:
    # only backfill within same patent and similar core scaffold cluster.
    for (pid, scaffold), g in out.groupby(["patent_id_for_text", "ligand_scaffold"]):
        if not pid or not scaffold:
            continue
        confident = g[
            (g["status"] == "ok")
            & (g["kimi_e3_ligase"].str.lower() != "unknown")
            & (g["kimi_confidence"] >= args.backfill_min_confidence)
        ]
        if len(confident) < args.backfill_min_count:
            continue
        vc = confident["kimi_e3_ligase"].value_counts()
        top_e3 = vc.index[0]
        top_ratio = vc.iloc[0] / len(confident)
        if top_ratio < args.backfill_majority_ratio:
            continue

        idx = g[
            (g["kimi_e3_ligase"].str.lower() == "unknown")
            | (g["status"].str.startswith("error"))
        ].index
        if len(idx) == 0:
            continue
        out.loc[idx, "final_e3_ligase"] = top_e3
        out.loc[idx, "final_source"] = "patent_scaffold_consistency_backfill"

    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
