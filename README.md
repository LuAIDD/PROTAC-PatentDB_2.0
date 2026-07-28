# PROTAC-PatentDB 2.0 — Code

Custom code accompanying the manuscript *"PROTAC-PatentDB 2.0: A PROTAC Patent
Compound Dataset with Pharmacological and Substructure Annotations"*.

This repository contains exactly the two parts of the manuscript that involve
custom code:

1. **`e3_ligase_assignment/`** — the automated stages (I and III) of the
   four-stage E3 ligase assignment workflow (Methods – "Assignment of E3 ligase
   identity"; stages II and IV were performed manually by the curators).
2. **`irak4_qsar/`** — the IRAK4 QSAR case study (Technical Validation –
   Case study 1).

Everything else reported in the manuscript is either produced by an external
tool (Derwent Innovation, SciFinder, PROTAC-Splitter, ADMETlab 3.0,
Open3DAlign, PROTAC-TS, TabPFN) or performed manually (patent retrieval and
screening, manual review and verification), and therefore has no custom code
to release. The deposited dataset itself is available from Figshare (see the
manuscript, Data availability).

## Repository structure

```
├── e3_ligase_assignment/
│   ├── infer_e3_from_ligand.py   # Stage I: canonical SMILES, Bemis–Murcko scaffold
│   │                             #   and Morgan fingerprint Tanimoto matching against
│   │                             #   the curated reference library, fused with patent
│   │                             #   text keywords; writes annotated sheet + audit trail
│   ├── infer_e3_with_kimi.py     # Stage III: re-resolves low-confidence rows with an
│   │                             #   LLM (KIMI_API_KEY required) plus patent/scaffold
│   │                             #   consistency backfill
│   └── data/                     # small sample inputs (E3 reference library,
│                                 #   patent bibliographic text, PROTAC input sheet)
└── irak4_qsar/
    ├── parse_lib.py              # programmatic conversion of heterogeneous DC50
    │                             #   representations (ranges, censored bounds,
    │                             #   scientific notation, dual read-outs) to µM
    ├── 01_series_selection.py    # congeneric-series selection for 3D-QSAR
    ├── 02_3d_qsar_modeling.py    # CoMFA/CoMSIA modelling with LOO and 5-fold CV
    ├── 03_final_experiments.py   # final field-based experiments reported in the paper
    ├── 04_comfa_contours.py      # CoMFA steric contour maps (Figure 3)
    ├── 05_ml_pharma_figures.py   # descriptor-based Extra-Trees model and SAR
    │                             #   panels (Figure 4)
    └── data/
        └── irak4_protac_sample.csv   # 160-row subset of the IRAK4 table, for
                                      #   syntax checking only
```

## Requirements

Python 3.10+; RDKit, pandas, numpy, scikit-learn, matplotlib, openpyxl,
requests (Stage III only).

## Usage

Stage I can be exercised end-to-end on the bundled sample data without any
external service (copy `data/prota_input_sample.xlsx` into a working directory
as `PROTAC-PatentDB 2.0-1.xlsx` first):

```bash
python e3_ligase_assignment/infer_e3_from_ligand.py \
    --e3-dir        <working_dir>/ \
    --reference-csv e3_ligase_assignment/data/e3_reference_sample.csv \
    --patent-info   e3_ligase_assignment/data/patent_info_sample.xlsx
```

Stage III expects the Stage I audit output and reads the Moonshot API key from
the `KIMI_API_KEY` environment variable.

The QSAR scripts expect `IRAK4_PROTAC_cleaned_v2.xlsx`, the cleaned IRAK4
activity table derived from the Figshare deposit (DC50 values parsed with
`parse_lib.py`); `data/irak4_protac_sample.csv` is a small subset for
verifying that the scripts run, not for reproducing the manuscript numbers.

## License

MIT (see `LICENSE`). If you use this code, please cite the companion
manuscript in *Scientific Data* (2026).
