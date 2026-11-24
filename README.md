# BRD9 classification QSAR case study

This minimal package reproduces the final BRD9 classification QSAR analysis used in the manuscript.

## Contents

- `code/`: the reproduction script and tested software versions.
- `data/`: 424 source-curated PROTAC records from four patent families.
- `results/`: frozen metrics, predictions and the final ROC panel.

## Endpoint and model

All records use comparable 6 h SYO-1 NanoLuc BRD9 reporter assays. Patent-reported activity bins were classified as `<10 nM` (positive) or `>=10 nM` (negative). The model is a fixed random forest using 2,048-dimensional radius-3 Morgan count fingerprints with chirality. Exact Bemis–Murcko scaffold groups are kept separate during validation.

## Reproduction

```bash
python code/run_brd9_qsar.py
```

The command writes metrics, predictions, the fitted model and the ROC plot to `output/`. Expected headline results are AUC 0.850 for five-fold scaffold cross-validation on the 338-compound development set and AUC 0.851, MCC 0.529 and balanced accuracy 0.743 for the 86-compound scaffold-separated test set.

This is a retrospective case study within the represented chemical and assay space; it is not evidence of transfer to unseen patent families or laboratories.
