"""Reproduce the final BRD9 classification QSAR case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold


SEED = 20261111


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=4,
    )


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict:
    labels = scores >= 0.5
    return {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "AUROC": float(roc_auc_score(y_true, scores)),
        "MCC": float(matthews_corrcoef(y_true, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, labels)),
        "AUPRC": float(average_precision_score(y_true, scores)),
        "confusion_matrix_TN_FP_FN_TP": confusion_matrix(
            y_true, labels, labels=[0, 1]
        ).ravel().tolist(),
    }


def fingerprints(smiles: pd.Series) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=3, fpSize=2048, includeChirality=True
    )
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    if any(molecule is None for molecule in molecules):
        raise ValueError("The input contains an invalid SMILES string.")
    return np.asarray(
        [generator.GetCountFingerprintAsNumPy(molecule) for molecule in molecules]
    )


def plot_roc(frame: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axis = plt.subplots(figsize=(3.54, 2.85))
    for split, label, color, style in (
        ("development", "CV", "#666666", "-"),
        ("test", "Test", "#0072B2", "--"),
    ):
        subset = frame[frame["split"] == split]
        fpr, tpr, _ = roc_curve(subset["y"], subset["probability"])
        auc = roc_auc_score(subset["y"], subset["probability"])
        axis.plot(fpr, tpr, color=color, linestyle=style, linewidth=1.6,
                  label=f"{label}  {auc:.3f}")
    axis.plot([0, 1], [0, 1], color="0.7", linewidth=0.8, linestyle="--")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    axis.legend(title="AUC", loc="lower right", frameon=False,
                fontsize=7, title_fontsize=7)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.19, top=0.94)
    fig.savefig(output_path, dpi=450, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "output"
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(base / "data" / "brd9_qsar_data.csv")
    y = data["class_label"].to_numpy(dtype=int)
    x = fingerprints(data["SMILES"])

    calculated_scaffolds = data["SMILES"].map(
        lambda value: MurckoScaffold.MurckoScaffoldSmiles(
            mol=Chem.MolFromSmiles(value), includeChirality=False
        )
    )
    if not calculated_scaffolds.equals(data["Murcko_scaffold"]):
        raise ValueError("Stored and recalculated scaffold assignments differ.")
    scaffolds = calculated_scaffolds.to_numpy()

    development = np.flatnonzero(data["split"].to_numpy() == "development")
    test = np.flatnonzero(data["split"].to_numpy() == "test")
    if set(scaffolds[development]) & set(scaffolds[test]):
        raise ValueError("Scaffold overlap detected between development and test sets.")

    scores = np.full(len(data), np.nan)
    fold_metrics = []
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (train_local, valid_local) in enumerate(
        cv.split(x[development], y[development], scaffolds[development])
    ):
        train = development[train_local]
        valid = development[valid_local]
        model = make_model()
        model.fit(x[train], y[train])
        scores[valid] = model.predict_proba(x[valid])[:, 1]
        fold_metrics.append({"fold": fold, **evaluate(y[valid], scores[valid])})

    final_model = make_model()
    final_model.fit(x[development], y[development])
    scores[test] = final_model.predict_proba(x[test])[:, 1]

    results = {
        "development_scaffold_CV": evaluate(y[development], scores[development]),
        "heldout_scaffold_test": evaluate(y[test], scores[test]),
        "development_fold_metrics": fold_metrics,
        "scaffold_overlap": 0,
        "full_molecule_overlap": 0,
    }
    predictions = data[
        ["record_id", "patent", "compound_id", "split", "class_label", "Murcko_scaffold"]
    ].copy()
    predictions["y"] = y
    predictions["probability"] = scores
    predictions["predicted_class"] = scores >= 0.5

    (output / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    predictions.to_csv(output / "predictions.csv", index=False)
    joblib.dump(final_model, output / "model.joblib")
    plot_roc(predictions, output / "Figure3a_ROC.png")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
