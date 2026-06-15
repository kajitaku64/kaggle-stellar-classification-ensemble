from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def score_grid(
    proba: np.ndarray,
    y: np.ndarray,
    qso_values: np.ndarray,
    star_values: np.ndarray,
) -> tuple[float, float, float]:
    best_score = -1.0
    best_qso = 1.0
    best_star = 1.0
    for qso_multiplier in qso_values:
        for star_multiplier in star_values:
            multipliers = np.array([1.0, qso_multiplier, star_multiplier])
            pred = (proba * multipliers).argmax(axis=1)
            score = balanced_accuracy_score(y, pred)
            if score > best_score:
                best_score = score
                best_qso = float(qso_multiplier)
                best_star = float(star_multiplier)
    return best_score, best_qso, best_star


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof",
        type=Path,
        default=Path("output/oof_lightgbm_300_estimators_lr_0p04_leaves_31_child_80_with_redshift_raw_bands.csv"),
    )
    args = parser.parse_args()

    oof = pd.read_csv(args.oof)
    labels = ["GALAXY", "QSO", "STAR"]
    proba = oof[[f"proba_{label}" for label in labels]].to_numpy()
    y = oof["target"].map({label: idx for idx, label in enumerate(labels)}).to_numpy()

    baseline_pred = proba.argmax(axis=1)
    baseline_score = balanced_accuracy_score(y, baseline_pred)
    print(f"baseline_oof={baseline_score:.6f}")

    coarse_score, coarse_qso, coarse_star = score_grid(
        proba,
        y,
        np.arange(0.80, 1.201, 0.01),
        np.arange(0.80, 1.201, 0.01),
    )
    print(f"coarse_best={coarse_score:.6f} GALAXY=1,QSO={coarse_qso:.3f},STAR={coarse_star:.3f}")

    fine_score, fine_qso, fine_star = score_grid(
        proba,
        y,
        np.arange(coarse_qso - 0.02, coarse_qso + 0.0201, 0.001),
        np.arange(coarse_star - 0.02, coarse_star + 0.0201, 0.001),
    )
    print(f"fine_best={fine_score:.6f} GALAXY=1,QSO={fine_qso:.3f},STAR={fine_star:.3f}")

    tuned_pred = (proba * np.array([1.0, fine_qso, fine_star])).argmax(axis=1)
    cm = confusion_matrix(y, tuned_pred, labels=np.arange(len(labels)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    print("\nTuned OOF confusion matrix:")
    print(cm_df.to_string())


if __name__ == "__main__":
    main()
