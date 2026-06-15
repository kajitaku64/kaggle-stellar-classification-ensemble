from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import balanced_accuracy_score


LABELS = ["GALAXY", "QSO", "STAR"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


def optimize_weights(proba: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    def loss(weights: np.ndarray) -> float:
        return -balanced_accuracy_score(y, (proba * weights).argmax(axis=1))

    result = differential_evolution(
        loss,
        [(0.1, 5.0)] * len(LABELS),
        seed=seed,
        popsize=8,
        tol=1e-6,
        polish=True,
        workers=1,
    )
    return result.x, -result.fun


def make_region_ids(alpha: np.ndarray, delta: np.ndarray, alpha_bins: int, delta_bins: int) -> np.ndarray:
    alpha_edges = np.linspace(alpha.min(), alpha.max(), alpha_bins + 1)
    delta_edges = np.linspace(delta.min(), delta.max(), delta_bins + 1)
    alpha_id = np.clip(np.digitize(alpha, alpha_edges[1:-1]), 0, alpha_bins - 1)
    delta_id = np.clip(np.digitize(delta, delta_edges[1:-1]), 0, delta_bins - 1)
    return alpha_id * delta_bins + delta_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--test-proba", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--alpha-bins", type=int, default=6)
    parser.add_argument("--delta-bins", type=int, default=4)
    parser.add_argument("--min-region-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv("input/train.csv", usecols=["id", "alpha", "delta"])
    test = pd.read_csv("input/test.csv", usecols=["id", "alpha", "delta"])
    oof = pd.read_csv(args.oof)
    test_proba = pd.read_csv(args.test_proba)
    base_submission = pd.read_csv(args.base_submission)

    proba_cols = [f"proba_{label}" for label in LABELS]
    oof_proba = oof[proba_cols].to_numpy()
    y = oof["target"].map(LABEL_TO_ID).to_numpy()
    test_proba_values = test_proba[proba_cols].to_numpy()

    global_weights, global_score = optimize_weights(oof_proba, y, args.seed)
    baseline_score = balanced_accuracy_score(y, oof_proba.argmax(axis=1))
    print(f"baseline={baseline_score:.6f}")
    print(f"global_calibrated={global_score:.6f}, weights={np.round(global_weights, 6)}")

    train_regions = make_region_ids(
        train["alpha"].to_numpy(),
        train["delta"].to_numpy(),
        args.alpha_bins,
        args.delta_bins,
    )
    test_regions = make_region_ids(
        test["alpha"].to_numpy(),
        test["delta"].to_numpy(),
        args.alpha_bins,
        args.delta_bins,
    )

    region_weights = {}
    tuned_oof_pred = np.empty(len(oof), dtype=np.int64)
    used_regions = 0
    for region in np.unique(train_regions):
        mask = train_regions == region
        if mask.sum() < args.min_region_rows or len(np.unique(y[mask])) < len(LABELS):
            weights = global_weights
        else:
            weights, _ = optimize_weights(oof_proba[mask], y[mask], args.seed + int(region) + 1)
            used_regions += 1
        region_weights[int(region)] = weights
        tuned_oof_pred[mask] = (oof_proba[mask] * weights).argmax(axis=1)

    regional_score = balanced_accuracy_score(y, tuned_oof_pred)
    print(
        f"regional_calibrated={regional_score:.6f}, "
        f"gain_vs_global={regional_score - global_score:.6f}, used_regions={used_regions}"
    )

    tuned_test_pred = np.empty(len(test), dtype=np.int64)
    for region in np.unique(test_regions):
        mask = test_regions == region
        weights = region_weights.get(int(region), global_weights)
        tuned_test_pred[mask] = (test_proba_values[mask] * weights).argmax(axis=1)

    submission = base_submission.copy()
    submission.iloc[:, 1] = [LABELS[idx] for idx in tuned_test_pred]
    submission.to_csv(args.output, index=False)
    print(f"wrote={args.output}")
    print(submission.iloc[:, 1].value_counts().to_string())


if __name__ == "__main__":
    main()
