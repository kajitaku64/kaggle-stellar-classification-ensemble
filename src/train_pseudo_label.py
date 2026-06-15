from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from train_tree_models import find_file, make_model, make_xy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.995)
    parser.add_argument("--pseudo-weight", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--include-spatial", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(find_file("train.csv"))
    test = pd.read_csv(find_file("test.csv"))
    sample_submission = pd.read_csv(find_file("sample_submission.csv"))
    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]

    x_train, x_test, y = make_xy(
        train,
        test,
        target_col,
        include_redshift=True,
        include_rest_colors=False,
        include_redshift_interactions=False,
        include_raw_bands=True,
        include_mag_stats=False,
        spatial_mode="all" if args.include_spatial else "none",
        include_notebook_features=False,
        include_catboost_ideas=False,
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    labels = list(label_encoder.classes_)

    base_model = make_model(
        "lightgbm",
        args.random_state,
        args.n_estimators,
        args.learning_rate,
        args.num_leaves,
        args.min_child_samples,
    )
    base_model.fit(x_train, y_encoded)
    test_proba = base_model.predict_proba(x_test)
    max_proba = test_proba.max(axis=1)
    pseudo_mask = max_proba >= args.threshold
    pseudo_y = test_proba[pseudo_mask].argmax(axis=1)

    print(f"threshold={args.threshold}", flush=True)
    print(f"pseudo_weight={args.pseudo_weight}", flush=True)
    print(f"include_spatial={args.include_spatial}", flush=True)
    print(f"pseudo_rows={int(pseudo_mask.sum())} / {len(test)}", flush=True)
    pseudo_counts = pd.Series(label_encoder.inverse_transform(pseudo_y)).value_counts().reindex(labels, fill_value=0)
    print("pseudo_counts:", flush=True)
    print(pseudo_counts.to_string(), flush=True)

    x_combined = pd.concat([x_train, x_test.loc[pseudo_mask]], axis=0, ignore_index=True)
    y_combined = np.concatenate([y_encoded, pseudo_y])
    sample_weight = np.concatenate(
        [
            np.ones(len(y_encoded), dtype=np.float32),
            np.full(len(pseudo_y), args.pseudo_weight, dtype=np.float32),
        ]
    )

    final_model = make_model(
        "lightgbm",
        args.random_state + 1000,
        args.n_estimators,
        args.learning_rate,
        args.num_leaves,
        args.min_child_samples,
    )
    final_model.fit(x_combined, y_combined, sample_weight=sample_weight)
    final_pred = final_model.predict_proba(x_test).argmax(axis=1)
    final_labels = label_encoder.inverse_transform(final_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_suffix = f"{args.threshold:g}".replace(".", "p")
    weight_suffix = f"{args.pseudo_weight:g}".replace(".", "p")
    spatial_suffix = "_spatial" if args.include_spatial else ""
    output_path = OUTPUT_DIR / f"submission_lgb{spatial_suffix}_pseudo_t{threshold_suffix}_w{weight_suffix}.csv"

    submission = sample_submission.copy()
    submission.iloc[:, 0] = test[id_col].values
    submission.iloc[:, 1] = final_labels
    submission.to_csv(output_path, index=False)

    print("submission_counts:", flush=True)
    print(submission.iloc[:, 1].value_counts().reindex(labels, fill_value=0).to_string(), flush=True)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
