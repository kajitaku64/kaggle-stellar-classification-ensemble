from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from train_tree_models import find_file, make_model, make_xy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"


def parse_seeds(value: str) -> list[int]:
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,7,2024")
    parser.add_argument("--pseudo-threshold", type=float, default=0.995)
    parser.add_argument("--pseudo-weight", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=80)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
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
        spatial_mode="none",
        include_notebook_features=False,
        include_catboost_ideas=False,
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    labels = list(label_encoder.classes_)
    ensemble_proba = np.zeros((len(x_test), len(labels)), dtype=np.float64)

    print(f"seeds={seeds}", flush=True)
    print(f"pseudo_threshold={args.pseudo_threshold}", flush=True)
    print(f"pseudo_weight={args.pseudo_weight}", flush=True)

    for seed in seeds:
        print(f"\nSeed {seed}: base model", flush=True)
        base_model = make_model(
            "lightgbm",
            seed,
            args.n_estimators,
            args.learning_rate,
            args.num_leaves,
            args.min_child_samples,
        )
        base_model.fit(x_train, y_encoded)
        base_test_proba = base_model.predict_proba(x_test)
        max_proba = base_test_proba.max(axis=1)
        pseudo_mask = max_proba >= args.pseudo_threshold
        pseudo_y = base_test_proba[pseudo_mask].argmax(axis=1)
        pseudo_counts = (
            pd.Series(label_encoder.inverse_transform(pseudo_y))
            .value_counts()
            .reindex(labels, fill_value=0)
        )
        print(f"Seed {seed}: pseudo_rows={int(pseudo_mask.sum())}", flush=True)
        print(pseudo_counts.to_string(), flush=True)

        x_combined = pd.concat([x_train, x_test.loc[pseudo_mask]], axis=0, ignore_index=True)
        y_combined = np.concatenate([y_encoded, pseudo_y])
        sample_weight = np.concatenate(
            [
                np.ones(len(y_encoded), dtype=np.float32),
                np.full(len(pseudo_y), args.pseudo_weight, dtype=np.float32),
            ]
        )

        print(f"Seed {seed}: final pseudo model", flush=True)
        final_model = make_model(
            "lightgbm",
            seed + 1000,
            args.n_estimators,
            args.learning_rate,
            args.num_leaves,
            args.min_child_samples,
        )
        final_model.fit(x_combined, y_combined, sample_weight=sample_weight)
        ensemble_proba += final_model.predict_proba(x_test) / len(seeds)

    final_pred = ensemble_proba.argmax(axis=1)
    final_labels = label_encoder.inverse_transform(final_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_suffix = "_".join(str(seed) for seed in seeds)
    threshold_suffix = f"{args.pseudo_threshold:g}".replace(".", "p")
    weight_suffix = f"{args.pseudo_weight:g}".replace(".", "p")
    output_path = OUTPUT_DIR / f"submission_lgb_seed_ensemble_{seed_suffix}_pseudo_t{threshold_suffix}_w{weight_suffix}.csv"

    submission = sample_submission.copy()
    submission.iloc[:, 0] = test[id_col].values
    submission.iloc[:, 1] = final_labels
    submission.to_csv(output_path, index=False)

    proba_path = OUTPUT_DIR / f"test_proba_lgb_seed_ensemble_{seed_suffix}_pseudo_t{threshold_suffix}_w{weight_suffix}.csv"
    proba_df = pd.DataFrame(ensemble_proba, columns=[f"proba_{label}" for label in labels])
    proba_df.insert(0, id_col, test[id_col].values)
    proba_df.to_csv(proba_path, index=False)

    print("\nsubmission_counts:", flush=True)
    print(submission.iloc[:, 1].value_counts().reindex(labels, fill_value=0).to_string(), flush=True)
    print(f"Wrote {output_path}", flush=True)
    print(f"Wrote {proba_path}", flush=True)


if __name__ == "__main__":
    main()
