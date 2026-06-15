from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from train_tree_models import find_file, make_model, make_xy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"


def read_oof(model_name: str, n_estimators: int) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / f"oof_{model_name}_{n_estimators}_estimators.csv")


def read_submission(model_name: str, n_estimators: int) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / f"submission_{model_name}_{n_estimators}_estimators.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgb-weight", type=float, default=0.5)
    parser.add_argument("--cat-weight", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    labels = ["GALAXY", "QSO", "STAR"]
    proba_cols = [f"proba_{label}" for label in labels]

    lgb_oof = read_oof("lightgbm", args.n_estimators)
    cat_oof = read_oof("catboost", args.n_estimators)
    if not lgb_oof["id"].equals(cat_oof["id"]):
        raise ValueError("OOF IDs do not match.")
    if not lgb_oof["target"].equals(cat_oof["target"]):
        raise ValueError("OOF targets do not match.")

    weights_sum = args.lgb_weight + args.cat_weight
    lgb_weight = args.lgb_weight / weights_sum
    cat_weight = args.cat_weight / weights_sum

    oof_proba = (
        lgb_weight * lgb_oof[proba_cols].to_numpy()
        + cat_weight * cat_oof[proba_cols].to_numpy()
    )
    oof_pred = np.array(labels, dtype=object)[oof_proba.argmax(axis=1)]
    score = balanced_accuracy_score(lgb_oof["target"], oof_pred)
    print(
        f"OOF balanced_accuracy={score:.6f} "
        f"(lightgbm={lgb_weight:.3f}, catboost={cat_weight:.3f})",
        flush=True,
    )

    cm = confusion_matrix(lgb_oof["target"], oof_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])
    print("\nOOF confusion matrix:", flush=True)
    print(cm_df.to_string(), flush=True)

    train = pd.read_csv(find_file("train.csv"))
    test = pd.read_csv(find_file("test.csv"))
    sample_submission = pd.read_csv(find_file("sample_submission.csv"))
    target_col = sample_submission.columns[1]
    id_col = sample_submission.columns[0]
    x_train, x_test, y = make_xy(train, test, target_col)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    lgb_model = make_model(
        "lightgbm",
        args.random_state,
        args.n_estimators,
        args.learning_rate,
        num_leaves=31,
        min_child_samples=80,
    )
    cat_model = make_model(
        "catboost",
        args.random_state,
        args.n_estimators,
        args.learning_rate,
        num_leaves=31,
        min_child_samples=80,
    )
    print("Fitting final LightGBM...", flush=True)
    lgb_model.fit(x_train, y_encoded)
    print("Fitting final CatBoost...", flush=True)
    cat_model.fit(x_train, y_encoded, cat_features=["spectral_type", "galaxy_population", "redshift_bin"])

    test_proba = (
        lgb_weight * lgb_model.predict_proba(x_test)
        + cat_weight * cat_model.predict_proba(x_test)
    )
    test_labels = label_encoder.inverse_transform(test_proba.argmax(axis=1))

    submission = sample_submission.copy()
    submission.iloc[:, 0] = test[id_col].values
    submission.iloc[:, 1] = test_labels

    suffix = f"lgb_{lgb_weight:.2f}_cat_{cat_weight:.2f}".replace(".", "p")
    submission_path = OUTPUT_DIR / f"submission_ensemble_{suffix}.csv"
    submission.to_csv(submission_path, index=False)

    oof = pd.DataFrame(oof_proba, columns=proba_cols)
    oof.insert(0, "id", lgb_oof["id"].values)
    oof["target"] = lgb_oof["target"].values
    oof["prediction"] = oof_pred
    oof_path = OUTPUT_DIR / f"oof_ensemble_{suffix}.csv"
    oof.to_csv(oof_path, index=False)

    print(f"\nWrote {submission_path}", flush=True)
    print(f"Wrote {oof_path}", flush=True)


if __name__ == "__main__":
    main()
