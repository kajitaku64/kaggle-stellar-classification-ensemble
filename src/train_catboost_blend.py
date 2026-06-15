from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import differential_evolution
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from train_tree_models import add_notebook_categorical_features, add_spatial_density_features, find_file, make_xy
from train_with_original import add_target_encoding, load_original


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
LABELS = ["GALAXY", "QSO", "STAR"]


def build_features(args: argparse.Namespace):
    train = pd.read_csv(find_file("train.csv"))
    test = pd.read_csv(find_file("test.csv"))
    original = load_original(args.original_data)
    target_col = "class"

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train[target_col])
    y_original = label_encoder.transform(original[target_col])

    x_train, x_test, _ = make_xy(
        train,
        test,
        target_col,
        include_redshift=True,
        include_rest_colors=False,
        include_redshift_interactions=False,
        include_raw_bands=True,
        include_mag_stats=False,
        spatial_mode="all",
        include_notebook_features=True,
        include_catboost_ideas=False,
    )
    x_original, _, _ = make_xy(
        original,
        test.iloc[:0].copy(),
        target_col,
        include_redshift=True,
        include_rest_colors=False,
        include_redshift_interactions=False,
        include_raw_bands=True,
        include_mag_stats=False,
        spatial_mode="all",
        include_notebook_features=True,
        include_catboost_ideas=False,
    )

    train_original = pd.concat([train, original], axis=0, ignore_index=True)
    x_train_original = pd.concat([x_train, x_original], axis=0, ignore_index=True)
    x_train_original, x_test = add_notebook_categorical_features(train_original, test, x_train_original, x_test)
    x_train_original, x_test = add_spatial_density_features(train_original, test, x_train_original, x_test)
    x_train = x_train_original.iloc[: len(train)].reset_index(drop=True)
    x_original = x_train_original.iloc[len(train) :].reset_index(drop=True)
    return train, test, x_train, x_original, x_test, y_train, y_original, label_encoder


def make_catboost(args: argparse.Namespace, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=seed,
        auto_class_weights="Balanced",
        task_type="CPU",
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def optimize_blend(lgb_oof: np.ndarray, cat_oof: np.ndarray, y: np.ndarray):
    def loss(params: np.ndarray) -> float:
        alpha = params[0]
        class_weights = params[1:]
        proba = (alpha * lgb_oof + (1.0 - alpha) * cat_oof) * class_weights
        return -balanced_accuracy_score(y, proba.argmax(axis=1))

    result = differential_evolution(
        loss,
        [(0.0, 1.0), (0.1, 5.0), (0.1, 5.0), (0.1, 5.0)],
        seed=63,
        popsize=10,
        tol=1e-7,
        polish=True,
        workers=1,
    )
    return result.x, -result.fun


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-data", type=Path, default=INPUT_DIR / "star_classification.csv")
    parser.add_argument("--original-weight", type=float, default=0.25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=6.0)
    parser.add_argument("--random-state", type=int, default=63)
    parser.add_argument(
        "--lgb-oof",
        type=Path,
        default=OUTPUT_DIR / "oof_lgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_10fold_average.csv",
    )
    parser.add_argument(
        "--lgb-test-proba",
        type=Path,
        default=OUTPUT_DIR / "test_proba_lgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_10fold_average.csv",
    )
    parser.add_argument(
        "--base-submission",
        type=Path,
        default=OUTPUT_DIR / "submission_lgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_10fold_average_calibrated.csv",
    )
    args = parser.parse_args()

    train, test, x_train, x_original, x_test, y_train, y_original, label_encoder = build_features(args)
    labels = list(label_encoder.classes_)
    cat_cols = [col for col in x_train.columns if str(x_train[col].dtype) == "category"]
    print(
        f"catboost folds={args.folds}, iterations={args.iterations}, "
        f"features={x_train.shape[1]}, cat_features={len(cat_cols)}",
        flush=True,
    )

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    oof_proba = np.zeros((len(x_train), len(labels)), dtype=np.float64)
    test_proba = np.zeros((len(x_test), len(labels)), dtype=np.float64)
    fold_scores = []

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train, y_train), start=1):
        x_fit = pd.concat([x_train.iloc[fit_idx], x_original], axis=0, ignore_index=True)
        y_fit = np.concatenate([y_train[fit_idx], y_original])
        x_valid = x_train.iloc[valid_idx].reset_index(drop=True)
        x_test_fold = x_test.reset_index(drop=True)
        x_fit, x_valid, x_test_fold = add_target_encoding(
            x_fit,
            y_fit,
            x_valid,
            x_test_fold,
            len(labels),
            args.folds,
            args.random_state + fold,
        )
        cat_features = [col for col in cat_cols if col in x_fit.columns]
        sample_weight = np.concatenate(
            [
                np.ones(len(fit_idx), dtype=np.float32),
                np.full(len(y_original), args.original_weight, dtype=np.float32),
            ]
        )
        model = make_catboost(args, args.random_state + fold)
        model.fit(x_fit, y_fit, cat_features=cat_features, sample_weight=sample_weight)
        valid_proba = model.predict_proba(x_valid)
        oof_proba[valid_idx] = valid_proba
        test_proba += model.predict_proba(x_test_fold) / args.folds
        score = balanced_accuracy_score(y_train[valid_idx], valid_proba.argmax(axis=1))
        fold_scores.append(score)
        print(f"Fold {fold}: balanced_accuracy={score:.6f}", flush=True)

    oof_score = balanced_accuracy_score(y_train, oof_proba.argmax(axis=1))
    print(f"CatBoost OOF: mean={np.mean(fold_scores):.6f}, oof={oof_score:.6f}", flush=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    suffix = f"catboost_original_w{args.original_weight:g}_te_{args.folds}fold"
    suffix = suffix.replace(".", "p")
    cat_oof_path = OUTPUT_DIR / f"oof_{suffix}.csv"
    cat_test_path = OUTPUT_DIR / f"test_proba_{suffix}.csv"
    oof_df = pd.DataFrame(oof_proba, columns=[f"proba_{label}" for label in labels])
    oof_df.insert(0, "id", train["id"].values)
    oof_df["target"] = train["class"].values
    oof_df.to_csv(cat_oof_path, index=False)
    test_df = pd.DataFrame(test_proba, columns=[f"proba_{label}" for label in labels])
    test_df.insert(0, "id", test["id"].values)
    test_df.to_csv(cat_test_path, index=False)
    print(f"Wrote {cat_oof_path}", flush=True)
    print(f"Wrote {cat_test_path}", flush=True)

    lgb_oof = pd.read_csv(args.lgb_oof)
    lgb_test = pd.read_csv(args.lgb_test_proba)
    base_submission = pd.read_csv(args.base_submission)
    proba_cols = [f"proba_{label}" for label in labels]
    y = lgb_oof["target"].map({label: idx for idx, label in enumerate(labels)}).to_numpy()
    params, blend_score = optimize_blend(
        lgb_oof[proba_cols].to_numpy(),
        oof_proba,
        y,
    )
    alpha = params[0]
    class_weights = params[1:]
    print(
        f"Blend OOF={blend_score:.6f}, alpha_lgb={alpha:.6f}, "
        f"class_weights={np.round(class_weights, 6)}",
        flush=True,
    )
    blend_test = (alpha * lgb_test[proba_cols].to_numpy() + (1.0 - alpha) * test_proba) * class_weights
    pred = blend_test.argmax(axis=1)
    submission = base_submission.copy()
    submission.iloc[:, 1] = [labels[idx] for idx in pred]
    blend_path = OUTPUT_DIR / f"submission_lgb_catboost_blend_{suffix}.csv"
    submission.to_csv(blend_path, index=False)
    print(f"Wrote {blend_path}", flush=True)
    print(submission.iloc[:, 1].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
