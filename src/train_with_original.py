from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.utils.class_weight import compute_class_weight

from train_tree_models import (
    add_notebook_categorical_features,
    add_spatial_density_features,
    find_file,
    make_model,
    make_xy,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
TE_COLS = ["alpha_cat__delta_cat__", "u_cat__z_cat__"]


def add_discussion_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type"] = pd.cut(
        out["r"] - out["g"],
        [-np.inf, -1, -0.5, 0, np.inf],
        labels=["M", "G/K", "A/F", "O/B"],
    ).astype(str)
    out["galaxy_population"] = pd.cut(
        out["u"] - out["r"],
        [-np.inf, 2.2, np.inf],
        labels=["Blue_Cloud", "Red_Sequence"],
    ).astype(str)
    return out


def load_original(path: Path) -> pd.DataFrame:
    original = pd.read_csv(path)
    original.columns = [col.strip() for col in original.columns]
    required_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift", "class"]
    missing_cols = [col for col in required_cols if col not in original.columns]
    if missing_cols:
        raise ValueError(f"Original data is missing columns: {missing_cols}")

    original = original[required_cols].copy()
    original.insert(0, "id", -np.arange(1, len(original) + 1))
    original = add_discussion_categories(original)
    return original


def add_target_encoding(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_valid: pd.DataFrame | None,
    x_test: pd.DataFrame,
    n_classes: int,
    folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    te_cols = [col for col in TE_COLS if col in x_fit.columns]
    if not te_cols:
        return x_fit, x_valid, x_test

    encoder = TargetEncoder(cv=folds, smooth="auto", shuffle=True, random_state=random_state)
    fit_encoded = encoder.fit_transform(x_fit[te_cols], y_fit)
    test_encoded = encoder.transform(x_test[te_cols])
    valid_encoded = encoder.transform(x_valid[te_cols]) if x_valid is not None else None

    te_names = [f"_{col}_te_cls{class_idx}" for col in te_cols for class_idx in range(n_classes)]
    x_fit = pd.concat(
        [x_fit.reset_index(drop=True), pd.DataFrame(fit_encoded, columns=te_names)],
        axis=1,
    )
    x_test = pd.concat(
        [x_test.reset_index(drop=True), pd.DataFrame(test_encoded, columns=te_names)],
        axis=1,
    )
    if x_valid is not None:
        x_valid = pd.concat(
            [x_valid.reset_index(drop=True), pd.DataFrame(valid_encoded, columns=te_names)],
            axis=1,
        )
    return x_fit, x_valid, x_test


def encode_categoricals_for_xgboost(
    x_fit: pd.DataFrame,
    x_valid: pd.DataFrame | None,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    lengths = [len(x_fit), 0 if x_valid is None else len(x_valid), len(x_test)]
    frames = [x_fit.reset_index(drop=True)]
    if x_valid is not None:
        frames.append(x_valid.reset_index(drop=True))
    frames.append(x_test.reset_index(drop=True))
    combined = pd.concat(frames, axis=0, ignore_index=True)

    for col in combined.columns:
        if pd.api.types.is_numeric_dtype(combined[col]):
            combined[col] = pd.to_numeric(combined[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            codes, _ = pd.factorize(combined[col].astype(str), sort=True)
            combined[col] = codes.astype(np.int32)

    fit_end = lengths[0]
    valid_end = fit_end + lengths[1]
    x_fit_out = combined.iloc[:fit_end].reset_index(drop=True)
    x_valid_out = None
    if x_valid is not None:
        x_valid_out = combined.iloc[fit_end:valid_end].reset_index(drop=True)
    x_test_out = combined.iloc[valid_end:].reset_index(drop=True)
    return x_fit_out, x_valid_out, x_test_out


def make_balanced_source_weights(y: np.ndarray, source_weights: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    class_weights = compute_class_weight("balanced", classes=classes, y=y)
    lookup = np.ones(int(classes.max()) + 1, dtype=np.float32)
    lookup[classes] = class_weights.astype(np.float32)
    return lookup[y] * source_weights.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-data", type=Path, default=INPUT_DIR / "star_classification.csv")
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--original-weight", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-min-child-weight", type=float, default=5)
    parser.add_argument("--xgb-reg-lambda", type=float, default=2.0)
    parser.add_argument("--spatial-mode", choices=["none", "raw", "trig", "all"], default="none")
    parser.add_argument("--include-spatial-density", action="store_true")
    parser.add_argument("--include-notebook-features", action="store_true")
    parser.add_argument("--include-green-valley-features", action="store_true")
    parser.add_argument("--include-target-encoding", action="store_true")
    parser.add_argument("--include-catboost-ideas", action="store_true")
    parser.add_argument("--use-fold-test-average", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(find_file("train.csv"))
    test = pd.read_csv(find_file("test.csv"))
    sample_submission = pd.read_csv(find_file("sample_submission.csv"))
    original = load_original(args.original_data)

    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(train[target_col])
    y_original_encoded = label_encoder.transform(original[target_col])
    labels = list(label_encoder.classes_)

    x_train, x_test, _ = make_xy(
        train,
        test,
        target_col,
        include_redshift=True,
        include_rest_colors=False,
        include_redshift_interactions=False,
        include_raw_bands=True,
        include_mag_stats=False,
        spatial_mode=args.spatial_mode,
        include_notebook_features=args.include_notebook_features,
        include_catboost_ideas=args.include_catboost_ideas,
        include_green_valley_features=args.include_green_valley_features,
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
        spatial_mode=args.spatial_mode,
        include_notebook_features=args.include_notebook_features,
        include_catboost_ideas=args.include_catboost_ideas,
        include_green_valley_features=args.include_green_valley_features,
    )
    if args.include_notebook_features:
        train_original = pd.concat([train, original], axis=0, ignore_index=True)
        x_train_original = pd.concat([x_train, x_original], axis=0, ignore_index=True)
        x_train_original, x_test = add_notebook_categorical_features(
            train_original,
            test,
            x_train_original,
            x_test,
        )
        x_train = x_train_original.iloc[: len(train)].reset_index(drop=True)
        x_original = x_train_original.iloc[len(train) :].reset_index(drop=True)

    if args.include_spatial_density:
        train_original = pd.concat([train, original], axis=0, ignore_index=True)
        x_train_original = pd.concat([x_train, x_original], axis=0, ignore_index=True)
        x_train_original, x_test = add_spatial_density_features(
            train_original,
            test,
            x_train_original,
            x_test,
        )
        x_train = x_train_original.iloc[: len(train)].reset_index(drop=True)
        x_original = x_train_original.iloc[len(train) :].reset_index(drop=True)

    print(f"train={train.shape}, original={original.shape}, test={test.shape}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"original_weight={args.original_weight}", flush=True)
    print(f"spatial_mode={args.spatial_mode}", flush=True)
    print(f"include_spatial_density={args.include_spatial_density}", flush=True)
    print(f"include_notebook_features={args.include_notebook_features}", flush=True)
    print(f"include_green_valley_features={args.include_green_valley_features}", flush=True)
    print(f"include_target_encoding={args.include_target_encoding}", flush=True)
    print(f"include_catboost_ideas={args.include_catboost_ideas}", flush=True)
    print(f"use_fold_test_average={args.use_fold_test_average}", flush=True)
    print(f"features={list(x_train.columns)}", flush=True)
    print("original class counts:", flush=True)
    print(original[target_col].value_counts().reindex(labels, fill_value=0).to_string(), flush=True)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    oof_proba = np.zeros((len(x_train), len(labels)), dtype=np.float64)
    test_proba_fold_average = np.zeros((len(x_test), len(labels)), dtype=np.float64)
    fold_scores = []

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train, y_train_encoded), start=1):
        model = make_model(
            args.model,
            args.random_state + fold,
            args.n_estimators,
            args.learning_rate,
            args.num_leaves,
            args.min_child_samples,
            args.xgb_max_depth,
            args.xgb_min_child_weight,
            args.xgb_reg_lambda,
        )
        x_fit = pd.concat([x_train.iloc[fit_idx], x_original], axis=0, ignore_index=True)
        y_fit = np.concatenate([y_train_encoded[fit_idx], y_original_encoded])
        x_valid = x_train.iloc[valid_idx].reset_index(drop=True)
        x_test_fold = x_test.reset_index(drop=True)
        if args.include_target_encoding:
            x_fit, x_valid, x_test_fold = add_target_encoding(
                x_fit,
                y_fit,
                x_valid,
                x_test_fold,
                len(labels),
                args.folds,
                args.random_state + fold,
            )
        source_weight = np.concatenate(
            [
                np.ones(len(fit_idx), dtype=np.float32),
                np.full(len(y_original_encoded), args.original_weight, dtype=np.float32),
            ]
        )
        if args.model == "xgboost":
            x_fit, x_valid, x_test_fold = encode_categoricals_for_xgboost(x_fit, x_valid, x_test_fold)
            sample_weight = make_balanced_source_weights(y_fit, source_weight)
        else:
            sample_weight = source_weight
        model.fit(x_fit, y_fit, sample_weight=sample_weight)
        valid_proba = model.predict_proba(x_valid)
        oof_proba[valid_idx] = valid_proba
        if args.use_fold_test_average:
            test_proba_fold_average += model.predict_proba(x_test_fold) / args.folds
        score = balanced_accuracy_score(y_train_encoded[valid_idx], valid_proba.argmax(axis=1))
        fold_scores.append(score)
        print(f"Fold {fold}: balanced_accuracy={score:.6f}", flush=True)

    oof_score = balanced_accuracy_score(y_train_encoded, oof_proba.argmax(axis=1))
    print(
        "CV balanced_accuracy: "
        f"mean={np.mean(fold_scores):.6f}, std={np.std(fold_scores):.6f}, oof={oof_score:.6f}",
        flush=True,
    )

    cm = confusion_matrix(y_train_encoded, oof_proba.argmax(axis=1), labels=np.arange(len(labels)))
    cm_df = pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])
    print("\nOOF confusion matrix:", flush=True)
    print(cm_df.to_string(), flush=True)

    if args.use_fold_test_average:
        test_proba = test_proba_fold_average
    else:
        final_model = make_model(
            args.model,
            args.random_state,
            args.n_estimators,
            args.learning_rate,
            args.num_leaves,
            args.min_child_samples,
            args.xgb_max_depth,
            args.xgb_min_child_weight,
            args.xgb_reg_lambda,
        )
        x_full = pd.concat([x_train, x_original], axis=0, ignore_index=True)
        y_full = np.concatenate([y_train_encoded, y_original_encoded])
        x_test_final = x_test.reset_index(drop=True)
        if args.include_target_encoding:
            x_full, _, x_test_final = add_target_encoding(
                x_full,
                y_full,
                None,
                x_test_final,
                len(labels),
                args.folds,
                args.random_state,
            )
        source_weight_full = np.concatenate(
            [
                np.ones(len(y_train_encoded), dtype=np.float32),
                np.full(len(y_original_encoded), args.original_weight, dtype=np.float32),
            ]
        )
        if args.model == "xgboost":
            x_full, _, x_test_final = encode_categoricals_for_xgboost(x_full, None, x_test_final)
            sample_weight_full = make_balanced_source_weights(y_full, source_weight_full)
        else:
            sample_weight_full = source_weight_full
        final_model.fit(x_full, y_full, sample_weight=sample_weight_full)
        test_proba = final_model.predict_proba(x_test_final)
    test_labels = label_encoder.inverse_transform(test_proba.argmax(axis=1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weight_suffix = f"{args.original_weight:g}".replace(".", "p")
    model_prefix = "lgb" if args.model == "lightgbm" else "xgb"
    suffix = f"{model_prefix}_original_w{weight_suffix}"
    if args.spatial_mode != "none":
        suffix = f"{suffix}_spatial_{args.spatial_mode}"
    if args.include_spatial_density:
        suffix = f"{suffix}_density"
    if args.include_notebook_features:
        suffix = f"{suffix}_notebook_features"
    if args.include_green_valley_features:
        suffix = f"{suffix}_green_valley"
    if args.include_target_encoding:
        suffix = f"{suffix}_target_encoding"
    if args.include_catboost_ideas:
        suffix = f"{suffix}_catboost_ideas"
    if args.use_fold_test_average:
        suffix = f"{suffix}_{args.folds}fold_average"
    if args.model == "xgboost":
        depth_suffix = str(args.xgb_max_depth).replace(".", "p")
        child_suffix = f"{args.xgb_min_child_weight:g}".replace(".", "p")
        lambda_suffix = f"{args.xgb_reg_lambda:g}".replace(".", "p")
        suffix = (
            f"{suffix}_depth{depth_suffix}_child{child_suffix}"
            f"_lambda{lambda_suffix}_seed{args.random_state}"
        )
    submission_path = OUTPUT_DIR / f"submission_{suffix}.csv"
    submission = sample_submission.copy()
    submission.iloc[:, 0] = test[id_col].values
    submission.iloc[:, 1] = test_labels
    submission.to_csv(submission_path, index=False)

    test_proba_df = pd.DataFrame(test_proba, columns=[f"proba_{label}" for label in labels])
    test_proba_df.insert(0, id_col, test[id_col].values)
    test_proba_path = OUTPUT_DIR / f"test_proba_{suffix}.csv"
    test_proba_df.to_csv(test_proba_path, index=False)

    oof = pd.DataFrame(oof_proba, columns=[f"proba_{label}" for label in labels])
    oof.insert(0, id_col, train[id_col].values)
    oof["target"] = train[target_col].values
    oof["prediction"] = label_encoder.inverse_transform(oof_proba.argmax(axis=1))
    oof_path = OUTPUT_DIR / f"oof_{suffix}.csv"
    oof.to_csv(oof_path, index=False)

    print(f"\nWrote {submission_path}", flush=True)
    print(f"Wrote {test_proba_path}", flush=True)
    print(f"Wrote {oof_path}", flush=True)


if __name__ == "__main__":
    main()
