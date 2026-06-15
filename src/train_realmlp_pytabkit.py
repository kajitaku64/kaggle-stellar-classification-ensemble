from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from train_tree_models import (
    CATEGORICAL_COLS,
    CATBOOST_IDEA_CAT_COLS,
    NOTEBOOK_CAT_COLS,
    add_notebook_categorical_features,
    add_spatial_density_features,
    find_file,
    make_xy,
)
from train_with_original import add_target_encoding, load_original


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


def sample_original(
    original: pd.DataFrame,
    target_col: str,
    frac: float,
    random_state: int,
) -> pd.DataFrame:
    if frac >= 1:
        return original.reset_index(drop=True)
    if frac <= 0:
        return original.iloc[:0].copy().reset_index(drop=True)
    return (
        original.groupby(target_col, group_keys=False)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )


def build_features(
    train: pd.DataFrame,
    original: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    include_target_encoding_base: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    del include_target_encoding_base
    x_train, x_test, cat_cols = make_xy(
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

    combined_raw = pd.concat([train, original], axis=0, ignore_index=True)
    combined_x = pd.concat([x_train, x_original], axis=0, ignore_index=True)
    combined_x, x_test = add_notebook_categorical_features(combined_raw, test, combined_x, x_test)
    combined_x, x_test = add_spatial_density_features(combined_raw, test, combined_x, x_test)

    x_train = combined_x.iloc[: len(train)].reset_index(drop=True)
    x_original = combined_x.iloc[len(train) :].reset_index(drop=True)

    cat_candidates = [
        *CATEGORICAL_COLS,
        *CATBOOST_IDEA_CAT_COLS,
        *NOTEBOOK_CAT_COLS,
        *cat_cols,
    ]
    cat_col_names = sorted({col for col in cat_candidates if col in x_train.columns})
    return x_train, x_original, x_test.reset_index(drop=True), cat_col_names


def normalize_dtypes(
    x_fit: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_col_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = sorted(x_fit.columns)
    x_fit = x_fit.reindex(columns=columns).copy()
    x_valid = x_valid.reindex(columns=columns).copy()
    x_test = x_test.reindex(columns=columns).copy()

    cat_set = set(cat_col_names)
    for df in [x_fit, x_valid, x_test]:
        for col in columns:
            if col in cat_set:
                df[col] = df[col].astype(str).fillna("__NA__")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x_fit, x_valid, x_test


def make_model(args: argparse.Namespace, seed: int) -> RealMLP_TD_Classifier:
    return RealMLP_TD_Classifier(
        device=args.device,
        random_state=seed,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        predict_batch_size=args.predict_batch_size,
        hidden_sizes=[args.hidden_width] * args.hidden_layers,
        n_ens=args.n_ens,
        embedding_size=args.embedding_size,
        max_one_hot_cat_size=args.onehot_thresh,
        p_drop=args.dropout,
        p_drop_sched="expm4t",
        add_front_scale=True,
        tfms=["median_center", "robust_scale"],
        lr=args.lr,
        mom=0.9,
        sq_mom=0.98,
        lr_sched="flat_cos",
        wd=args.weight_decay,
        use_ls=True,
        ls_eps=args.label_smoothing,
        ls_eps_sched="cos",
        use_early_stopping=False,
        verbosity=args.verbosity,
        n_threads=args.n_threads,
        tmp_folder=str(OUTPUT_DIR / "realmlp_tmp"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--original-frac", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--n-ens", type=int, default=1)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--hidden-width", type=int, default=512)
    parser.add_argument("--embedding-size", type=int, default=8)
    parser.add_argument("--onehot-thresh", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.063)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.013)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=10240)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--include-target-encoding", action="store_true")
    parser.add_argument("--random-state", type=int, default=63)
    parser.add_argument("--verbosity", type=int, default=1)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(find_file("train.csv"))
    test = pd.read_csv(find_file("test.csv"))
    sample_submission = pd.read_csv(find_file("sample_submission.csv"))
    original = load_original(INPUT_DIR / "star_classification.csv")
    original = sample_original(original, sample_submission.columns[1], args.original_frac, args.random_state)

    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train[target_col])
    y_original = label_encoder.transform(original[target_col])
    labels = list(label_encoder.classes_)

    x_train, x_original, x_test, cat_col_names = build_features(train, original, test, target_col)

    print(f"train={x_train.shape}, original_sample={x_original.shape}, test={x_test.shape}", flush=True)
    print(f"classes={labels}", flush=True)
    print(f"cat_features={len(cat_col_names)}", flush=True)
    print(f"features={x_train.shape[1]}", flush=True)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    oof = np.zeros((len(x_train), len(labels)), dtype=np.float32)
    test_proba = np.zeros((len(x_test), len(labels)), dtype=np.float32)
    fold_scores: list[float] = []

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train, y_train), start=1):
        if args.max_folds is not None and fold > args.max_folds:
            break
        print(f"\nfold {fold}/{args.folds}", flush=True)
        x_fit = pd.concat([x_train.iloc[fit_idx], x_original], axis=0, ignore_index=True)
        y_fit = np.concatenate([y_train[fit_idx], y_original])
        x_valid = x_train.iloc[valid_idx].reset_index(drop=True)
        x_test_fold = x_test.copy()

        current_cat_cols = cat_col_names.copy()
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
        current_cat_cols = [col for col in current_cat_cols if col in x_fit.columns]
        x_fit, x_valid, x_test_fold = normalize_dtypes(x_fit, x_valid, x_test_fold, current_cat_cols)

        model = make_model(args, args.random_state + fold)
        model.fit(x_fit, y_fit, X_val=x_valid, y_val=y_train[valid_idx], cat_col_names=current_cat_cols)
        valid_proba = model.predict_proba(x_valid)
        oof[valid_idx] = valid_proba
        test_proba += model.predict_proba(x_test_fold) / (args.max_folds or args.folds)

        score = balanced_accuracy_score(y_train[valid_idx], valid_proba.argmax(axis=1))
        fold_scores.append(score)
        print(f"fold_score={score:.6f}", flush=True)

    covered = np.flatnonzero(oof.sum(axis=1) > 0)
    oof_score = balanced_accuracy_score(y_train[covered], oof[covered].argmax(axis=1))
    print(f"\noof_score={oof_score:.6f}", flush=True)
    print(f"fold_scores={[round(s, 6) for s in fold_scores]}", flush=True)

    suffix = (
        f"realmlp_pytabkit_orig{args.original_frac:g}_{args.folds}fold"
        f"_epochs{args.epochs}_ens{args.n_ens}"
    )
    if args.max_folds is not None:
        suffix += f"_max{args.max_folds}"
    if args.include_target_encoding:
        suffix += "_te"

    oof_df = pd.DataFrame({id_col: train[id_col]})
    for i, label in enumerate(labels):
        oof_df[f"proba_{label}"] = oof[:, i]
    oof_df[target_col] = train[target_col]
    oof_df.to_csv(OUTPUT_DIR / f"oof_{suffix}.csv", index=False)

    test_df = pd.DataFrame({id_col: test[id_col]})
    for i, label in enumerate(labels):
        test_df[f"proba_{label}"] = test_proba[:, i]
    test_df.to_csv(OUTPUT_DIR / f"test_proba_{suffix}.csv", index=False)

    sub = pd.DataFrame({id_col: test[id_col], target_col: test_proba.argmax(axis=1)})
    sub[target_col] = sub[target_col].map({i: label for i, label in enumerate(labels)})
    sub_path = OUTPUT_DIR / f"submission_{suffix}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"saved={sub_path}", flush=True)
    print(sub[target_col].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
