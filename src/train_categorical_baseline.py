from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
REDSHIFT_BINS = [-np.inf, 0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
REDSHIFT_BIN_LABELS = [
    "z_neg",
    "z_0_0.05",
    "z_0.05_0.1",
    "z_0.1_0.2",
    "z_0.2_0.5",
    "z_0.5_1",
    "z_1_2",
    "z_2_3",
    "z_3_5",
    "z_5_plus",
]


def find_file(name: str) -> Path:
    candidates = [
        INPUT_DIR / name,
        ROOT / name,
        ROOT / "data" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {name}. Put competition files in {INPUT_DIR}.")


def infer_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> tuple[str, str]:
    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]

    if target_col not in train.columns:
        possible_targets = [col for col in train.columns if col not in test.columns]
        if len(possible_targets) == 1:
            target_col = possible_targets[0]
        elif "class" in train.columns:
            target_col = "class"
        else:
            raise ValueError("Could not infer target column. Pass --target-col explicitly.")

    if id_col not in test.columns:
        possible_ids = [col for col in test.columns if col.lower() == "id"]
        if possible_ids:
            id_col = possible_ids[0]
        else:
            raise ValueError("Could not infer ID column. Pass --id-col explicitly.")

    return id_col, target_col


def fit_category_lookup(
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[dict[tuple[object, ...], str], str]:
    global_label = train[target_col].mode().iat[0]
    grouped = train.groupby(feature_cols, dropna=False)[target_col]
    lookup = grouped.agg(lambda s: s.value_counts().idxmax()).to_dict()
    return lookup, global_label


def predict_from_lookup(
    data: pd.DataFrame,
    feature_cols: list[str],
    lookup: dict[tuple[object, ...], str],
    fallback_label: str,
) -> np.ndarray:
    keys = data[feature_cols].itertuples(index=False, name=None)
    return np.array([lookup.get(key, fallback_label) for key in keys], dtype=object)


def add_redshift_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["redshift_bin"] = pd.cut(
        out["redshift"],
        bins=REDSHIFT_BINS,
        labels=REDSHIFT_BIN_LABELS,
        right=False,
    ).astype(str)
    return out


def add_color_features(df: pd.DataFrame, color_features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for feature in color_features:
        if feature == "u_minus_z":
            out[feature] = out["u"] - out["z"]
        elif feature == "u_minus_g":
            out[feature] = out["u"] - out["g"]
        elif feature == "u_minus_r":
            out[feature] = out["u"] - out["r"]
        elif feature == "r_minus_g":
            out[feature] = out["r"] - out["g"]
        elif feature == "r_minus_i":
            out[feature] = out["r"] - out["i"]
        elif feature == "i_minus_z":
            out[feature] = out["i"] - out["z"]
        else:
            raise ValueError(f"Unsupported color feature: {feature}")
    return out


def cross_validate(
    train: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    target_col: str,
    folds: int,
    random_state: int,
    model: str,
    regularization_c: float,
) -> tuple[np.ndarray, dict[str, float]]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    y = train[target_col].to_numpy()
    oof_pred = np.empty(len(train), dtype=object)
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(train, y), start=1):
        fold_train = train.iloc[train_idx]
        fold_valid = train.iloc[valid_idx]
        if model == "lookup":
            lookup, fallback_label = fit_category_lookup(fold_train, categorical_cols, target_col)
            valid_pred = predict_from_lookup(fold_valid, categorical_cols, lookup, fallback_label)
        else:
            model_cols = categorical_cols + numeric_cols
            estimator = make_categorical_model(
                categorical_cols,
                numeric_cols,
                random_state,
                regularization_c,
            )
            estimator.fit(fold_train[model_cols], fold_train[target_col])
            valid_pred = estimator.predict(fold_valid[model_cols])
        oof_pred[valid_idx] = valid_pred
        score = balanced_accuracy_score(fold_valid[target_col], valid_pred)
        fold_scores.append(score)
        print(f"Fold {fold}: balanced_accuracy={score:.6f}", flush=True)

    metrics = {
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof": float(balanced_accuracy_score(y, oof_pred)),
    }
    return oof_pred, metrics


def make_categorical_model(
    categorical_cols: list[str],
    numeric_cols: list[str],
    random_state: int,
    regularization_c: float,
) -> object:
    transformers = [
        ("category", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
    if numeric_cols:
        transformers.append(("numeric", StandardScaler(), numeric_cols))

    return make_pipeline(
        ColumnTransformer(
            transformers,
            remainder="drop",
        ),
        LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
    )


def print_lookup_table(train: pd.DataFrame, feature_cols: list[str], target_col: str) -> None:
    table = (
        train.groupby(feature_cols + [target_col], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    table["pct_within_group"] = (
        table["count"] / table.groupby(feature_cols, dropna=False)["count"].transform("sum") * 100
    )
    print("\nCategory combination distribution:", flush=True)
    print(
        table.sort_values(feature_cols + ["count"], ascending=[True] * len(feature_cols) + [False])
        .to_string(index=False, formatters={"pct_within_group": "{:.2f}".format}),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument(
        "--features",
        nargs="+",
        default=["spectral_type", "galaxy_population"],
        help="Categorical feature columns to use.",
    )
    parser.add_argument(
        "--numeric-features",
        nargs="*",
        default=["redshift"],
        help="Numeric feature columns to use with logreg. Ignored by lookup.",
    )
    parser.add_argument(
        "--redshift-mode",
        choices=["continuous", "bin", "none"],
        default="continuous",
        help="How to use redshift: continuous numeric, binned categorical, or not at all.",
    )
    parser.add_argument(
        "--color-features",
        nargs="*",
        default=[],
        choices=[
            "u_minus_z",
            "u_minus_g",
            "u_minus_r",
            "r_minus_g",
            "r_minus_i",
            "i_minus_z",
        ],
        help="Color-index numeric features to add.",
    )
    parser.add_argument(
        "--print-category-table",
        action="store_true",
        help="Print the category combination distribution table.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument(
        "--model",
        choices=["lookup", "logreg"],
        default="lookup",
        help="lookup uses each category combination majority class; logreg uses one-hot logistic regression.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_path = args.train or find_file("train.csv")
    test_path = args.test or find_file("test.csv")
    sample_path = args.sample_submission or find_file("sample_submission.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)

    id_col, target_col = infer_columns(train, test, sample_submission)
    id_col = args.id_col or id_col
    target_col = args.target_col or target_col
    categorical_cols = args.features
    numeric_cols = args.numeric_features
    color_features = args.color_features

    if color_features:
        train = add_color_features(train, color_features)
        test = add_color_features(test, color_features)
        numeric_cols = [*numeric_cols, *color_features]

    if args.redshift_mode == "bin":
        train = add_redshift_bin(train)
        test = add_redshift_bin(test)
        categorical_cols = [*categorical_cols, "redshift_bin"]
        numeric_cols = [col for col in numeric_cols if col != "redshift"]
    elif args.redshift_mode == "none":
        numeric_cols = [col for col in numeric_cols if col != "redshift"]

    all_feature_cols = categorical_cols + numeric_cols
    missing_features = [col for col in all_feature_cols if col not in train.columns or col not in test.columns]
    if missing_features:
        raise ValueError(f"Feature columns missing from train/test: {missing_features}")

    print(f"train={train.shape}, test={test.shape}", flush=True)
    print(f"id_col={id_col}, target_col={target_col}", flush=True)
    print(f"categorical_features={categorical_cols}", flush=True)
    print(f"numeric_features={numeric_cols}", flush=True)
    print(f"color_features={color_features}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"regularization_c={args.regularization_c}", flush=True)
    if args.model == "lookup" and numeric_cols:
        print("lookup model ignores numeric_features.", flush=True)
    if args.print_category_table:
        print_lookup_table(train, categorical_cols, target_col)

    oof_pred, metrics = cross_validate(
        train,
        categorical_cols,
        numeric_cols,
        target_col,
        args.folds,
        args.random_state,
        args.model,
        args.regularization_c,
    )
    print(
        "\nCV balanced_accuracy: "
        f"mean={metrics['cv_mean']:.6f}, std={metrics['cv_std']:.6f}, oof={metrics['oof']:.6f}",
        flush=True,
    )

    labels = sorted(train[target_col].unique())
    cm = confusion_matrix(train[target_col], oof_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])
    print("\nOOF confusion matrix:", flush=True)
    print(cm_df.to_string(), flush=True)

    if args.model == "lookup":
        lookup, fallback_label = fit_category_lookup(train, categorical_cols, target_col)
        test_pred = predict_from_lookup(test, categorical_cols, lookup, fallback_label)
    else:
        model_cols = categorical_cols + numeric_cols
        estimator = make_categorical_model(
            categorical_cols,
            numeric_cols,
            args.random_state,
            args.regularization_c,
        )
        estimator.fit(train[model_cols], train[target_col])
        test_pred = estimator.predict(test[model_cols])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission = sample_submission.copy()
    submission.iloc[:, 0] = test[id_col].values
    submission.iloc[:, 1] = test_pred
    if args.redshift_mode == "bin":
        suffix_parts = ["redshift_bin", *color_features]
        suffix = "_".join(suffix_parts)
    elif numeric_cols and args.model != "lookup":
        suffix = "_".join(numeric_cols)
    else:
        suffix = "categorical"
    if args.model != "lookup":
        suffix = f"{suffix}_c_{args.regularization_c:g}".replace(".", "p")
    submission_path = OUTPUT_DIR / f"submission_categorical_{args.model}_{suffix}_baseline.csv"
    submission.to_csv(submission_path, index=False)

    oof = train[[id_col, *categorical_cols, *numeric_cols, target_col]].copy()
    oof["prediction"] = oof_pred
    oof_path = OUTPUT_DIR / f"oof_categorical_{args.model}_{suffix}_baseline.csv"
    oof.to_csv(oof_path, index=False)

    print(f"\nWrote {submission_path}", flush=True)
    print(f"Wrote {oof_path}", flush=True)


if __name__ == "__main__":
    main()
