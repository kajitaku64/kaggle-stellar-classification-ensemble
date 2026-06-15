from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


def find_file(name: str) -> Path:
    candidates = [
        INPUT_DIR / name,
        ROOT / name,
        ROOT / "data" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {name}. Put competition files in {INPUT_DIR}."
    )


def infer_columns(train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame) -> tuple[str, str]:
    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]

    if target_col not in train.columns:
        possible_targets = [col for col in train.columns if col not in test.columns]
        if len(possible_targets) == 1:
            target_col = possible_targets[0]
        elif "class" in train.columns:
            target_col = "class"
        else:
            raise ValueError(
                "Could not infer target column. Pass --target-col explicitly."
            )

    if id_col not in test.columns:
        possible_ids = [col for col in test.columns if col.lower() == "id"]
        if possible_ids:
            id_col = possible_ids[0]
        else:
            raise ValueError("Could not infer ID column. Pass --id-col explicitly.")

    return id_col, target_col


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    bands = [band for band in ["u", "g", "r", "i", "z"] if band in out.columns]
    for left, right in zip(bands, bands[1:]):
        out[f"{left}_minus_{right}"] = out[left] - out[right]
    if {"u", "z"}.issubset(out.columns):
        out["u_minus_z"] = out["u"] - out["z"]

    if bands:
        band_values = out[bands]
        out["mag_mean"] = band_values.mean(axis=1)
        out["mag_std"] = band_values.std(axis=1)
        out["mag_min"] = band_values.min(axis=1)
        out["mag_max"] = band_values.max(axis=1)
        out["mag_range"] = out["mag_max"] - out["mag_min"]

    if "redshift" in out.columns:
        redshift = out["redshift"].astype(float)
        out["redshift_abs"] = redshift.abs()
        out["redshift_sq"] = redshift**2
        out["redshift_cube"] = redshift**3
        out["redshift_log1p_abs"] = np.log1p(redshift.abs())
        out["redshift_sqrt_abs"] = np.sqrt(redshift.abs())
        for col in ["u_minus_z", "g_minus_r", "r_minus_i"]:
            if col in out.columns:
                out[f"redshift_x_{col}"] = redshift * out[col]

    if {"alpha", "delta"}.issubset(out.columns):
        alpha_rad = np.deg2rad(out["alpha"].astype(float))
        delta_rad = np.deg2rad(out["delta"].astype(float))
        out["alpha_sin"] = np.sin(alpha_rad)
        out["alpha_cos"] = np.cos(alpha_rad)
        out["delta_sin"] = np.sin(delta_rad)
        out["delta_cos"] = np.cos(delta_rad)
        out["sky_x"] = np.cos(delta_rad) * np.cos(alpha_rad)
        out["sky_y"] = np.cos(delta_rad) * np.sin(alpha_rad)
        out["sky_z"] = np.sin(delta_rad)

    if {"plate", "fiber_ID"}.issubset(out.columns):
        out["plate_fiber_ratio"] = out["plate"] / (out["fiber_ID"].abs() + 1)
    if {"MJD", "plate"}.issubset(out.columns):
        out["mjd_minus_plate"] = out["MJD"] - out["plate"]

    return out


def make_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    y = train[target_col].copy()
    train_x = train.drop(columns=[target_col])
    test_x = test.copy()

    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    combined = add_features(combined)

    drop_cols = [col for col in [id_col] if col in combined.columns]
    combined = combined.drop(columns=drop_cols)

    categorical_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        combined = pd.get_dummies(combined, columns=categorical_cols, dummy_na=True)

    combined = combined.replace([np.inf, -np.inf], np.nan)

    train_features = combined.iloc[: len(train)].reset_index(drop=True)
    test_features = combined.iloc[len(train) :].reset_index(drop=True)
    return train_features, test_features, y


def build_models(random_state: int, preset: str) -> list[tuple[str, object]]:
    hgb = (
        "hgb",
        make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=260,
                l2_regularization=0.02,
                random_state=random_state,
            ),
        ),
    )

    if preset == "fast":
        return [hgb]

    return [
        hgb,
        (
            "extra_trees",
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesClassifier(
                    n_estimators=180,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ),
    ]


def cross_validate(
    x: pd.DataFrame,
    y_encoded: np.ndarray,
    models: list[tuple[str, object]],
    folds: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, float]]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    n_classes = len(np.unique(y_encoded))
    oof_proba = np.zeros((len(x), n_classes), dtype=float)
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y_encoded), start=1):
        estimators = [(name, clone(model)) for name, model in models]
        ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
        ensemble.fit(x.iloc[train_idx], y_encoded[train_idx])
        valid_proba = ensemble.predict_proba(x.iloc[valid_idx])
        oof_proba[valid_idx] = valid_proba
        valid_pred = valid_proba.argmax(axis=1)
        score = balanced_accuracy_score(y_encoded[valid_idx], valid_pred)
        fold_scores.append(score)
        print(f"Fold {fold}: balanced_accuracy={score:.6f}", flush=True)

    overall = balanced_accuracy_score(y_encoded, oof_proba.argmax(axis=1))
    metrics = {
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof": float(overall),
    }
    return oof_proba, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--preset", choices=["fast", "ensemble"], default="fast")
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

    print(f"train={train.shape}, test={test.shape}", flush=True)
    print(f"id_col={id_col}, target_col={target_col}", flush=True)

    x_train, x_test, y = make_features(train, test, id_col, target_col)
    print(f"features={x_train.shape[1]}", flush=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    models = build_models(args.random_state, args.preset)

    oof_proba, metrics = cross_validate(
        x_train, y_encoded, models, args.folds, args.random_state
    )
    print(
        "CV balanced_accuracy: "
        f"mean={metrics['cv_mean']:.6f}, std={metrics['cv_std']:.6f}, oof={metrics['oof']:.6f}",
        flush=True,
    )

    final_ensemble = VotingClassifier(
        estimators=[(name, clone(model)) for name, model in models],
        voting="soft",
        n_jobs=1,
    )
    final_ensemble.fit(x_train, y_encoded)
    test_pred = final_ensemble.predict(x_test)
    test_labels = label_encoder.inverse_transform(test_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission = sample_submission.copy()
    submission.iloc[:, 0] = test[id_col].values
    submission.iloc[:, 1] = test_labels
    submission_path = OUTPUT_DIR / "submission_baseline.csv"
    submission.to_csv(submission_path, index=False)

    oof = pd.DataFrame(oof_proba, columns=[f"proba_{label}" for label in label_encoder.classes_])
    if id_col in train.columns:
        oof.insert(0, id_col, train[id_col].values)
    oof["target"] = y.values
    oof["prediction"] = label_encoder.inverse_transform(oof_proba.argmax(axis=1))
    oof_path = OUTPUT_DIR / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)

    print(f"Wrote {submission_path}")
    print(f"Wrote {oof_path}")


if __name__ == "__main__":
    main()
