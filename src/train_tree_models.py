from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


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

CATEGORICAL_COLS = ["spectral_type", "galaxy_population", "redshift_bin"]
NUMERIC_COLS = [
    "u_minus_z",
    "u_minus_r",
    "r_minus_g",
    "u_minus_g",
    "r_minus_i",
    "i_minus_z",
]
RAW_BAND_COLS = ["u", "g", "r", "i", "z"]
MAG_STAT_COLS = ["mag_mean", "mag_std", "mag_min", "mag_max", "mag_range"]
SPATIAL_RAW_COLS = ["alpha", "delta"]
SPATIAL_TRIG_COLS = ["alpha_sin", "alpha_cos", "delta_sin", "delta_cos"]
SPATIAL_COLS = [*SPATIAL_RAW_COLS, *SPATIAL_TRIG_COLS]
SPHERE_COLS = ["sphere_x", "sphere_y", "sphere_z"]
CATBOOST_IDEA_CAT_COLS = ["spectral_galaxy_interaction"]
REST_COLOR_COLS = [f"{col}_over_1p_redshift" for col in NUMERIC_COLS]
REDSHIFT_INTERACTION_COLS = [f"redshift_x_{col}" for col in NUMERIC_COLS]
KNN_K_VALUES = [30, 100]
SPATIAL_DENSITY_GRIDS = [10, 20, 40]
NOTEBOOK_COLOR_PAIRS = [
    ("u", "g"),
    ("g", "r"),
    ("i", "z"),
    ("r", "z"),
    ("u", "z"),
    ("r", "i"),
    ("u", "r"),
    ("g", "i"),
    ("g", "z"),
]
NOTEBOOK_NUMERIC_COLS = [
    "_g_div_redshift",
    "_i_div_redshift",
    "_z_div_redshift",
    "_u_div_redshift",
    *[f"_{a}_{b}_color" for a, b in NOTEBOOK_COLOR_PAIRS],
    "_flux_u",
    "_flux_g",
    "_flux_r",
    "_flux_i",
    "_flux_z",
    "_mag_mean",
    "_mag_std",
    "_mag_range",
    "_redshift_decimal",
    "_sym_sq_z_sq_g",
    "_sym_log_z_log_g",
    "_sym_sqrt_z_sqrt_g",
    "_sym_cbrt_z_cbrt_g",
    "_sym_log_z_minus_g",
    "_sym_cbrt_z_minus_g",
    "_sym_sqrt_z_minus_g",
    "_sym_log_i_cube_ratio",
    "_sym_log_z_cbrt_g",
    "_sym_cube_z_cos_r_cube_g",
]
GREEN_VALLEY_NUMERIC_COLS = [
    "_green_valley_margin",
    "_green_valley_abs_margin",
    "_green_valley_margin_x_mag_mean",
]
NOTEBOOK_CAT_COLS = [
    "alpha_cat_",
    "delta_cat_",
    "u_cat_",
    "g_cat_",
    "r_cat_",
    "i_cat_",
    "z_cat_",
    "redshift_cat_",
    "delta_100_quantile_bin_",
    "delta_500_quantile_bin_",
    "alpha_cat__delta_cat__",
    "u_cat__z_cat__",
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["redshift_bin"] = pd.cut(
        out["redshift"],
        bins=REDSHIFT_BINS,
        labels=REDSHIFT_BIN_LABELS,
        right=False,
    ).astype(str)
    out["u_minus_z"] = out["u"] - out["z"]
    out["u_minus_r"] = out["u"] - out["r"]
    out["r_minus_g"] = out["r"] - out["g"]
    out["u_minus_g"] = out["u"] - out["g"]
    out["r_minus_i"] = out["r"] - out["i"]
    out["i_minus_z"] = out["i"] - out["z"]

    one_plus_redshift = 1 + out["redshift"]
    for color_col, rest_col in zip(NUMERIC_COLS, REST_COLOR_COLS):
        out[rest_col] = out[color_col] / one_plus_redshift
    for color_col, interaction_col in zip(NUMERIC_COLS, REDSHIFT_INTERACTION_COLS):
        out[interaction_col] = out["redshift"] * out[color_col]

    bands = out[RAW_BAND_COLS]
    out["mag_mean"] = bands.mean(axis=1)
    out["mag_std"] = bands.std(axis=1)
    out["mag_min"] = bands.min(axis=1)
    out["mag_max"] = bands.max(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]

    alpha_rad = np.deg2rad(out["alpha"])
    delta_rad = np.deg2rad(out["delta"])
    out["alpha_sin"] = np.sin(alpha_rad)
    out["alpha_cos"] = np.cos(alpha_rad)
    out["delta_sin"] = np.sin(delta_rad)
    out["delta_cos"] = np.cos(delta_rad)
    out["sphere_x"] = out["delta_cos"] * out["alpha_cos"]
    out["sphere_y"] = out["delta_cos"] * out["alpha_sin"]
    out["sphere_z"] = out["delta_sin"]
    out["spectral_galaxy_interaction"] = (
        out["spectral_type"].astype(str) + "_" + out["galaxy_population"].astype(str)
    )

    for col in [*CATEGORICAL_COLS, *CATBOOST_IDEA_CAT_COLS]:
        out[col] = out[col].astype("category")
    return out


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / (b + 1e-6)


def add_notebook_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for band in RAW_BAND_COLS:
        clipped = np.clip(out[band].to_numpy(dtype=np.float64), -30, 30)
        out[f"_flux_{band}"] = np.power(10.0, -0.4 * clipped).astype(np.float32)

    for a, b in NOTEBOOK_COLOR_PAIRS:
        out[f"_{a}_{b}_color"] = (out[a] - out[b]).astype(np.float32)

    for band in ["g", "i", "z", "u"]:
        out[f"_{band}_div_redshift"] = safe_div(
            out[band].to_numpy(dtype=np.float64),
            out["redshift"].to_numpy(dtype=np.float64),
        ).astype(np.float32)

    mags = out[RAW_BAND_COLS].to_numpy(dtype=np.float32)
    out["_mag_mean"] = np.nanmean(mags, axis=1).astype(np.float32)
    out["_mag_std"] = np.nanstd(mags, axis=1).astype(np.float32)
    out["_mag_range"] = (np.nanmax(mags, axis=1) - np.nanmin(mags, axis=1)).astype(np.float32)
    green_valley_margin = (out["u"].to_numpy(dtype=np.float32) - out["r"].to_numpy(dtype=np.float32)) - 2.2
    out["_green_valley_margin"] = green_valley_margin.astype(np.float32)
    out["_green_valley_abs_margin"] = np.abs(green_valley_margin).astype(np.float32)
    out["_green_valley_margin_x_mag_mean"] = (
        green_valley_margin * out["_mag_mean"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    redshift_abs = np.abs(out["redshift"].to_numpy(dtype=np.float64))
    out["_redshift_decimal"] = (redshift_abs - np.floor(redshift_abs)).astype(np.float32)

    g = out["g"].to_numpy(dtype=np.float64)
    r = out["r"].to_numpy(dtype=np.float64)
    i = out["i"].to_numpy(dtype=np.float64)
    z = out["z"].to_numpy(dtype=np.float64)
    sym_features = {
        "_sym_sq_z_sq_g": safe_div(z**2, g**2),
        "_sym_log_z_log_g": safe_div(np.log(np.abs(z) + 1e-6), np.log(np.abs(g) + 1e-6)),
        "_sym_sqrt_z_sqrt_g": safe_div(np.sqrt(np.abs(z)), np.sqrt(np.abs(g))),
        "_sym_cbrt_z_cbrt_g": safe_div(np.cbrt(z), np.cbrt(g)),
        "_sym_log_z_minus_g": np.log(np.abs(z) + 1e-6) - np.log(np.abs(g) + 1e-6),
        "_sym_cbrt_z_minus_g": np.cbrt(z) - np.cbrt(g),
        "_sym_sqrt_z_minus_g": np.sqrt(np.abs(z)) - np.sqrt(np.abs(g)),
        "_sym_log_i_cube_ratio": np.log(np.abs(i) + 1e-6) - safe_div(z**3, g**3),
        "_sym_log_z_cbrt_g": safe_div(np.log(np.abs(z) + 1e-6), np.cbrt(g)),
        "_sym_cube_z_cos_r_cube_g": safe_div(z**3 + np.cos(r), g**3),
    }
    for col, values in sym_features.items():
        arr = np.asarray(values, dtype=np.float32)
        out[col] = np.where(np.isfinite(arr), arr, 0.0)
    return out


def make_xy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    include_redshift: bool,
    include_rest_colors: bool,
    include_redshift_interactions: bool,
    include_raw_bands: bool,
    include_mag_stats: bool,
    spatial_mode: str,
    include_notebook_features: bool,
    include_catboost_ideas: bool,
    include_green_valley_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train_features = add_features(train)
    test_features = add_features(test)
    if include_notebook_features:
        train_features = add_notebook_numeric_features(train_features)
        test_features = add_notebook_numeric_features(test_features)
    numeric_cols = [*NUMERIC_COLS]
    if include_redshift:
        numeric_cols.append("redshift")
    if include_rest_colors:
        numeric_cols.extend(REST_COLOR_COLS)
    if include_redshift_interactions:
        numeric_cols.extend(REDSHIFT_INTERACTION_COLS)
    if include_raw_bands:
        numeric_cols.extend(RAW_BAND_COLS)
    if include_mag_stats:
        numeric_cols.extend(MAG_STAT_COLS)
    if spatial_mode == "raw":
        numeric_cols.extend(SPATIAL_RAW_COLS)
    elif spatial_mode == "trig":
        numeric_cols.extend(SPATIAL_TRIG_COLS)
    elif spatial_mode == "all":
        numeric_cols.extend(SPATIAL_COLS)
    if include_notebook_features:
        numeric_cols.extend(NOTEBOOK_NUMERIC_COLS)
    if include_green_valley_features:
        numeric_cols.extend(GREEN_VALLEY_NUMERIC_COLS)
    categorical_cols = [*CATEGORICAL_COLS]
    if include_catboost_ideas:
        numeric_cols.extend(SPHERE_COLS)
        categorical_cols.extend(CATBOOST_IDEA_CAT_COLS)
    feature_cols = categorical_cols + numeric_cols
    return train_features[feature_cols], test_features[feature_cols], train[target_col]


def make_model(
    model_name: str,
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    xgb_max_depth: int = 4,
    xgb_min_child_weight: float = 5,
    xgb_reg_lambda: float = 2.0,
):
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=xgb_max_depth,
            min_child_weight=xgb_min_child_weight,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=xgb_reg_lambda,
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=n_estimators,
            learning_rate=learning_rate,
            depth=6,
            l2_leaf_reg=6.0,
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def encode_for_xgboost(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=CATEGORICAL_COLS, dummy_na=False)
    return (
        combined.iloc[: len(x_train)].reset_index(drop=True),
        combined.iloc[len(x_train) :].reset_index(drop=True),
    )


def parse_prediction_multipliers(value: str | None, labels: list[str]) -> np.ndarray:
    multipliers = {label: 1.0 for label in labels}
    if value:
        for item in value.split(","):
            label, multiplier = item.split("=", maxsplit=1)
            label = label.strip()
            if label not in multipliers:
                raise ValueError(f"Unknown class in --prediction-multipliers: {label}")
            multipliers[label] = float(multiplier)
    return np.array([multipliers[label] for label in labels], dtype=float)


def make_knn_feature_names(labels: list[str], k_values: list[int] = KNN_K_VALUES) -> list[str]:
    return [f"knn{k}_{label}_ratio" for k in k_values for label in labels]


def ratios_from_neighbor_indices(
    neighbor_indices: np.ndarray,
    y_reference: np.ndarray,
    n_classes: int,
    k_values: list[int] = KNN_K_VALUES,
) -> np.ndarray:
    features = np.zeros((neighbor_indices.shape[0], len(k_values) * n_classes), dtype=np.float32)
    offset = 0
    neighbor_labels = y_reference[neighbor_indices]
    for k in k_values:
        labels_k = neighbor_labels[:, :k]
        for class_idx in range(n_classes):
            features[:, offset] = (labels_k == class_idx).mean(axis=1)
            offset += 1
    return features


def make_knn_features(
    x_reference: pd.DataFrame,
    x_query: pd.DataFrame,
    y_reference: np.ndarray,
    numeric_cols: list[str],
    n_classes: int,
    exclude_self: bool,
    k_values: list[int] = KNN_K_VALUES,
) -> np.ndarray:
    scaler = StandardScaler()
    reference_scaled = scaler.fit_transform(x_reference[numeric_cols])
    query_scaled = scaler.transform(x_query[numeric_cols])

    extra_neighbor = 1 if exclude_self else 0
    model = NearestNeighbors(n_neighbors=max(k_values) + extra_neighbor, algorithm="auto", n_jobs=-1)
    model.fit(reference_scaled)
    neighbor_indices = model.kneighbors(query_scaled, return_distance=False)
    if exclude_self:
        neighbor_indices = neighbor_indices[:, 1:]
    return ratios_from_neighbor_indices(neighbor_indices, y_reference, n_classes, k_values)


def add_spatial_density_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    grids: list[int] = SPATIAL_DENSITY_GRIDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined_position = pd.concat(
        [train[["alpha", "delta"]], test[["alpha", "delta"]]],
        axis=0,
        ignore_index=True,
    )
    alpha = combined_position["alpha"].to_numpy()
    delta = combined_position["delta"].to_numpy()
    train_density = pd.DataFrame(index=np.arange(len(train)))
    test_density = pd.DataFrame(index=np.arange(len(test)))

    for grid in grids:
        alpha_edges = np.linspace(alpha.min(), alpha.max(), grid + 1)
        delta_edges = np.linspace(delta.min(), delta.max(), grid + 1)
        alpha_bin = np.clip(np.digitize(alpha, alpha_edges[1:-1], right=False), 0, grid - 1)
        delta_bin = np.clip(np.digitize(delta, delta_edges[1:-1], right=False), 0, grid - 1)
        bin_id = alpha_bin * grid + delta_bin
        counts = np.bincount(bin_id, minlength=grid * grid)
        density = np.log1p(counts[bin_id]).astype(np.float32)
        col = f"spatial_density_grid_{grid}"
        train_density[col] = density[: len(train)]
        test_density[col] = density[len(train) :]

    return (
        pd.concat([x_train.reset_index(drop=True), train_density], axis=1),
        pd.concat([x_test.reset_index(drop=True), test_density], axis=1),
    )


def add_notebook_categorical_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_cat = pd.DataFrame(index=np.arange(len(train)))
    test_cat = pd.DataFrame(index=np.arange(len(test)))
    numeric_source_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]

    for col in numeric_source_cols:
        train_codes, uniques = pd.factorize(np.floor(train[col]), sort=False)
        code_map = {value: idx for idx, value in enumerate(uniques)}
        test_codes = np.floor(test[col]).map(code_map).fillna(-1).astype(np.int32)
        cat_col = f"{col}_cat_"
        train_cat[cat_col] = pd.Series(train_codes, dtype=np.int32).astype("category")
        test_cat[cat_col] = test_codes.astype("category")

    for n_bins in [100, 500]:
        col = f"delta_{n_bins}_quantile_bin_"
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        train_cat[col] = (
            discretizer.fit_transform(train[["delta"]]).ravel().astype(np.int32)
        )
        test_cat[col] = discretizer.transform(test[["delta"]]).ravel().astype(np.int32)
        train_cat[col] = train_cat[col].astype("category")
        test_cat[col] = test_cat[col].astype("category")

    for left, right, combo_col in [
        ("alpha_cat_", "delta_cat_", "alpha_cat__delta_cat__"),
        ("u_cat_", "z_cat_", "u_cat__z_cat__"),
    ]:
        train_combo = train_cat[left].astype(str) + "_" + train_cat[right].astype(str)
        test_combo = test_cat[left].astype(str) + "_" + test_cat[right].astype(str)
        train_codes, uniques = pd.factorize(train_combo, sort=False)
        code_map = {value: idx for idx, value in enumerate(uniques)}
        test_codes = test_combo.map(code_map).fillna(-1).astype(np.int32)
        train_cat[combo_col] = pd.Series(train_codes, dtype=np.int32).astype("category")
        test_cat[combo_col] = test_codes.astype("category")

    return (
        pd.concat([x_train.reset_index(drop=True), train_cat[NOTEBOOK_CAT_COLS]], axis=1),
        pd.concat([x_test.reset_index(drop=True), test_cat[NOTEBOOK_CAT_COLS]], axis=1),
    )


def cross_validate(
    model_name: str,
    x: pd.DataFrame,
    y_encoded: np.ndarray,
    label_encoder: LabelEncoder,
    folds: int,
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    include_knn_features: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    n_classes = len(label_encoder.classes_)
    oof_proba = np.zeros((len(x), n_classes), dtype=float)
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y_encoded), start=1):
        model = make_model(
            model_name,
            random_state + fold,
            n_estimators,
            learning_rate,
            num_leaves,
            min_child_samples,
        )
        x_train = x.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_train = y_encoded[train_idx]
        y_valid = y_encoded[valid_idx]

        if include_knn_features:
            numeric_cols = [col for col in x.columns if col not in CATEGORICAL_COLS]
            n_classes = len(label_encoder.classes_)
            knn_cols = make_knn_feature_names(list(label_encoder.classes_))
            train_knn = make_knn_features(
                x_train,
                x_train,
                y_train,
                numeric_cols,
                n_classes,
                exclude_self=True,
            )
            valid_knn = make_knn_features(
                x_train,
                x_valid,
                y_train,
                numeric_cols,
                n_classes,
                exclude_self=False,
            )
            x_train = pd.concat(
                [x_train.reset_index(drop=True), pd.DataFrame(train_knn, columns=knn_cols)],
                axis=1,
            )
            x_valid = pd.concat(
                [x_valid.reset_index(drop=True), pd.DataFrame(valid_knn, columns=knn_cols)],
                axis=1,
            )

        fit_kwargs = {}
        if model_name == "catboost":
            fit_kwargs["cat_features"] = CATEGORICAL_COLS
        elif model_name == "xgboost":
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)

        model.fit(x_train, y_train, **fit_kwargs)
        valid_proba = model.predict_proba(x_valid)
        oof_proba[valid_idx] = valid_proba
        valid_pred = valid_proba.argmax(axis=1)
        score = balanced_accuracy_score(y_valid, valid_pred)
        fold_scores.append(score)
        print(f"Fold {fold}: balanced_accuracy={score:.6f}", flush=True)

    metrics = {
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof": float(balanced_accuracy_score(y_encoded, oof_proba.argmax(axis=1))),
    }
    return oof_proba, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "catboost"], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--include-redshift", action="store_true")
    parser.add_argument("--include-rest-colors", action="store_true")
    parser.add_argument("--include-redshift-interactions", action="store_true")
    parser.add_argument("--include-raw-bands", action="store_true")
    parser.add_argument("--include-mag-stats", action="store_true")
    parser.add_argument("--include-spatial", action="store_true")
    parser.add_argument("--spatial-mode", choices=["none", "raw", "trig", "all"], default="none")
    parser.add_argument("--include-spatial-density", action="store_true")
    parser.add_argument("--include-notebook-features", action="store_true")
    parser.add_argument("--include-catboost-ideas", action="store_true")
    parser.add_argument("--include-knn-features", action="store_true")
    parser.add_argument(
        "--prediction-multipliers",
        help="Comma-separated class multipliers, for example GALAXY=1,QSO=1.02,STAR=0.98.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(find_file("train.csv"))
    test = pd.read_csv(find_file("test.csv"))
    sample_submission = pd.read_csv(find_file("sample_submission.csv"))
    if args.include_spatial and args.spatial_mode == "none":
        args.spatial_mode = "all"

    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]
    x_train, x_test, y = make_xy(
        train,
        test,
        target_col,
        args.include_redshift,
        args.include_rest_colors,
        args.include_redshift_interactions,
        args.include_raw_bands,
        args.include_mag_stats,
        args.spatial_mode,
        args.include_notebook_features,
        args.include_catboost_ideas,
    )
    if args.include_notebook_features:
        x_train, x_test = add_notebook_categorical_features(train, test, x_train, x_test)
    if args.include_spatial_density:
        x_train, x_test = add_spatial_density_features(train, test, x_train, x_test)

    if args.model == "xgboost":
        x_train, x_test = encode_for_xgboost(x_train, x_test)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"train={train.shape}, test={test.shape}", flush=True)
    print(
        f"model={args.model}, folds={args.folds}, "
        f"n_estimators={args.n_estimators}, learning_rate={args.learning_rate}",
        flush=True,
    )
    if args.model == "lightgbm":
        print(
            f"num_leaves={args.num_leaves}, min_child_samples={args.min_child_samples}",
            flush=True,
        )
    print(f"include_redshift={args.include_redshift}", flush=True)
    print(f"include_rest_colors={args.include_rest_colors}", flush=True)
    print(f"include_redshift_interactions={args.include_redshift_interactions}", flush=True)
    print(f"include_raw_bands={args.include_raw_bands}", flush=True)
    print(f"include_mag_stats={args.include_mag_stats}", flush=True)
    print(f"spatial_mode={args.spatial_mode}", flush=True)
    print(f"include_spatial_density={args.include_spatial_density}", flush=True)
    print(f"include_notebook_features={args.include_notebook_features}", flush=True)
    print(f"include_catboost_ideas={args.include_catboost_ideas}", flush=True)
    print(f"include_knn_features={args.include_knn_features}", flush=True)
    print(f"features={list(x_train.columns)}", flush=True)

    oof_proba, metrics = cross_validate(
        args.model,
        x_train,
        y_encoded,
        label_encoder,
        args.folds,
        args.random_state,
        args.n_estimators,
        args.learning_rate,
        args.num_leaves,
        args.min_child_samples,
        args.include_knn_features,
    )
    print(
        "CV balanced_accuracy: "
        f"mean={metrics['cv_mean']:.6f}, std={metrics['cv_std']:.6f}, oof={metrics['oof']:.6f}",
        flush=True,
    )

    labels = list(label_encoder.classes_)
    prediction_multipliers = parse_prediction_multipliers(args.prediction_multipliers, labels)
    tuned_oof_pred = (oof_proba * prediction_multipliers).argmax(axis=1)
    if args.prediction_multipliers:
        tuned_score = balanced_accuracy_score(y_encoded, tuned_oof_pred)
        print(f"prediction_multipliers={dict(zip(labels, prediction_multipliers))}", flush=True)
        print(f"Tuned OOF balanced_accuracy: {tuned_score:.6f}", flush=True)

    cm = confusion_matrix(y_encoded, tuned_oof_pred, labels=np.arange(len(labels)))
    cm_df = pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])
    print("\nOOF confusion matrix:", flush=True)
    print(cm_df.to_string(), flush=True)

    final_model = make_model(
        args.model,
        args.random_state,
        args.n_estimators,
        args.learning_rate,
        args.num_leaves,
        args.min_child_samples,
    )
    fit_kwargs = {}
    if args.model == "catboost":
        fit_kwargs["cat_features"] = CATEGORICAL_COLS
    elif args.model == "xgboost":
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_encoded)
    if args.include_knn_features:
        numeric_cols = [col for col in x_train.columns if col not in CATEGORICAL_COLS]
        n_classes = len(label_encoder.classes_)
        knn_cols = make_knn_feature_names(labels)
        train_knn = make_knn_features(
            x_train,
            x_train,
            y_encoded,
            numeric_cols,
            n_classes,
            exclude_self=True,
        )
        test_knn = make_knn_features(
            x_train,
            x_test,
            y_encoded,
            numeric_cols,
            n_classes,
            exclude_self=False,
        )
        x_train = pd.concat(
            [x_train.reset_index(drop=True), pd.DataFrame(train_knn, columns=knn_cols)],
            axis=1,
        )
        x_test = pd.concat(
            [x_test.reset_index(drop=True), pd.DataFrame(test_knn, columns=knn_cols)],
            axis=1,
        )

    final_model.fit(x_train, y_encoded, **fit_kwargs)
    test_proba = final_model.predict_proba(x_test)
    test_pred = (test_proba * prediction_multipliers).argmax(axis=1)
    test_labels = label_encoder.inverse_transform(test_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission = sample_submission.copy()
    submission.iloc[:, 0] = test[id_col].values
    submission.iloc[:, 1] = test_labels
    learning_rate_suffix = f"{args.learning_rate:g}".replace(".", "p")
    suffix = f"{args.model}_{args.n_estimators}_estimators_lr_{learning_rate_suffix}"
    if args.model == "lightgbm":
        suffix = f"{suffix}_leaves_{args.num_leaves}_child_{args.min_child_samples}"
    if args.include_redshift:
        suffix = f"{suffix}_with_redshift"
    if args.include_rest_colors:
        suffix = f"{suffix}_rest_colors"
    if args.include_redshift_interactions:
        suffix = f"{suffix}_redshift_interactions"
    if args.include_raw_bands:
        suffix = f"{suffix}_raw_bands"
    if args.include_mag_stats:
        suffix = f"{suffix}_mag_stats"
    if args.spatial_mode != "none":
        suffix = f"{suffix}_spatial_{args.spatial_mode}"
    if args.include_spatial_density:
        suffix = f"{suffix}_density"
    if args.include_notebook_features:
        suffix = f"{suffix}_notebook_features"
    if args.include_catboost_ideas:
        suffix = f"{suffix}_catboost_ideas"
    if args.include_knn_features:
        suffix = f"{suffix}_knn_features"
    if args.prediction_multipliers:
        suffix = f"{suffix}_tuned_thresholds"
    submission_path = OUTPUT_DIR / f"submission_{suffix}.csv"
    submission.to_csv(submission_path, index=False)

    test_proba_df = pd.DataFrame(test_proba, columns=[f"proba_{label}" for label in labels])
    test_proba_df.insert(0, id_col, test[id_col].values)
    test_proba_path = OUTPUT_DIR / f"test_proba_{suffix}.csv"
    test_proba_df.to_csv(test_proba_path, index=False)

    oof = pd.DataFrame(oof_proba, columns=[f"proba_{label}" for label in labels])
    oof.insert(0, id_col, train[id_col].values)
    oof["target"] = y.values
    oof["prediction"] = label_encoder.inverse_transform(tuned_oof_pred)
    oof_path = OUTPUT_DIR / f"oof_{suffix}.csv"
    oof.to_csv(oof_path, index=False)

    print(f"\nWrote {submission_path}", flush=True)
    print(f"Wrote {test_proba_path}", flush=True)
    print(f"Wrote {oof_path}", flush=True)


if __name__ == "__main__":
    main()
