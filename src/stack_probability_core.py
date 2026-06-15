from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


CLASSES = ["GALAXY", "QSO", "STAR"]
C2I = {label: idx for idx, label in enumerate(CLASSES)}
I2C = {idx: label for label, idx in C2I.items()}
PROBA_COLS = [f"proba_{label}" for label in CLASSES]


DEFAULT_MODELS = {
    "lgb": (
        "output/oof_lgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_10fold_average.csv",
        "output/test_proba_lgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_10fold_average.csv",
    ),
    "xgb2_seed63": (
        "output/oof_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child3_lambda3_seed63.csv",
        "output/test_proba_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child3_lambda3_seed63.csv",
    ),
    "xgb2_seed71": (
        "output/oof_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child3_lambda3_seed71.csv",
        "output/test_proba_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child3_lambda3_seed71.csv",
    ),
    "xgb_green_valley": (
        "output/oof_xgb_original_w0p25_spatial_all_density_notebook_features_green_valley_target_encoding_5fold_average_depth5_child3_lambda3_seed63.csv",
        "output/test_proba_xgb_original_w0p25_spatial_all_density_notebook_features_green_valley_target_encoding_5fold_average_depth5_child3_lambda3_seed63.csv",
    ),
    "xgb3": (
        "output/oof_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child5_lambda4.csv",
        "output/test_proba_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child5_lambda4.csv",
    ),
}


def load_pair(name: str, oof_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not oof_path.exists():
        raise FileNotFoundError(f"{name}: missing OOF file {oof_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"{name}: missing test file {test_path}")
    oof = pd.read_csv(oof_path)
    test = pd.read_csv(test_path)
    for df, kind in [(oof, "oof"), (test, "test")]:
        missing = {"id", *PROBA_COLS} - set(df.columns)
        if missing:
            raise ValueError(f"{name} {kind}: missing columns {sorted(missing)}")
    if "target" not in oof.columns:
        raise ValueError(f"{name}: OOF file must include target column")
    return oof.sort_values("id").reset_index(drop=True), test.sort_values("id").reset_index(drop=True)


def build_stack(model_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    oof_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    ref_oof_id: np.ndarray | None = None
    ref_test_id: np.ndarray | None = None
    y: np.ndarray | None = None

    for name in model_names:
        oof_path, test_path = DEFAULT_MODELS[name]
        oof, test = load_pair(name, Path(oof_path), Path(test_path))
        if ref_oof_id is None:
            ref_oof_id = oof["id"].to_numpy()
            ref_test_id = test["id"].to_numpy()
            y = oof["target"].map(C2I).to_numpy(np.int8)
        else:
            if not np.array_equal(ref_oof_id, oof["id"].to_numpy()):
                raise ValueError(f"{name}: OOF id mismatch")
            if not np.array_equal(ref_test_id, test["id"].to_numpy()):
                raise ValueError(f"{name}: test id mismatch")
            if not np.array_equal(y, oof["target"].map(C2I).to_numpy(np.int8)):
                raise ValueError(f"{name}: target mismatch")
        oof_parts.append(oof[PROBA_COLS].to_numpy(np.float32))
        test_parts.append(test[PROBA_COLS].to_numpy(np.float32))

    assert ref_test_id is not None and y is not None
    return np.hstack(oof_parts), np.hstack(test_parts), y, ref_test_id


def evaluate_stacker(
    x_oof: np.ndarray,
    y: np.ndarray,
    c_value: float,
    class_weight: str | None,
    folds: int,
    random_state: int,
) -> tuple[float, np.ndarray]:
    model = LogisticRegression(
        max_iter=3000,
        C=c_value,
        class_weight=class_weight,
        multi_class="auto",
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    pred_proba = cross_val_predict(model, x_oof, y, cv=cv, method="predict_proba")
    score = balanced_accuracy_score(y, pred_proba.argmax(axis=1))
    return float(score), pred_proba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgb", "xgb2_seed63"],
        choices=sorted(DEFAULT_MODELS),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=63)
    parser.add_argument("--output", type=Path, default=Path("output/submission_probability_stacker.csv"))
    args = parser.parse_args()

    x_oof, x_test, y, test_id = build_stack(args.models)
    print(f"models={args.models}")
    print(f"x_oof={x_oof.shape}, x_test={x_test.shape}")

    candidates: list[tuple[float, str | None]] = []
    for c_value in [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]:
        candidates.append((c_value, None))
        candidates.append((c_value, "balanced"))

    best_score = -1.0
    best_params: tuple[float, str | None] | None = None
    best_oof: np.ndarray | None = None
    for c_value, class_weight in candidates:
        score, oof_proba = evaluate_stacker(
            x_oof,
            y,
            c_value,
            class_weight,
            args.folds,
            args.random_state,
        )
        print(f"C={c_value:g}, class_weight={class_weight}: oof={score:.6f}")
        if score > best_score:
            best_score = score
            best_params = (c_value, class_weight)
            best_oof = oof_proba

    assert best_params is not None and best_oof is not None
    print(f"best_oof={best_score:.6f}, C={best_params[0]:g}, class_weight={best_params[1]}")

    final_model = LogisticRegression(
        max_iter=3000,
        C=best_params[0],
        class_weight=best_params[1],
        multi_class="auto",
    )
    final_model.fit(x_oof, y)
    test_proba = final_model.predict_proba(x_test)
    pred = test_proba.argmax(axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": test_id, "class": [I2C[int(idx)] for idx in pred]})
    sub.to_csv(args.output, index=False)

    suffix = args.output.stem.replace("submission_", "")
    oof_path = args.output.with_name(f"oof_{suffix}.csv")
    test_path = args.output.with_name(f"test_proba_{suffix}.csv")
    oof_df = pd.DataFrame(best_oof, columns=PROBA_COLS)
    oof_df.insert(0, "id", np.arange(len(best_oof)))
    oof_df["target"] = [I2C[int(idx)] for idx in y]
    oof_df.to_csv(oof_path, index=False)
    test_df = pd.DataFrame(test_proba, columns=PROBA_COLS)
    test_df.insert(0, "id", test_id)
    test_df.to_csv(test_path, index=False)

    print(f"wrote={args.output}")
    print(f"wrote={oof_path}")
    print(f"wrote={test_path}")
    print(sub["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
