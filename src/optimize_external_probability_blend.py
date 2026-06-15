from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ProbaModel:
    name: str
    oof_path: Path
    test_path: Path


MODELS = [
    ProbaModel(
        "external5",
        Path("output/oof_external_5model_blend.csv"),
        Path("output/test_proba_external_5model_blend.csv"),
    ),
    ProbaModel(
        "lgb10",
        Path("output/oof_lgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_10fold_average.csv"),
        Path("output/test_proba_lgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_10fold_average.csv"),
    ),
    ProbaModel(
        "xgb2",
        Path("output/oof_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child3_lambda3_seed63.csv"),
        Path("output/test_proba_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child3_lambda3_seed63.csv"),
    ),
    ProbaModel(
        "xgb_seed71",
        Path("output/oof_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child3_lambda3_seed71.csv"),
        Path("output/test_proba_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child3_lambda3_seed71.csv"),
    ),
    ProbaModel(
        "xgb_green",
        Path("output/oof_xgb_original_w0p25_spatial_all_density_notebook_features_green_valley_target_encoding_5fold_average_depth5_child3_lambda3_seed63.csv"),
        Path("output/test_proba_xgb_original_w0p25_spatial_all_density_notebook_features_green_valley_target_encoding_5fold_average_depth5_child3_lambda3_seed63.csv"),
    ),
    ProbaModel(
        "cat",
        Path("output/oof_catboost_original_w0p25_te_5fold.csv"),
        Path("output/test_proba_catboost_original_w0p25_te_5fold.csv"),
    ),
]


def load_model(model: ProbaModel) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not model.oof_path.exists() or not model.test_path.exists():
        raise FileNotFoundError(f"{model.name}: missing probability files")
    oof = pd.read_csv(model.oof_path).sort_values("id").reset_index(drop=True)
    test = pd.read_csv(model.test_path).sort_values("id").reset_index(drop=True)
    if "target" not in oof.columns:
        raise ValueError(f"{model.name}: OOF file needs target column")
    for df, kind in [(oof, "oof"), (test, "test")]:
        missing = {"id", *PROBA_COLS} - set(df.columns)
        if missing:
            raise ValueError(f"{model.name} {kind}: missing {sorted(missing)}")
    return oof, test


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, 1e-12, None)
    return proba / proba.sum(axis=1, keepdims=True)


def score(proba: np.ndarray, y: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, proba.argmax(axis=1)))


def write_submission(
    name: str,
    test_id: np.ndarray,
    test_proba: np.ndarray,
    oof_id: np.ndarray,
    oof_proba: np.ndarray,
    y: np.ndarray,
    baseline_pred: np.ndarray,
) -> None:
    out_dir = Path("output")
    pred = test_proba.argmax(axis=1)
    sub = pd.DataFrame({"id": test_id, "class": [I2C[int(idx)] for idx in pred]})
    sub_path = out_dir / f"submission_{name}.csv"
    sub.to_csv(sub_path, index=False)

    oof_df = pd.DataFrame(oof_proba, columns=PROBA_COLS)
    oof_df.insert(0, "id", oof_id)
    oof_df["target"] = [I2C[int(idx)] for idx in y]
    oof_df.to_csv(out_dir / f"oof_{name}.csv", index=False)

    test_df = pd.DataFrame(test_proba, columns=PROBA_COLS)
    test_df.insert(0, "id", test_id)
    test_df.to_csv(out_dir / f"test_proba_{name}.csv", index=False)

    changed = int((pred != baseline_pred).sum())
    print(f"saved {sub_path} | changed_vs_external={changed:,} | counts={sub['class'].value_counts().to_dict()}")


def main() -> None:
    loaded = []
    ref_oof_id = ref_test_id = None
    y = None
    for model in MODELS:
        oof, test = load_model(model)
        if ref_oof_id is None:
            ref_oof_id = oof["id"].to_numpy()
            ref_test_id = test["id"].to_numpy()
            y = oof["target"].map(C2I).to_numpy(np.int8)
        else:
            if not np.array_equal(ref_oof_id, oof["id"].to_numpy()):
                raise ValueError(f"{model.name}: OOF id mismatch")
            if not np.array_equal(ref_test_id, test["id"].to_numpy()):
                raise ValueError(f"{model.name}: test id mismatch")
            if not np.array_equal(y, oof["target"].map(C2I).to_numpy(np.int8)):
                raise ValueError(f"{model.name}: target mismatch")
        loaded.append((model.name, oof[PROBA_COLS].to_numpy(np.float64), test[PROBA_COLS].to_numpy(np.float64)))

    assert ref_oof_id is not None and ref_test_id is not None and y is not None
    external_oof = loaded[0][1]
    external_test = loaded[0][2]
    external_test_pred = external_test.argmax(axis=1)

    print("Individual OOF:")
    for name, oof_proba, _ in loaded:
        print(f"  {name:12s}: {score(oof_proba, y):.6f}")
    print(f"external baseline: {score(external_oof, y):.6f}")

    # 1) External-only calibration. This is the lowest-risk candidate because it
    # only changes rows near the current decision boundaries.
    best = (-1.0, 1.0, 1.0, 1.0)
    for gamma in np.arange(0.90, 1.101, 0.05):
        powered = normalize_proba(external_oof**gamma)
        for qso in np.arange(0.94, 1.141, 0.02):
            for star in np.arange(0.96, 1.261, 0.02):
                mult = np.array([1.0, qso, star])
                tuned = normalize_proba(powered * mult)
                s = score(tuned, y)
                if s > best[0]:
                    best = (s, gamma, qso, star)
    best_score, best_gamma, best_qso, best_star = best
    print(
        "best external calibration: "
        f"oof={best_score:.6f}, gamma={best_gamma:.3f}, qso={best_qso:.3f}, star={best_star:.3f}"
    )
    ext_cal_oof = normalize_proba((external_oof**best_gamma) * np.array([1.0, best_qso, best_star]))
    ext_cal_test = normalize_proba((external_test**best_gamma) * np.array([1.0, best_qso, best_star]))
    write_submission(
        f"external5_calibrated_g{best_gamma:.3f}_q{best_qso:.3f}_s{best_star:.3f}".replace(".", "p"),
        ref_test_id,
        ext_cal_test,
        ref_oof_id,
        ext_cal_oof,
        y,
        external_test_pred,
    )

    # 2) Tiny probability blend around the external model. The search is
    # intentionally constrained: local models are allowed to help only a little.
    model_names = [name for name, _, _ in loaded]
    oof_stack = np.stack([oof for _, oof, _ in loaded], axis=0)
    test_stack = np.stack([test for _, _, test in loaded], axis=0)
    rng = np.random.default_rng(63)
    best_blend = (-1.0, None, None, None)
    candidates = [
        np.array([0.90, 0.04, 0.03, 0.02, 0.01, 0.00]),
        np.array([0.85, 0.06, 0.05, 0.02, 0.02, 0.00]),
        np.array([0.80, 0.10, 0.06, 0.02, 0.02, 0.00]),
        np.array([0.75, 0.12, 0.08, 0.03, 0.02, 0.00]),
    ]
    for _ in range(80):
        tail = rng.dirichlet(np.ones(len(loaded) - 1)) * rng.uniform(0.02, 0.25)
        w = np.r_[1.0 - tail.sum(), tail]
        candidates.append(w)

    class_grid = [(1.0, 1.0), (1.06, 1.10), (1.10, 1.20)]
    gamma_grid = [0.95, 1.00]
    for w in candidates:
        blend_oof_raw = np.tensordot(w, oof_stack, axes=(0, 0))
        for gamma in gamma_grid:
            powered = normalize_proba(blend_oof_raw**gamma)
            for qso, star in class_grid:
                tuned = normalize_proba(powered * np.array([1.0, qso, star]))
                s = score(tuned, y)
                if s > best_blend[0]:
                    best_blend = (s, w.copy(), gamma, (qso, star))
    blend_score, blend_w, blend_gamma, blend_class = best_blend
    assert blend_w is not None and blend_class is not None
    print(f"best constrained blend: oof={blend_score:.6f}, gamma={blend_gamma:.3f}, qso={blend_class[0]:.3f}, star={blend_class[1]:.3f}")
    for name, w in zip(model_names, blend_w):
        print(f"  {name:12s}: {w:.5f}")
    blend_oof = normalize_proba((np.tensordot(blend_w, oof_stack, axes=(0, 0)) ** blend_gamma) * np.array([1.0, *blend_class]))
    blend_test = normalize_proba((np.tensordot(blend_w, test_stack, axes=(0, 0)) ** blend_gamma) * np.array([1.0, *blend_class]))
    write_submission(
        "external5_constrained_probability_blend",
        ref_test_id,
        blend_test,
        ref_oof_id,
        blend_oof,
        y,
        external_test_pred,
    )

    # 3) Meta-stacker diagnostic. This usually changes more rows, so treat it as
    # a higher-risk candidate even if OOF looks good.
    x_oof = np.hstack([oof for _, oof, _ in loaded])
    x_test = np.hstack([test for _, _, test in loaded])
    best_lr = (-1.0, None, None, None)
    for c_value in [0.03, 0.3, 3.0]:
        for class_weight in [None, "balanced"]:
            lr = LogisticRegression(max_iter=3000, C=c_value, class_weight=class_weight)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=63)
            oof_pred = cross_val_predict(lr, x_oof, y, cv=cv, method="predict_proba")
            s = score(oof_pred, y)
            print(f"stacker C={c_value:g}, class_weight={class_weight}: oof={s:.6f}")
            if s > best_lr[0]:
                best_lr = (s, c_value, class_weight, oof_pred)
    lr_score, c_value, class_weight, lr_oof = best_lr
    assert lr_oof is not None
    print(f"best stacker: oof={lr_score:.6f}, C={c_value:g}, class_weight={class_weight}")
    final_lr = LogisticRegression(max_iter=3000, C=c_value, class_weight=class_weight)
    final_lr.fit(x_oof, y)
    lr_test = final_lr.predict_proba(x_test)
    write_submission(
        "external5_local_probability_stacker",
        ref_test_id,
        lr_test,
        ref_oof_id,
        lr_oof,
        y,
        external_test_pred,
    )


if __name__ == "__main__":
    main()
