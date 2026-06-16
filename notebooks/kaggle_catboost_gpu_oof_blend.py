"""
Kaggle Notebook用:

1. Competition dataset を Input に追加
   /kaggle/input/competitions/playground-series-s6e6/

2. Original dataset を Input に追加
   Stellar Classification Dataset - SDSS17
   star_classification.csv を含む dataset

3. 可能なら外部5モデルの確率CSVも Input に追加
   - oof_external_5model_blend.csv
   - test_proba_external_5model_blend.csv

このセルをKaggle Notebookに貼って実行すると、以下を保存します。

- oof_catboost_gpu_diverse.csv
- test_proba_catboost_gpu_diverse.csv
- submission_catboost_gpu_diverse.csv
- submission_external_catboost_blend.csv  (外部5モデル確率が見つかった場合)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


SEED = 63
FOLDS = 5
ORIGINAL_WEIGHT = 0.25
CLASSES = ["GALAXY", "QSO", "STAR"]
PROBA_COLS = [f"proba_{c}" for c in CLASSES]


def find_file(name: str) -> Path:
    roots = [Path("/kaggle/input"), Path(".")]
    for root in roots:
        if not root.exists():
            continue
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find {name}")


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


def load_original() -> pd.DataFrame:
    path = find_file("star_classification.csv")
    orig = pd.read_csv(path)
    orig.columns = [c.strip() for c in orig.columns]
    keep = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift", "class"]
    missing = [c for c in keep if c not in orig.columns]
    if missing:
        raise ValueError(f"Original data missing columns: {missing}")
    orig = orig[keep].copy()
    orig.insert(0, "id", -np.arange(1, len(orig) + 1))
    return add_discussion_categories(orig)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_discussion_categories(df)
    eps = 1e-6

    color_pairs = [
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("u", "r"),
        ("u", "z"),
        ("g", "i"),
        ("g", "z"),
        ("r", "z"),
    ]
    for a, b in color_pairs:
        out[f"{a}_{b}_color"] = (out[a] - out[b]).astype(np.float32)

    one_plus_z = 1.0 + out["redshift"].astype(float)
    for a, b in color_pairs:
        col = f"{a}_{b}_color"
        out[f"{col}_rest"] = (out[col] / one_plus_z).astype(np.float32)
        out[f"{col}_x_redshift"] = (out[col] * out["redshift"]).astype(np.float32)

    bands = ["u", "g", "r", "i", "z"]
    out["mag_mean"] = out[bands].mean(axis=1).astype(np.float32)
    out["mag_std"] = out[bands].std(axis=1).astype(np.float32)
    out["mag_min"] = out[bands].min(axis=1).astype(np.float32)
    out["mag_max"] = out[bands].max(axis=1).astype(np.float32)
    out["mag_range"] = (out["mag_max"] - out["mag_min"]).astype(np.float32)

    alpha_rad = np.deg2rad(out["alpha"].astype(float))
    delta_rad = np.deg2rad(out["delta"].astype(float))
    out["alpha_sin"] = np.sin(alpha_rad).astype(np.float32)
    out["alpha_cos"] = np.cos(alpha_rad).astype(np.float32)
    out["delta_sin"] = np.sin(delta_rad).astype(np.float32)
    out["delta_cos"] = np.cos(delta_rad).astype(np.float32)
    out["sphere_x"] = (out["delta_cos"] * out["alpha_cos"]).astype(np.float32)
    out["sphere_y"] = (out["delta_cos"] * out["alpha_sin"]).astype(np.float32)
    out["sphere_z"] = out["delta_sin"].astype(np.float32)

    out["green_valley_margin"] = (out["u"] - out["r"] - 2.2).astype(np.float32)
    out["green_valley_abs_margin"] = np.abs(out["green_valley_margin"]).astype(np.float32)

    out["spectral_galaxy_interaction"] = (
        out["spectral_type"].astype(str) + "_" + out["galaxy_population"].astype(str)
    )
    out["redshift_bin"] = pd.cut(
        out["redshift"],
        [-np.inf, 0, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, np.inf],
        labels=["z_neg", "z_0_0.05", "z_0.05_0.1", "z_0.1_0.2", "z_0.2_0.5",
                "z_0.5_1", "z_1_2", "z_2_3", "z_3_5", "z_5_plus"],
    ).astype(str)

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            out[col] = out[col].astype(str).fillna("missing")
    return out


def load_external_proba() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    try:
        oof_path = find_file("oof_external_5model_blend.csv")
        test_path = find_file("test_proba_external_5model_blend.csv")
    except FileNotFoundError:
        return None
    return pd.read_csv(oof_path), pd.read_csv(test_path)


train = pd.read_csv(find_file("train.csv"))
test = pd.read_csv(find_file("test.csv"))
sample = pd.read_csv(find_file("sample_submission.csv"))
orig = load_original()

le = LabelEncoder()
y_train = le.fit_transform(train["class"])
y_orig = le.transform(orig["class"])
labels = list(le.classes_)

train_fe = add_features(train)
test_fe = add_features(test)
orig_fe = add_features(orig)

drop_cols = ["id", "class"]
features = [c for c in train_fe.columns if c not in drop_cols]
cat_cols = [
    c for c in features
    if train_fe[c].dtype == object or c in ["spectral_type", "galaxy_population", "spectral_galaxy_interaction", "redshift_bin"]
]
cat_indices = [features.index(c) for c in cat_cols]

print("train:", train.shape)
print("test :", test.shape)
print("orig :", orig.shape)
print("features:", len(features))
print("cat features:", cat_cols)

oof = np.zeros((len(train), len(labels)), dtype=np.float64)
test_pred = np.zeros((len(test), len(labels)), dtype=np.float64)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (tr_idx, va_idx) in enumerate(skf.split(train_fe, y_train), 1):
    X_tr = pd.concat(
        [train_fe.iloc[tr_idx][features], orig_fe[features]],
        axis=0,
        ignore_index=True,
    )
    y_tr = np.concatenate([y_train[tr_idx], y_orig])
    X_va = train_fe.iloc[va_idx][features].reset_index(drop=True)
    y_va = y_train[va_idx]
    X_te = test_fe[features].reset_index(drop=True)

    sw = compute_sample_weight("balanced", y_tr).astype(np.float32)
    sw[len(tr_idx):] *= ORIGINAL_WEIGHT

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=3000,
        learning_rate=0.035,
        depth=7,
        l2_leaf_reg=8.0,
        random_strength=1.2,
        bagging_temperature=0.35,
        auto_class_weights=None,
        task_type="GPU",
        devices="0",
        random_seed=SEED + fold,
        verbose=250,
        early_stopping_rounds=250,
        allow_writing_files=False,
    )

    try:
        model.fit(
            Pool(X_tr, y_tr, cat_features=cat_indices, weight=sw),
            eval_set=Pool(X_va, y_va, cat_features=cat_indices),
            use_best_model=True,
        )
    except Exception as e:
        print("GPU failed, fallback to CPU:", repr(e))
        model.set_params(task_type="CPU", thread_count=-1)
        model.fit(
            Pool(X_tr, y_tr, cat_features=cat_indices, weight=sw),
            eval_set=Pool(X_va, y_va, cat_features=cat_indices),
            use_best_model=True,
        )

    oof[va_idx] = model.predict_proba(X_va)
    test_pred += model.predict_proba(X_te) / FOLDS
    fold_score = balanced_accuracy_score(y_va, oof[va_idx].argmax(1))
    print(f"fold {fold}: BA={fold_score:.6f}")

raw_oof = balanced_accuracy_score(y_train, oof.argmax(1))
print("CatBoost OOF:", raw_oof)

oof_df = pd.DataFrame(oof, columns=PROBA_COLS)
oof_df.insert(0, "id", train["id"])
oof_df["target"] = train["class"].values
oof_df.to_csv("oof_catboost_gpu_diverse.csv", index=False)

test_df = pd.DataFrame(test_pred, columns=PROBA_COLS)
test_df.insert(0, "id", test["id"])
test_df.to_csv("test_proba_catboost_gpu_diverse.csv", index=False)

sub = sample.copy()
sub["class"] = le.inverse_transform(test_pred.argmax(1))
sub.to_csv("submission_catboost_gpu_diverse.csv", index=False)
print("saved CatBoost files")
print(sub["class"].value_counts())


external = load_external_proba()
if external is not None:
    ext_oof, ext_test = external
    ext_oof = ext_oof.sort_values("id").reset_index(drop=True)
    ext_test = ext_test.sort_values("id").reset_index(drop=True)
    oof_df_sorted = oof_df.sort_values("id").reset_index(drop=True)
    test_df_sorted = test_df.sort_values("id").reset_index(drop=True)

    y_eval = ext_oof["target"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    ext_p = ext_oof[PROBA_COLS].to_numpy()
    cat_p = oof_df_sorted[PROBA_COLS].to_numpy()
    ext_tp = ext_test[PROBA_COLS].to_numpy()
    cat_tp = test_df_sorted[PROBA_COLS].to_numpy()

    best = (-1, None, None, None)
    for w_cat in np.arange(0.00, 0.251, 0.01):
        p = (1 - w_cat) * ext_p + w_cat * cat_p
        for qso in np.arange(1.00, 1.161, 0.02):
            for star in np.arange(1.00, 1.261, 0.02):
                tuned = p * np.array([1.0, qso, star])
                score = balanced_accuracy_score(y_eval, tuned.argmax(1))
                if score > best[0]:
                    best = (score, w_cat, qso, star)

    score, w_cat, qso, star = best
    print(f"best external+cat blend OOF={score:.6f}, cat_weight={w_cat:.3f}, qso={qso:.3f}, star={star:.3f}")
    final_test = ((1 - w_cat) * ext_tp + w_cat * cat_tp) * np.array([1.0, qso, star])
    blend = sample.copy()
    blend["class"] = [CLASSES[i] for i in final_test.argmax(1)]
    blend.to_csv("submission_external_catboost_blend.csv", index=False)
    print("saved submission_external_catboost_blend.csv")
    print(blend["class"].value_counts())
