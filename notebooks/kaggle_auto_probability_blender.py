"""
Kaggle Notebook用: OOF/test probability auto blender

目的:
- 外部5モデルなどの OOF/test 確率ファイルを Input に追加する
- このコードを実行する
- 使える確率ファイルを自動検出して、OOF balanced accuracy が高くなるblendを作る

対応する確率ファイル:
- CSV:
  - OOF: id + proba_GALAXY/proba_QSO/proba_STAR + target
  - Test: id + proba_GALAXY/proba_QSO/proba_STAR
- NPY:
  - OOF: shape = (577347, 3)
  - Test: shape = (247435, 3)

出力:
- submission_auto_probability_blend.csv
- oof_auto_probability_blend.csv
- test_proba_auto_probability_blend.csv
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


SEED = 63
CLASSES = ["GALAXY", "QSO", "STAR"]
C2I = {c: i for i, c in enumerate(CLASSES)}
I2C = {i: c for c, i in C2I.items()}
PROBA_COLS = [f"proba_{c}" for c in CLASSES]
KAGGLE_INPUT = Path("/kaggle/input")


def find_file(name: str) -> Path:
    for root in [KAGGLE_INPUT, Path(".")]:
        if not root.exists():
            continue
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def load_competition():
    train = pd.read_csv(find_file("train.csv"))
    test = pd.read_csv(find_file("test.csv"))
    sample = pd.read_csv(find_file("sample_submission.csv"))
    y = train["class"].map(C2I).to_numpy(np.int8)
    return train, test, sample, y


def score(p: np.ndarray, y: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, p.argmax(axis=1)))


def csv_kind(path: Path) -> str | None:
    try:
        cols = pd.read_csv(path, nrows=1).columns
    except Exception:
        return None
    has_proba = set(PROBA_COLS).issubset(cols)
    if not has_proba:
        return None
    if "target" in cols:
        return "oof"
    if "id" in cols:
        return "test"
    return None


def npy_kind(path: Path, n_train: int, n_test: int) -> str | None:
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    if arr.shape[0] == n_train:
        return "oof"
    if arr.shape[0] == n_test:
        return "test"
    return None


def stem_key(path: Path) -> str:
    s = path.stem.lower()
    s = re.sub(r"^(oof|train_oof|train|test_proba|test_preds|test|submission)[_-]*", "", s)
    s = re.sub(r"(_?oof|_?test|_?preds|_?proba|_?predictions)$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def discover_probability_pairs(n_train: int, n_test: int):
    oof_files: dict[str, Path] = {}
    test_files: dict[str, Path] = {}

    for path in sorted(KAGGLE_INPUT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".csv":
            kind = csv_kind(path)
        elif path.suffix.lower() == ".npy":
            kind = npy_kind(path, n_train, n_test)
        else:
            continue
        if kind is None:
            continue
        key = stem_key(path)
        if kind == "oof":
            oof_files[key] = path
        else:
            test_files[key] = path

    pairs = []
    for key, oof_path in oof_files.items():
        if key in test_files:
            pairs.append((key, oof_path, test_files[key]))
            continue
        # forgiving match for notebook folders where names differ slightly
        for test_key, test_path in test_files.items():
            if key in test_key or test_key in key:
                pairs.append((key, oof_path, test_path))
                break

    # Known public notebook artifact names. Some notebooks use short names such
    # as oof_lgb.npy / test_lgb.npy, which are easy to miss if a folder contains
    # several unrelated files. Add these pairs explicitly when present.
    known_pairs = {
        "lgb_public": ("oof_lgb.npy", "test_lgb.npy"),
        "mlp_public": ("oof_mlp.npy", "test_mlp.npy"),
        "external5": ("oof_external_5model_blend.csv", "test_proba_external_5model_blend.csv"),
    }
    existing_oof_paths = {str(oof_path) for _, oof_path, _ in pairs}
    for key, (oof_name, test_name) in known_pairs.items():
        try:
            oof_path = find_file(oof_name)
            test_path = find_file(test_name)
        except FileNotFoundError:
            continue
        if str(oof_path) not in existing_oof_paths:
            pairs.append((key, oof_path, test_path))
            existing_oof_paths.add(str(oof_path))
    return pairs


def load_proba(path: Path, train: pd.DataFrame, test: pd.DataFrame, kind: str):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "id" in df.columns:
            ref = train if kind == "oof" else test
            df = df.sort_values("id").reset_index(drop=True)
            ref_id = ref["id"].sort_values().to_numpy()
            if not np.array_equal(df["id"].to_numpy(), ref_id):
                raise ValueError(f"id mismatch: {path}")
        return normalize(df[PROBA_COLS].to_numpy())

    arr = np.load(path)
    if kind == "oof" and arr.shape[0] > len(train):
        print(f"trim augmented OOF {path.name}: {arr.shape[0]} -> {len(train)}")
        arr = arr[: len(train)]
    return normalize(arr)


def optimize_weighted_average(oof_list, y):
    n = len(oof_list)
    stack = np.stack(oof_list, axis=0)

    def objective(x):
        raw_w = np.exp(x[:n])
        w = raw_w / raw_w.sum()
        qso = np.exp(x[n])
        star = np.exp(x[n + 1])
        gamma = np.exp(x[n + 2])
        p = np.tensordot(w, stack, axes=(0, 0))
        p = normalize((p**gamma) * np.array([1.0, qso, star]))
        return -score(p, y)

    bounds = [(-4, 4)] * n + [(-0.25, 0.25), (-0.35, 0.35), (-0.20, 0.20)]
    result = differential_evolution(
        objective,
        bounds,
        seed=SEED,
        popsize=8,
        maxiter=80,
        tol=1e-5,
        polish=True,
        workers=1,
    )

    x = result.x
    w = np.exp(x[:n])
    w = w / w.sum()
    qso = float(np.exp(x[n]))
    star = float(np.exp(x[n + 1]))
    gamma = float(np.exp(x[n + 2]))
    best = -float(result.fun)
    return best, w, qso, star, gamma


def make_blend(oof_list, test_list, w, qso, star, gamma):
    oof_stack = np.stack(oof_list, axis=0)
    test_stack = np.stack(test_list, axis=0)
    oof = np.tensordot(w, oof_stack, axes=(0, 0))
    tst = np.tensordot(w, test_stack, axes=(0, 0))
    mult = np.array([1.0, qso, star])
    return normalize((oof**gamma) * mult), normalize((tst**gamma) * mult)


def fit_meta_stacker(oof_list, test_list, y):
    X = np.hstack(oof_list)
    Xt = np.hstack(test_list)
    best = (-1.0, None, None, None)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for C in [0.03, 0.1, 0.3, 1.0, 3.0]:
        for class_weight in [None, "balanced"]:
            lr = LogisticRegression(max_iter=3000, C=C, class_weight=class_weight)
            pred = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")
            s = score(pred, y)
            print(f"stacker C={C:g}, class_weight={class_weight}: OOF={s:.6f}")
            if s > best[0]:
                best = (s, C, class_weight, pred)

    _, C, class_weight, oof_pred = best
    final = LogisticRegression(max_iter=3000, C=C, class_weight=class_weight)
    final.fit(X, y)
    test_pred = final.predict_proba(Xt)
    return best[0], C, class_weight, normalize(oof_pred), normalize(test_pred)


train, test, sample, y = load_competition()
pairs = discover_probability_pairs(len(train), len(test))

print(f"discovered pairs: {len(pairs)}")
for key, oof_path, test_path in pairs:
    print(f"  {key}:")
    print(f"    oof : {oof_path}")
    print(f"    test: {test_path}")

names = []
oof_list = []
test_list = []
for key, oof_path, test_path in pairs:
    try:
        oof_p = load_proba(oof_path, train, test, "oof")
        test_p = load_proba(test_path, train, test, "test")
    except Exception as e:
        print(f"skip {key}: {e}")
        continue
    if oof_p.shape != (len(train), 3) or test_p.shape != (len(test), 3):
        print(f"skip {key}: shape mismatch {oof_p.shape}, {test_p.shape}")
        continue
    names.append(key)
    oof_list.append(oof_p)
    test_list.append(test_p)
    print(f"{key}: OOF BA={score(oof_p, y):.6f}")

if len(oof_list) < 2:
    raise RuntimeError("Need at least two OOF/test probability pairs. Add more probability files as Kaggle Input.")

avg_score, w, qso, star, gamma = optimize_weighted_average(oof_list, y)
blend_oof, blend_test = make_blend(oof_list, test_list, w, qso, star, gamma)

print("\nWeighted average blend")
print(f"OOF BA={avg_score:.6f}, qso={qso:.4f}, star={star:.4f}, gamma={gamma:.4f}")
for name, weight in sorted(zip(names, w), key=lambda x: -x[1]):
    print(f"  {name:35s}: {weight:.5f}")

stack_score, C, class_weight, stack_oof, stack_test = fit_meta_stacker(oof_list, test_list, y)
print("\nMeta stacker")
print(f"OOF BA={stack_score:.6f}, C={C}, class_weight={class_weight}")

if stack_score > avg_score:
    final_name = "auto_probability_stacker"
    final_oof = stack_oof
    final_test = stack_test
    final_score = stack_score
else:
    final_name = "auto_probability_blend"
    final_oof = blend_oof
    final_test = blend_test
    final_score = avg_score

print(f"\nSelected: {final_name}, OOF BA={final_score:.6f}")

oof_df = pd.DataFrame(final_oof, columns=PROBA_COLS)
oof_df.insert(0, "id", train["id"])
oof_df["target"] = train["class"].values
oof_df.to_csv(f"oof_{final_name}.csv", index=False)

test_df = pd.DataFrame(final_test, columns=PROBA_COLS)
test_df.insert(0, "id", test["id"])
test_df.to_csv(f"test_proba_{final_name}.csv", index=False)

sub = sample.copy()
sub["class"] = [I2C[int(i)] for i in final_test.argmax(axis=1)]
sub.to_csv(f"submission_{final_name}.csv", index=False)

print(f"saved submission_{final_name}.csv")
print(sub["class"].value_counts())


# Final polish: hard-vote candidates.
# These are riskier than probability blending, but sometimes work better on
# public LB because they only alter rows where models disagree.
label_mat = np.column_stack([p.argmax(axis=1) for p in test_list])
model_scores = np.array([score(p, y) for p in oof_list], dtype=np.float64)
model_weights = np.exp((model_scores - model_scores.max()) / 0.0005)
model_weights = model_weights / model_weights.sum()

final_label = final_test.argmax(axis=1)
vote_scores = np.zeros((len(test), len(CLASSES)), dtype=np.float64)
for j in range(label_mat.shape[1]):
    np.add.at(vote_scores, (np.arange(len(test)), label_mat[:, j]), model_weights[j])
weighted_vote_label = vote_scores.argmax(axis=1)

agreement_with_vote = (label_mat == weighted_vote_label[:, None]).sum(axis=1)
final_margin = np.sort(final_test, axis=1)[:, -1] - np.sort(final_test, axis=1)[:, -2]

hard_weighted = sample.copy()
hard_weighted["class"] = [I2C[int(i)] for i in weighted_vote_label]
hard_weighted.to_csv("submission_hard_vote_weighted.csv", index=False)

conservative_label = final_label.copy()
change_mask = (
    (weighted_vote_label != final_label)
    & (agreement_with_vote >= max(3, len(names) - 1))
    & (final_margin < 0.20)
)
conservative_label[change_mask] = weighted_vote_label[change_mask]
hard_conservative = sample.copy()
hard_conservative["class"] = [I2C[int(i)] for i in conservative_label]
hard_conservative.to_csv("submission_hard_vote_conservative.csv", index=False)

print("\nHard vote candidates")
print("model hard-vote weights:")
for name, s, w in sorted(zip(names, model_scores, model_weights), key=lambda x: -x[2]):
    print(f"  {name:35s}: oof={s:.6f}, weight={w:.5f}")
print(f"weighted hard vote saved: submission_hard_vote_weighted.csv")
print(hard_weighted["class"].value_counts())
print(
    "conservative hard vote saved: submission_hard_vote_conservative.csv "
    f"| changed_vs_probability_blend={int(change_mask.sum())}"
)
print(hard_conservative["class"].value_counts())


# Public LB probe candidates.
# The OOF optimum is already very tight, so these files intentionally make
# small class-prior and sharpness moves around the selected probability blend.
def adjusted_proba(p: np.ndarray, qso_mult: float, star_mult: float, gamma_mult: float) -> np.ndarray:
    mult = np.array([1.0, qso_mult, star_mult], dtype=np.float64)
    return normalize((p**gamma_mult) * mult)


probe_grid = []
for qso_mult in [0.970, 0.985, 1.000, 1.015, 1.030, 1.045]:
    for star_mult in [0.940, 0.960, 0.980, 1.000, 1.020, 1.040]:
        for gamma_mult in [0.985, 1.000, 1.015]:
            if qso_mult == 1.0 and star_mult == 1.0 and gamma_mult == 1.0:
                continue
            adj_oof = adjusted_proba(final_oof, qso_mult, star_mult, gamma_mult)
            probe_grid.append(
                (
                    score(adj_oof, y),
                    qso_mult,
                    star_mult,
                    gamma_mult,
                )
            )

# Save focused Public LB probes in submit order.
probe_grid = sorted(probe_grid, key=lambda x: -x[0])
submit_probes = [
    # Safest QSO-up candidates. These move enough rows to matter without
    # drifting far from the OOF optimum.
    ("submit_01_qso_up_safe", 1.030, 1.000, 0.985),
    ("submit_02_qso_up_mid", 1.045, 1.000, 1.000),
    # Different direction: slightly fewer QSO/STAR with a sharper distribution.
    ("submit_03_star_down_alt", 0.990, 0.970, 1.010),
    # More aggressive STAR-down candidate.
    ("submit_04_star_down_attack", 1.025, 0.965, 1.000),
    # Keep the previous conservative hard vote as a named submit candidate too.
    ("submit_05_hard_vote_conservative", None, None, None),
]

print("\nPublic LB submit candidates")
for tag, qso_mult, star_mult, gamma_mult in submit_probes:
    if qso_mult is None:
        filename = f"submission_{tag}.csv"
        hard_conservative.to_csv(filename, index=False)
        print(f"{filename}: copied conservative hard vote, changed_vs_final={int(change_mask.sum())}")
        continue

    adj_oof = adjusted_proba(final_oof, qso_mult, star_mult, gamma_mult)
    adj_test = adjusted_proba(final_test, qso_mult, star_mult, gamma_mult)
    s = score(adj_oof, y)
    probe_sub = sample.copy()
    probe_sub["class"] = [I2C[int(k)] for k in adj_test.argmax(axis=1)]
    filename = (
        f"submission_{tag}_"
        f"oof{s:.6f}_q{qso_mult:.3f}_star{star_mult:.3f}_gamma{gamma_mult:.3f}.csv"
    )
    probe_sub.to_csv(filename, index=False)
    changed = int((adj_test.argmax(axis=1) != final_label).sum())
    print(
        f"{filename}: OOF={s:.6f}, changed_vs_final={changed}, "
        f"qso={qso_mult:.3f}, star={star_mult:.3f}, gamma={gamma_mult:.3f}"
    )

print("\nExtra OOF-safe probe reference")
for rank, (s, qso_mult, star_mult, gamma_mult) in enumerate(probe_grid[:10], start=1):
    adj_test = adjusted_proba(final_test, qso_mult, star_mult, gamma_mult)
    changed = int((adj_test.argmax(axis=1) != final_label).sum())
    print(
        f"rank={rank:02d}, OOF={s:.6f}, changed_vs_final={changed}, "
        f"qso={qso_mult:.3f}, star={star_mult:.3f}, gamma={gamma_mult:.3f}"
    )
