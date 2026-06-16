"""
Kaggle Notebook用: scored submission hard blender

目的:
- Input 内の submission CSV を集める
- ファイル名に含まれる Public LB score を重みにする
- 全員一致している行はそのまま
- 割れている行だけ weighted hard vote で決める

必要なInput例:
- competition data
- 0.97092.csv, 0.97080.csv ... のような scored submission CSV が入った dataset
- 自分の best submission を追加したい場合は、ファイル名を 0.97044_external5.csv のようにする

出力:
- submission_scored_hard_vote_softmax.csv
- submission_scored_hard_vote_shifted.csv
- submission_scored_hard_vote_topk.csv
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


CLASSES = ["GALAXY", "QSO", "STAR"]
C2I = {c: i for i, c in enumerate(CLASSES)}
I2C = {i: c for c, i in C2I.items()}
KAGGLE_INPUT = Path("/kaggle/input")


def find_file(name: str) -> Path:
    for root in [KAGGLE_INPUT, Path(".")]:
        if not root.exists():
            continue
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


def parse_score(path: Path) -> float | None:
    m = re.search(r"(0\.\d{4,6})", path.stem)
    if not m:
        return None
    return float(m.group(1))


def is_submission_csv(path: Path) -> bool:
    if path.name in {"train.csv", "test.csv", "sample_submission.csv"}:
        return False
    score = parse_score(path)
    if score is None:
        return False
    try:
        cols = pd.read_csv(path, nrows=2).columns
    except Exception:
        return False
    return {"id", "class"}.issubset(cols)


def make_weights(scores: np.ndarray, mode: str, temperature: float = 0.0004) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if mode == "raw":
        w = scores.copy()
    elif mode == "shifted":
        w = scores - scores.min() + 1e-4
    elif mode == "softmax":
        w = np.exp((scores - scores.max()) / temperature)
    else:
        raise ValueError(mode)
    return w / w.sum()


def uniqueness(label_matrix: np.ndarray) -> np.ndarray:
    n = label_matrix.shape[1]
    if n <= 1:
        return np.ones(n)
    agreement = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            agreement[i, j] = (label_matrix[:, i] == label_matrix[:, j]).mean()
    np.fill_diagonal(agreement, np.nan)
    mean_agreement = np.nanmean(agreement, axis=1)
    uniq = 1.0 / np.clip(mean_agreement, 1e-9, None)
    return uniq / uniq.mean()


def weighted_vote(label_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vote = np.zeros((label_matrix.shape[0], len(CLASSES)), dtype=np.float64)
    for j in range(label_matrix.shape[1]):
        np.add.at(vote, (np.arange(label_matrix.shape[0]), label_matrix[:, j]), weights[j])
    return vote.argmax(axis=1)


def save_submission(name: str, ids: np.ndarray, pred: np.ndarray) -> None:
    sub = pd.DataFrame({"id": ids, "class": [I2C[int(i)] for i in pred]})
    sub.to_csv(name, index=False)
    print(f"saved {name}")
    print(sub["class"].value_counts())


sample = pd.read_csv(find_file("sample_submission.csv"))
sample_ids = sample["id"].to_numpy()

files = [p for p in sorted(KAGGLE_INPUT.rglob("*.csv")) if is_submission_csv(p)]
if not files:
    raise RuntimeError("No scored submission CSVs found. Add a dataset with files like 0.97092.csv.")

names = []
scores = []
labels = []
for path in files:
    score = parse_score(path)
    assert score is not None
    df = pd.read_csv(path)[["id", "class"]].sort_values("id").reset_index(drop=True)
    if not np.array_equal(df["id"].to_numpy(), np.sort(sample_ids)):
        print(f"skip id mismatch: {path}")
        continue
    bad = sorted(set(df["class"].dropna()) - set(CLASSES))
    if bad:
        print(f"skip bad classes {bad}: {path}")
        continue
    names.append(path.stem)
    scores.append(score)
    labels.append(df["class"].map(C2I).to_numpy(np.int8))

if len(labels) < 2:
    raise RuntimeError("Need at least two valid scored submissions.")

scores = np.asarray(scores, dtype=np.float64)
label_matrix = np.column_stack(labels)
ids = np.sort(sample_ids)

print(f"valid submissions: {len(names)}")
for name, score in sorted(zip(names, scores), key=lambda x: -x[1]):
    print(f"  {score:.5f}  {name}")

unanimous = (label_matrix == label_matrix[:, [0]]).all(axis=1)
print(f"unanimous rows: {int(unanimous.sum())} ({unanimous.mean() * 100:.2f}%)")
print(f"disagree rows : {int((~unanimous).sum())} ({(~unanimous).mean() * 100:.2f}%)")

uniq = uniqueness(label_matrix)
for mode in ["softmax", "shifted"]:
    base_w = make_weights(scores, mode=mode)
    w = base_w * uniq
    w = w / w.sum()
    pred = weighted_vote(label_matrix, w)
    pred[unanimous] = label_matrix[unanimous, 0]
    print(f"\nmode={mode}")
    for name, score, weight in sorted(zip(names, scores, w), key=lambda x: -x[2]):
        print(f"  weight={weight:.5f} score={score:.5f} {name}")
    save_submission(f"submission_scored_hard_vote_{mode}.csv", ids, pred)

for k in [3, 5, 8, 12]:
    if len(scores) < k:
        continue
    keep = np.argsort(scores)[-k:]
    lm = label_matrix[:, keep]
    sc = scores[keep]
    un = (lm == lm[:, [0]]).all(axis=1)
    w = make_weights(sc, mode="softmax")
    pred = weighted_vote(lm, w)
    pred[un] = lm[un, 0]
    print(f"\ntop{k} softmax")
    for idx, weight in sorted(zip(keep, w), key=lambda x: -x[1]):
        print(f"  weight={weight:.5f} score={scores[idx]:.5f} {names[idx]}")
    save_submission(f"submission_scored_hard_vote_top{k}.csv", ids, pred)
