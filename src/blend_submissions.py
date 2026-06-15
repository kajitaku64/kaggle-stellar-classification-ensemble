from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ["GALAXY", "QSO", "STAR"]
C2I = {label: idx for idx, label in enumerate(CLASSES)}
I2C = {idx: label for label, idx in C2I.items()}


def parse_score(path: Path) -> float:
    match = re.match(r"^(\d+(?:\.\d+)?)", path.stem)
    if not match:
        raise ValueError(f"{path.name}: filename must start with a score, e.g. 0.96883_best.csv")
    return float(match.group(1))


def make_weights(scores: np.ndarray, mode: str, temperature: float) -> np.ndarray:
    scores = scores.astype(float)
    if mode == "raw":
        return scores.copy()
    if mode == "shifted":
        return scores - scores.min() + 1e-3
    if mode == "softmax":
        exp_scores = np.exp((scores - scores.max()) / temperature)
        return exp_scores / exp_scores.sum()
    raise ValueError(f"Unknown weight mode: {mode}")


def uniqueness(label_matrix: np.ndarray) -> np.ndarray:
    n_models = label_matrix.shape[1]
    if n_models == 1:
        return np.ones(1, dtype=float)
    agreement = np.array(
        [
            [(label_matrix[:, left] == label_matrix[:, right]).mean() for right in range(n_models)]
            for left in range(n_models)
        ],
        dtype=float,
    )
    np.fill_diagonal(agreement, np.nan)
    mean_agreement = np.nanmean(agreement, axis=1)
    uniq = 1.0 / mean_agreement
    return uniq / uniq.mean()


def weighted_vote(label_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = np.zeros((label_matrix.shape[0], len(CLASSES)), dtype=np.float64)
    for class_idx in range(len(CLASSES)):
        scores[:, class_idx] = ((label_matrix == class_idx) * weights).sum(axis=1)
    return scores.argmax(axis=1)


def load_submissions(submissions_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    files = sorted(submissions_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV submissions found in {submissions_dir}")

    names: list[str] = []
    scores: list[float] = []
    labels: list[np.ndarray] = []
    ref_id: np.ndarray | None = None

    for path in files:
        score = parse_score(path)
        df = pd.read_csv(path)
        missing = {"id", "class"} - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        invalid = sorted(set(df["class"].dropna()) - set(CLASSES))
        if invalid:
            raise ValueError(f"{path.name}: unexpected classes {invalid}")
        if not df["id"].is_unique:
            raise ValueError(f"{path.name}: duplicate id values")

        df = df[["id", "class"]].sort_values("id").reset_index(drop=True)
        if ref_id is None:
            ref_id = df["id"].to_numpy()
        elif not np.array_equal(ref_id, df["id"].to_numpy()):
            raise ValueError(f"{path.name}: id mismatch")

        names.append(path.stem)
        scores.append(score)
        labels.append(df["class"].map(C2I).to_numpy(np.int8))

    assert ref_id is not None
    return ref_id, np.column_stack(labels), names, np.asarray(scores, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions-dir", type=Path, default=Path("output/submissions_for_blend"))
    parser.add_argument("--output", type=Path, default=Path("output/submission_hard_vote_blend.csv"))
    parser.add_argument("--weight-mode", choices=["raw", "shifted", "softmax"], default="softmax")
    parser.add_argument("--temperature", type=float, default=0.0005)
    parser.add_argument("--no-decorrelation", action="store_true")
    args = parser.parse_args()

    ids, label_matrix, names, scores = load_submissions(args.submissions_dir)
    base_weights = make_weights(scores, args.weight_mode, args.temperature)
    uniq = np.ones_like(base_weights) if args.no_decorrelation else uniqueness(label_matrix)
    weights = base_weights * uniq
    weights = weights / weights.sum()

    unanimous = (label_matrix == label_matrix[:, [0]]).all(axis=1)
    pred = weighted_vote(label_matrix, weights)
    pred[unanimous] = label_matrix[unanimous, 0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": ids, "class": [I2C[int(idx)] for idx in pred]})
    sub.to_csv(args.output, index=False)

    print(f"submissions={len(names)}, rows={len(ids)}")
    print(
        f"weight_mode={args.weight_mode}, temperature={args.temperature}, "
        f"decorrelation={not args.no_decorrelation}"
    )
    print(f"unanimous_rows={int(unanimous.sum())} ({unanimous.mean():.4%})")
    print(f"disagreement_rows={int((~unanimous).sum())} ({(~unanimous).mean():.4%})")
    print()
    for name, score, raw_w, uniq_w, final_w in zip(names, scores, base_weights, uniq, weights):
        counts = pd.Series(label_matrix[:, names.index(name)]).map(I2C).value_counts().to_dict()
        print(
            f"{name}: score={score:.5f} raw_w={raw_w:.6f} "
            f"uniq={uniq_w:.3f} final_w={final_w:.6f} counts={counts}"
        )
    print()
    print(f"wrote={args.output}")
    print(sub["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
