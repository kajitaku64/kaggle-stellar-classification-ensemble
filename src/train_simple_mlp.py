from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_tree_models import add_notebook_categorical_features, add_spatial_density_features, find_file, make_xy
from train_with_original import add_target_encoding, load_original


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"


class RobustScalerLite:
    def fit(self, x: np.ndarray) -> "RobustScalerLite":
        self.median_ = np.nanmedian(x, axis=0)
        q75 = np.nanpercentile(x, 75, axis=0)
        q25 = np.nanpercentile(x, 25, axis=0)
        scale = q75 - q25
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.median_) / self.scale_
        x = np.clip(x, -20, 20)
        return np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)


class TabularMLP(nn.Module):
    def __init__(self, n_num: int, cat_dims: list[int], emb_dim: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, min(emb_dim, max(2, (dim + 1) // 2))) for dim in cat_dims]
        )
        emb_total = sum(emb.embedding_dim for emb in self.embeddings)
        in_dim = n_num + emb_total
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if self.embeddings:
            emb = [layer(x_cat[:, idx]) for idx, layer in enumerate(self.embeddings)]
            x = torch.cat([x_num, *emb], dim=1)
        else:
            x = x_num
        return self.net(x)


def encode_categoricals(
    x_fit: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    fit_codes = []
    valid_codes = []
    test_codes = []
    cat_dims = []
    for col in cat_cols:
        combined = pd.concat([x_fit[col], x_valid[col], x_test[col]], axis=0).astype(str)
        _, uniques = pd.factorize(combined, sort=False)
        code_map = {value: idx + 1 for idx, value in enumerate(uniques)}
        fit_col = x_fit[col].astype(str).map(code_map).fillna(0).astype(np.int64).to_numpy()
        valid_col = x_valid[col].astype(str).map(code_map).fillna(0).astype(np.int64).to_numpy()
        test_col = x_test[col].astype(str).map(code_map).fillna(0).astype(np.int64).to_numpy()
        fit_codes.append(fit_col)
        valid_codes.append(valid_col)
        test_codes.append(test_col)
        cat_dims.append(len(uniques) + 1)
    if not cat_cols:
        return (
            np.zeros((len(x_fit), 0), dtype=np.int64),
            np.zeros((len(x_valid), 0), dtype=np.int64),
            np.zeros((len(x_test), 0), dtype=np.int64),
            [],
        )
    return (
        np.vstack(fit_codes).T,
        np.vstack(valid_codes).T,
        np.vstack(test_codes).T,
        cat_dims,
    )


def build_features(args: argparse.Namespace):
    train = pd.read_csv(find_file("train.csv"))
    test = pd.read_csv(find_file("test.csv"))
    original = load_original(args.original_data)
    target_col = "class"

    le = LabelEncoder()
    y_train = le.fit_transform(train[target_col])
    y_original = le.transform(original[target_col])

    x_train, x_test, _ = make_xy(
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
    train_original = pd.concat([train, original], axis=0, ignore_index=True)
    x_train_original = pd.concat([x_train, x_original], axis=0, ignore_index=True)
    x_train_original, x_test = add_notebook_categorical_features(train_original, test, x_train_original, x_test)
    x_train_original, x_test = add_spatial_density_features(train_original, test, x_train_original, x_test)
    x_train = x_train_original.iloc[: len(train)].reset_index(drop=True)
    x_original = x_train_original.iloc[len(train) :].reset_index(drop=True)
    return train, test, x_train, x_original, x_test, y_train, y_original, le


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-data", type=Path, default=INPUT_DIR / "star_classification.csv")
    parser.add_argument("--original-weight", type=float, default=0.36)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.06)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=63)
    args = parser.parse_args()

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    device = torch.device("cpu")
    train, test, x_train, x_original, x_test, y_train, y_original, le = build_features(args)
    labels = list(le.classes_)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train, y_train), start=1):
        if fold > args.max_folds:
            break
        x_fit = pd.concat([x_train.iloc[fit_idx], x_original], axis=0, ignore_index=True)
        y_fit = np.concatenate([y_train[fit_idx], y_original])
        x_valid = x_train.iloc[valid_idx].reset_index(drop=True)
        x_test_fold = x_test.reset_index(drop=True)
        x_fit, x_valid, x_test_fold = add_target_encoding(
            x_fit,
            y_fit,
            x_valid,
            x_test_fold,
            len(labels),
            args.folds,
            args.random_state + fold,
        )

        cat_cols = [col for col in x_fit.columns if str(x_fit[col].dtype) == "category"]
        num_cols = [col for col in x_fit.columns if col not in cat_cols]
        fit_cat, valid_cat, _, cat_dims = encode_categoricals(x_fit, x_valid, x_test_fold, cat_cols)
        scaler = RobustScalerLite().fit(x_fit[num_cols].to_numpy(dtype=np.float32))
        fit_num = scaler.transform(x_fit[num_cols].to_numpy(dtype=np.float32))
        valid_num = scaler.transform(x_valid[num_cols].to_numpy(dtype=np.float32))

        sample_weight = np.concatenate(
            [
                np.ones(len(fit_idx), dtype=np.float32),
                np.full(len(y_original), args.original_weight, dtype=np.float32),
            ]
        )
        train_ds = TensorDataset(
            torch.as_tensor(fit_num),
            torch.as_tensor(fit_cat, dtype=torch.long),
            torch.as_tensor(y_fit, dtype=torch.long),
            torch.as_tensor(sample_weight, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        model = TabularMLP(
            n_num=fit_num.shape[1],
            cat_dims=cat_dims,
            emb_dim=args.emb_dim,
            hidden=args.hidden,
            n_classes=len(labels),
            dropout=args.dropout,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(
            f"fold={fold}, fit={len(fit_num)}, valid={len(valid_num)}, "
            f"num={fit_num.shape[1]}, cat={len(cat_cols)}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_weight = 0.0
            for xb_num, xb_cat, yb, wb in train_loader:
                xb_num = xb_num.to(device)
                xb_cat = xb_cat.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                opt.zero_grad()
                logits = model(xb_num, xb_cat)
                loss_each = nn.functional.cross_entropy(logits, yb, reduction="none", label_smoothing=0.04)
                loss = (loss_each * wb).sum() / wb.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.2)
                opt.step()
                total_loss += float((loss_each * wb).sum().detach().cpu())
                total_weight += float(wb.sum().detach().cpu())

            model.eval()
            val_probs = []
            with torch.no_grad():
                for start in range(0, len(valid_num), 8192):
                    xb_num = torch.as_tensor(valid_num[start : start + 8192]).to(device)
                    xb_cat = torch.as_tensor(valid_cat[start : start + 8192], dtype=torch.long).to(device)
                    val_probs.append(torch.softmax(model(xb_num, xb_cat), dim=1).cpu().numpy())
            val_probs = np.vstack(val_probs)
            score = balanced_accuracy_score(y_train[valid_idx], val_probs.argmax(axis=1))
            print(f"epoch={epoch}, loss={total_loss / total_weight:.5f}, valid_bal_acc={score:.6f}", flush=True)


if __name__ == "__main__":
    main()
