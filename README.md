# Kaggle Stellar Classification Ensemble Analysis

Kaggle Playground Series S6E6 "Predicting Stellar Class" の実験リポジトリです。

天体を `GALAXY` / `QSO` / `STAR` に分類する多クラス分類タスクに対して、特徴量生成、LightGBM / XGBoost / CatBoost / MLP 系モデル比較、OOF ベースの probability blending、stacking、hard vote、Public LB probe の検証を行いました。

## Results

| Setup | OOF | Public LB | Notes |
| --- | ---: | ---: | --- |
| Local LightGBM + XGBoost blend | 0.968112 | 0.96883 | 自作モデル中心の最高水準 |
| Public 5-model blend | 0.969853 | 0.97044 | 公開Notebook由来の予測を利用 |
| Public + local blend | 0.969940 | 0.97035 | OOFは改善したがLBは低下 |
| Auto probability blend | 0.970106 | 0.97024 | OOF/test probabilityを自動検出して最適化 |
| Conservative hard vote | - | 0.97026 | 82行のみ変更する後処理 |

最終的な最高 Public LB は **0.97044** ですが、これは公開Notebook由来の外部予測を含みます。そのため、このリポジトリではスコアだけでなく、再現可能な検証基盤、OOF と LB の乖離分析、予測ファイルの自動読み込み、ブレンド手法の比較を主な成果として整理しています。

詳しい振り返りは [docs/portfolio_case_study.md](docs/portfolio_case_study.md) にまとめています。

## What I Built

- SDSS photometry からの色指数、赤方偏移、空間座標、密度系特徴量
- LightGBM / XGBoost / CatBoost / MLP 系モデルの比較実験
- OOF balanced accuracy を直接最大化する weighted probability blender
- CSV / NPY の OOF・test probability ファイルの自動検出
- augmented OOF 配列の shape mismatch 検出と trimming
- logistic stacking、hard vote、class-prior probe の比較
- OOF スコア、Public LB、変更行数を並べた実験ログ

## Repository Structure

```text
notebooks/
  kaggle_auto_probability_blender.py      # Kaggle上で外部OOF/test確率を自動検出してブレンド
  kaggle_catboost_gpu_oof_blend.py        # CatBoost系OOF実験
  kaggle_submission_hard_blender.py       # submission hard-vote候補生成

src/
  train_categorical_baseline.py           # カテゴリ特徴量を含むlocal baseline
  train_tree_models.py                    # tree model向け特徴量生成・学習
  optimize_external_probability_blend.py  # 外部予測とlocal予測のOOFブレンド検証

docs/
  experiment_log.md                       # 実験ログ
  portfolio_case_study.md                 # ポートフォリオ向けケーススタディ
```

## Data

Kaggle の competition files はリポジトリに含めていません。ローカルで実行する場合は以下に配置します。

```text
input/train.csv
input/test.csv
input/sample_submission.csv
```

`input/`, `data/`, `output/` は `.gitignore` で除外しています。

## Run Examples

Baseline:

```bash
python3 src/train_categorical_baseline.py
```

Kaggle probability auto blender:

```bash
python3 notebooks/kaggle_auto_probability_blender.py
```

The auto blender expects OOF/test probability artifacts such as:

```text
OOF CSV : id + proba_GALAXY/proba_QSO/proba_STAR + target
Test CSV: id + proba_GALAXY/proba_QSO/proba_STAR
OOF NPY : shape = (n_train, 3)
Test NPY: shape = (n_test, 3)
```

## Notes

This project intentionally separates:

- 自作モデルで到達したスコア
- 公開Notebook由来の予測を利用したスコア
- OOFでは良くてもPublic LBで悪化した後処理

Kaggle の discussion / public notebook を活用した部分は、ポートフォリオ上でもその旨を明記する前提で整理しています。
