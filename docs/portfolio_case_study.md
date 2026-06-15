# Kaggle Stellar Classification: OOF-Based Ensemble Analysis

## 概要

Kaggle Playground Series S6E6「Predicting Stellar Class」に取り組み、恒星天体を `GALAXY` / `QSO` / `STAR` に分類する多クラス分類パイプラインを構築した。評価指標は balanced accuracy。

最終的な公開LBベストは **0.97044**。このスコアは公開Notebook由来の外部予測を含むため、ポートフォリオ上では「公開予測を活用したアンサンブル検証結果」として扱う。

## 自分で実装したこと

- SDSSの測光特徴量から、色指数・赤方偏移変換・空間座標・密度特徴量を生成
- LightGBM / XGBoost / CatBoost / MLP系モデルを比較
- original datasetを利用した追加学習と特徴量整合を検証
- OOF予測を用いた balanced accuracy ベースの重み最適化
- CSV / NPY の OOF・test probability ファイルを自動検出する Kaggle Notebook を作成
- shape mismatch のある augmented OOF を検出し、competition train 部分へ整形
- probability blending、logistic stacking、hard vote、class-prior probe を比較
- OOFスコアとPublic LBの乖離を記録し、過学習しやすい後処理を切り分け

## 主な結果

| 実験 | OOF | Public LB | 備考 |
| --- | ---: | ---: | --- |
| 自力 LightGBM + XGBoost blend | 0.968112 | 0.96883 | 自力モデル中心の最高水準 |
| 外部5モデル blend | 0.969853 | 0.97044 | 公開予測を利用した最高スコア |
| 外部 + local blend | 0.969940 | 0.97035 | OOFは上がったがLBは低下 |
| Kaggle auto probability blend | 0.970106 | 0.97024 | 公開OOF/test確率5本を自動ブレンド |
| Conservative hard vote | - | 0.97026 | 82行だけ変更する後処理 |
| Class-prior probes | 約0.9700 | 0.97021 | QSO/STAR倍率調整は効果なし |

## 技術的な工夫

### OOFベースのブレンド最適化

単純なsubmission votingではなく、各モデルのOOF確率を使って balanced accuracy を直接最大化した。重みは softmax 形式で正規化し、クラスごとの倍率と確率分布の sharpness も同時に探索した。

### 汎用的なprobability loader

公開Notebookごとに成果物の形式が異なるため、以下を自動判定する仕組みを作成した。

- `id + proba_GALAXY/proba_QSO/proba_STAR + target` のOOF CSV
- `id + proba_GALAXY/proba_QSO/proba_STAR` のtest CSV
- `(n_train, 3)` / `(n_test, 3)` のNPY probability
- augmented trainingを含むOOF配列の trimming

### OOFとLBの乖離分析

OOFが改善してもPublic LBが下がるケースが複数あったため、後処理を「採用」ではなく「検証対象」として扱った。

例:

- 外部 + local blend は OOF `0.969940` まで上がったが、LBは `0.97035` に低下
- class-prior probe は OOFを大きく崩さなかったが、LBは `0.97021` で頭打ち
- hard vote は変更行数を82行まで絞った場合のみ微改善

## 学び

このコンペでは、上位スコアに近づくほど新しい特徴量や小さな後処理よりも、誤り方の異なる強いモデルを追加することが重要だった。特に、OOF改善がそのままPublic LB改善につながらない場面が多く、OOF・LB・変更行数をセットで管理する必要があった。

また、公開Notebookやdiscussionの成果物を利用する場合は、スコアだけを自分の成果として示すのではなく、再現・検証・自動化・失敗分析の部分を明確に切り分けることが重要だと分かった。

## 成果物

- `notebooks/kaggle_auto_probability_blender.py`
  - Kaggle上でOOF/test probabilityを自動検出し、weighted blend / logistic stacker / hard vote候補を生成
- `src/train_tree_models.py`
  - LightGBM / XGBoost向け特徴量生成と学習
- `src/optimize_external_probability_blend.py`
  - 外部予測とlocalモデルのOOFブレンド検証
- `docs/experiment_log.md`
  - 実験結果、採用・不採用理由、スコア推移の記録

## まとめ

このプロジェクトでは、恒星分類コンペに対して自作モデルの学習から公開予測を含むアンサンブル検証までを行った。

最終スコアだけを見ると公開予測の影響が大きいため、ポートフォリオでは「高スコアを出した」ことよりも、以下を成果として示すのが適切だと考えている。

- 自作モデルで Public LB `0.96883` まで到達したこと
- OOF/test probability を自動検出してブレンドできる仕組みを作ったこと
- OOF と Public LB のズレを記録し、後処理の採用可否を検証したこと
- 公開Notebook由来の予測を使った部分と、自分で実装した部分を切り分けて説明できる形にしたこと
