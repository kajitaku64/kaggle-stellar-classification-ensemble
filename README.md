# Predicting Stellar Class

Kaggle Playground Series S6E6: Predicting Stellar Class starter workspace.

## Data

Put the competition files here:

```text
input/train.csv
input/test.csv
input/sample_submission.csv
```

Expected task: classify stellar objects as `GALAXY`, `STAR`, or `QSO`. The public competition listing describes the metric as balanced accuracy.

## Run Baseline

```bash
python3 src/train_baseline.py
```

Outputs:

```text
output/submission_baseline.csv
output/oof_predictions.csv
```

The baseline uses only packages already available in this environment: `pandas` and `scikit-learn`.

## Notes

The script adds astronomy-oriented features:

- SDSS color indices: `u-g`, `g-r`, `r-i`, `i-z`, `u-z`
- photometric summary stats across `u`, `g`, `r`, `i`, `z`
- redshift transforms
- alpha/delta trigonometric and Cartesian projections
- simple observation ID interactions when columns are present

## Portfolio Case Study

See [docs/portfolio_case_study.md](docs/portfolio_case_study.md) for a concise, honest write-up of the ensemble analysis, including which parts used public notebook predictions and which parts were implemented in this workspace.
