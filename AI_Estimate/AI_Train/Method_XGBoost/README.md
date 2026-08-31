# Method_XGBoost

Vanilla XGBoost (`gbtree`, squared-error objective) baseline for the
`Data_Estimate_2` travel-time task.

- Uses the same 17 input features as the MLP.
- Uses raw feature values because decision trees do not require Z-score scaling.
- Applies `log1p` to each target during training and `expm1` before reporting seconds.
- Trains three independent regressors for minimum, mean, and maximum travel time.
- Does not reorder the three predictions by default; raw vanilla behavior is retained.
- Writes UI-compatible `test_eval/predictions.csv` plus provenance manifests and hashes.

Safe configuration check (default):

```bash
python AI_Estimate/AI_Train/Method_XGBoost/run_pipeline.py --stage plan
```

Train and evaluate are explicit operations:

```bash
python AI_Estimate/AI_Train/Method_XGBoost/run_pipeline.py --stage train
python AI_Estimate/AI_Train/Method_XGBoost/run_pipeline.py --stage evaluate --run-path <run_dir>
```

Install the method dependency in the active project environment:

```bash
python -m pip install -r AI_Estimate/AI_Train/Method_XGBoost/requirements.txt
```

On macOS, the XGBoost wheel also requires the OpenMP runtime:

```bash
brew install libomp
```

Training does not evaluate the test split. The test split is read only during the
explicit evaluation stage.
