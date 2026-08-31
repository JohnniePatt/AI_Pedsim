# AI_Estimate

Predict travel time summaries from HouseGAN topology features and Social Force simulation outputs.

Targets:
- `min_agent_time_s`
- `mean_agent_time_s`
- `max_agent_time_s`

Inputs come from:
- `Geo_scenario/Topo_HouseGAN/time_summary/all_route_time_summary.csv`
- `Geo_scenario/Topo_HouseGAN/route_information/all_route_information.csv`

Formatted data is split into isolated files:
- `Dataset/Data_Estimate/Train/data_estimate.csv`
- `Dataset/Data_Estimate/Val/data_estimate.csv`
- `Dataset/Data_Estimate/Test/data_estimate.csv`

The split is plan-level to avoid leaking the same geometry into Train, Val, and Test.

The canonical estimate dataset used by current Data Estimate 2 comparisons is:

- `Dataset/Data_Estimate_2/Train/data_estimate.csv` (2,603 scenarios / 412 plans)
- `Dataset/Data_Estimate_2/Val/data_estimate.csv` (439 scenarios / 60 plans)
- `Dataset/Data_Estimate_2/Test/data_estimate.csv` (862 scenarios / 117 plans)

Data split is expected to be prepared already under:
- `Dataset/Data_Estimate/Train/data_estimate.csv`
- `Dataset/Data_Estimate/Val/data_estimate.csv`
- `Dataset/Data_Estimate/Test/data_estimate.csv`

Run training (PyTorch):
```bash
python AI_Estimate/AI_Train/Method_MLP_PyTorch/train_time_estimator.py --config AI_Estimate/AI_Train/Method_MLP_PyTorch/config_train.json
```

Run testing (PyTorch):
```bash
python AI_Estimate/AI_Train/Method_MLP_PyTorch/test_time_estimator.py --config AI_Estimate/AI_Train/Method_MLP_PyTorch/config_train.json
```

Run training (Keras):
```bash
python AI_Estimate/AI_Train/Method_MLP_Keras/train_time_estimator.py --config AI_Estimate/AI_Train/Method_MLP_Keras/config_train.json
```

Run testing (Keras):
```bash
python AI_Estimate/AI_Train/Method_MLP_Keras/test_time_estimator.py --config AI_Estimate/AI_Train/Method_MLP_Keras/config_train.json
```

Run the vanilla XGBoost baseline (configuration check is the safe default):

```bash
python AI_Estimate/AI_Train/Method_XGBoost/run_pipeline.py --stage plan
python AI_Estimate/AI_Train/Method_XGBoost/run_pipeline.py --stage train
python AI_Estimate/AI_Train/Method_XGBoost/run_pipeline.py --stage evaluate --run-path <run_dir>
```

Create result plots (choose method):
```bash
python AI_Estimate/AI_Train/Method_MLP_PyTorch/visual_time_estimator.py --run-dir AI_Estimate/AI_result/Method_MLP_PyTorch/outputs/<run_name>
python AI_Estimate/AI_Train/Method_MLP_Keras/visual_time_estimator.py --run-dir AI_Estimate/AI_result/Method_MLP_Keras/outputs/<run_name>
```

Open Streamlit:
```bash
streamlit run AI_Estimate/Streamlit_ui/app.py
```

Outputs are now isolated by method:
- `AI_Estimate/AI_result/Method_MLP_PyTorch/outputs/<run_name>`
- `AI_Estimate/AI_result/Method_MLP_Keras/outputs/<run_name>`
- `AI_Estimate/AI_result/Method_XGBoost/outputs/<run_name>`
