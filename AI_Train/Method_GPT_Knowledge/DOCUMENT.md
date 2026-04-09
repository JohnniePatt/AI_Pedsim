# Method_GPT_Knowledge

This method uses the train dataset as a knowledge base instead of training a new neural network.

## Files
- `prepare_geometry_gpt_knowledge.py`: geometry loading, case loading, path resampling, and local-frame transforms
- `build_knowledge.py`: build the scene retrieval index from the train split
- `generate_gpt_knowledge.py`: retrieve similar cases and transfer trajectories into a target scene
- `validate_gpt_knowledge.py`: evaluate generated scenes on a chosen split

## Outputs
The baseline writes:
- `AI_Result/Method_GPT_Knowledge/knowledge/.../scene_index.parquet`
- `AI_Result/Method_GPT_Knowledge/knowledge/.../scene_index.csv`
- `AI_Result/Method_GPT_Knowledge/knowledge/.../knowledge_manifest.json`
- `AI_Result/Method_GPT_Knowledge/outputs/.../AI_pred_<case>.parquet`
- `AI_Result/Method_GPT_Knowledge/outputs/.../retrieved_cases.json`
- `AI_Result/Method_GPT_Knowledge/outputs/.../planner_prompt.txt`
- `AI_Result/Method_GPT_Knowledge/evaluation/.../*_scene_metrics.csv`
- `AI_Result/Method_GPT_Knowledge/evaluation/.../*_agent_metrics.csv`

## Notes
- The first version is retrieval-based and does not call an external GPT API yet.
- It already prepares a `planner_prompt.txt` so the GPT planner can be added later without changing the data pipeline.
- Validation uses progress-normalized path ADE/FDE and separates duration error as its own metric.

## Suggested Workflow
1. Build the knowledge index
2. Generate one sample scene
3. Run validation on test split

### Build
```bash
python AI_Train/Method_GPT_Knowledge/build_knowledge.py --config AI_Train/Method_GPT_Knowledge/config_build.json
```

### Generate
```bash
python AI_Train/Method_GPT_Knowledge/generate_gpt_knowledge.py --config AI_Train/Method_GPT_Knowledge/config_generate.json
```

### Validate
```bash
python AI_Train/Method_GPT_Knowledge/validate_gpt_knowledge.py --config AI_Train/Method_GPT_Knowledge/config_validate.json
```

To validate only part of the split, change:
- `validation_percent`: percentage of cases to evaluate, for example `10.0` or `25.0`
- `validation_seed`: random seed for reproducible case sampling
- `max_cases`: optional hard cap after percentage sampling
