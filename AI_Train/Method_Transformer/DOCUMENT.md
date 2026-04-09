# Method_Transformer (GPT-2 Goal Conditioned)

This internal architecture allows AI Pedsim to predict entire pedestrian trajectories from an observation point all the way to a designated destination (Whole Path prediction). 

Instead of traditional LSTMs or Vanilla GANs which suffer from error-accumulation over long distances, this method implements a **Goal-Conditioned GPT-2 Transformer** optimized by the Hugging Face `Trainer` API.

---

## 🚀 Key Features

*   **Whole Path Prediction**: Traditional methods (like SGAN) predict just the next 10 frames. This transformer looks at the current movement, the environmental walls (`geo_mask`), and the final destination (`end_pt`), then maps out the *entire remaining journey* in a single shot.
*   **Hugging Face Backend**: Training is managed by the industry-standard `Trainer` API, granting features like mixed precision, advanced checkpointing, gradient accumulation, and excellent memory management.
*   **Geometry Aware (GeoEncoder)**: A Convolutional Neural Network (CNN) feature extractor compresses the 2D environmental map (`geo_mask`) and feeds it directly into the Transformer's context block, ensuring pedestrians "see" the walls.

---

## 🧠 Architecture Setup 

The underlying model is `GoalConditionedGPT2`, defined in `model.py`.

### 1. Variables & Hyperparameters (`config_train.json`)
*   **`d_model` (Embedding Dimension)**: Default `128`. Determines the "brain size" of the Transformer. Higher values allow it to learn more complex environments but require exponentially more VRAM.
*   **`max_seq_len` (Context Window)**: Default `1024`. Important! Ensure this number is strictly larger than the longest expected trajectory in your dataset. If pedestrians take more than 1024 steps to reach their destination, the model will throw an index out of bounds error.
*   **`batch_size`**: Default `4`. Given the long sequence nature, keep this low to avoid CUDA OOM (Out of Memory) errors. The script automatically uses `gradient_accumulation_steps=4` to simulate an effective batch size of 16 without exploding the VRAM.

### 2. The Data Flow
1. **Input Encoding**: The past trajectory (`obs_traj`) is embedded.
2. **Context Fusion**: The Start Point, End Point, and CNN-encoded Map (`geo_mask`) are concatenated into a `context_embeds` vector.
3. **Sequence Modeling**: The Context and Inputs are fed into the standard GPT-2 block.
4. **Teacher Forcing**: During training, the entire `labels` trajectory is provided to compute the mask-aware `SmoothL1Loss` instantly.

---

## 🏃 Running the Pipeline

### Training
Execute via the UI or `train_transformer.py`.
The custom `StreamlitProgressCallback` intercepts Hugging Face's eval loops and writes to `logs/training_history.csv` and `progress.json` to keep the Streamlit dashboard animated.

### Testing & Evaluation
Execute via `test_transformer.py`.
The script reads `labels`, masks out different length trajectories properly via a custom `seq_collate`, and computes:
*   **ADE (Average Displacement Error)**: Mean error across the entire path.
*   **FDE (Final Displacement Error)**: Error at the very final step before destruction.

> ⚠️ Note on Test Speed: For heavily populated datasets (e.g., Millions of trajectories), the `test_transformer.py` includes a downsampler mechanism that evaluates every 100th sample to prevent the evaluation from taking hundreds of hours.
