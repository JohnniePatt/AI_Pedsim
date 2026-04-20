import json
from io import BytesIO
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from utils.executor import ProcessManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "AI_Estimate"
TRAIN_ROOT = MODULE_ROOT / "AI_Train"
DATA_ROOT = PROJECT_ROOT / "Dataset" / "Data_Estimate"
GEO_HOUSEGAN_ROOT = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN" / "geo"

METHODS = {
    "Method_MLP_PyTorch": {
        "name": "MLP PyTorch",
        "train_dir": TRAIN_ROOT / "Method_MLP_PyTorch",
        "result_root": MODULE_ROOT / "AI_result" / "Method_MLP_PyTorch" / "outputs",
        "checkpoint_name": "best_result.pth",
    },
    "Method_MLP_Keras": {
        "name": "MLP Keras",
        "train_dir": TRAIN_ROOT / "Method_MLP_Keras",
        "result_root": MODULE_ROOT / "AI_result" / "Method_MLP_Keras" / "outputs",
        "checkpoint_name": "best_result.keras",
    },
}

st.set_page_config(page_title="AI_Estimate", layout="wide")


def read_json(path, fallback=None):
    path = Path(path)
    if not path.exists():
        return fallback
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_manager(key):
    if key not in st.session_state:
        st.session_state[key] = ProcessManager()
    return st.session_state[key]


def render_process_output(manager, key):
    output_key = f"{key}_output"
    if output_key not in st.session_state:
        st.session_state[output_key] = ""
    for line in manager.get_output():
        st.session_state[output_key] += line
    st.code(st.session_state[output_key] or "waiting for output...", language="text")


def run_button_row(manager, start_label, stop_label, command, output_key):
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(start_label, type="primary", use_container_width=True, disabled=manager.is_running):
            st.session_state[f"{output_key}_output"] = ""
            started = manager.start_process(command, cwd=str(PROJECT_ROOT))
            if not started:
                st.warning("Process is already running.")
    with col2:
        if st.button(stop_label, use_container_width=True, disabled=not manager.is_running):
            manager.stop_process()
    render_process_output(manager, output_key)


def list_runs(method_cfg):
    output_root = method_cfg["result_root"]
    if not output_root.exists():
        return []
    runs = [path for path in output_root.iterdir() if path.is_dir()]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def latest_checkpoint(run_dir, method_cfg):
    if not run_dir:
        return None
    best = run_dir / method_cfg["checkpoint_name"]
    if best.exists():
        return best
    if method_cfg["checkpoint_name"].endswith(".pth"):
        checkpoints = sorted(run_dir.glob("epoch_*.pth"))
        return checkpoints[-1] if checkpoints else None
    return run_dir / method_cfg["checkpoint_name"] if (run_dir / method_cfg["checkpoint_name"]).exists() else None


def config_path_for(method_cfg):
    return method_cfg["train_dir"] / "config_train.json"


def show_config_summary(config, method_cfg):
    train_cfg = config.get("train", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Epochs", train_cfg.get("epochs", "-"))
    c2.metric("Batch size", train_cfg.get("batch_size", "-"))
    c3.metric("Learning rate", train_cfg.get("learning_rate", "-"))
    c4.metric("Device", train_cfg.get("device", "auto"))
    st.caption(f"Config: {config_path_for(method_cfg)}")
    st.caption(f"Data source: {DATA_ROOT} (pre-split by user)")
    with st.expander("Current config JSON"):
        st.json(config)


def data_estimate_manifest():
    return read_json(DATA_ROOT / "data_estimate_manifest.json", {})


def split_csv_path(split_name):
    return DATA_ROOT / split_name / "data_estimate.csv"


def render_data_estimate_manifest():
    manifest = data_estimate_manifest()
    st.subheader("Data_Estimate status")
    if not manifest:
        st.info("No manifest found. This is okay if you manage split files manually.")
        rows = []
        for label in ["Train", "Val", "Test"]:
            path = split_csv_path(label)
            rows.append({"split": label, "exists": path.exists(), "csv": str(path)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

    splits = manifest.get("splits", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source rows", manifest.get("source_rows", 0))
    c2.metric("Train rows", splits.get("train", {}).get("rows", 0))
    c3.metric("Val rows", splits.get("val", {}).get("rows", 0))
    c4.metric("Test rows", splits.get("test", {}).get("rows", 0))

    rows = []
    for split_name, label in [("train", "Train"), ("val", "Val"), ("test", "Test")]:
        item = splits.get(split_name, {})
        rows.append({"split": label, "rows": item.get("rows", 0), "plans": item.get("plans", 0), "csv": item.get("csv", "")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("Raw Data_Estimate manifest (optional)"):
        st.json(manifest)


def render_data_estimate_preview():
    st.subheader("Data_Estimate preview")
    train_tab, val_tab, test_tab = st.tabs(["Train", "Val", "Test"])
    for tab, split_name in [(train_tab, "Train"), (val_tab, "Val"), (test_tab, "Test")]:
        with tab:
            path = split_csv_path(split_name)
            st.caption(str(path))
            if not path.exists():
                st.info(f"No {split_name} data yet.")
                continue
            df = pd.read_csv(path, nrows=80)
            st.dataframe(df, use_container_width=True, hide_index=True)


def load_train_dataframe():
    train_path = DATA_ROOT / "Train" / "data_estimate.csv"
    if not train_path.exists():
        return None
    return pd.read_csv(train_path)


def build_housegan_collage(image_paths, width=2000, height=1000, padding=8):
    if not image_paths:
        return None
    count = len(image_paths)
    aspect = width / max(1, height)
    cols = max(1, int((count * aspect) ** 0.5))
    rows = (count + cols - 1) // cols
    if rows > 0 and (cols - 1) * rows >= count:
        cols = max(1, cols - 1)
        rows = (count + cols - 1) // cols

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    cell_w = max(8, (width - padding * (cols + 1)) // cols)
    cell_h = max(8, (height - padding * (rows + 1)) // rows)

    for idx, image_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)
        try:
            with Image.open(image_path) as im:
                tile = im.convert("RGB")
        except Exception:
            continue

        ratio = min(cell_w / max(1, tile.width), cell_h / max(1, tile.height))
        new_size = (
            max(1, int(tile.width * ratio)),
            max(1, int(tile.height * ratio)),
        )
        tile = tile.resize(new_size, Image.Resampling.LANCZOS)
        dx = x + (cell_w - tile.width) // 2
        dy = y + (cell_h - tile.height) // 2
        canvas.paste(tile, (dx, dy))

    return canvas


def page_view_summary(method_cfg):
    st.header("View Summary")
    st.subheader("Topology Preview Collage (HouseGAN)")
    preview_paths = sorted(GEO_HOUSEGAN_ROOT.glob("plan*/preview.png"))
    if not preview_paths:
        st.info(f"No preview images found in: {GEO_HOUSEGAN_ROOT}")
    else:
        collage = build_housegan_collage(preview_paths, width=2000, height=1000, padding=8)
        if collage is not None:
            st.caption(f"Loaded {len(preview_paths)} previews into collage size 2000x1000")
            st.image(collage, use_container_width=True)
            collage_buf = BytesIO()
            collage.save(collage_buf, format="PNG")
            collage_buf.seek(0)
            st.download_button(
                "Export collage (.png)",
                data=collage_buf.getvalue(),
                file_name="housegan_preview_collage_2000x1000.png",
                mime="image/png",
            )

    st.divider()
    st.subheader("Dataset Input Summary (Train)")

    config = read_json(config_path_for(method_cfg), {})
    if not config:
        st.error(f"Missing config: {config_path_for(method_cfg)}")
        return

    feature_numeric = list(config.get("features", {}).get("numeric", []))
    feature_categorical = list(config.get("features", {}).get("categorical", []))
    target_cols = list(config.get("features", {}).get("target", []))

    c1, c2, c3 = st.columns(3)
    c1.metric("Numeric inputs", len(feature_numeric))
    c2.metric("Categorical inputs", len(feature_categorical))
    c3.metric("Targets", len(target_cols))

    st.caption("Inputs used for training from config")
    st.json(
        {
            "numeric": feature_numeric,
            "categorical": feature_categorical,
            "target": target_cols,
        }
    )

    train_df = load_train_dataframe()
    if train_df is None or train_df.empty:
        st.warning(f"Train data not found or empty at: {DATA_ROOT / 'Train' / 'data_estimate.csv'}")
        return

    st.divider()
    st.subheader("Distance Distribution (Sorted)")
    st.caption("X = sorted row index, Y = distance (m)")

    distance_candidates = [
        "topology_centerline_distance_m",
        "straight_distance_m",
        "distance_gap_m",
    ]
    available = [col for col in distance_candidates if col in train_df.columns]
    if not available:
        st.error("No distance columns found in Train data.")
        return

    sort_col = st.selectbox("Sort by data", available, index=0)
    plot_col = st.selectbox("Distance on Y axis", available, index=0, key="summary_plot_col")

    df = train_df.copy()
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[sort_col, plot_col]).sort_values(sort_col).reset_index(drop=True)
    df["sorted_index"] = df.index + 1

    c4, c5 = st.columns(2)
    c4.metric("Rows used", len(df))
    c5.metric(f"Mean {plot_col}", f"{df[plot_col].mean():.2f} m")

    st.caption("Matplotlib graph (clearer line) + instant export")
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(
        df["sorted_index"],
        df[plot_col],
        color="#2563eb",
        linewidth=1.8,
        alpha=0.95,
    )
    ax.set_xlabel("number data sampling")
    ax.set_ylabel(f"{plot_col} (m)")
    ax.set_title(f"Distance Profile (sorted by {sort_col})")
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    export_buf = BytesIO()
    fig.savefig(export_buf, format="png", dpi=180, bbox_inches="tight")
    export_buf.seek(0)
    st.download_button(
        "Export graph (.png)",
        data=export_buf.getvalue(),
        file_name=f"view_summary_{sort_col}_vs_{plot_col}.png",
        mime="image/png",
        use_container_width=False,
    )
    plt.close(fig)

    with st.expander("Preview sorted table (top 120 rows)"):
        preview_candidates = ["plan", "route_index", "start_node", "end_node", sort_col, plot_col]
        preview_cols = []
        for col in preview_candidates:
            if col in df.columns and col not in preview_cols:
                preview_cols.append(col)
        st.dataframe(df[preview_cols].head(120), use_container_width=True, hide_index=True)


def page_utilities():
    st.header("Utilities")
    st.subheader("Use Existing Split Data")
    st.caption("Workflow is locked to your prepared split folders. No auto split is performed here.")
    render_data_estimate_manifest()
    st.divider()
    render_data_estimate_preview()


def page_train(method_key, method_cfg):
    st.header(f"Train AI_Estimate ({method_cfg['name']})")
    config_path = config_path_for(method_cfg)
    config = read_json(config_path, {})
    if not config:
        st.error(f"Missing config: {config_path}")
        return
    show_config_summary(config, method_cfg)

    st.divider()
    st.subheader("Execute training")
    st.caption(f"Output will be saved under {method_cfg['result_root']}.")
    command = [sys.executable, str(method_cfg["train_dir"] / "train_time_estimator.py"), "--config", str(config_path)]
    manager = get_manager(f"estimate_train_process_{method_key}")
    run_button_row(manager, "Start Train", "Stop Train", command, f"estimate_train_{method_key}")

    st.divider()
    render_run_overview(method_cfg)


def page_test(method_key, method_cfg):
    st.header(f"Test AI_Estimate ({method_cfg['name']})")
    runs = list_runs(method_cfg)
    if not runs:
        st.info("No training run yet. Train a model first.")
        return

    run_names = [path.name for path in runs]
    selected_name = st.selectbox("Training run", run_names)
    run_dir = runs[run_names.index(selected_name)]
    checkpoint = latest_checkpoint(run_dir, method_cfg)
    if checkpoint:
        st.caption(f"Checkpoint: {checkpoint}")
    else:
        st.error("No checkpoint found in selected run.")
        return

    config_path = config_path_for(method_cfg)
    output_dir = run_dir / "test_eval"
    command = [
        sys.executable,
        str(method_cfg["train_dir"] / "test_time_estimator.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
    ]
    manager = get_manager(f"estimate_test_process_{method_key}")
    run_button_row(manager, "Start Test", "Stop Test", command, f"estimate_test_{method_key}")

    st.divider()
    render_test_outputs(run_dir)


def render_run_overview(method_cfg):
    runs = list_runs(method_cfg)
    st.subheader("Latest runs")
    if not runs:
        st.caption("No runs yet.")
        return
    rows = []
    for run_dir in runs[:12]:
        manifest = read_json(run_dir / "dataset_manifest.json", {})
        metrics = read_json(run_dir / "metrics.json", {})
        final_test = metrics.get("final_test", {}) if isinstance(metrics, dict) else {}
        rows.append(
            {
                "run": run_dir.name,
                "rows": manifest.get("rows"),
                "train_rows": manifest.get("train_rows"),
                "val_rows": manifest.get("val_rows"),
                "test_rows": manifest.get("test_rows"),
                "final_test_mae_s": final_test.get("mae_overall_s"),
                "path": str(run_dir),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_history(run_dir):
    history_path = run_dir / "training_history.csv"
    if not history_path.exists():
        st.caption("No training_history.csv in this run.")
        return
    history = pd.read_csv(history_path)
    st.subheader("Training history")
    if "epoch" in history.columns:
        metric_cols = [col for col in ["train_mae_overall_s", "val_mae_overall_s", "val_rmse_overall_s", "train_loss", "val_loss"] if col in history.columns]
        if metric_cols:
            st.line_chart(history.set_index("epoch")[metric_cols], use_container_width=True)
    st.dataframe(history.tail(20), use_container_width=True, hide_index=True)


def render_metrics(run_dir):
    manifest = read_json(run_dir / "dataset_manifest.json", {})
    metrics = read_json(run_dir / "metrics.json", {})
    st.subheader("Run summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", manifest.get("rows", 0))
    c2.metric("Train rows", manifest.get("train_rows", 0))
    c3.metric("Val rows", manifest.get("val_rows", 0))
    c4.metric("Test rows", manifest.get("test_rows", 0))
    with st.expander("Dataset manifest"):
        st.json(manifest)
    with st.expander("Metrics JSON"):
        st.json(metrics)


def render_test_outputs(run_dir):
    metrics_path = run_dir / "test_eval" / "test_metrics.json"
    predictions_path = run_dir / "test_eval" / "predictions.csv"
    if metrics_path.exists():
        metrics = read_json(metrics_path, {})
        st.subheader("Test metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", metrics.get("rows", 0))
        c2.metric("MAE overall", f"{metrics.get('mae_overall_s', 0):.2f}s")
        c3.metric("RMSE overall", f"{metrics.get('rmse_overall_s', 0):.2f}s")
        with st.expander("Raw test metrics"):
            st.json(metrics)
    if predictions_path.exists():
        df = pd.read_csv(predictions_path)
        st.subheader("Prediction preview")
        groups = split_prediction_groups(df)
        if list(groups.keys()) == ["All"]:
            render_prediction_scatter(df)
            st.dataframe(df.head(80), use_container_width=True, hide_index=True)
        else:
            tabs = st.tabs(list(groups.keys()))
            for tab, label in zip(tabs, groups.keys()):
                with tab:
                    group_df = groups[label]
                    st.caption(f"Rows: {len(group_df)}")
                    render_prediction_scatter(group_df)
                    st.dataframe(group_df.head(80), use_container_width=True, hide_index=True)


def normalize_variant_label(raw_value):
    value = str(raw_value).strip()
    key = value.lower().replace(" ", "")
    if "n/2" in key or "half" in key:
        return "N/2 Agent"
    if key.startswith("1") or "single" in key:
        return "1 Agent"
    if key.startswith("n") or "full" in key:
        return "N Agent"
    return value or "Unknown"


def split_prediction_groups(df):
    if "variant_label" not in df.columns:
        return {"All": df}
    grouped_df = df.copy()
    grouped_df["__variant_group"] = grouped_df["variant_label"].apply(normalize_variant_label)
    grouped = {label: chunk.drop(columns="__variant_group") for label, chunk in grouped_df.groupby("__variant_group")}
    preferred = ["N Agent", "N/2 Agent", "1 Agent"]
    ordered_labels = [label for label in preferred if label in grouped]
    ordered_labels.extend(sorted(label for label in grouped if label not in preferred))
    return {label: grouped[label] for label in ordered_labels}


def render_prediction_scatter(df):
    if df.empty:
        st.caption("No prediction rows for this group.")
        return
    pairs = [("min_agent_time_s", "Fastest"), ("mean_agent_time_s", "Average"), ("max_agent_time_s", "Slowest")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (target, title) in zip(axes, pairs):
        true_col = f"true_{target}"
        pred_col = f"pred_{target}"
        if true_col not in df or pred_col not in df:
            ax.axis("off")
            continue
        pair_df = df[[true_col, pred_col]].dropna()
        if pair_df.empty:
            ax.axis("off")
            continue
        ax.scatter(pair_df[true_col], pair_df[pred_col], s=18, alpha=0.65)
        low = min(pair_df[true_col].min(), pair_df[pred_col].min())
        high = max(pair_df[true_col].max(), pair_df[pred_col].max())
        ax.plot([low, high], [low, high], color="#ef4444", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Ground truth (s)")
        ax.set_ylabel("Predicted time (s)")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_visual_report(run_dir, method_key, method_cfg):
    st.subheader("Visual report")
    images = [run_dir / "training_curves.png", run_dir / "prediction_scatter.png", run_dir / "error_histogram.png"]
    existing = [path for path in images if path.exists()]
    manager = get_manager(f"estimate_visual_process_{method_key}")
    command = [sys.executable, str(method_cfg["train_dir"] / "visual_time_estimator.py"), "--run-dir", str(run_dir)]
    if st.button("Generate visual report", use_container_width=True, disabled=manager.is_running):
        st.session_state[f"estimate_visual_{method_key}_output"] = ""
        manager.start_process(command, cwd=str(PROJECT_ROOT))
    render_process_output(manager, f"estimate_visual_{method_key}")
    if existing:
        cols = st.columns(min(3, len(existing)))
        for idx, path in enumerate(existing):
            with cols[idx % len(cols)]:
                st.image(str(path), caption=path.name, use_container_width=True)
    else:
        st.caption("No visual images yet. Click Generate visual report.")


def render_sample_time_report(run_dir, method_key, method_cfg):
    st.subheader("Sample Time Estimator (Per Test File)")
    manager = get_manager(f"estimate_sample_process_{method_key}")
    command = [sys.executable, str(method_cfg["train_dir"] / "sample_time_estimator.py"), "--run-dir", str(run_dir)]
    if st.button("Generate sample time report", use_container_width=True, disabled=manager.is_running):
        st.session_state[f"estimate_sample_{method_key}_output"] = ""
        manager.start_process(command, cwd=str(PROJECT_ROOT))
    render_process_output(manager, f"estimate_sample_{method_key}")

    report_path = run_dir / "test_eval" / "sample_time_report.json"
    if not report_path.exists():
        st.caption("No sample report yet. Click Generate sample time report.")
        return

    report = read_json(report_path, {})
    rows = report.get("rows", [])
    c1, c2 = st.columns(2)
    c1.metric("Test files", report.get("files", 0))
    c2.metric("Rows (total in report)", sum(int(item.get("rows", 0)) for item in rows))

    for idx, item in enumerate(rows):
        title = item.get("display_name") or item.get("file_name", "unknown_file")
        with st.expander(f"{idx + 1}. {title}", expanded=(idx == 0)):
            table = pd.DataFrame(
                [
                    {
                        "row": "min",
                        "real": item.get("min_real_s", 0.0),
                        "AI": item.get("min_ai_s", 0.0),
                        "error": item.get("min_error_s", 0.0),
                    },
                    {
                        "row": "mean",
                        "real": item.get("mean_real_s", 0.0),
                        "AI": item.get("mean_ai_s", 0.0),
                        "error": item.get("mean_error_s", 0.0),
                    },
                    {
                        "row": "max",
                        "real": item.get("max_real_s", 0.0),
                        "AI": item.get("max_ai_s", 0.0),
                        "error": item.get("max_error_s", 0.0),
                    },
                ]
            )
            st.dataframe(table, use_container_width=True, hide_index=True)


def page_results(method_key, method_cfg):
    st.header(f"View AI_Estimate results ({method_cfg['name']})")
    runs = list_runs(method_cfg)
    if not runs:
        st.info("No training run yet.")
        return
    run_names = [path.name for path in runs]
    selected_name = st.selectbox("Training run", run_names)
    run_dir = runs[run_names.index(selected_name)]
    st.caption(str(run_dir))

    summary_tab, history_tab, test_tab, visual_tab, sample_tab = st.tabs(
        ["Summary", "Training history", "Test predictions", "Visual report", "Sample time report"]
    )
    with summary_tab:
        render_metrics(run_dir)
    with history_tab:
        render_history(run_dir)
    with test_tab:
        render_test_outputs(run_dir)
    with visual_tab:
        render_visual_report(run_dir, method_key, method_cfg)
    with sample_tab:
        render_sample_time_report(run_dir, method_key, method_cfg)


def sidebar():
    st.sidebar.title("AI_Estimate")
    method_key = st.sidebar.selectbox("Method", list(METHODS.keys()), format_func=lambda key: METHODS[key]["name"])
    page = st.sidebar.radio("Page", ["Utilities", "Train model", "Testing model", "View results", "View summary"])
    st.sidebar.divider()
    st.sidebar.caption("Formatted data is isolated under")
    st.sidebar.code("Dataset/Data_Estimate")
    st.sidebar.caption("Outputs are isolated under")
    st.sidebar.code(f"AI_Estimate/AI_result/{method_key}/outputs")
    return page, method_key


def main():
    page, method_key = sidebar()
    method_cfg = METHODS[method_key]
    if page == "Utilities":
        page_utilities()
    elif page == "Train model":
        page_train(method_key, method_cfg)
    elif page == "Testing model":
        page_test(method_key, method_cfg)
    elif page == "View summary":
        page_view_summary(method_cfg)
    else:
        page_results(method_key, method_cfg)


if __name__ == "__main__":
    main()
