import json
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

from utils.executor import ProcessManager


st.set_page_config(page_title="AI Pedsim | Grid Trajectory", layout="wide")

GRID_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = GRID_ROOT.parent
TOOL_DIR = PROJECT_ROOT / "Tool_utility"
GEO_ROOT = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN" / "geo"
HOUSEGAN_ROOT = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN"
AI_TRAIN_DIR = GRID_ROOT / "AI_Train"
AI_RESULT_DIR = GRID_ROOT / "AI_Result"
DATASET_ROOT = PROJECT_ROOT / "Dataset" / "Data_TrajectoryGrid" / "Topo_HouseGAN"
TRAJECTORY_LINE_ROOT = HOUSEGAN_ROOT / "trajectory_line"


PYTHON_BIN_CANDIDATES = [
    PROJECT_ROOT.parent / "AI_Pedsim-env" / "bin" / "python3",
    PROJECT_ROOT / "AI_Pedsim-env" / "bin" / "python3",
    PROJECT_ROOT / ".venv_sim" / "bin" / "python",
]


def get_python_executable() -> str:
    for python_bin in PYTHON_BIN_CANDIDATES:
        if python_bin.exists():
            return str(python_bin)
    return "python3"


def format_float3(value: float) -> str:
    return f"{value:.3f}"


def load_json_file(path: pathlib.Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def get_available_methods() -> list[str]:
    return sorted([p.name for p in AI_TRAIN_DIR.glob("Method_*") if p.is_dir()])


def get_method_runs(method_name: str) -> list[pathlib.Path]:
    method_result_dir = AI_RESULT_DIR / method_name
    if not method_result_dir.exists():
        return []
    return sorted([p for p in method_result_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)


def get_config_path(method_path: pathlib.Path, config_name: str) -> pathlib.Path:
    return method_path / config_name


def get_plan_options() -> list[str]:
    return sorted([p.name for p in GEO_ROOT.glob("plan_*") if p.is_dir()])


def get_generated_plan_options() -> list[str]:
    return sorted(
        [
            p.name
            for p in GEO_ROOT.glob("plan_*")
            if p.is_dir() and (p / "walkablearea_grid.json").exists()
        ]
    )


def load_grid_payload(plan_name: str) -> dict:
    with (GEO_ROOT / plan_name / "walkablearea_grid.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def grid_to_array(grid_rows: list[str]) -> np.ndarray:
    if not grid_rows:
        return np.zeros((1, 1), dtype=np.uint8)

    height = len(grid_rows)
    width = len(grid_rows[0])
    grid_bytes = "".join(grid_rows).encode("ascii")
    return (np.frombuffer(grid_bytes, dtype=np.uint8).reshape(height, width) == ord("1")).astype(np.uint8)


def plot_grid_payload(payload: dict, show_axes: bool):
    meta = payload["meta"]
    grid = grid_to_array(payload["grid"])
    width = int(meta["width"])
    height = int(meta["height"])
    aspect = max(width / max(height, 1), 0.25)
    fig_width = min(18.0, max(7.0, 7.0 * aspect))
    fig_height = min(12.0, max(4.5, fig_width / aspect))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    cmap = ListedColormap(["#111820", "#f4f7fa"])

    # Plot strictly from JSON grid cells (not from source geometry).
    # pcolormesh draws one quad per cell, with explicit cell borders.
    grid_plot = np.flipud(grid)  # JSON rows are top->bottom; matplotlib y grows upward.
    x_edges = np.arange(width + 1, dtype=np.float64)
    y_edges = np.arange(height + 1, dtype=np.float64)
    ax.pcolormesh(
        x_edges,
        y_edges,
        grid_plot,
        cmap=cmap,
        vmin=0,
        vmax=1,
        shading="flat",
        edgecolors="#8a93a0",
        linewidth=0.15,
        antialiased=True,
    )
    ax.set_aspect("equal", adjustable="box")

    # Convert axis ticks from cell-index space to metric coordinates.
    cell = float(meta["cell_size_m"])

    if show_axes:
        # Keep labels readable by showing a limited number of ticks.
        tick_count = 8
        xt = np.linspace(0, width, tick_count)
        yt = np.linspace(0, height, tick_count)
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        x_labels = [f"{meta['min_x'] + (t * cell):.2f}" for t in xt]
        y_labels = [f"{meta['min_y'] + (t * cell):.2f}" for t in yt]
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    else:
        ax.set_axis_off()

    ax.set_title(f"{meta['plan_name']} walkable grid", pad=10)
    fig.tight_layout()
    return fig


def plot_rollout(input_dir: pathlib.Path, rollout_path: pathlib.Path):
    payload = load_json_file(input_dir / "walkablearea_grid.json")
    rollout_df = pd.read_parquet(rollout_path)
    meta = payload["meta"]
    grid = grid_to_array(payload["grid"])
    width = int(meta["width"])
    height = int(meta["height"])
    aspect = max(width / max(height, 1), 0.25)
    fig_width = min(18.0, max(7.0, 7.0 * aspect))
    fig_height = min(12.0, max(4.5, fig_width / aspect))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    cmap = ListedColormap(["#101820", "#f3f6f8"])
    ax.imshow(grid, cmap=cmap, origin="upper", interpolation="nearest")
    if not rollout_df.empty:
        for _, agent_df in rollout_df.groupby("agent_id"):
            ordered = agent_df.sort_values("frame")
            moved = bool(((ordered["grid_x"].diff().fillna(0) != 0) | (ordered["grid_row"].diff().fillna(0) != 0)).any())
            if moved:
                ax.plot(ordered["grid_x"], ordered["grid_row"], linewidth=1.2, alpha=0.75)
                ax.scatter(ordered["grid_x"].iloc[-1], ordered["grid_row"].iloc[-1], s=12, c="#ef4444", zorder=4)
            else:
                ax.scatter(
                    ordered["grid_x"].iloc[0],
                    ordered["grid_row"].iloc[0],
                    s=60,
                    facecolors="none",
                    edgecolors="#ef4444",
                    linewidths=1.0,
                    zorder=4,
                )
            ax.scatter(ordered["grid_x"].iloc[0], ordered["grid_row"].iloc[0], s=12, c="#22c55e", zorder=5)
    ax.set_title("Rollout on walkable grid")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def show_rollout_sample_or_plot(output_dir: pathlib.Path, input_dir: pathlib.Path, rollout_path: pathlib.Path):
    sample_path = output_dir / "samples" / "rollout_preview.png"
    if sample_path.exists():
        st.image(str(sample_path), caption=f"Saved sample: {sample_path}", use_container_width=True)
        return
    fig = plot_rollout(input_dir, rollout_path)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def sqlite_stem_to_raw_trajectory_image(sqlite_stem: str) -> str:
    stem = str(sqlite_stem)
    if stem.startswith("plan_sim_"):
        stem = stem[len("plan_sim_") :]
    return f"trajectory_{stem}.png"


def raw_ground_truth_image_path(plan_name: str, sqlite_stem: str) -> pathlib.Path:
    return TRAJECTORY_LINE_ROOT / str(plan_name) / sqlite_stem_to_raw_trajectory_image(str(sqlite_stem))


def find_manifest_row(split: str, plan_name: str, sqlite_stem: str) -> pd.Series | None:
    manifest_path = DATASET_ROOT / "manifest_trajectory_grid.csv"
    if not manifest_path.exists():
        return None
    manifest_df = pd.read_csv(manifest_path)
    matches = manifest_df[
        (manifest_df["split"] == split)
        & (manifest_df["plan_name"] == plan_name)
        & (manifest_df["sqlite_stem"] == sqlite_stem)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def render_output_compare_row(plan_name: str, sqlite_stem: str, output_dir: pathlib.Path, split: str | None = None):
    raw_path = raw_ground_truth_image_path(plan_name, sqlite_stem)
    rollout_path = output_dir / "rollout.parquet"
    ai_preview_path = output_dir / "samples" / "rollout_preview.png"
    manifest_row = None
    if split is not None:
        manifest_row = find_manifest_row(split, plan_name, sqlite_stem)

    st.markdown(f"#### {plan_name}/{sqlite_stem}")
    col_raw, col_grid, col_ai = st.columns(3)

    with col_raw:
        st.markdown("**Ground truth raw**")
        if raw_path.exists():
            st.image(str(raw_path), use_container_width=True)
        else:
            st.warning(f"Missing raw image: {raw_path.name}")

    with col_grid:
        st.markdown("**Ground truth on grid**")
        if manifest_row is not None:
            input_dir = pathlib.Path(manifest_row["input_dir"])
            target_dir = pathlib.Path(manifest_row["target_dir"])
            if input_dir.exists() and target_dir.exists():
                if (target_dir / "trajectory.parquet").exists():
                    target_agent_count = pd.read_parquet(target_dir / "trajectory.parquet", columns=["agent_id"])["agent_id"].nunique()
                    st.caption(f"{target_agent_count:,} target agents")
                fig = plot_case_preview(input_dir, target_dir=target_dir)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.warning("Missing prepared grid/target directory.")
        else:
            st.warning("Case not found in manifest.")

    with col_ai:
        st.markdown("**AI rollout output**")
        if ai_preview_path.exists():
            st.image(str(ai_preview_path), use_container_width=True)
        elif rollout_path.exists() and manifest_row is not None:
            input_dir = pathlib.Path(manifest_row["input_dir"])
            fig = plot_rollout(input_dir, rollout_path)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.warning("No AI rollout preview yet.")


def render_rollout_summary(summary: dict):
    cols = st.columns(6)
    cols[0].metric("Frames", summary.get("frames", 0))
    cols[1].metric("Agents", summary.get("agents", 0))
    cols[2].metric("Moving agents", summary.get("moving_agents", 0))
    cols[3].metric("Movement steps", summary.get("movement_steps", 0))
    cols[4].metric("Wait steps", summary.get("wait_steps", 0))
    cols[5].metric("Stopped", summary.get("stopped_agents", 0))

    cols_b = st.columns(4)
    cols_b[0].metric("Walkable Ratio", f"{summary.get('walkable_ratio', 0):.3f}")
    cols_b[1].metric("Collisions", summary.get("collision_count", 0))
    cols_b[2].metric("Blocked wall", summary.get("blocked_by_wall_steps", 0))
    cols_b[3].metric("Blocked collision", summary.get("blocked_by_collision_steps", 0))

    if int(summary.get("movement_steps", 0)) == 0:
        st.error("This rollout has no movement. The model selected wait or got blocked for every step, so no trajectory line can appear.")


def render_action_trace(output_dir: pathlib.Path):
    trace_path = output_dir / "action_trace.parquet"
    if not trace_path.exists():
        return
    trace_df = pd.read_parquet(trace_path)
    if trace_df.empty:
        return
    st.subheader("Action Trace")
    counts = trace_df["action_name"].value_counts().reset_index()
    counts.columns = ["action", "count"]
    st.dataframe(counts, use_container_width=True)
    with st.expander("Action trace rows", expanded=False):
        st.dataframe(trace_df.head(1000), use_container_width=True)


def render_batch_rollout_table(run_path: pathlib.Path):
    rollouts_root = run_path / "rollouts"
    if not rollouts_root.exists():
        return
    batch_files = sorted(rollouts_root.glob("batch_rollout_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not batch_files:
        return
    st.subheader("Batch Rollout Samples")
    selected_batch = st.selectbox("Batch summary", batch_files, format_func=lambda p: p.name)
    batch_df = pd.read_csv(selected_batch)
    st.dataframe(batch_df, use_container_width=True)
    if not batch_df.empty:
        st.header("Output preview and compare ground truth")
        for row in batch_df.head(20).itertuples():
            render_output_compare_row(
                plan_name=str(row.plan_name),
                sqlite_stem=str(row.sqlite_stem),
                output_dir=pathlib.Path(row.output_dir),
                split=str(row.split),
            )


def world_to_grid_plot(x: float, y: float, meta: dict) -> tuple[float, float]:
    cell = float(meta["cell_size_m"])
    gx = (float(x) - float(meta["origin_x"])) / cell
    gy = (float(y) - float(meta["origin_y"])) / cell
    row = int(meta["height"]) - gy
    return gx, row


def plot_case_preview(input_dir: pathlib.Path, target_dir: pathlib.Path | None = None, max_paths: int | None = None):
    payload = load_json_file(input_dir / "walkablearea_grid.json")
    exit_payload = load_json_file(input_dir / "exit_room.json", default={})
    spawn_df = pd.read_parquet(input_dir / "spawn_agent.parquet")
    meta = payload["meta"]
    grid = grid_to_array(payload["grid"])
    width = int(meta["width"])
    height = int(meta["height"])
    aspect = max(width / max(height, 1), 0.25)
    fig_width = min(18.0, max(7.0, 7.0 * aspect))
    fig_height = min(12.0, max(4.5, fig_width / aspect))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    cmap = ListedColormap(["#101820", "#f3f6f8"])
    ax.imshow(grid, cmap=cmap, origin="upper", interpolation="nearest")

    if exit_payload.get("exit_node", {}).get("polygon"):
        poly_points = [world_to_grid_plot(x, y, meta) for x, y in exit_payload["exit_node"]["polygon"]]
        xs = [p[0] for p in poly_points]
        ys = [p[1] for p in poly_points]
        ax.fill(xs, ys, color="#f59e0b", alpha=0.35, label="exit room")
        ax.plot(xs, ys, color="#f97316", linewidth=1.4)

    if target_dir is not None and (target_dir / "trajectory.parquet").exists():
        traj_df = pd.read_parquet(target_dir / "trajectory.parquet", columns=["agent_id", "grid_x", "grid_row"])
        for i, (_, agent_df) in enumerate(traj_df.groupby("agent_id")):
            if max_paths is not None and i >= max_paths:
                break
            ax.plot(agent_df["grid_x"], agent_df["grid_row"], color="#38bdf8", linewidth=0.65, alpha=0.32)

    ax.scatter(spawn_df["grid_x"], spawn_df["grid_row"], s=18, c="#22c55e", edgecolors="#052e16", linewidths=0.4, label="spawn")
    ax.set_title("Case preview: walkable grid, spawn, exit, target traces")
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    return fig


def find_rollout_manifest_row(rollout_dir: pathlib.Path) -> pd.Series | None:
    manifest_path = DATASET_ROOT / "manifest_trajectory_grid.csv"
    if not manifest_path.exists():
        return None
    name = rollout_dir.name
    manifest_df = pd.read_csv(manifest_path)
    for split in ["train", "val", "test"]:
        prefix = f"{split}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        for row in manifest_df[manifest_df["split"] == split].itertuples():
            key = f"{row.plan_name}_{row.sqlite_stem}"
            if rest == key or rest.startswith(f"{key}_"):
                return pd.Series(row._asdict())
    return None


def find_rollout_input_dir(rollout_dir: pathlib.Path) -> pathlib.Path | None:
    row = find_rollout_manifest_row(rollout_dir)
    if row is None:
        return None
    return pathlib.Path(row["input_dir"])


def render_run_metrics(run_path: pathlib.Path):
    metrics_path = run_path / "metrics.csv"
    if not metrics_path.exists():
        st.warning("No `metrics.csv` found.")
        return
    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty:
        st.info("Metrics file is empty.")
        return
    latest = metrics_df.iloc[-1]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Epoch", int(latest["epoch"]))
    metric_cols[1].metric("Val Loss", f"{latest['val_loss']:.4f}")
    metric_cols[2].metric("Val Action Acc", f"{latest['val_action_acc']:.3f}")
    metric_cols[3].metric("Val Stop Acc", f"{latest['val_stop_acc']:.3f}")
    if {"val_action_loss", "val_stop_loss"}.issubset(metrics_df.columns):
        loss_cols = st.columns(4)
        loss_cols[0].metric("Val Action Loss", f"{latest['val_action_loss']:.4f}")
        loss_cols[1].metric("Val Stop Loss", f"{latest['val_stop_loss']:.4f}")
        loss_cols[2].metric("Train Action Loss", f"{latest['train_action_loss']:.4f}")
        loss_cols[3].metric("Train Stop Loss", f"{latest['train_stop_loss']:.4f}")
    chart_cols = [c for c in ["train_loss", "val_loss", "train_action_acc", "val_action_acc"] if c in metrics_df.columns]
    st.line_chart(metrics_df.set_index("epoch")[chart_cols])
    with st.expander("Metrics table", expanded=False):
        st.dataframe(metrics_df, use_container_width=True)


def render_action_space_preview(run_path: pathlib.Path):
    action_path = run_path / "action_space.json"
    if not action_path.exists():
        return
    payload = load_json_file(action_path, default={})
    actions = payload.get("actions", [])
    if actions:
        st.dataframe(pd.DataFrame(actions), use_container_width=True)


def render_process_controls(command: list[str], cwd: pathlib.Path, run_label: str, stop_label: str, log_key: str, running_message: str):
    st.code(" ".join(str(x) for x in command), language="bash")
    manager = st.session_state.process_manager
    run_col, stop_col = st.columns([1, 1])
    with run_col:
        if st.button(run_label, type="primary", disabled=manager.is_running, use_container_width=True):
            st.session_state[log_key] = ""
            started = manager.start_process(command, cwd=str(cwd))
            if not started:
                st.warning("A process is already running.")
            st.rerun()
    with stop_col:
        if st.button(stop_label, disabled=not manager.is_running, use_container_width=True, key=f"{log_key}_stop"):
            manager.stop_process()
            st.rerun()

    for line in manager.get_output():
        st.session_state[log_key] = st.session_state.get(log_key, "") + line

    if manager.is_running:
        st.info(running_message)
        st.code(st.session_state.get(log_key, ""), language="bash")
        time.sleep(1)
        st.rerun()

    st.text_area("Output", st.session_state.get(log_key, ""), height=320, key=f"{log_key}_output")


if "process_manager" not in st.session_state:
    st.session_state.process_manager = ProcessManager()
if "process_log" not in st.session_state:
    st.session_state.process_log = ""
if "current_nav" not in st.session_state:
    st.session_state.current_nav = "Training model"


st.sidebar.markdown("# AI Pedsim Grid")
available_methods = get_available_methods()
if not available_methods:
    st.sidebar.error("No grid training methods found.")
    st.stop()

selected_method = st.sidebar.selectbox("Current AI Method", available_methods, label_visibility="collapsed")
method_path = AI_TRAIN_DIR / selected_method
result_method_path = AI_RESULT_DIR / selected_method

nav_options = [
    "Training model",
    "Testing model",
    "View results",
    "Prepare Walkable Grid",
    "Prepare Trajectory Grid Dataset",
]
nav_index = nav_options.index(st.session_state.current_nav) if st.session_state.current_nav in nav_options else 0
selected_nav = st.sidebar.radio("Navigation", nav_options, index=nav_index)
if selected_nav != st.session_state.current_nav:
    st.session_state.current_nav = selected_nav
    st.rerun()

st.title("AI_GenerateTrajectoryGrid")
st.caption("Grid-based trajectory generation workspace.")
nav = st.session_state.current_nav

if nav == "Training model":
    st.subheader(f"Training: {selected_method}")
    config_path = get_config_path(method_path, "config_train.json")
    config_train = load_json_file(config_path, default={})
    if not config_train:
        st.warning("No `config_train.json` found for this method.")

    train_scripts = sorted(method_path.glob("train_*.py"))
    runs = get_method_runs(selected_method)
    latest_run = runs[0] if runs else None

    top_cols = st.columns(4)
    top_cols[0].metric("Method", selected_method.replace("Method_", ""))
    top_cols[1].metric("Runs", len(runs))
    top_cols[2].metric("Train batch", config_train.get("batch_size", "-"))
    top_cols[3].metric("Workers", config_train.get("num_workers", "-"))

    tab_config, tab_execute, tab_preview = st.tabs(["Config", "Execute", "Run Preview"])

    with tab_config:
        st.write("Training configuration")
        config_text = st.text_area("JSON Editor", value=json.dumps(config_train, indent=2), height=360, key="train_config_editor")
        save_col, smoke_col = st.columns([1, 1])
        with save_col:
            if st.button("Save Training Configuration", type="primary", use_container_width=True):
                try:
                    save_json_file(config_path, json.loads(config_text))
                    st.success("Training configuration saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Invalid JSON: {exc}")
        with smoke_col:
            smoke_path = method_path / "config_smoke.json"
            if smoke_path.exists() and st.button("Load Smoke Config", use_container_width=True):
                st.session_state.train_config_editor = json.dumps(load_json_file(smoke_path, default={}), indent=2)
                st.rerun()

    with tab_execute:
        st.write("Execute training script")
        if not train_scripts:
            st.error("No `train_*.py` script found.")
        else:
            selected_script = st.selectbox("Training script", [p.name for p in train_scripts])
            script_path = method_path / selected_script
            command = [get_python_executable(), str(script_path)]
            if "--config" in script_path.read_text(encoding="utf-8", errors="ignore"):
                command.extend(["--config", "config_train.json"])
            render_process_controls(
                command=command,
                cwd=method_path,
                run_label="Start Training",
                stop_label="Stop Training",
                log_key="training_logs",
                running_message="Training is running...",
            )
            if st.button("Clear Training Log", use_container_width=True):
                st.session_state.training_logs = ""
                st.rerun()

    with tab_preview:
        if latest_run is None:
            st.info("No training run found yet.")
        else:
            st.write("Latest run")
            st.code(str(latest_run))
            render_run_metrics(latest_run)
            st.subheader("Action Space")
            render_action_space_preview(latest_run)
            with st.expander("Files", expanded=False):
                st.code(str(latest_run / "checkpoints" / "best.pt"))
                st.code(str(latest_run / "checkpoints" / "last.pt"))

if nav == "Testing model":
    st.subheader(f"Testing: {selected_method}")
    runs = get_method_runs(selected_method)
    if not runs:
        st.info("No training runs found yet.")
    else:
        run_names = [p.name for p in runs]
        selected_run_name = st.selectbox("Run", run_names)
        run_path = result_method_path / selected_run_name
        checkpoint_options = []
        for name in ["best.pt", "last.pt"]:
            candidate = run_path / "checkpoints" / name
            if candidate.exists():
                checkpoint_options.append(candidate)
        if not checkpoint_options:
            st.warning("No checkpoints found in this run.")
        else:
            checkpoint_path = st.selectbox("Checkpoint", checkpoint_options, format_func=lambda p: p.name)
            manifest_path = DATASET_ROOT / "manifest_trajectory_grid.csv"
            if not manifest_path.exists():
                st.error(f"Dataset manifest not found: {manifest_path}")
            else:
                manifest_df = pd.read_csv(manifest_path)
                tab_case, tab_run, tab_preview = st.tabs(["Case Preview", "Run Test", "Rollout Preview"])

                with tab_case:
                    split = st.selectbox("Split", ["train", "val", "test"], index=1)
                    split_df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
                    case_labels = [f"{r.plan_name}/{r.sqlite_stem}" for r in split_df.itertuples()]
                    selected_case_label = st.selectbox("Input case", case_labels)
                    selected_case = split_df.iloc[case_labels.index(selected_case_label)]
                    input_dir = pathlib.Path(selected_case["input_dir"])
                    target_dir = pathlib.Path(selected_case["target_dir"])

                    case_cols = st.columns(5)
                    case_cols[0].metric("Agents", int(float(selected_case["agent_count"])))
                    case_cols[1].metric("Frames", int(float(selected_case["frame_count"])))
                    case_cols[2].metric("Rows", int(float(selected_case["trajectory_rows"])))
                    case_cols[3].metric("Start", selected_case["start_node"])
                    case_cols[4].metric("Exit", selected_case["exit_node"])

                    if (target_dir / "trajectory.parquet").exists():
                        target_agent_count = pd.read_parquet(target_dir / "trajectory.parquet", columns=["agent_id"])["agent_id"].nunique()
                        st.caption(f"Grid target preview plots all {target_agent_count:,} trajectory agents.")

                    fig = plot_case_preview(input_dir, target_dir=target_dir)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                output_dir = run_path / "rollouts" / f"{split}_{selected_case['plan_name']}_{selected_case['sqlite_stem']}"
                summary_path = output_dir / "summary.json"
                rollout_path = output_dir / "rollout.parquet"

                with tab_run:
                    col_a, col_b, col_c, col_d, col_e = st.columns(5)
                    with col_a:
                        max_steps = st.number_input("Max steps", min_value=1, max_value=5000, value=300, step=50)
                    with col_b:
                        stop_threshold = st.number_input("Stop threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
                    with col_c:
                        crop_size = st.number_input("Crop size", min_value=9, max_value=129, value=33, step=2)
                    with col_d:
                        wait_logit_bias = st.number_input("Wait logit bias", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)
                    with col_e:
                        sample_count = st.number_input("Samples", min_value=1, max_value=20, value=10, step=1)
                    disable_wait = st.checkbox(
                        "Disable wait action for diagnostic rollout",
                        value=False,
                        help="Use only to inspect whether the trained movement head can produce a path. Default rollout should keep this off.",
                    )
                    if int(sample_count) * int(max_steps) > 3000:
                        st.warning("This is a heavy preview run. Runtime scales roughly with Samples x Max steps x active agents.")

                    if int(sample_count) == 1:
                        command = [
                            get_python_executable(),
                            str(method_path / "rollout.py"),
                            "--checkpoint",
                            str(checkpoint_path),
                            "--input-dir",
                            str(input_dir),
                            "--output-dir",
                            str(output_dir),
                            "--max-steps",
                            str(int(max_steps)),
                            "--stop-threshold",
                            str(float(stop_threshold)),
                            "--crop-size",
                            str(int(crop_size)),
                            "--wait-logit-bias",
                            str(float(wait_logit_bias)),
                        ]
                    else:
                        command = [
                            get_python_executable(),
                            str(method_path / "rollout_batch.py"),
                            "--checkpoint",
                            str(checkpoint_path),
                            "--dataset-root",
                            str(DATASET_ROOT),
                            "--output-root",
                            str(run_path / "rollouts"),
                            "--split",
                            split,
                            "--sample-count",
                            str(int(sample_count)),
                            "--start-index",
                            str(int(split_df.index[case_labels.index(selected_case_label)])),
                            "--max-steps",
                            str(int(max_steps)),
                            "--stop-threshold",
                            str(float(stop_threshold)),
                            "--crop-size",
                            str(int(crop_size)),
                            "--wait-logit-bias",
                            str(float(wait_logit_bias)),
                        ]
                    if disable_wait:
                        command.append("--disable-wait")
                    render_process_controls(
                        command=command,
                        cwd=method_path,
                        run_label="Start Testing Rollout",
                        stop_label="Stop Testing",
                        log_key="testing_logs",
                        running_message="Testing rollout is running...",
                    )
                    if st.button("Clear Testing Log", use_container_width=True):
                        st.session_state.testing_logs = ""
                        st.rerun()

                with tab_preview:
                    if summary_path.exists():
                        st.subheader("Rollout Summary")
                        summary = load_json_file(summary_path, default={})
                        render_rollout_summary(summary)
                        if rollout_path.exists():
                            show_rollout_sample_or_plot(output_dir, input_dir, rollout_path)
                            render_action_trace(output_dir)
                            with st.expander("Rollout rows", expanded=False):
                                st.dataframe(pd.read_parquet(rollout_path).head(500), use_container_width=True)
                        render_batch_rollout_table(run_path)
                    else:
                        st.info("No rollout result for this case yet. Run the test first.")
                        render_batch_rollout_table(run_path)

if nav == "View results":
    st.subheader(f"Results: {selected_method}")
    runs = get_method_runs(selected_method)
    if not runs:
        st.info("No result runs found.")
    else:
        selected_run_name = st.selectbox("Run", [p.name for p in runs], key="result_run")
        run_path = result_method_path / selected_run_name
        st.caption(str(run_path))

        metrics_path = run_path / "metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            st.subheader("Training Metrics")
            if not metrics_df.empty:
                latest = metrics_df.iloc[-1]
                metric_cols = st.columns(4)
                metric_cols[0].metric("Epoch", int(latest["epoch"]))
                metric_cols[1].metric("Val Loss", f"{latest['val_loss']:.4f}")
                metric_cols[2].metric("Val Action Acc", f"{latest['val_action_acc']:.3f}")
                metric_cols[3].metric("Val Stop Acc", f"{latest['val_stop_acc']:.3f}")
                st.line_chart(metrics_df.set_index("epoch")[["train_loss", "val_loss"]])
                st.dataframe(metrics_df, use_container_width=True)
        else:
            st.warning("No `metrics.csv` found.")

        c1, c2, c3 = st.columns(3)
        c1.write("Best checkpoint")
        c1.code(str(run_path / "checkpoints" / "best.pt"))
        c2.write("Last checkpoint")
        c2.code(str(run_path / "checkpoints" / "last.pt"))
        c3.write("Action space")
        c3.code(str(run_path / "action_space.json"))

        rollouts_root = run_path / "rollouts"
        if rollouts_root.exists():
            render_batch_rollout_table(run_path)
            rollout_dirs = sorted([p for p in rollouts_root.iterdir() if p.is_dir()], key=lambda p: p.name)
            if rollout_dirs:
                st.subheader("Rollouts")
                selected_rollout = st.selectbox("Rollout", rollout_dirs, format_func=lambda p: p.name)
                summary = load_json_file(selected_rollout / "summary.json", default={})
                if summary:
                    render_rollout_summary(summary)
                rollout_path = selected_rollout / "rollout.parquet"
                if rollout_path.exists():
                    manifest_row = find_rollout_manifest_row(selected_rollout)
                    if manifest_row is not None:
                        st.header("Output preview and compare ground truth")
                        render_output_compare_row(
                            plan_name=str(manifest_row["plan_name"]),
                            sqlite_stem=str(manifest_row["sqlite_stem"]),
                            output_dir=selected_rollout,
                            split=str(manifest_row["split"]),
                        )
                    else:
                        input_dir = find_rollout_input_dir(selected_rollout)
                        if input_dir is not None and input_dir.exists():
                            show_rollout_sample_or_plot(selected_rollout, input_dir, rollout_path)
                    render_action_trace(selected_rollout)
                    with st.expander("Rollout rows", expanded=False):
                        rollout_df = pd.read_parquet(rollout_path)
                        st.dataframe(rollout_df.head(500), use_container_width=True)
                with st.expander("Summary JSON", expanded=False):
                    st.json(summary)

if nav == "Prepare Walkable Grid":
    st.subheader("Prepare Layout to Grid Walkable Area")
    st.write("Create `walkablearea_grid.json` inside each Topo_HouseGAN plan folder.")

    tab_prepare, tab_preview = st.tabs(["Prepare Grid", "Preview Grid"])

    with tab_prepare:
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            cell_size_m = st.number_input(
                "Cell size (m)",
                min_value=0.001,
                max_value=1.000,
                value=0.125,
                step=0.001,
                format="%.3f",
            )
        with col_b:
            padding_m = st.number_input(
                "Padding (m)",
                min_value=0.000,
                max_value=5.000,
                value=0.000,
                step=0.001,
                format="%.3f",
            )
        with col_c:
            wall_thickness_m = st.number_input(
                "Wall thickness (m)",
                min_value=0.001,
                max_value=2.000,
                value=0.125,
                step=0.001,
                format="%.3f",
            )
        with col_d:
            overwrite = st.checkbox("Overwrite existing files", value=True)

        plan_options = ["All plans"] + get_plan_options()
        selected_plan = st.selectbox("Plan", plan_options)

        command = [
            get_python_executable(),
            str(TOOL_DIR / "prepare_layout_to_grid_walkablearea.py"),
            "--geo-root",
            str(GEO_ROOT),
            "--cell-size-m",
            format_float3(cell_size_m),
            "--padding-m",
            format_float3(padding_m),
            "--wall-thickness-m",
            format_float3(wall_thickness_m),
        ]
        if selected_plan != "All plans":
            command.extend(["--plan-name", selected_plan])
        if overwrite:
            command.append("--overwrite")

        st.code(" ".join(command), language="bash")

        manager = st.session_state.process_manager
        run_col, stop_col = st.columns([1, 1])
        with run_col:
            if st.button("Run Utility", type="primary", disabled=manager.is_running):
                st.session_state.process_log = ""
                started = manager.start_process(command, cwd=str(PROJECT_ROOT))
                if not started:
                    st.warning("A process is already running.")
                st.rerun()
        with stop_col:
            if st.button("Stop", disabled=not manager.is_running):
                manager.stop_process()
                st.rerun()

        for line in manager.get_output():
            st.session_state.process_log += line

        if manager.is_running:
            st.info("Utility is running...")
            time.sleep(1)
            st.rerun()

        st.text_area("Output", st.session_state.process_log, height=360)

    with tab_preview:
        generated_plan_options = get_generated_plan_options()
        if not generated_plan_options:
            st.info("No `walkablearea_grid.json` files found yet.")
        else:
            preview_plan = st.selectbox("Preview plan", generated_plan_options)
            payload = load_grid_payload(preview_plan)
            meta = payload["meta"]
            summary = payload["grid_summary"]

            metric_cols = st.columns(5)
            metric_cols[0].metric("Cell size", f"{meta['cell_size_m']:.3f} m")
            metric_cols[1].metric("Grid size", f"{meta['width']} x {meta['height']}")
            metric_cols[2].metric("Walkable cells", f"{summary['walkable_cells']:,}")
            metric_cols[3].metric("Total cells", f"{summary['total_cells']:,}")
            metric_cols[4].metric("Walkable ratio", f"{summary['walkable_ratio']:.3f}")

            show_axes = st.checkbox("Show metric axes", value=True)

            fig = plot_grid_payload(payload, show_axes=show_axes)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            with st.expander("Grid metadata"):
                st.json(
                    {
                        "meta": meta,
                        "geometry_summary": payload.get("geometry_summary", {}),
                        "grid_summary": summary,
                    }
                )

if nav == "Prepare Trajectory Grid Dataset":
    st.subheader("Prepare Trajectory Grid Dataset")
    st.write("Create A/B training data directly from Topo_HouseGAN `dataswarm`, `geo`, and route metadata.")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        cell_size_m = st.number_input(
            "Cell size (m)",
            min_value=0.001,
            max_value=1.000,
            value=0.125,
            step=0.001,
            format="%.3f",
            key="trajectory_grid_cell_size_m",
        )
    with col_b:
        padding_m = st.number_input(
            "Padding (m)",
            min_value=0.000,
            max_value=5.000,
            value=0.000,
            step=0.001,
            format="%.3f",
            key="trajectory_grid_padding_m",
        )
    with col_c:
        wall_thickness_m = st.number_input(
            "Wall thickness (m)",
            min_value=0.001,
            max_value=2.000,
            value=0.125,
            step=0.001,
            format="%.3f",
            key="trajectory_grid_wall_thickness_m",
        )
    with col_d:
        overwrite = st.checkbox("Overwrite existing files", value=True, key="trajectory_grid_overwrite")

    col_e, col_f, col_g = st.columns(3)
    with col_e:
        plan_options = ["All plans"] + get_plan_options()
        selected_plan = st.selectbox("Plan", plan_options, key="trajectory_grid_plan")
    with col_f:
        max_cases = st.number_input(
            "Max cases (0 = all)",
            min_value=0,
            max_value=100000,
            value=0,
            step=1,
            key="trajectory_grid_max_cases",
        )
    with col_g:
        regenerate_grid = st.checkbox("Regenerate grid before copy", value=False, key="trajectory_grid_regenerate")

    source_root = HOUSEGAN_ROOT
    output_root = PROJECT_ROOT / "Dataset" / "Data_TrajectoryGrid" / "Topo_HouseGAN"
    command = [
        get_python_executable(),
        str(TOOL_DIR / "prepare_dataset_trajectory_grid.py"),
        "--source-root",
        str(source_root),
        "--output-root",
        str(output_root),
        "--cell-size-m",
        format_float3(cell_size_m),
        "--padding-m",
        format_float3(padding_m),
        "--wall-thickness-m",
        format_float3(wall_thickness_m),
    ]
    if selected_plan != "All plans":
        command.extend(["--plan-name", selected_plan])
    if max_cases > 0:
        command.extend(["--max-cases", str(int(max_cases))])
    if overwrite:
        command.append("--overwrite")
    if regenerate_grid:
        command.append("--regenerate-grid")

    st.code(" ".join(command), language="bash")

    manager = st.session_state.process_manager
    run_col, stop_col = st.columns([1, 1])
    with run_col:
        if st.button("Run Dataset Utility", type="primary", disabled=manager.is_running):
            st.session_state.process_log = ""
            started = manager.start_process(command, cwd=str(PROJECT_ROOT))
            if not started:
                st.warning("A process is already running.")
            st.rerun()
    with stop_col:
        if st.button("Stop", disabled=not manager.is_running, key="trajectory_grid_stop"):
            manager.stop_process()
            st.rerun()

    for line in manager.get_output():
        st.session_state.process_log += line

    if manager.is_running:
        st.info("Dataset utility is running...")
        time.sleep(1)
        st.rerun()

    report_path = output_root / "manifest_trajectory_grid.csv"
    if report_path.exists():
        st.caption(f"Report: {report_path}")
        try:
            report_df = pd.read_csv(report_path)
            st.dataframe(report_df, use_container_width=True)
        except Exception as exc:
            st.warning(f"Unable to load report: {exc}")

    st.text_area("Output", st.session_state.process_log, height=360, key="trajectory_grid_output")
