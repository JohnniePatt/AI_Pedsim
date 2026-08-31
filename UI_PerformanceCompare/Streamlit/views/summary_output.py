from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = PROJECT_ROOT / "AI_Estimate" / "AI_result"

TARGETS = {
    "Fastest": "min_agent_time_s",
    "Average": "mean_agent_time_s",
    "Slowest": "max_agent_time_s",
}
CONDITION_ORDER = ["N Agents", "N/2 Agents", "1 Agent"]
METHOD_NAMES = {
    "Method_MLP_Keras": "MLP",
    "Method_GNN": "GNN",
    "Method_MLP_Keras_dataestimate2": "MLP",
    "Method_GNN_dataestimate2": "GNN",
    "Method_XGBoost": "XGBoost",
}
METHOD_DATASETS = {
    "Method_MLP_Keras": "Data Estimate 1",
    "Method_GNN": "Data Estimate 1",
    "Method_MLP_Keras_dataestimate2": "Data Estimate 2",
    "Method_GNN_dataestimate2": "Data Estimate 2",
    "Method_XGBoost": "Data Estimate 2",
}
DEFAULT_RUNS = {
    "Method_MLP_Keras_dataestimate2": "run_20260820_102231",
    "Method_GNN_dataestimate2": "run_20260820_102427",
    "Method_XGBoost": "run_20260831T085815Z_seed042",
}
MODEL_COLORS = {"MLP": "#2563eb", "GNN": "#f97316", "XGBoost": "#16a34a"}


@dataclass(frozen=True)
class EstimateRun:
    method: str
    run_name: str
    path: Path

    @property
    def model(self) -> str:
        return METHOD_NAMES.get(self.method, self.method)

    @property
    def dataset(self) -> str:
        return METHOD_DATASETS.get(self.method, "Unknown dataset")

    @property
    def label(self) -> str:
        return f"{self.model} · {self.dataset} / {self.run_name}"


def _discover_runs() -> list[EstimateRun]:
    runs = []
    if not RESULT_ROOT.exists():
        return runs
    for method_dir in sorted(path for path in RESULT_ROOT.iterdir() if path.is_dir()):
        if method_dir.name not in METHOD_NAMES:
            continue
        output_dir = method_dir / "outputs"
        if not output_dir.exists():
            continue
        for run_dir in output_dir.iterdir():
            if (run_dir / "test_eval" / "predictions.csv").exists():
                runs.append(EstimateRun(method_dir.name, run_dir.name, run_dir))
    return sorted(runs, key=lambda run: run.path.stat().st_mtime, reverse=True)


def _default_labels(runs: list[EstimateRun]) -> list[str]:
    labels = []
    for method, run_name in DEFAULT_RUNS.items():
        match = next((run for run in runs if run.method == method and run.run_name == run_name), None)
        if match:
            labels.append(match.label)
    return labels or [run.label for run in runs[:2]]


@st.cache_data(show_spinner=False)
def _load_predictions(path: str, label: str, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "variant_label",
        *[f"true_{target}" for target in TARGETS.values()],
        *[f"pred_{target}" for target in TARGETS.values()],
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.copy()
    frame["run"] = label
    frame["model"] = model
    frame["condition"] = frame["variant_label"].map(
        {"N Agent": "N Agents", "N/2 Agent": "N/2 Agents", "1 Agent": "1 Agent"}
    ).fillna(frame["variant_label"])
    return frame


def _combine_predictions(runs: list[EstimateRun]) -> pd.DataFrame:
    frames = [
        _load_predictions(str(run.path / "test_eval" / "predictions.csv"), run.label, run.model)
        for run in runs
    ]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _metric_row(frame: pd.DataFrame, model: str, run: str, condition: str = "All") -> dict:
    errors = []
    row = {"Model": model, "Run": run, "Condition": condition, "Scenarios": len(frame)}
    for display, target in TARGETS.items():
        error = frame[f"pred_{target}"] - frame[f"true_{target}"]
        errors.append(error.to_numpy())
        row[f"{display} MAE (s)"] = float(error.abs().mean())
        row[f"{display} RMSE (s)"] = float(np.sqrt(np.square(error).mean()))
    flattened = np.concatenate(errors)
    row["MAE (s)"] = float(np.abs(flattened).mean())
    row["MSE (s²)"] = float(np.square(flattened).mean())
    row["RMSE (s)"] = float(np.sqrt(np.square(flattened).mean()))
    return row


def _overall_metrics(frame: pd.DataFrame, runs: list[EstimateRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        subset = frame[frame["run"] == run.label]
        if not subset.empty:
            rows.append(_metric_row(subset, run.model, run.run_name))
    return pd.DataFrame(rows)


def _target_metrics(frame: pd.DataFrame, runs: list[EstimateRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        subset = frame[frame["run"] == run.label]
        if subset.empty:
            continue
        for display, target in TARGETS.items():
            error = subset[f"pred_{target}"] - subset[f"true_{target}"]
            rows.append(
                {
                    "Model": run.model,
                    "Run": run.run_name,
                    "Output": display,
                    "Scenarios": len(subset),
                    "MAE (s)": float(error.abs().mean()),
                    "MSE (s²)": float(np.square(error).mean()),
                    "RMSE (s)": float(np.sqrt(np.square(error).mean())),
                }
            )
    return pd.DataFrame(rows)


def _condition_metrics(frame: pd.DataFrame, runs: list[EstimateRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        for condition in CONDITION_ORDER:
            subset = frame[(frame["run"] == run.label) & (frame["condition"] == condition)]
            if not subset.empty:
                rows.append(_metric_row(subset, run.model, run.run_name, condition))
    return pd.DataFrame(rows)


def _metric_table(metrics: pd.DataFrame):
    columns = ["Model", "Run", "Scenarios", "MAE (s)", "MSE (s²)", "RMSE (s)"]

    def highlight_best(series: pd.Series):
        numeric = pd.to_numeric(series, errors="coerce")
        if str(series.name) not in {"MAE (s)", "MSE (s²)", "RMSE (s)"} or numeric.dropna().empty:
            return ["" for _ in series]
        best = numeric.min()
        return ["font-weight:700;background-color:#dcfce7" if pd.notna(value) and value == best else "" for value in numeric]

    st.dataframe(
        metrics[columns].style.format(
            {"Scenarios": "{:,.0f}", "MAE (s)": "{:.2f}", "MSE (s²)": "{:.2f}", "RMSE (s)": "{:.2f}"}
        ).apply(highlight_best, axis=0),
        use_container_width=True,
        hide_index=True,
    )


def _summary_cards(metrics: pd.DataFrame):
    columns = st.columns(len(metrics))
    for column, (_, row) in zip(columns, metrics.iterrows()):
        with column:
            st.markdown(f"**{row['Model']}**")
            metric_columns = st.columns(3)
            metric_columns[0].metric("MAE", f"{row['MAE (s)']:.2f} s")
            metric_columns[1].metric("MSE", f"{row['MSE (s²)']:.2f} s²")
            metric_columns[2].metric("RMSE", f"{row['RMSE (s)']:.2f} s")
            st.caption(f"{row['Run']} · {int(row['Scenarios'])} test scenarios")


@st.cache_data(show_spinner=False)
def _load_computational_efficiency(path: str) -> dict | None:
    artifact = Path(path) / "test_eval" / "computational_efficiency.json"
    if not artifact.exists():
        return None
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _render_computational_efficiency(runs: list[EstimateRun]):
    rows = []
    available = 0
    for run in runs:
        payload = _load_computational_efficiency(str(run.path))
        if payload is None:
            rows.append(
                {
                    "Model": run.model,
                    "Run": run.run_name,
                    "Scenarios": "—",
                    "JuPedSim wall time (s)": "—",
                    "AI inference wall time (s)": "—",
                    "AI latency/scenario (ms)": "—",
                    "Speed-up": "—",
                    "Status": "Not evaluated",
                }
            )
            continue

        available += 1
        rows.append(
            {
                "Model": run.model,
                "Run": run.run_name,
                "Scenarios": payload.get("scenario_count", "—"),
                "JuPedSim wall time (s)": payload.get("simulation_wall_time_total_s", "—"),
                "AI inference wall time (s)": payload.get("ai_inference_wall_time_total_s", "—"),
                "AI latency/scenario (ms)": payload.get("ai_latency_per_scenario_ms", "—"),
                "Speed-up": payload.get("speedup_vs_simulation", "—"),
                "Status": "Available" if payload.get("research_valid") is True else "Preliminary",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if available == 0:
        st.info(
            "Computational-efficiency results have not been evaluated for the selected runs. "
            "The existing simulation_duration_s field is simulated travel duration, not JuPedSim wall-clock runtime, "
            "so it is not used to calculate AI speed-up."
        )
    elif available < len(runs):
        st.warning(
            f"Timing artifacts are available for {available}/{len(runs)} selected runs. "
            "Rows without a comparable benchmark remain marked Not evaluated."
        )
    st.caption(
        "A valid comparison requires the same 862-scenario test set, declared hardware, warm-up policy, "
        "repeat count, timing scope, and wall-clock measurements for both JuPedSim and AI inference."
    )


def _render_target_order_status(frame: pd.DataFrame, runs: list[EstimateRun]):
    xgboost_runs = [run for run in runs if run.model == "XGBoost"]
    for run in xgboost_runs:
        subset = frame[frame["run"] == run.label]
        if subset.empty:
            continue
        valid = (
            (subset["pred_min_agent_time_s"] <= subset["pred_mean_agent_time_s"])
            & (subset["pred_mean_agent_time_s"] <= subset["pred_max_agent_time_s"])
        )
        violations = int((~valid).sum())
        rate = float((~valid).mean())
        if violations:
            st.warning(
                f"XGBoost vanilla output has {violations:,}/{len(subset):,} rows "
                f"({rate:.2%}) where predicted min ≤ mean ≤ max is not satisfied. "
                "The comparison uses uncorrected raw predictions; no ordering post-processing is applied."
            )
        else:
            st.success("XGBoost predictions satisfy min ≤ mean ≤ max for every selected test row.")


@st.cache_data(show_spinner=False)
def _parity_figure_png(frame: pd.DataFrame, color: str) -> bytes:
    """Render one report-ready, non-interactive PNG with all three targets."""
    figure = Figure(figsize=(16, 4.8), dpi=160, facecolor="white")
    axes = figure.subplots(1, 3)
    for axis, (title, target) in zip(axes, TARGETS.items()):
        true_col = f"true_{target}"
        pred_col = f"pred_{target}"
        chart_data = frame[[true_col, pred_col]].dropna()
        true = chart_data[true_col].to_numpy(dtype=float)
        predicted = chart_data[pred_col].to_numpy(dtype=float)
        low = float(min(true.min(), predicted.min()))
        high = float(max(true.max(), predicted.max()))
        margin = max((high - low) * 0.025, 0.5)
        axis.scatter(true, predicted, s=18, alpha=0.62, color=color)
        axis.plot([low, high], [low, high], color="#ff4040", linewidth=1.6)
        axis.set_xlim(low - margin, high + margin)
        axis.set_ylim(low - margin, high + margin)
        axis.set_title(title, fontsize=15, pad=10)
        axis.set_xlabel("Ground truth (s)", fontsize=11)
        axis.set_ylabel("Predicted time (s)", fontsize=11)
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.tick_params(labelsize=10)
    figure.tight_layout(w_pad=2.2)
    output = BytesIO()
    figure.savefig(output, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    return output.getvalue()


def _render_parity_plots(frame: pd.DataFrame, runs: list[EstimateRun]):
    for run in runs:
        color = MODEL_COLORS.get(run.model, "#475569")
        st.markdown(f"### {run.model}")
        st.caption(f"Run: `{run.run_name}` · Dataset: {run.dataset}")
        tabs = st.tabs(CONDITION_ORDER)
        for tab, condition in zip(tabs, CONDITION_ORDER):
            with tab:
                subset = frame[(frame["run"] == run.label) & (frame["condition"] == condition)]
                if subset.empty:
                    st.info(f"No {condition} predictions in this run.")
                    continue
                png = _parity_figure_png(
                    subset[[
                        *[f"true_{target}" for target in TARGETS.values()],
                        *[f"pred_{target}" for target in TARGETS.values()],
                    ]],
                    color,
                )
                st.image(png, use_container_width=True)
                safe_model = run.model.lower().replace(" ", "_")
                safe_condition = condition.lower().replace("/", "_").replace(" ", "_")
                st.download_button(
                    "Download parity plot (PNG)",
                    data=png,
                    file_name=f"parity_{safe_model}_{safe_condition}.png",
                    mime="image/png",
                    key=f"parity_download_{run.method}_{run.run_name}_{safe_condition}",
                )
                st.caption(f"{condition}: {len(subset)} scenarios · red line = ideal prediction")


@st.cache_data(show_spinner=False)
def _error_distribution_png(
    frame: pd.DataFrame,
    run_labels: tuple[tuple[str, str, str], ...],
) -> bytes:
    figure = Figure(figsize=(11, 5.2), dpi=160, facecolor="white")
    axis = figure.subplots(1, 1)
    output_names = list(TARGETS)
    model_count = len(run_labels)
    group_width = 0.72
    box_width = group_width / max(model_count, 1) * 0.78
    legend_handles = []
    legend_labels = []
    model_counts = {}
    for model, _, _ in run_labels:
        model_counts[model] = model_counts.get(model, 0) + 1
    for model_index, (model, run_label, run_name) in enumerate(run_labels):
        subset = frame[frame["run"] == run_label]
        values = [
            (subset[f"pred_{target}"] - subset[f"true_{target}"]).abs().dropna().to_numpy()
            for target in TARGETS.values()
        ]
        positions = [
            output_index - group_width / 2 + (model_index + 0.5) * group_width / model_count
            for output_index in range(len(output_names))
        ]
        color = MODEL_COLORS.get(model, "#475569")
        boxes = axis.boxplot(
            values,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            flierprops={"markersize": 2.5, "alpha": 0.35, "markeredgecolor": color},
            medianprops={"color": "#111827", "linewidth": 1.2},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.7)
        legend_handles.append(boxes["boxes"][0])
        legend_labels.append(model if model_counts[model] == 1 else f"{model} · {run_name}")
    axis.set_xticks(range(len(output_names)), output_names)
    axis.set_ylabel("Absolute error (s)")
    axis.set_title("Absolute Error Distribution by Predicted Output")
    axis.grid(True, axis="y", alpha=0.22, linewidth=0.7)
    axis.legend(legend_handles, legend_labels, loc="upper left", frameon=False)
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    return output.getvalue()


def _render_error_distribution(frame: pd.DataFrame, runs: list[EstimateRun]):
    columns = ["run"]
    for target in TARGETS.values():
        columns.extend([f"true_{target}", f"pred_{target}"])
    run_labels = tuple((run.model, run.label, run.run_name) for run in runs)
    png = _error_distribution_png(frame[columns], run_labels)
    st.image(png, use_container_width=True)
    st.download_button(
        "Download error distribution (PNG)",
        data=png,
        file_name="travel_time_error_distribution.png",
        mime="image/png",
        key="summary_error_distribution_download",
    )


def _case_evidence(frame: pd.DataFrame):
    model_options = frame["run"].drop_duplicates().tolist()
    selected_run = st.selectbox("Model run", model_options, key="summary_case_run")
    subset = frame[frame["run"] == selected_run].copy()
    selected_condition = st.selectbox("Occupancy", ["All", *CONDITION_ORDER], key="summary_case_condition")
    if selected_condition != "All":
        subset = subset[subset["condition"] == selected_condition]
    columns = ["plan", "variant_label", "computed_agents", "topology_centerline_distance_m"]
    for target in TARGETS.values():
        columns.extend([f"true_{target}", f"pred_{target}", f"abs_error_{target}"])
    columns = [column for column in columns if column in subset]
    st.dataframe(subset[columns], use_container_width=True, hide_index=True, height=380)
    st.download_button(
        "Download selected prediction evidence (CSV)",
        subset[columns].to_csv(index=False).encode("utf-8"),
        file_name="travel_time_prediction_evidence.csv",
        mime="text/csv",
    )


def render_summary_output():
    st.markdown('<div class="pc-title">Summary Output</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pc-subtitle">MLP, GNN, and XGBoost surrogate-model results for fastest, average, and slowest pedestrian travel-time prediction.</div>',
        unsafe_allow_html=True,
    )

    runs = _discover_runs()
    if not runs:
        st.error("No AI_Estimate prediction artifacts were found.")
        return

    labels = [run.label for run in runs]
    selected_labels = st.multiselect(
        "Select model runs to compare",
        labels,
        default=_default_labels(runs),
        help="Pick one or more evaluated MLP, GNN, or XGBoost runs with test predictions.",
    )
    selected_runs = [run for run in runs if run.label in selected_labels]
    method_order = {"MLP": 0, "GNN": 1, "XGBoost": 2}
    selected_runs.sort(key=lambda run: (method_order.get(run.model, 99), run.run_name))
    if not selected_runs:
        st.info("Select at least one evaluated run to inspect.")
        return

    predictions = _combine_predictions(selected_runs)
    if predictions.empty:
        st.error("The selected prediction files do not match the AI_Estimate output schema.")
        return
    metrics = _overall_metrics(predictions, selected_runs)

    st.markdown("## Model Performance Comparison")
    st.caption("Test-set errors recomputed from true/predicted travel times in predictions.csv; lower is better.")
    _metric_table(metrics)

    st.markdown("### Run Summary")
    _summary_cards(metrics)
    _render_target_order_status(predictions, selected_runs)

    st.markdown("### Computational Efficiency Comparison (Simulation vs AI)")
    _render_computational_efficiency(selected_runs)

    st.markdown("## Performance by Predicted Output")
    target_metrics = _target_metrics(predictions, selected_runs)
    st.dataframe(
        target_metrics.style.format(
            {
                "Scenarios": "{:,.0f}",
                "MAE (s)": "{:.2f}",
                "MSE (s²)": "{:.2f}",
                "RMSE (s)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## Parity Plot Analysis")
    st.markdown(
        "Ground truth from JuPedSim is shown on the x-axis and AI-predicted travel time on the y-axis. "
        "Each cached Matplotlib image combines fastest, average, and slowest outputs and is separated by model and occupancy condition. "
        "The static PNG can be downloaded directly for report use."
    )
    _render_parity_plots(predictions, selected_runs)

    st.markdown("## Error Distribution")
    _render_error_distribution(predictions, selected_runs)

    st.markdown("## Performance by Occupancy Condition")
    condition_metrics = _condition_metrics(predictions, selected_runs)
    display_columns = ["Model", "Condition", "Scenarios", "MAE (s)", "MSE (s²)", "RMSE (s)"]
    st.dataframe(
        condition_metrics[display_columns].style.format(
            {"Scenarios": "{:,.0f}", "MAE (s)": "{:.2f}", "MSE (s²)": "{:.2f}", "RMSE (s)": "{:.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## Per-Scenario Prediction Evidence")
    _case_evidence(predictions)

    st.markdown("## Result Interpretation")
    ranked = metrics.sort_values("MAE (s)").reset_index(drop=True)
    best = ranked.iloc[0]
    comparison = ", ".join(
        f"{row['Model']} {row['MAE (s)']:.2f} s" for _, row in ranked.iterrows()
    )
    st.markdown(
        f"For the selected runs, **{best['Model']} records the lowest overall MAE**. "
        f"The selected-run MAE values are {comparison}. The parity plots and per-condition tables "
        "should be used to inspect where the differences occur, especially for slowest-agent and high-density cases."
    )
    scenario_summary = ", ".join(
        f"{row['Model']} {int(row['Scenarios']):,}" for _, row in metrics.iterrows()
    )
    st.caption(
        f"Scenario counts come directly from each selected predictions.csv ({scenario_summary}). "
        "Data Estimate 2 follows the canonical Image-based HouseGAN split: 2,603 train, 439 validation, and 862 test scenarios."
    )
