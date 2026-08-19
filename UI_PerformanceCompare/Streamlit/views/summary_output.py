from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import altair as alt
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
}
DEFAULT_RUNS = {
    "Method_MLP_Keras": "run_20260414_000936",
    "Method_GNN": "run_20260421_161900",
}
MODEL_COLORS = {"MLP": "#2563eb", "GNN": "#f97316"}


@dataclass(frozen=True)
class EstimateRun:
    method: str
    run_name: str
    path: Path

    @property
    def model(self) -> str:
        return METHOD_NAMES.get(self.method, self.method)

    @property
    def label(self) -> str:
        return f"{self.model} / {self.run_name}"


def _discover_runs() -> list[EstimateRun]:
    runs = []
    if not RESULT_ROOT.exists():
        return runs
    for method_dir in sorted(path for path in RESULT_ROOT.iterdir() if path.is_dir()):
        if method_dir.name not in {"Method_MLP_Keras", "Method_GNN"}:
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


def _parity_chart(frame: pd.DataFrame, target: str, title: str, color: str):
    true_col = f"true_{target}"
    pred_col = f"pred_{target}"
    chart_data = frame[["plan", "variant_label", true_col, pred_col]].dropna()
    low = float(min(chart_data[true_col].min(), chart_data[pred_col].min()))
    high = float(max(chart_data[true_col].max(), chart_data[pred_col].max()))
    ideal = pd.DataFrame({"actual": [low, high], "predicted": [low, high]})
    points = alt.Chart(chart_data).mark_circle(size=42, opacity=0.72, color=color).encode(
        x=alt.X(f"{true_col}:Q", title="Ground truth (s)", scale=alt.Scale(domain=[low, high])),
        y=alt.Y(f"{pred_col}:Q", title="Predicted time (s)", scale=alt.Scale(domain=[low, high])),
        tooltip=["plan:N", "variant_label:N", alt.Tooltip(f"{true_col}:Q", format=".2f"), alt.Tooltip(f"{pred_col}:Q", format=".2f")],
    )
    line = alt.Chart(ideal).mark_line(color="#ef4444", strokeWidth=1.6).encode(
        x=alt.X("actual:Q", scale=alt.Scale(domain=[low, high])),
        y=alt.Y("predicted:Q", scale=alt.Scale(domain=[low, high])),
    )
    st.altair_chart((line + points).properties(height=285, title=title).interactive(), use_container_width=True)


def _render_parity_plots(frame: pd.DataFrame, runs: list[EstimateRun]):
    for run in runs:
        color = MODEL_COLORS.get(run.model, "#475569")
        st.markdown(f"### {run.model}")
        tabs = st.tabs(CONDITION_ORDER)
        for tab, condition in zip(tabs, CONDITION_ORDER):
            with tab:
                subset = frame[(frame["run"] == run.label) & (frame["condition"] == condition)]
                if subset.empty:
                    st.info(f"No {condition} predictions in this run.")
                    continue
                columns = st.columns(3)
                for column, (display, target) in zip(columns, TARGETS.items()):
                    with column:
                        _parity_chart(subset, target, display, color)
                st.caption(f"{condition}: {len(subset)} scenarios · red line = ideal prediction")


def _error_distribution(frame: pd.DataFrame, runs: list[EstimateRun]):
    rows = []
    for run in runs:
        subset = frame[frame["run"] == run.label]
        for display, target in TARGETS.items():
            error = (subset[f"pred_{target}"] - subset[f"true_{target}"]).abs()
            rows.extend({"Model": run.model, "Output": display, "Absolute error (s)": value} for value in error)
    chart_data = pd.DataFrame(rows)
    chart = alt.Chart(chart_data).mark_boxplot(size=30, extent="min-max").encode(
        x=alt.X("Output:N", sort=list(TARGETS), title=None),
        y=alt.Y("Absolute error (s):Q", scale=alt.Scale(zero=True)),
        color=alt.Color("Model:N", scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))),
        xOffset="Model:N",
        tooltip=["Model:N", "Output:N"],
    ).properties(height=360)
    st.altair_chart(chart, use_container_width=True)


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
        '<div class="pc-subtitle">MLP and GNN surrogate-model results for fastest, average, and slowest pedestrian travel-time prediction.</div>',
        unsafe_allow_html=True,
    )

    runs = _discover_runs()
    if not runs:
        st.error("No AI_Estimate prediction artifacts were found.")
        return

    mlp_runs = [run for run in runs if run.method == "Method_MLP_Keras"]
    gnn_runs = [run for run in runs if run.method == "Method_GNN"]
    if not mlp_runs or not gnn_runs:
        st.error("Summary Output requires both an MLP run and a GNN run with test predictions.")
        return

    defaults = _default_labels(runs)
    default_mlp = next((run.label for run in mlp_runs if run.label in defaults), mlp_runs[0].label)
    default_gnn = next((run.label for run in gnn_runs if run.label in defaults), gnn_runs[0].label)
    selector_columns = st.columns(2)
    with selector_columns[0]:
        mlp_label = st.selectbox(
            "MLP run",
            [run.label for run in mlp_runs],
            index=[run.label for run in mlp_runs].index(default_mlp),
        )
    with selector_columns[1]:
        gnn_label = st.selectbox(
            "GNN run",
            [run.label for run in gnn_runs],
            index=[run.label for run in gnn_runs].index(default_gnn),
        )
    selected_runs = [
        next(run for run in mlp_runs if run.label == mlp_label),
        next(run for run in gnn_runs if run.label == gnn_label),
    ]

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

    st.markdown("## Parity Plot Analysis")
    st.markdown(
        "Ground truth from JuPedSim is shown on the x-axis and AI-predicted travel time on the y-axis. "
        "Plots follow the paper structure: each model is separated by occupancy condition and by fastest, average, and slowest output."
    )
    _render_parity_plots(predictions, selected_runs)

    st.markdown("## Error Distribution")
    _error_distribution(predictions, selected_runs)

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
    if {"MLP", "GNN"}.issubset(set(metrics["Model"])):
        mlp = metrics[metrics["Model"] == "MLP"].iloc[0]
        gnn = metrics[metrics["Model"] == "GNN"].iloc[0]
        ratio = gnn["MAE (s)"] / mlp["MAE (s)"] if mlp["MAE (s)"] else np.nan
        st.markdown(
            f"For the selected paper runs, **MLP records lower overall error than GNN**: "
            f"MAE {mlp['MAE (s)']:.2f} s versus {gnn['MAE (s)']:.2f} s, "
            f"or approximately {ratio:.1f}x lower error. The parity plots should be used to inspect where this gap occurs, "
            "especially for slowest-agent and high-density cases."
        )
    else:
        st.info("Select the paper MLP and GNN runs together to show the comparative interpretation.")
    st.caption(
        "This page reports only the 167 test scenarios present in each selected predictions.csv. "
        "It does not claim the 862-case trajectory-evaluation protocol used by other modules."
    )
