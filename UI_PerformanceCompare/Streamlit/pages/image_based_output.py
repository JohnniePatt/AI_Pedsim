from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from utils.image_paths import image_triplet
from utils.metrics_loader import (
    METRIC_ORDER,
    attach_run_label,
    load_per_image,
    load_summary,
    metric_summary_from_per_image,
)
from utils.result_scanner import RunInfo, discover_runs, get_run_by_label


LOWER_IS_BETTER = {"MAE", "MSE", "RMSE", "LPIPS"}
HIGHER_IS_BETTER = {"SSIM", "PSNR"}


def _format_metric(value: float, metric: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    if metric == "PSNR":
        return f"{value:.2f}"
    return f"{value:.4f}"


def _default_runs(runs: list[RunInfo]) -> list[str]:
    if not runs:
        return []
    image_ready = [
        r for r in runs
        if (r.path / "test_evaluation_per_image.csv").exists()
        and (r.path / "test_results" / "predictions").exists()
    ]
    labels = [r.label for r in (image_ready or runs)]
    return labels[: min(2, len(labels))]


def _summary_cards(run: RunInfo, per_image: pd.DataFrame):
    st.markdown(f"**{run.label}**")
    summary = metric_summary_from_per_image(per_image)
    if summary.empty:
        summary_file = load_summary(run.path)
        if summary_file.empty:
            st.info("No metric summary found.")
            return
        cols = st.columns(min(3, len(summary_file)))
        for idx, row in enumerate(summary_file.itertuples()):
            cols[idx % len(cols)].metric(str(row.metric), _format_metric(float(row.value), str(row.metric)))
        return

    cols = st.columns(3)
    key_metrics = ["MAE", "RMSE", "SSIM", "PSNR", "LPIPS"]
    shown = 0
    for metric in key_metrics:
        row = summary[summary["metric"] == metric]
        if row.empty:
            continue
        mean_value = float(row.iloc[0]["mean"])
        cols[shown % 3].metric(metric, _format_metric(mean_value, metric))
        shown += 1

    with st.expander("Mean / median / p95 / worst", expanded=False):
        st.dataframe(summary, use_container_width=True)


def _histogram(df: pd.DataFrame, metric: str):
    chart_df = df[["run", metric]].dropna()
    if chart_df.empty:
        st.info(f"No data for {metric}.")
        return

    bins = 24
    min_v = float(chart_df[metric].min())
    max_v = float(chart_df[metric].max())
    if min_v == max_v:
        st.info(f"{metric} has one repeated value.")
        return

    chart_df = chart_df.copy()
    chart_df["bin"] = pd.cut(chart_df[metric], bins=bins)
    grouped = chart_df.groupby(["bin", "run"], observed=False).size().reset_index(name="count")
    grouped["bin"] = grouped["bin"].astype(str)
    pivot = grouped.pivot(index="bin", columns="run", values="count").fillna(0)
    st.bar_chart(pivot, height=300)


def _scatter(df: pd.DataFrame, x_metric: str, y_metric: str):
    chart_df = df[["run", "file_name", x_metric, y_metric]].dropna()
    if chart_df.empty:
        st.info("No scatter data available for the selected metrics.")
        return
    st.scatter_chart(chart_df, x=x_metric, y=y_metric, color="run", height=360)


def _combined_per_image(selected_runs: list[RunInfo]) -> pd.DataFrame:
    frames = []
    for run in selected_runs:
        per_image = load_per_image(run.path)
        if not per_image.empty:
            frames.append(attach_run_label(per_image, run.label))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _show_image_compare(selected_runs: list[RunInfo], file_name: str):
    if not file_name:
        return

    st.markdown("### Image Comparison")
    first_input = None
    first_target = None
    predictions = []

    for run in selected_runs:
        input_path, pred_path, target_path = image_triplet(run.path, file_name)
        if first_input is None and input_path is not None:
            first_input = input_path
        if first_target is None and target_path is not None:
            first_target = target_path
        predictions.append((run.label, pred_path))

    total_cols = 2 + len(predictions)
    cols = st.columns(total_cols)
    cols[0].caption("INPUT")
    if first_input:
        cols[0].image(str(first_input), use_container_width=True)
    else:
        cols[0].info("Missing input")

    for idx, (label, pred_path) in enumerate(predictions, start=1):
        cols[idx].caption(label)
        if pred_path:
            cols[idx].image(str(pred_path), use_container_width=True)
        else:
            cols[idx].info("Missing prediction")

    cols[-1].caption("TARGET")
    if first_target:
        cols[-1].image(str(first_target), use_container_width=True)
    else:
        cols[-1].info("Missing target")


def render_image_based_output():
    st.markdown('<div class="pc-title">Image Based Output</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pc-subtitle">Compare image-generating model runs with per-image metrics and visual samples.</div>',
        unsafe_allow_html=True,
    )

    runs = discover_runs()
    if not runs:
        st.error("No evaluated runs found under AI_GenerateTrajectory/AI_Result.")
        return

    labels = [r.label for r in runs]
    selected_labels = st.multiselect(
        "Select model runs to compare",
        labels,
        default=_default_runs(runs),
        help="Pick one or more evaluated runs that contain test metrics and image outputs.",
    )

    selected_runs = [get_run_by_label(runs, label) for label in selected_labels]
    selected_runs = [run for run in selected_runs if run is not None]

    if not selected_runs:
        st.info("Select at least one run to inspect.")
        return

    st.markdown("### Run Summary")
    summary_cols = st.columns(len(selected_runs))
    for col, run in zip(summary_cols, selected_runs):
        with col:
            _summary_cards(run, load_per_image(run.path))

    combined = _combined_per_image(selected_runs)
    if combined.empty:
        st.warning("Selected runs do not have test_evaluation_per_image.csv yet.")
        return

    st.markdown("### Per-image Metrics")
    table_cols = ["run", "file_name"] + [m for m in METRIC_ORDER if m in combined.columns]
    st.dataframe(combined[table_cols], use_container_width=True, height=360)

    st.markdown("### Metric Graphs")
    graph_col1, graph_col2 = st.columns([1, 1])
    metric_options = [m for m in METRIC_ORDER if m in combined.columns]
    with graph_col1:
        hist_metric = st.selectbox("Distribution metric", metric_options, index=metric_options.index("RMSE") if "RMSE" in metric_options else 0)
        _histogram(combined, hist_metric)
    with graph_col2:
        x_metric = st.selectbox("Scatter X", metric_options, index=metric_options.index("RMSE") if "RMSE" in metric_options else 0)
        y_default = metric_options.index("SSIM") if "SSIM" in metric_options else min(1, len(metric_options) - 1)
        y_metric = st.selectbox("Scatter Y", metric_options, index=y_default)
        _scatter(combined, x_metric, y_metric)

    st.markdown("### Worst / Best Case Viewer")
    rank_metric = st.selectbox("Rank by metric", metric_options, index=metric_options.index("LPIPS") if "LPIPS" in metric_options else 0)
    ascending = rank_metric in HIGHER_IS_BETTER
    ranked = combined.dropna(subset=[rank_metric]).sort_values(rank_metric, ascending=ascending)

    mode = st.radio("Case set", ["Worst cases", "Best cases"], horizontal=True)
    if mode == "Best cases":
        ranked = ranked.sort_values(rank_metric, ascending=not ascending)

    top_n = st.slider("Rows to show", min_value=5, max_value=50, value=20, step=5)
    ranked_view = ranked.head(top_n)
    st.dataframe(ranked_view[table_cols], use_container_width=True, height=280)

    file_names = sorted(combined["file_name"].dropna().unique().tolist())
    default_file = ranked_view.iloc[0]["file_name"] if not ranked_view.empty else file_names[0]
    selected_file = st.selectbox(
        "Select image file to compare",
        file_names,
        index=file_names.index(default_file) if default_file in file_names else 0,
    )
    _show_image_compare(selected_runs, selected_file)
