from __future__ import annotations

import math

import altair as alt
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
RUN_COLOR_RANGE = [
    "#e11d48",
    "#2563eb",
    "#16a34a",
    "#f97316",
    "#7c3aed",
    "#0891b2",
    "#ca8a04",
    "#4b5563",
]


def _format_metric(value: float, metric: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    if metric == "PSNR":
        return f"{value:.2f}"
    return f"{value:.4f}"


def _default_runs(runs: list[RunInfo]) -> list[str]:
    if not runs:
        return []
    preferred_labels = [
        "Method_pix2pixHD / run_HD_20260517_133538_BestForBW",
        "Method_PlainUnet / run_PlainUNet_20260708_211818",
        "Method_pix2pixhd_No_D / run_HD_NoD_20260709_180550",
        "Method_CVAE / run_CVAE_20260627_193237_config2"
    ]
    defaults = [label for label in preferred_labels if any(r.label == label for r in runs)]
    if defaults:
        return defaults
    labels = [r.label for r in runs]
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


def _histogram(df: pd.DataFrame, metric: str, run_colors: dict[str, str]):
    chart_df = df[["run", metric]].dropna()
    if chart_df.empty:
        st.info(f"No data for {metric}.")
        return

    domain = list(run_colors.keys())
    range_colors = list(run_colors.values())
    color_scale = alt.Scale(domain=domain, range=range_colors)

    chart = (
        alt.Chart(chart_df)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=24), title=metric),
            y=alt.Y("count()", stack=True, title="Count"),
            color=alt.Color("run:N", scale=color_scale, title="Run"),
            tooltip=[alt.Tooltip("run:N"), alt.Tooltip("count()")]
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def _scatter(df: pd.DataFrame, x_metric: str, y_metric: str, run_colors: dict[str, str]):
    chart_df = df[["run", "file_name", x_metric, y_metric]].dropna()
    if chart_df.empty:
        st.info("No scatter data available for the selected metrics.")
        return
        
    domain = list(run_colors.keys())
    range_colors = list(run_colors.values())
    color_scale = alt.Scale(domain=domain, range=range_colors)
    
    chart = (
        alt.Chart(chart_df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(f"{x_metric}:Q", scale=alt.Scale(zero=False)),
            y=alt.Y(f"{y_metric}:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("run:N", scale=color_scale, title="Run"),
            tooltip=["file_name:N", "run:N", f"{x_metric}:Q", f"{y_metric}:Q"]
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def _short_run_labels(run_labels: list[str]) -> dict[str, str]:
    short_labels = [label.split(" / ")[-1] for label in run_labels]
    if len(short_labels) == len(set(short_labels)):
        return dict(zip(run_labels, short_labels))
    return dict(zip(run_labels, run_labels))


def _metric_compare_scores(
    df: pd.DataFrame,
    metric_options: list[str],
    run_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    detail_rows = []

    for metric in metric_options:
        chart_df = df[["run", "file_name", metric]].dropna()
        if chart_df.empty:
            continue

        pivot = chart_df.pivot_table(
            index="file_name",
            columns="run",
            values=metric,
            aggfunc="mean",
        )
        compared_runs = [label for label in run_labels if label in pivot.columns]
        if len(compared_runs) < 2:
            continue

        pivot = pivot[compared_runs].dropna()
        if pivot.empty:
            continue

        lower_is_better = metric in LOWER_IS_BETTER
        best_values = pivot.min(axis=1) if lower_is_better else pivot.max(axis=1)
        winners = pivot.eq(best_values, axis=0)

        metric_winners = []
        for file_name, winner_flags in winners.iterrows():
            winning_runs = winner_flags[winner_flags].index.tolist()
            winner = winning_runs[0] if len(winning_runs) == 1 else "Tie"
            row_values = pivot.loc[file_name].to_dict()
            metric_winners.append(winner)
            detail_row = {
                "metric": metric,
                "file_name": file_name,
                "winner": winner,
                "best_value": best_values.loc[file_name],
                "direction": "lower" if lower_is_better else "higher",
            }
            detail_row.update({f"value__{run_label}": row_values[run_label] for run_label in compared_runs})
            detail_rows.append(detail_row)

        counts = pd.Series(metric_winners).value_counts()
        winner_order = compared_runs + (["Tie"] if "Tie" in counts.index else [])
        counts = counts.reindex(winner_order, fill_value=0)
        total_images = int(counts.sum())
        for winner, score in counts.items():
            score_rows.append(
                {
                    "metric": metric,
                    "winner": winner,
                    "score": int(score),
                    "total_images": total_images,
                    "win_rate": float(score) / total_images if total_images else 0.0,
                }
            )

    return pd.DataFrame(score_rows), pd.DataFrame(detail_rows)


def _donut_chart(score_df: pd.DataFrame, metric: str, display_labels: dict[str, str], run_colors: dict[str, str]):
    metric_scores = score_df[score_df["metric"] == metric].sort_values("score", ascending=False)
    if metric_scores.empty:
        st.info(f"No compare data for {metric}.")
        return

    chart_df = metric_scores.copy()
    chart_df["winner_label"] = chart_df["winner"].map(lambda label: display_labels.get(str(label), str(label)))
    chart_df["score"] = chart_df["score"].astype(int)
    total = int(chart_df["score"].sum())
    if total <= 0:
        st.info(f"No compare data for {metric}.")
        return

    chart_df["win_rate"] = (chart_df["score"] / total * 100).round(2)
    chart_df = chart_df[chart_df["score"] > 0]

    # Map display_label (short label) to its chosen color
    domain = []
    range_colors = []
    for raw_label, color in run_colors.items():
        short_label = display_labels.get(raw_label, raw_label)
        domain.append(short_label)
        range_colors.append(color)

    color_scale = alt.Scale(domain=domain, range=range_colors)

    st.markdown(f"**{metric}**")
    chart = (
        alt.Chart(chart_df)
        .mark_arc(innerRadius=58, outerRadius=92)
        .encode(
            theta=alt.Theta("score:Q", stack=True),
            color=alt.Color(
                "winner_label:N",
                title=None,
                scale=color_scale,
            ),
            tooltip=[
                alt.Tooltip("winner_label:N", title="Winner"),
                alt.Tooltip("score:Q", title="Score"),
                alt.Tooltip("total_images:Q", title="Images"),
                alt.Tooltip("win_rate:Q", title="Win rate (%)"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def _show_metric_compare(df: pd.DataFrame, metric_options: list[str], selected_runs: list[RunInfo], run_colors: dict[str, str]):
    st.markdown("### Metric compare")

    run_labels = [run.label for run in selected_runs]
    if len(run_labels) < 2:
        st.info("Select at least two runs to compare metric winners.")
        return

    score_df, detail_df = _metric_compare_scores(df, metric_options, run_labels)
    if score_df.empty:
        st.info("No matching per-image metric rows found for the selected runs.")
        return

    display_labels = _short_run_labels(run_labels)
    display_labels["Tie"] = "Tie"

    st.caption("Each image gives 1 point per metric to the winning run. MAE, MSE, RMSE and LPIPS use lower-is-better; SSIM and PSNR use higher-is-better.")

    champion_rows = []
    for metric in metric_options:
        metric_scores = score_df[score_df["metric"] == metric]
        if metric_scores.empty:
            continue
        leader = metric_scores.sort_values(["score", "winner"], ascending=[False, True]).iloc[0]
        champion_rows.append(
            {
                "metric": metric,
                "winner": display_labels.get(str(leader["winner"]), str(leader["winner"])),
                "score": int(leader["score"]),
                "total_images": int(leader["total_images"]),
                "win_rate": round(float(leader["win_rate"]) * 100, 2),
            }
        )

    if champion_rows:
        st.dataframe(pd.DataFrame(champion_rows), use_container_width=True, height=250)

    for start in range(0, len(metric_options), 3):
        cols = st.columns(3)
        for col, metric in zip(cols, metric_options[start:start + 3]):
            with col:
                _donut_chart(score_df, metric, display_labels, run_colors)

    score_view = score_df.copy()
    score_view["winner"] = score_view["winner"].map(lambda label: display_labels.get(str(label), str(label)))
    score_view["win_rate"] = (score_view["win_rate"] * 100).round(2)

    st.markdown("#### Score by metric")
    st.dataframe(
        score_view.sort_values(["metric", "score"], ascending=[True, False]),
        use_container_width=True,
        height=260,
    )

    if not detail_df.empty:
        detail_view = detail_df.copy()
        detail_view["winner"] = detail_view["winner"].map(lambda label: display_labels.get(str(label), str(label)))
        detail_metric = st.selectbox(
            "Per-image winner metric",
            metric_options,
            index=metric_options.index("MAE") if "MAE" in metric_options else 0,
        )
        detail_view = detail_view[detail_view["metric"] == detail_metric].copy()
        value_rename = {
            f"value__{run_label}": display_labels.get(run_label, run_label)
            for run_label in run_labels
            if f"value__{run_label}" in detail_view.columns
        }
        detail_view = detail_view.rename(columns=value_rename)
        ordered_cols = [
            "file_name",
            *value_rename.values(),
            "winner",
            "best_value",
            "direction",
        ]
        ordered_cols = [col for col in ordered_cols if col in detail_view.columns]
        st.markdown("#### Per-image winners")
        st.dataframe(detail_view[ordered_cols], use_container_width=True, height=360)


def _sample_metric_scatter(df: pd.DataFrame, metric: str, sample_order: dict[str, int], run_colors: dict[str, str]):
    chart_df = df[["run", "file_name", metric]].dropna().copy()
    if chart_df.empty:
        st.info(f"No sample scatter data for {metric}.")
        return

    chart_df["sample"] = chart_df["file_name"].map(sample_order)
    chart_df = chart_df.dropna(subset=["sample"])
    chart_df["sample"] = chart_df["sample"].astype(int)

    domain = list(run_colors.keys())
    range_colors = list(run_colors.values())
    color_scale = alt.Scale(domain=domain, range=range_colors)

    st.markdown(f"**{metric} by sample**")
    chart = (
        alt.Chart(chart_df)
        .mark_circle(size=38, opacity=0.72)
        .encode(
            x=alt.X("sample:Q", title="Sample"),
            y=alt.Y(f"{metric}:Q", title=metric),
            color=alt.Color("run:N", title="Run", scale=color_scale),
            tooltip=[
                alt.Tooltip("sample:Q", title="Sample"),
                alt.Tooltip("file_name:N", title="File"),
                alt.Tooltip("run:N", title="Run"),
                alt.Tooltip(f"{metric}:Q", title=metric, format=".6f"),
            ],
        )
        .interactive()
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def _show_sample_metric_scatters(df: pd.DataFrame, metric_options: list[str], run_colors: dict[str, str]):
    st.markdown("### Metric sample scatter")
    st.caption("X is the sample index from each image file. Y is the selected metric value, with points colored by run.")

    file_names = sorted(df["file_name"].dropna().unique().tolist())
    sample_order = {file_name: idx + 1 for idx, file_name in enumerate(file_names)}
    if not sample_order:
        st.info("No samples found for metric scatter plots.")
        return

    for start in range(0, len(metric_options), 2):
        cols = st.columns(2)
        for col, metric in zip(cols, metric_options[start:start + 2]):
            with col:
                _sample_metric_scatter(df, metric, sample_order, run_colors)


def _show_occupancy_level_analysis(
    df: pd.DataFrame,
    selected_runs: list[RunInfo],
    run_colors: dict[str, str],
    metric_options: list[str]
):
    st.markdown("## Occupancy Level Analysis")
    st.caption("Detailed metric analysis and sample scatter plots broken down by occupancy levels (N, N-half, and 1-agent) shown sequentially for print reporting.")

    # 1. Parse occupancy levels from file name
    df_copy = df.copy()
    def get_occupancy_level(fname):
        fname_str = str(fname).lower()
        if "full" in fname_str:
            return "N (Full)"
        elif "half" in fname_str:
            return "N-half (Half)"
        elif "single" in fname_str:
            return "1-agent (Single)"
        return "Unknown"

    df_copy["occupancy_level"] = df_copy["file_name"].apply(get_occupancy_level)
    df_copy = df_copy[df_copy["occupancy_level"] != "Unknown"]

    METHOD_DISPLAY_NAMES = {
        "Method_pix2pixHD": "pix2pixHD",
        "Method_PlainUnet": "Plain U-Net",
        "Method_pix2pixhd_No_D": "pix2pixHD (No D)",
        "Method_CVAE": "CVAE"
    }

    levels = [
        ("N (Full)", "👥 N (Full occupancy)"),
        ("N-half (Half)", "🌗 N-half (Half occupancy)"),
        ("1-agent (Single)", "👤 1-agent (Single agent)")
    ]

    for level_key, level_title in levels:
        st.markdown(f"### {level_title}")
        level_df = df_copy[df_copy["occupancy_level"] == level_key]
        if level_df.empty:
            st.info(f"No data available for {level_key}.")
            st.markdown("---")
            continue

        # 2. Compute and Display average metrics table
        level_summary = []
        for run in selected_runs:
            run_data = level_df[level_df["run"] == run.label]
            row = {"Model": METHOD_DISPLAY_NAMES.get(run.method, run.method)}
            for m in metric_options:
                if not run_data.empty and m in run_data.columns:
                    row[m] = float(run_data[m].mean())
                else:
                    row[m] = float("nan")
            level_summary.append(row)

        df_level_summary = pd.DataFrame(level_summary)

        # Highlight best values
        best_vals = {}
        for m in metric_options:
            col_vals = df_level_summary[m].dropna()
            if col_vals.empty:
                continue
            best_vals[m] = col_vals.min() if m in LOWER_IS_BETTER else col_vals.max()

        formatted_rows = []
        for _, row in df_level_summary.iterrows():
            f_row = {"Model": row["Model"]}
            for m in metric_options:
                val = row[m]
                if pd.isna(val):
                    f_row[m] = "—"
                else:
                    is_best = (m in best_vals and abs(val - best_vals[m]) < 1e-9)
                    if m == "PSNR":
                        formatted_val = f"{val:.2f}"
                    else:
                        formatted_val = f"{val:.4f}"
                    f_row[m] = f"**{formatted_val}**" if is_best else formatted_val
            formatted_rows.append(f_row)

        display_cols = ["Model"] + [m for m in metric_options if m in df_level_summary.columns]
        st.markdown(f"**Average Metrics Summary Table ({level_key})**")
        st.dataframe(pd.DataFrame(formatted_rows)[display_cols], use_container_width=True)

        # 3. Plot sample scatter plots for ALL metrics in this level
        st.markdown(f"**Metric Scatter Plots by Sample ({level_key})**")
        level_file_names = sorted(level_df["file_name"].dropna().unique().tolist())
        level_sample_order = {file_name: idx + 1 for idx, file_name in enumerate(level_file_names)}
        
        # We will plot all metrics in 2 columns
        for start in range(0, len(metric_options), 2):
            cols = st.columns(2)
            for col, metric in zip(cols, metric_options[start:start + 2]):
                with col:
                    chart_df = level_df[["run", "file_name", metric]].dropna().copy()
                    if not chart_df.empty and level_sample_order:
                        chart_df["sample"] = chart_df["file_name"].map(level_sample_order)
                        chart_df = chart_df.dropna(subset=["sample"])
                        chart_df["sample"] = chart_df["sample"].astype(int)

                        domain = list(run_colors.keys())
                        range_colors = list(run_colors.values())
                        color_scale = alt.Scale(domain=domain, range=range_colors)

                        st.markdown(f"*{metric} by sample ({level_key})*")
                        chart = (
                            alt.Chart(chart_df)
                            .mark_circle(size=38, opacity=0.72)
                            .encode(
                                x=alt.X("sample:Q", title="Sample"),
                                y=alt.Y(f"{metric}:Q", title=metric),
                                color=alt.Color("run:N", title="Run", scale=color_scale),
                                tooltip=[
                                    alt.Tooltip("sample:Q", title="Sample"),
                                    alt.Tooltip("file_name:N", title="File"),
                                    alt.Tooltip("run:N", title="Run"),
                                    alt.Tooltip(f"{metric}:Q", title=metric, format=".6f"),
                                ],
                            )
                            .interactive()
                            .properties(height=260)
                        )
                        st.altair_chart(chart, use_container_width=True)
        st.markdown("---")


@st.cache_data
def _load_and_get_file_distances(file_names: tuple[str, ...]):
    import json
    import pandas as pd
    from utils.result_scanner import PROJECT_ROOT
    
    scenario_root = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN"
    ri_path = scenario_root / 'route_information' / 'all_route_information.csv'
    
    if not ri_path.exists():
        return {}
        
    try:
        df_route = pd.read_csv(ri_path)
    except Exception:
        return {}
        
    df_route['pair_key'] = df_route.apply(
        lambda row: tuple(sorted([str(row['start_node']), str(row['end_node'])])), 
        axis=1
    )
    
    route_map = {}
    for _, row in df_route.iterrows():
        plan = row['plan']
        if plan not in route_map:
            route_map[plan] = {}
        route_map[plan][row['pair_key']] = float(row['topology_centerline_distance_m'])
        
    file_distances = {}
    for fname in file_names:
        parts = fname.split('__')
        if len(parts) < 2:
            continue
        plan_name = parts[0]
        suffix = parts[1].replace('.png', '')
        sub_parts = suffix.split('_')
        try:
            if sub_parts[-1] in ['full', 'half', 'single']:
                route_idx = int(sub_parts[-2])
            else:
                route_idx = int(sub_parts[-1])
                
            meta_path = scenario_root / 'metadata' / plan_name / f'route_{route_idx:02d}.json'
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                sn = meta.get('start_node')
                en = meta.get('end_node')
                if sn and en:
                    pair_key = tuple(sorted([str(sn), str(en)]))
                    dist = route_map.get(plan_name, {}).get(pair_key)
                    if dist is not None:
                        file_distances[fname] = dist
        except Exception:
            pass
            
    return file_distances


def _show_error_vs_route_length_analysis(
    df: pd.DataFrame,
    selected_runs: list[RunInfo],
    run_colors: dict[str, str],
    metric_options: list[str]
):
    st.markdown("## Error & Quality vs Route Length Analysis")
    st.caption("Investigating the relationship between the route physical length (meters) and the model prediction error / similarity metrics on the test set.")
    
    # 1. Resolve route distances for files
    unique_files = tuple(df["file_name"].dropna().unique().tolist())
    file_distances = _load_and_get_file_distances(unique_files)
    
    if not file_distances:
        st.info("No route distance data could be mapped to test set images.")
        return
        
    df_copy = df.copy()
    df_copy["route_length_m"] = df_copy["file_name"].map(file_distances)
    df_copy = df_copy.dropna(subset=["route_length_m"])
    
    # Use all metrics (MAE, MSE, RMSE, SSIM, PSNR, LPIPS)
    error_metrics = metric_options
    
    # 2. Compute unified Pearson correlation coefficient r table
    st.markdown("**Correlation Analysis (Pearson Correlation Coefficient $r$)**")
    st.markdown(
        "Pearson correlation measures the linear relationship. "
        "For error metrics (MAE, MSE, RMSE, LPIPS), $r > 0$ indicates that longer routes lead to higher error (worse performance). "
        "For similarity metrics (SSIM, PSNR), $r < 0$ indicates that longer routes lead to lower similarity/quality (worse performance)."
    )
    
    METHOD_DISPLAY_NAMES = {
        "Method_pix2pixHD": "pix2pixHD",
        "Method_PlainUnet": "Plain U-Net",
        "Method_pix2pixhd_No_D": "pix2pixHD (No D)",
        "Method_CVAE": "CVAE"
    }
    
    corr_rows = []
    for run in selected_runs:
        run_df = df_copy[df_copy["run"] == run.label]
        row = {"Model": METHOD_DISPLAY_NAMES.get(run.method, run.method)}
        for m in error_metrics:
            col_name = f"{m}_r"
            if not run_df.empty:
                row[col_name] = float(run_df["route_length_m"].corr(run_df[m]))
            else:
                row[col_name] = float("nan")
        corr_rows.append(row)
        
    df_corr = pd.DataFrame(corr_rows)
    st.dataframe(df_corr, use_container_width=True)
        
    # 3. Scatter Plot Grid for ALL metrics (2 columns)
    st.markdown("**Scatter Plots: Route Length (m) vs. Evaluation Metrics**")
    
    domain = list(run_colors.keys())
    range_colors = list(run_colors.values())
    color_scale = alt.Scale(domain=domain, range=range_colors)
    
    for start in range(0, len(error_metrics), 2):
        cols = st.columns(2)
        for col, metric in zip(cols, error_metrics[start:start + 2]):
            with col:
                st.markdown(f"*Scatter Plot: Route Length (m) vs. {metric}*")
                chart = (
                    alt.Chart(df_copy)
                    .mark_circle(size=50, opacity=0.6)
                    .encode(
                        x=alt.X("route_length_m:Q", title="Route Length (meters)", scale=alt.Scale(zero=False)),
                        y=alt.Y(f"{metric}:Q", title=f"{metric} Value"),
                        color=alt.Color("run:N", title="Run", scale=color_scale),
                        tooltip=["file_name:N", "run:N", "route_length_m:Q", f"{metric}:Q"]
                    )
                    .properties(height=300)
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
                
    # 4. Narrative Conclusion
    st.markdown("**Analytical Conclusion (บทวิเคราะห์สรุปแนวโน้ม):**")
    for row in corr_rows:
        model = row["Model"]
        r_mae = row.get("MAE_r")
        r_rmse = row.get("RMSE_r")
        r_ssim = row.get("SSIM_r")
        r_psnr = row.get("PSNR_r")
        
        desc_list = []
        if r_mae is not None and not pd.isna(r_mae):
            desc_list.append(f"MAE $r$ = {r_mae:.3f}")
        if r_rmse is not None and not pd.isna(r_rmse):
            desc_list.append(f"RMSE $r$ = {r_rmse:.3f}")
        if r_ssim is not None and not pd.isna(r_ssim):
            desc_list.append(f"SSIM $r$ = {r_ssim:.3f}")
        if r_psnr is not None and not pd.isna(r_psnr):
            desc_list.append(f"PSNR $r$ = {r_psnr:.3f}")
            
        desc_str = ", ".join(desc_list)
        
        # Decide performance trend based on MAE/RMSE (lower-is-better, so r > 0 means worse)
        # and SSIM/PSNR (higher-is-better, so r < 0 means worse)
        main_r_err = r_mae if r_mae is not None and not pd.isna(r_mae) else r_rmse
        main_r_sim = r_ssim if r_ssim is not None and not pd.isna(r_ssim) else r_psnr
        
        is_worse_with_distance = False
        if main_r_err is not None and main_r_err > 0.15:
            is_worse_with_distance = True
        elif main_r_sim is not None and main_r_sim < -0.15:
            is_worse_with_distance = True
            
        if main_r_err is None and main_r_sim is None:
            st.markdown(f"- **{model}**: ไม่พบข้อมูลเปรียบเทียบความสัมพันธ์")
        elif is_worse_with_distance:
            st.markdown(f"- **{model}** ({desc_str}): มี **แนวโน้มประสิทธิภาพลดลงชัดเจนเมื่อเส้นทางยาวขึ้น (Performance degrades with distance)** โดยมีค่าสหสัมพันธ์ของ Error เพิ่มขึ้น (Positive $r$ สำหรับ MAE/RMSE) และค่าความคล้ายคลึงลดลง (Negative $r$ สำหรับ SSIM/PSNR)")
        else:
            st.markdown(f"- **{model}** ({desc_str}): **ไม่มีแนวโน้มความเสื่อมถอยอย่างมีนัยสำคัญ (Stable performance across distances)** ความยาวเส้นทางไม่มีผลกระทบที่รุนแรงต่อความแม่นยำและการทำนายผังความหนาแน่น")
            
    st.markdown("---")


def _combined_per_image(selected_runs: list[RunInfo]) -> pd.DataFrame:
    frames = []
    for run in selected_runs:
        per_image = load_per_image(run.path)
        if not per_image.empty:
            frames.append(attach_run_label(per_image, run.label))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _get_layout_borders(input_path, thickness=1):
    import json
    import pathlib
    import numpy as np
    from PIL import Image
    from utils.result_scanner import PROJECT_ROOT
    try:
        input_path = pathlib.Path(input_path)
        target_img = Image.open(input_path)
        target_w, target_h = target_img.size
        
        orig_img_path = None
        
        # 1. Try to resolve the dataset root and topology name from run_config_snapshot.json
        dataset_root = None
        topology_name = "Topo_HouseGAN"
        for parent in input_path.parents:
            snapshot_path = parent / "run_config_snapshot.json"
            if snapshot_path.exists():
                try:
                    with open(snapshot_path, "r", encoding="utf-8") as f:
                        snapshot = json.load(f)
                    dataset_root_raw = snapshot.get("DATASET_ROOT", snapshot.get("dataset_root", ""))
                    if dataset_root_raw:
                        dataset_root = pathlib.Path(dataset_root_raw)
                        if not dataset_root.is_absolute():
                            dataset_root = (PROJECT_ROOT / dataset_root).resolve()
                        topology_name = dataset_root.name
                        break
                except Exception:
                    pass

        # 2. Try to fetch the high-resolution clean layout image from the dataset root splits
        if dataset_root and dataset_root.exists():
            for split in ("test", "train", "validation"):
                candidate = dataset_root / "A" / split / input_path.name
                if candidate.exists():
                    orig_img_path = candidate
                    break

        # 3. Fallback to default dataset candidates in PROJECT_ROOT
        if not orig_img_path:
            candidates_dirs = [
                PROJECT_ROOT / "Dataset" / "Data_ImageUNet" / "DensityMap_dataset" / topology_name,
                PROJECT_ROOT / "Dataset" / "Data_ImageUNet" / "DensityMap_COLORJET_dataset" / topology_name,
                PROJECT_ROOT / "Dataset" / "Data_ImageUNet" / "Trajectory_line_dataset" / topology_name,
            ]
            for d in candidates_dirs:
                if d.exists():
                    for split in ("test", "train", "validation"):
                        candidate = d / "A" / split / input_path.name
                        if candidate.exists():
                            orig_img_path = candidate
                            break
                if orig_img_path:
                    break
                    
        if orig_img_path:
            img = Image.open(orig_img_path).convert("RGB")
        else:
            img = target_img.convert("RGB")
            
        arr = np.array(img)
        walkable_highres = (arr[:, :, 0] > 50) | (arr[:, :, 1] > 50) | (arr[:, :, 2] > 50)
        
        walkable_pil = Image.fromarray(walkable_highres.astype(np.uint8) * 255)
        walkable_resized = np.array(walkable_pil.resize((target_w, target_h), Image.NEAREST)) > 128
        
        current = walkable_resized.copy()
        for _ in range(thickness):
            eroded = current.copy()
            eroded[1:, :] &= current[:-1, :]
            eroded[:-1, :] &= current[1:, :]
            eroded[:, 1:] &= current[:, :-1]
            eroded[:, :-1] &= current[:, 1:]
            current = eroded
            
        borders = walkable_resized & ~current
        return borders
    except Exception:
        return None


def _apply_jet_colormap(image_path_or_pil):
    import cv2
    import numpy as np
    from PIL import Image
    from pathlib import Path

    try:
        if isinstance(image_path_or_pil, (str, Path)):
            img = Image.open(image_path_or_pil)
        else:
            img = image_path_or_pil

        img_gray = img.convert("L")
        arr = np.array(img_gray)
        jet_arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
        jet_rgb = cv2.cvtColor(jet_arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(jet_rgb)
    except Exception:
        if isinstance(image_path_or_pil, (str, Path)):
            return Image.open(image_path_or_pil)
        return image_path_or_pil


def _overlay_borders(image_path, borders, color=[220, 220, 220]):
    import numpy as np
    from PIL import Image
    from pathlib import Path
    try:
        if isinstance(image_path, (str, Path)):
            img = Image.open(image_path)
        else:
            img = image_path
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb)
        
        h, w = arr.shape[:2]
        bh, bw = borders.shape
        if (h != bh) or (w != bw):
            borders_pil = Image.fromarray(borders.astype(np.uint8) * 255)
            borders = np.array(borders_pil.resize((w, h), Image.NEAREST)) > 128
            
        arr[borders] = color
        return Image.fromarray(arr)
    except Exception:
        return Image.open(image_path) if isinstance(image_path, (str, Path)) else image_path


def _show_image_compare(selected_runs: list[RunInfo], file_name: str, show_layout: bool = True):
    if not file_name:
        return
    
    first_input = None
    first_target = None
    predictions = []

    for run in selected_runs:
        input_path, pred_path, target_path = image_triplet(run.path, file_name)
        if first_input is None and input_path is not None:
            first_input = input_path
        if first_target is None and target_path is not None:
            first_target = target_path
        predictions.append((run.label, pred_path, run.method, run.run_name))

    # Sort predictions consistently by method order: pix2pixHD -> Plain U-Net -> pix2pixHD (No D) -> CVAE
    METHOD_ORDER = {
        "Method_pix2pixHD": 0,
        "Method_PlainUnet": 1,
        "Method_pix2pixhd_No_D": 2,
        "Method_CVAE": 3
    }
    predictions.sort(key=lambda x: METHOD_ORDER.get(x[2], 99))

    borders = _get_layout_borders(first_input) if (show_layout and first_input) else None

    total_cols = 2 + len(predictions)
    cols = st.columns(total_cols)
    
    # Column 0: INPUT
    cols[0].markdown("**INPUT**")
    if first_input:
        cols[0].image(str(first_input), use_container_width=True)
    else:
        cols[0].info("Missing input")

    # Column 1: GROUND TRUTH
    cols[1].markdown("**GROUND TRUTH**")
    if first_target:
        target_img = _apply_jet_colormap(first_target)
        if borders is not None:
            img_with_layout = _overlay_borders(target_img, borders)
            cols[1].image(img_with_layout, use_container_width=True)
        else:
            cols[1].image(target_img, use_container_width=True)
    else:
        cols[1].info("Missing target")

    # Columns 2+: PREDICTIONS
    METHOD_SHORT_NAMES = {
        "Method_pix2pixHD": "pix2pixHD",
        "Method_PlainUnet": "Plain U-Net",
        "Method_pix2pixhd_No_D": "pix2pixHD (No D)",
        "Method_CVAE": "CVAE"
    }

    for idx, (label, pred_path, method, run_name) in enumerate(predictions, start=2):
        short_name = METHOD_SHORT_NAMES.get(method, method)
        cols[idx].markdown(f"**{short_name}**")
        if pred_path:
            pred_img = _apply_jet_colormap(pred_path)
            if borders is not None:
                img_with_layout = _overlay_borders(pred_img, borders)
                cols[idx].image(img_with_layout, use_container_width=True)
            else:
                cols[idx].image(pred_img, use_container_width=True)
        else:
            cols[idx].info("Missing prediction")
        
        # Display the long run folder name underneath the image to preserve grid alignment
        cols[idx].caption(run_name)



def _show_summary_table(combined: pd.DataFrame, selected_runs: list[RunInfo]):
    METHOD_DISPLAY_NAMES = {
        "Method_pix2pixHD": "pix2pixHD",
        "Method_PlainUnet": "Plain U-Net",
        "Method_pix2pixhd_No_D": "pix2pixHD (No D)",
        "Method_CVAE": "CVAE"
    }
    
    METRIC_HEADERS = {
        "MAE": "MAE ↓",
        "MSE": "MSE ↓",
        "RMSE": "RMSE ↓",
        "SSIM": "SSIM ↑",
        "PSNR": "PSNR ↑",
        "LPIPS": "LPIPS ↓"
    }

    # Identify metric columns that exist in the dataframe
    available_metrics = [m for m in METRIC_ORDER if m in combined.columns]
    
    # Compute mean for each run
    table_rows = []
    for run in selected_runs:
        run_data = combined[combined["run"] == run.label]
        row = {"Model": METHOD_DISPLAY_NAMES.get(run.method, run.method)}
        for m in available_metrics:
            if not run_data.empty and m in run_data.columns:
                row[m] = float(run_data[m].mean())
            else:
                row[m] = float("nan")
        table_rows.append(row)
        
    df_summary = pd.DataFrame(table_rows)

    # Find the best value for each metric
    best_vals = {}
    for m in available_metrics:
        col_vals = df_summary[m].dropna()
        if col_vals.empty:
            continue
        best_vals[m] = col_vals.min() if m in LOWER_IS_BETTER else col_vals.max()

    # Format the cells
    formatted_rows = []
    for _, row in df_summary.iterrows():
        f_row = {"Model": row["Model"]}
        for m in available_metrics:
            val = row[m]
            header = METRIC_HEADERS.get(m, m)
            if pd.isna(val):
                f_row[header] = "—"
            else:
                is_best = False
                if m in best_vals and abs(val - best_vals[m]) < 1e-9:
                    is_best = True
                
                # Format to appropriate decimals
                if m == "PSNR":
                    formatted_val = f"{val:.2f}"
                else:
                    formatted_val = f"{val:.4f}"
                    
                if is_best:
                    f_row[header] = f"**{formatted_val}**"
                else:
                    f_row[header] = formatted_val
        formatted_rows.append(f_row)

    # Build Markdown Table
    headers = ["Model"] + [METRIC_HEADERS.get(m, m) for m in available_metrics]
    aligns = [":---"] + [":---:" for _ in available_metrics]
    
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(aligns) + " |")
    
    for f_row in formatted_rows:
        row_vals = [str(f_row.get(h, "—")) for h in headers]
        md_lines.append("| " + " | ".join(row_vals) + " |")
        
    table_md = "\n".join(md_lines)
    
    st.markdown("### Model Benchmark Summary (Average)")
    st.markdown(table_md)
    st.markdown("")


def _show_failure_case_analysis(combined: pd.DataFrame, selected_runs: list):
    st.markdown("### Failure-Case Analysis")
    st.markdown("This section highlights 3 best cases (lowest error, representative) and 3 worst cases (highest error) dynamically chosen from the test set.")
    
    if not selected_runs:
        st.warning("No models selected.")
        return
        
    run_labels = [r.label for r in selected_runs]
    ranking_options = ["Average (All Selected Models)"] + run_labels
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        ranking_basis = st.selectbox("Select ranking basis (SSIM):", ranking_options, index=0)
    with col2:
        flow_types = st.multiselect(
            "Select Flow Case Types:",
            options=["1 agent (single)", "N half (half)", "N full (full)"],
            default=["1 agent (single)", "N half (half)", "N full (full)"]
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True) # visual alignment
        show_layout_overlay = st.checkbox("Overlay room layout borders on predictions and targets", value=True, key="failure_case_layout")
    
    if not flow_types:
        st.warning("Please select at least one Flow Case Type.")
        return

    type_map = {
        "1 agent (single)": "single",
        "N half (half)": "half",
        "N full (full)": "full"
    }
    selected_suffixes = [type_map[ft] for ft in flow_types]

    def check_suffix(fname):
        if not isinstance(fname, str):
            return False
        base = fname.lower().split(".png")[0]
        for s in selected_suffixes:
            if base.endswith(f"_{s}"):
                return True
        return False

    filtered_combined = combined[combined["file_name"].apply(check_suffix)]

    if ranking_basis == "Average (All Selected Models)":
        agg_df = filtered_combined.dropna(subset=["SSIM"]).groupby("file_name")["SSIM"].mean().reset_index()
    else:
        agg_df = filtered_combined[filtered_combined["run"] == ranking_basis].dropna(subset=["SSIM"])
        
    if agg_df.empty:
        st.warning("No SSIM data available for ranking under current filters.")
        return

    sorted_df = agg_df.sort_values(by="SSIM", ascending=True)
    worst_cases = sorted_df.head(3)["file_name"].tolist()
    best_cases = sorted_df.tail(3)["file_name"].tolist()
    
    def render_cases(cases, title):
        st.markdown(f"#### {title}")
        for case in cases:
            st.markdown(f"**Sample**: `{case}`")
            _show_image_compare(selected_runs, case, show_layout=show_layout_overlay)
            st.markdown("---")
            
    render_cases(best_cases, "✅ 3 Cases Where Model Performed Best (Highest SSIM)")
    render_cases(worst_cases, "❌ 3 Cases Where Model Performed Worst (Lowest SSIM)")



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

    combined = _combined_per_image(selected_runs)
    if combined.empty:
        st.warning("Selected runs do not have test_evaluation_per_image.csv yet.")
        return

    metric_options = [m for m in METRIC_ORDER if m in combined.columns]

    # Color Settings
    DEFAULT_COLORS = {
        "Method_PlainUnet": "#ff4d4d",      # Bright Red
        "Method_pix2pixHD": "#f43f5e",      # Rose Pink
        "Method_pix2pixhd_No_D": "#ffd166", # Soft Yellow
        "Method_CVAE": "#f97316",           # Soft Orange
        "Method_GNN_CVAE": "#ca8a04",       # Yellow
        "Method_GNN_CVAE2": "#0891b2",      # Cyan
        "Method_LSTM_01": "#f97316",        # Orange
        "Method_Transformer": "#06b6d4",    # Teal
    }
    
    METHOD_DISPLAY_NAMES = {
        "Method_pix2pixHD": "pix2pixHD",
        "Method_PlainUnet": "Plain U-Net",
        "Method_pix2pixhd_No_D": "pix2pixHD (No D)",
        "Method_CVAE": "CVAE"
    }

    run_colors = {}
    with st.expander("🎨 Chart Color Settings", expanded=False):
        st.markdown("Customize chart colors for each model/run:")
        
        # Tie color (for winner donut charts)
        if "color_tie" not in st.session_state:
            st.session_state["color_tie"] = "#9ca3af"
        tie_color = st.color_picker("Tie (Draw)", key="color_tie")
        run_colors["Tie"] = tie_color
        
        color_cols = st.columns(min(4, len(selected_runs)))
        for idx, run in enumerate(selected_runs):
            col = color_cols[idx % len(color_cols)]
            with col:
                method_name = run.method
                method_display = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                default_color = DEFAULT_COLORS.get(method_name, "#4b5563")
                state_key = f"color_{run.label}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = default_color
                chosen_color = st.color_picker(
                    method_display,
                    key=state_key
                )
                run_colors[run.label] = chosen_color

    # Show Average Summary Table
    _show_summary_table(combined, selected_runs)

    st.markdown("### Run Summary")
    summary_cols = st.columns(len(selected_runs))
    for col, run in zip(summary_cols, selected_runs):
        with col:
            _summary_cards(run, load_per_image(run.path))

    st.markdown("### Per-image Metrics")
    table_cols = ["run", "file_name"] + [m for m in METRIC_ORDER if m in combined.columns]
    st.dataframe(combined[table_cols], use_container_width=True, height=360)

    st.markdown("### Graph Visualizations")
    graph_col1, graph_col2 = st.columns(2)
    with graph_col1:
        hist_metric = st.selectbox("Distribution metric", metric_options, index=metric_options.index("RMSE") if "RMSE" in metric_options else 0)
        _histogram(combined, hist_metric, run_colors)
    with graph_col2:
        x_metric = st.selectbox("Scatter X", metric_options, index=metric_options.index("RMSE") if "RMSE" in metric_options else 0)
        y_default = metric_options.index("SSIM") if "SSIM" in metric_options else min(1, len(metric_options) - 1)
        y_metric = st.selectbox("Scatter Y", metric_options, index=y_default)
        _scatter(combined, x_metric, y_metric, run_colors)

    _show_metric_compare(combined, metric_options, selected_runs, run_colors)
    _show_sample_metric_scatters(combined, metric_options, run_colors)
    _show_occupancy_level_analysis(combined, selected_runs, run_colors, metric_options)
    _show_error_vs_route_length_analysis(combined, selected_runs, run_colors, metric_options)
    _show_failure_case_analysis(combined, selected_runs)

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
    
    show_all_files = st.checkbox("Show all test files (including those with missing predictions on disk)", value=False)
    if not show_all_files:
        available_files = []
        for f in file_names:
            all_exist = True
            for run in selected_runs:
                _, pred_path, _ = image_triplet(run.path, f)
                if pred_path is None:
                    all_exist = False
                    break
            if all_exist:
                available_files.append(f)
        if available_files:
            file_names = sorted(available_files)

    st.markdown("### Image Comparison")
    
    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        show_layout = st.checkbox("Overlay room layout borders on predictions and targets", value=True)
    with col_ctrl2:
        show_all_samples = st.checkbox("Show all samples in a list", value=False)
        
    if show_all_samples:
        limit_samples = st.slider("Number of samples to display", min_value=5, max_value=len(file_names), value=min(20, len(file_names)), step=5)
        for idx_file, f in enumerate(file_names[:limit_samples]):
            st.markdown(f"#### Sample {idx_file + 1}: `{f}`")
            _show_image_compare(selected_runs, f, show_layout=show_layout)
            st.markdown("---")
    else:
        default_file = ranked_view.iloc[0]["file_name"] if not ranked_view.empty else file_names[0]
        selected_file = st.selectbox(
            "Select image file to compare",
            file_names,
            index=file_names.index(default_file) if default_file in file_names else 0,
        )
        _show_image_compare(selected_runs, selected_file, show_layout=show_layout)
