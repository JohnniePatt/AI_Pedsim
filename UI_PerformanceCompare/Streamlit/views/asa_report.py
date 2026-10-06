from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import altair as alt

from utils.image_paths import image_triplet
from utils.metrics_loader import (
    METRIC_ORDER,
    attach_run_label,
    load_per_image,
    load_summary,
    load_walkable_per_image,
)
from utils.result_scanner import PROJECT_ROOT, RunInfo
from views.image_based_output import (
    LOWER_IS_BETTER,
    HIGHER_IS_BETTER,
    _load_jupedsim_runtime,
    _get_layout_borders,
    _apply_jet_colormap,
    _overlay_borders,
    _get_dataset_highres_paths,
    _histogram,
    _scatter,
    _show_metric_compare,
    _show_sample_metric_scatters,
    _show_occupancy_level_analysis,
    _show_error_vs_route_length_analysis,
    _show_failure_case_analysis,
    _get_path_rooms_count,
)

# -----------------------------------------------------------------------------
# Configuration for the 3 ASA Report Models
# -----------------------------------------------------------------------------
ASA_RUNS = [
    RunInfo(
        method="Method_pix2pixHD",
        run_name="run_HD_20260517_133538_BestForBW_model_evaluate_256",
        path=(PROJECT_ROOT / "AI_GenerateImage/AI_Result/Method_pix2pixHD/outputs/run_HD_20260517_133538_BestForBW_model_evaluate_256").resolve(),
    ),
    RunInfo(
        method="Method_CVAE_Real",
        run_name="run_CVAE_Real_20260925_235753",
        path=(PROJECT_ROOT / "AI_GenerateImage/AI_Result/Method_CVAE_Real/outputs/run_CVAE_Real_20260925_235753").resolve(),
    ),
    RunInfo(
        method="Method_CVAE",
        run_name="run_CVAE_20260627_193237_config2",
        path=(PROJECT_ROOT / "AI_GenerateImage/AI_Result/Method_CVAE/outputs/run_CVAE_20260627_193237_config2").resolve(),
    ),
]

ASA_DISPLAY_NAMES = {
    ASA_RUNS[0].label: "Pix2PixHD (Paper Winner, Adversarial)",
    ASA_RUNS[1].label: "CVAE Real (Proper Latent z ~ N(0, I), Seed 42)",
    ASA_RUNS[2].label: "CVAE Legacy (Paper Baseline, Fixed z = 0)",
}

ASA_SHORT_NAMES = {
    ASA_RUNS[0].label: "Pix2PixHD",
    ASA_RUNS[1].label: "CVAE Real",
    ASA_RUNS[2].label: "CVAE Legacy (z=0)",
}

ASA_DEFAULT_COLORS = {
    ASA_RUNS[0].label: "#f43f5e",  # Rose Pink
    ASA_RUNS[1].label: "#0ea5e9",  # Sky Blue
    ASA_RUNS[2].label: "#eab308",  # Amber / Yellow
}


def _asa_display_name(run: RunInfo) -> str:
    return ASA_DISPLAY_NAMES.get(run.label, run.label)


def _load_ai_runtime(run: RunInfo) -> dict:
    runtime_path = run.path / "test_runtime.csv"
    if not runtime_path.exists():
        return {}
    try:
        frame = pd.read_csv(runtime_path)
    except Exception:
        return {}
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    required = ["sample_count", "test_pipeline_wall_time_s"]
    if any(col not in frame.columns for col in required):
        return {}
    try:
        sample_count = int(row["sample_count"])
        total_s = float(row["test_pipeline_wall_time_s"])
        time_gen = float(row.get("Time Generate", np.nan))
        return {
            "sample_count": sample_count,
            "total_s": total_s,
            "average_s": total_s / sample_count,
            "time_gen_s": time_gen,
            "avg_gen_s": time_gen / sample_count if not np.isnan(time_gen) else np.nan,
            "device": str(row.get("device_name", row.get("device_type", "Unknown GPU"))),
            "measured_at_utc": str(row.get("measured_at_utc", "")),
        }
    except (TypeError, ValueError):
        return {}


def _combined_asa_per_image(runs: list[RunInfo]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        df = load_per_image(run.path)
        if not df.empty:
            df = df.copy()
            df["run_label"] = run.label
            df["model_name"] = _asa_display_name(run)
            frames.append(attach_run_label(df, run.label))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _combined_asa_walkable(runs: list[RunInfo]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        df = load_walkable_per_image(run.path)
        if not df.empty:
            df = df.copy()
            df["run_label"] = run.label
            df["model_name"] = _asa_display_name(run)
            frames.append(attach_run_label(df, run.label))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _show_asa_summary_table(combined: pd.DataFrame, runs: list[RunInfo]):
    METRIC_HEADERS = {
        "MAE": "MAE ↓",
        "MSE": "MSE ↓",
        "RMSE": "RMSE ↓",
        "SSIM": "SSIM ↑",
        "PSNR": "PSNR ↑",
        "LPIPS": "LPIPS ↓",
        "FOREGROUND_MAE": "Foreground MAE ↓",
        "HOTSPOT_IOU": "Hotspot IoU ↑",
    }
    core_metrics = list(METRIC_ORDER) + ["FOREGROUND_MAE", "HOTSPOT_IOU"]

    table_rows = []
    for run in runs:
        row = {"Model": _asa_display_name(run)}
        summary_csv = run.path / "test_evaluation_summary.csv"
        if summary_csv.exists():
            try:
                df_s = pd.read_csv(summary_csv)
                for _, s_row in df_s.iterrows():
                    m_name = str(s_row.iloc[0]).strip().upper()
                    try:
                        m_val = float(s_row.iloc[1])
                        if not np.isnan(m_val):
                            row[m_name] = m_val
                    except Exception:
                        pass
            except Exception:
                pass

        # Fallback to mean of per_image if summary didn't have metric
        run_data = combined[combined["run"] == run.label]
        for m in core_metrics:
            if m not in row and m in run_data.columns:
                val = run_data[m].dropna().mean()
                if not np.isnan(val):
                    row[m] = float(val)
        table_rows.append(row)

    if not table_rows:
        return

    df_summary = pd.DataFrame(table_rows)
    present_metrics = [m for m in core_metrics if m in df_summary.columns]

    best_vals = {}
    for m in present_metrics:
        col_vals = df_summary[m].dropna()
        if not col_vals.empty:
            best_vals[m] = col_vals.min() if m in LOWER_IS_BETTER or m == "FOREGROUND_MAE" else col_vals.max()

    formatted_rows = []
    for _, row in df_summary.iterrows():
        f_row = {"Model": row["Model"]}
        for m in present_metrics:
            val = row.get(m, np.nan)
            header = METRIC_HEADERS.get(m, m)
            if pd.isna(val):
                f_row[header] = "—"
            else:
                is_best = (m in best_vals and abs(val - best_vals[m]) < 1e-6)
                if m == "PSNR":
                    formatted_val = f"{val:.2f} dB"
                elif m == "HOTSPOT_IOU":
                    formatted_val = f"{val * 100:.1f}%"
                else:
                    formatted_val = f"{val:.4f}" if val >= 0.01 else f"{val:.6f}"
                f_row[header] = f"**{formatted_val}**" if is_best else formatted_val
        formatted_rows.append(f_row)

    headers = ["Model"] + [METRIC_HEADERS.get(m, m) for m in present_metrics]
    aligns = [":---"] + [":---:" for _ in headers[1:]]

    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for f_row in formatted_rows:
        md_lines.append("| " + " | ".join([str(f_row.get(h, "—")) for h in headers]) + " |")

    st.markdown("### Model Benchmark Summary (Average)")
    st.markdown("\n".join(md_lines))
    st.caption("Foreground MAE: Error restricted exclusively to non-zero density regions. Hotspot IoU: Overlap on high congestion zones (density ≥ 0.20).")


def _show_asa_walkable_table(runs: list[RunInfo]):
    METRIC_HEADERS = {
        "MAE": "MAE ↓",
        "MSE": "MSE ↓",
        "RMSE": "RMSE ↓",
        "SSIM": "SSIM ↑",
        "PSNR": "PSNR ↑",
        "LPIPS": "LPIPS ↓",
    }
    available_metrics = list(METRIC_ORDER)
    table_rows = []
    for run in runs:
        row = {"Model": _asa_display_name(run)}
        for p in [run.path / "test_evaluation_walkable_summary.csv", run.path / "logs" / "test_evaluation_walkable_summary.csv"]:
            if p.exists():
                try:
                    df_s = pd.read_csv(p)
                    for _, s_row in df_s.iterrows():
                        m_name = str(s_row.iloc[0]).strip().upper()
                        val = float(s_row.iloc[1])
                        if m_name in available_metrics:
                            row[m_name] = val
                    break
                except Exception:
                    pass
        table_rows.append(row)

    if not table_rows or not any(len(r) > 1 for r in table_rows):
        return

    df_summary = pd.DataFrame(table_rows)
    best_vals = {}
    for m in available_metrics:
        if m in df_summary.columns:
            col_vals = df_summary[m].dropna()
            if not col_vals.empty:
                best_vals[m] = col_vals.min() if m in LOWER_IS_BETTER else col_vals.max()

    formatted_rows = []
    for _, row in df_summary.iterrows():
        f_row = {"Model": row["Model"]}
        for m in available_metrics:
            val = row.get(m, np.nan)
            header = METRIC_HEADERS.get(m, m)
            if pd.isna(val):
                f_row[header] = "—"
            else:
                is_best = (m in best_vals and abs(val - best_vals[m]) < 1e-6)
                formatted_val = f"{val:.2f} dB" if m == "PSNR" else f"{val:.4f}"
                f_row[header] = f"**{formatted_val}**" if is_best else formatted_val
        formatted_rows.append(f_row)

    headers = ["Model"] + [METRIC_HEADERS.get(m, m) for m in available_metrics if any(METRIC_HEADERS.get(m, m) in r for r in formatted_rows)]
    aligns = [":---"] + [":---:" for _ in headers[1:]]

    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for f_row in formatted_rows:
        md_lines.append("| " + " | ".join([str(f_row.get(h, "—")) for h in headers]) + " |")

    st.markdown("### Model Benchmark Summary only walkable area (Average)")
    st.markdown("\n".join(md_lines))


def _show_asa_computational_efficiency(runs: list[RunInfo]):
    st.markdown("### Computational Efficiency Comparison (Simulation vs AI)")
    jupedsim = _load_jupedsim_runtime()
    if not jupedsim:
        st.warning("Canonical JuPedSim runtime is missing or does not contain exactly 862 successful test cases.")
        return

    rows = [{
        "name": "JuPedSim (simulation + outputs)",
        **jupedsim,
        "speedup": 1.0,
    }]
    missing = []
    for run in runs:
        runtime = _load_ai_runtime(run)
        if not runtime or runtime.get("sample_count") != 862:
            missing.append(_asa_display_name(run))
            continue
        rows.append({
            "name": _asa_display_name(run),
            **runtime,
            "speedup": jupedsim["average_s"] / runtime["average_s"],
        })

    md_lines = [
        "| Method / Model | Samples | Total Runtime (s) | Avg Total Runtime / Sample (s) | Speedup |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        speedup = "1.0×" if row["speedup"] == 1.0 else f"**{row['speedup']:,.1f}×**"
        md_lines.append(
            f"| **{row['name']}** | {row['sample_count']:,} | {row['total_s']:,.6f} | "
            f"{row['average_s']:.9f} | {speedup} |"
        )
    st.markdown("\n".join(md_lines))
    st.caption(
        "Artifact-backed total-runtime comparison: JuPedSim uses total_wall_time_s, including "
        "setup, simulation, SQLite output, trajectory plotting, and density-heatmap generation. "
        "AI models use test_pipeline_wall_time_s, including inference, metrics, post-processing, "
        "and image output within the timing scope recorded by each run."
    )
    ai_devices = sorted({row["device"] for row in rows[1:] if "device" in row})
    if ai_devices:
        st.caption("AI device recorded in test_runtime.csv: " + ", ".join(ai_devices))
    if missing:
        st.warning("Runtime artifact missing or not canonical (862 cases): " + ", ".join(missing))


def _show_asa_image_compare(runs: list[RunInfo], file_name: str, show_layout: bool = True):
    if not file_name:
        return

    first_input = None
    first_target = None
    preds = []

    for run in runs:
        in_p, p_p, t_p = image_triplet(run.path, file_name)
        if first_input is None and in_p:
            first_input = in_p
        if first_target is None and t_p:
            first_target = t_p
        preds.append((run, p_p))

    borders = _get_layout_borders(first_input) if (show_layout and first_input) else None
    ds_input, ds_target = _get_dataset_highres_paths(first_input, file_name)
    final_input = ds_input if ds_input else first_input
    final_target = ds_target if ds_target else first_target

    total_cols = 2 + len(preds)
    cols = st.columns(total_cols)

    # Input
    cols[0].markdown("**INPUT**<br><small>Layout + Scenario</small>", unsafe_allow_html=True)
    if final_input and final_input.exists():
        img = Image.open(final_input).resize((512, 512), Image.BILINEAR)
        cols[0].image(img, use_container_width=True)
    else:
        cols[0].info("Missing Input")

    # Ground Truth Target
    cols[1].markdown("**GROUND TRUTH**<br><small>JuPedSim Target</small>", unsafe_allow_html=True)
    if final_target and final_target.exists():
        target_img = _apply_jet_colormap(final_target)
        if borders is not None:
            target_img = _overlay_borders(target_img, borders)
        cols[1].image(target_img.resize((512, 512), Image.BILINEAR), use_container_width=True)
    else:
        cols[1].info("Missing Target")

    # Predictions
    for idx, (run, pred_path) in enumerate(preds, start=2):
        name_short = ASA_SHORT_NAMES.get(run.label, run.method)
        cols[idx].markdown(f"**{name_short}**<br><small>{run.run_name[:20]}...</small>", unsafe_allow_html=True)
        if pred_path and pred_path.exists():
            pred_img = _apply_jet_colormap(pred_path)
            if borders is not None:
                pred_img = _overlay_borders(pred_img, borders)
            cols[idx].image(pred_img.resize((512, 512), Image.BILINEAR), use_container_width=True)
        else:
            cols[idx].info("Missing prediction")


# -----------------------------------------------------------------------------
# Main Render Function for ASA Report
# -----------------------------------------------------------------------------
def render_asa_report():
    st.markdown('<div class="pc-section-label">Ablation & Peer-Review Analysis</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="pc-title">ASA Report: Pix2PixHD vs. Probabilistic CVAE</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pc-subtitle">'
        "Direct response to Reviewer 3 & Reviewer 4: Evaluating the impact of learned latent space, "
        "KL regularisation, and adversarial training under identically matched conditions (50 Epochs, Batch Size 8)."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ About This ASA Report (Research Context & Reviewer Defense)", expanded=True):
        st.markdown(
            """
            This report compares the **three critical model configurations** from the paper and revision:
            1. **Pix2PixHD (Paper Winner)**: Full adversarial cGAN model with 9-block ResNet generator and multi-scale PatchGAN discriminator.
            2. **CVAE Real (Proper Latent)**: Retrained true Conditional Variational Autoencoder featuring active latent reparameterization ($z \\sim q(z|x, y)$), Gaussian KL divergence regularization ($D_{\\mathrm{KL}}$) with annealing, and reproducible inference with random seed 42. Controlled under matched 50 Epochs & Batch Size 8.
            3. **CVAE Legacy (Paper Baseline)**: The original baseline described in the submitted manuscript with fixed $z=0$ and $\\mathrm{KL}=0$ (deterministic conditional autoencoder).

            **Key Academic Findings**:
            * **Addressing Reviewer 3 & 4 on CVAE Definition**: CVAE Real restores true probabilistic latent-variable operation with positive KL loss and stochastic inference capability.
            * **Adversarial Loss Advantage**: Pix2PixHD produces significantly sharper congestion boundaries (SSIM 0.9630 vs 0.8737) and lower perceptual distance (LPIPS 0.047 vs 0.069), confirming the necessity of adversarial training.
            * **Active Region Defense**: Evaluating on active pedestrian zones shows **Foreground MAE of 0.0215** and **Hotspot IoU of 77.9%**, proving model quality is not an artifact of empty background space.
            """
        )

    # Check existence of runs
    valid_runs = [r for r in ASA_RUNS if r.path.exists()]
    if len(valid_runs) < len(ASA_RUNS):
        st.warning(f"Note: Found {len(valid_runs)}/{len(ASA_RUNS)} configured runs on disk.")

    combined = _combined_asa_per_image(valid_runs)
    if combined.empty:
        st.error("Could not load per-image evaluation metrics for ASA models.")
        return

    # User customizable chart colors
    run_colors = {}
    with st.expander("🎨 Model Colors", expanded=False):
        c_cols = st.columns(len(valid_runs) + 1)
        tie_c = c_cols[0].color_picker("Tie (Draw)", "#9ca3af", key="asa_color_tie")
        run_colors["Tie"] = tie_c
        for idx, run in enumerate(valid_runs, start=1):
            def_c = ASA_DEFAULT_COLORS.get(run.label, "#4b5563")
            chosen_c = c_cols[idx].color_picker(_asa_display_name(run), def_c, key=f"asa_c_{run.label}")
            run_colors[run.label] = chosen_c

    # 1. Summary Tables
    _show_asa_summary_table(combined, valid_runs)
    st.markdown("---")
    _show_asa_walkable_table(valid_runs)
    st.markdown("---")

    # 2. Computational Efficiency
    _show_asa_computational_efficiency(valid_runs)
    st.markdown("---")

    # 3. Metric Comparison & Winner Analysis
    metric_options = [m for m in METRIC_ORDER if m in combined.columns]
    _show_metric_compare(combined, metric_options, valid_runs, run_colors, title="Metric Winner & Comparison Analysis", key_prefix="asa_metric")

    walkable_combined = _combined_asa_walkable(valid_runs)
    if not walkable_combined.empty:
        walkable_metrics = [m for m in METRIC_ORDER if m in walkable_combined.columns]
        _show_metric_compare(walkable_combined, walkable_metrics, valid_runs, run_colors, title="Metric Compare (Walkable Area)", key_prefix="asa_walkable")

    st.markdown("---")

    # 4. Error Distributions (Histogram & Scatter)
    st.markdown("### 📈 Distribution & Correlation Analysis")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        dist_m = st.selectbox("Distribution metric", metric_options, index=metric_options.index("RMSE") if "RMSE" in metric_options else 0, key="asa_hist_m")
        _histogram(combined, dist_m, run_colors)
    with d_col2:
        scat_x = st.selectbox("Scatter X", metric_options, index=metric_options.index("RMSE") if "RMSE" in metric_options else 0, key="asa_scat_x")
        scat_y_def = metric_options.index("SSIM") if "SSIM" in metric_options else 0
        scat_y = st.selectbox("Scatter Y", metric_options, index=scat_y_def, key="asa_scat_y")
        _scatter(combined, scat_x, scat_y, run_colors)
    st.markdown("---")
    _show_sample_metric_scatters(combined, metric_options, run_colors)
    _show_occupancy_level_analysis(combined, valid_runs, run_colors, metric_options)
    _show_error_vs_route_length_analysis(combined, valid_runs, run_colors, metric_options)
    _show_failure_case_analysis(combined, valid_runs)

    st.markdown("---")

    # 5. Worst / Best Case Viewer
    st.markdown("### 🔍 Case Explorer (Best vs. Worst Layouts)")
    rank_m = st.selectbox("Rank layouts by metric", metric_options, index=metric_options.index("SSIM") if "SSIM" in metric_options else 0, key="asa_rank_m")
    ascending = rank_m in HIGHER_IS_BETTER
    ranked = combined.dropna(subset=[rank_m]).sort_values(rank_m, ascending=ascending)

    c_mode = st.radio("Case filter", ["Worst cases (Lowest SSIM / Highest Error)", "Best cases (Highest SSIM / Lowest Error)"], horizontal=True, key="asa_case_mode")
    if "Best" in c_mode:
        ranked = ranked.sort_values(rank_m, ascending=not ascending)

    top_n = st.slider("Number of cases to inspect", min_value=5, max_value=30, value=15, step=5, key="asa_top_n")
    table_cols = ["run", "file_name"] + [m for m in METRIC_ORDER if m in combined.columns]
    st.dataframe(ranked.head(top_n)[table_cols], use_container_width=True, height=280)

    st.markdown("---")

    # 6. Interactive Visual Comparison
    st.markdown("### 🖼️ Side-by-Side Heatmap Visualizer")
    file_names = sorted(combined["file_name"].dropna().unique().tolist())
    
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        show_layout = st.checkbox("Overlay room layout borders", value=True, key="asa_layout_check")
    with col_c2:
        show_all_list = st.checkbox("Show sequential sample list", value=False, key="asa_show_all")

    if show_all_list:
        lim = st.slider("Samples to render", 3, 20, 5, key="asa_lim")
        for idx_f, f in enumerate(file_names[:lim]):
            st.markdown(f"#### Case {idx_f + 1}: `{f}`")
            _show_asa_image_compare(valid_runs, f, show_layout=show_layout)
            st.markdown("---")
    else:
        def_file = ranked.iloc[0]["file_name"] if not ranked.empty else file_names[0]
        sel_file = st.selectbox("Choose floor plan test case to inspect", file_names, index=file_names.index(def_file) if def_file in file_names else 0, key="asa_file_select")
        _show_asa_image_compare(valid_runs, sel_file, show_layout=show_layout)
