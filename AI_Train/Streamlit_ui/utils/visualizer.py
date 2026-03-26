import streamlit as st
import pandas as pd
import pathlib
from PIL import Image

def plot_training_history(csv_path):
    """
    Reads the CSV and plots the loss curves using st.line_chart.
    """
    if not pathlib.Path(csv_path).exists():
        st.warning(f"No log file found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            st.info("Log file is empty.")
            return

        # Sidebar to select which columns to plot
        cols = [c for c in df.columns if any(w in c.lower() for w in ['loss', 'mae', 'l1', 'g_adv', 'fm'])]
        if not cols:
            cols = [c for c in df.columns if c != 'epoch']
            
        selected_cols = st.multiselect("Select Metrics to Plot", options=cols, default=cols[:2])
        
        if selected_cols:
            st.line_chart(df.set_index('epoch')[selected_cols])
        else:
            st.info("Select metrics to view the chart.")
            
    except Exception as e:
        st.error(f"Error reading log file: {e}")

def show_sample_images(sample_dir):
    """
    Displays the sample images from the given directory in a grid.
    """
    path = pathlib.Path(sample_dir)
    if not path.exists():
        st.warning(f"No samples folder found at {sample_dir}")
        return

    images = sorted(list(path.glob("*.png")), key=lambda x: x.stat().st_mtime, reverse=True)
    if not images:
        st.info("No sample images found yet.")
        return

    st.subheader(f"Samples ({len(images)})")
    
    # Simple grid using columns
    cols_per_row = 3
    num_rows = (len(images) + cols_per_row - 1) // cols_per_row
    
    for i in range(num_rows):
        st_cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < len(images):
                img_path = images[idx]
                st_cols[j].image(str(img_path), caption=img_path.name, use_container_width=True)

def show_test_evaluation(run_dir):
    """
    Displays the test evaluation summary CSV and the side-by-side test samples.
    """
    run_path = pathlib.Path(run_dir)
    score_path = run_path / "test_evaluation_summary.csv"
    test_results_dir = run_path / "test_results"

    if score_path.exists():
        st.subheader("📊 Final Test Scores")
        df_score = pd.read_csv(score_path)
        st.table(df_score)

    if test_results_dir.exists():
        st.subheader("🖼️ Test Evaluation Samples (Input | Target | Prediction)")
        images = sorted(list(test_results_dir.glob("*.png")), key=lambda x: x.name)
        if images:
            cols_per_row = 1 # Test samples are wide (3 images side-by-side)
            for img_path in images:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)
        else:
            st.info("No test evaluation samples found.")
