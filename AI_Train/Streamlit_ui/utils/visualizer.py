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
    Displays the test evaluation summary and the side-by-side test samples with 3-column layout.
    Strictly following mockup: RESULT TEST -> Header Row (MAE, MSE, ...) -> Divider -> Value Row -> Divider -> Header Labels -> Images.
    """
    run_path = pathlib.Path(run_dir)
    score_path = run_path / "test_results" / "test_evaluation_summary.csv"
    if not score_path.exists(): score_path = run_path / "test_evaluation_summary.csv" # Fallback
    
    test_results_dir = run_path / "test_results"

    st.write("### RESULT TEST")

    # 1. Metrics Header & Values
    if score_path.exists():
        try:
            df_score = pd.read_csv(score_path)
            # Filter and pivot to get headers and values
            m_data = df_score[df_score['metric'].str.contains('MAE|MSE|RMSE|L1|mse|mae', case=False, na=False)]
            if not m_data.empty:
                # Row 1: Metric Names
                m_cols_h = st.columns(max(3, len(m_data)))
                for idx, row in enumerate(m_data.itertuples()):
                    if idx < len(m_cols_h):
                        m_cols_h[idx].write(f"**{row.metric.upper()}**")
                
                st.divider() # Line 1
                
                # Row 2: Metric Values
                m_cols_v = st.columns(max(3, len(m_data)))
                for idx, row in enumerate(m_data.itertuples()):
                    if idx < len(m_cols_v):
                        m_cols_v[idx].write(f"{float(row.value):.6f}")
                
            else:
                st.table(df_score)
        except Exception as e:
            st.error(f"Error loading scores: {e}")
    
    st.divider() # Line 2

    # 2. Evaluation Samples (3 separate columns per row)
    if test_results_dir.exists():
        images = sorted(list(test_results_dir.glob("*.png")), key=lambda x: x.name)
        if images:
            # Header Row for Labels
            h_col1, h_col2, h_col3 = st.columns(3)
            h_col1.markdown("<h5 style='text-align: center;'>INPUT</h5>", unsafe_allow_html=True)
            h_col2.markdown("<h5 style='text-align: center;'>GROUND TRUTH</h5>", unsafe_allow_html=True)
            h_col3.markdown("<h5 style='text-align: center;'>AI</h5>", unsafe_allow_html=True)
            st.divider() # Line 3

            for img_path in images:
                try:
                    img = Image.open(str(img_path))
                    w, h = img.size
                    
                    # Split logic: Assuming hstack [res_a, res_b, res_f]
                    # Check if w is roughly 3x h to confirm it's an hstack
                    if w >= (h * 2.5):
                        unit_w = w // 3
                        input_img = img.crop((0, 0, unit_w, h))
                        gt_img = img.crop((unit_w, 0, unit_w * 2, h))
                        ai_img = img.crop((unit_w * 2, 0, w, h))
                        
                        r_col1, r_col2, r_col3 = st.columns(3)
                        r_col1.image(input_img, use_container_width=True)
                        r_col2.image(gt_img, use_container_width=True)
                        r_col3.image(ai_img, use_container_width=True)
                        st.divider()
                    else:
                        st.image(img, caption=img_path.name, use_container_width=True)
                        st.divider()
                except Exception as e:
                    st.error(f"Error loading {img_path.name}: {e}")
        else:
            st.info("No test evaluation samples found.")
    else:
        st.info("No test results directory found.")
