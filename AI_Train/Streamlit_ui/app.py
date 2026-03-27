import streamlit as st
import pathlib
import os
import json
import time

# Import utilities
from utils.config_loader import get_available_methods, load_config, save_config, get_method_runs
from utils.executor import ProcessManager
from utils.visualizer import plot_training_history, show_sample_images, show_test_evaluation

# Initialization
st.set_page_config(page_title="AI Pedsim | Dashboard", page_icon="🚀", layout="wide")

# --- UI STYLE INJECTION ---
def inject_custom_css():
    st.markdown("""
    <style>
        /* 1. Sidebar Foundation */
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.1);
        }
        
        /* 2. Navigation Labels / Headers */
        .sidebar-header {
            font-size: 0.75rem;
            font-weight: 700;
            color: rgba(128, 128, 128, 0.9);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin: 1.5rem 0 0.5rem 0;
            padding-left: 8px;
        }

        /* 3. The Nuclear Fix for Radio Button Dots */
        /* Hide the container for the radio dot completely */
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        
        /* Ensure the content is properly aligned after hiding the dot */
        [data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] {
            padding-left: 0 !important;
        }

        /* 4. Menu Item Styling */
        div[role="radiogroup"] {
            gap: 4px;
        }
        
        div[role="radiogroup"] label {
            background-color: transparent !important;
            border-radius: 10px;
            padding: 8px 16px !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            margin: 0 !important;
            border: none !important;
        }
        
        /* Active Item: Use a theme-neutral alpha background */
        div[role="radiogroup"] label[data-selected="true"] {
            background-color: rgba(128, 128, 128, 0.2) !important;
            font-weight: 600 !important;
        }
        
        /* Hover State */
        div[role="radiogroup"] label:hover {
            background-color: rgba(128, 128, 128, 0.1) !important;
        }

        /* Text Sizing */
        div[role="radiogroup"] label p {
            font-size: 0.98rem !important;
            margin: 0 !important;
        }

        /* 5. Tidy up the UI */
        section[data-testid="stSidebar"] hr {
            margin: 1rem 0 !important;
            opacity: 0.15 !important;
        }
    </style>
    """, unsafe_allow_html=True)

if "process_manager" not in st.session_state:
    st.session_state.process_manager = ProcessManager()

# --- SIDEBAR: Premium Navigation ---
inject_custom_css()

# Branding
st.sidebar.markdown("# 🚀 AI Pedsim")

# 📁 Section: Workspace
st.sidebar.markdown('<p class="sidebar-header">Workspace</p>', unsafe_allow_html=True)
AI_TRAIN_DIR = pathlib.Path(__file__).parent.parent
available_methods = get_available_methods(AI_TRAIN_DIR)

if not available_methods:
    st.sidebar.error("No methods found.")
    st.stop()

# Method selection
selected_method = st.sidebar.selectbox("Current AI Method", available_methods, label_visibility="collapsed")
method_path = AI_TRAIN_DIR / selected_method
st.sidebar.markdown(f"📍 Method: **{selected_method}**")

# ⚙️ Section: Pipeline
st.sidebar.markdown('<p class="sidebar-header">Training Pipeline</p>', unsafe_allow_html=True)
navigation = st.sidebar.radio(
    "Pipeline",
    ["🚀 Training model", "🔬 Testing model", "📈 View results"],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.caption("v1.2.5 | JohnniePatt build")

# --- MAIN CONTENT LOGIC ---

# --- PAGE: Training model ---
if navigation == "🚀 Training model":
    st.header(f"Training: {selected_method}")
    
    # 📝 1. Configuration Section (Integrated)
    st.subheader("🛠 Training Configuration")
    config_train = load_config(method_path, config_type="train")
    if not config_train:
        config_train = {"epochs": 100, "batch_size": 4, "learning_rate": 0.0002}
    
    with st.expander("📝 Edit Parameters", expanded=True):
        updated_config_str = st.text_area("JSON Editor (config_train.json)", value=json.dumps(config_train, indent=4), height=250)
        if st.button("💾 Save Training Configuration", use_container_width=True):
            try:
                new_config = json.loads(updated_config_str)
                save_config(method_path, new_config, config_type="train")
                st.success("✅ Training configuration saved!")
                st.rerun()
            except Exception as e: st.error(f"❌ Invalid JSON: {e}")

    st.divider()

    # 🏃 2. Execution Section
    st.subheader("🏃 Execute Training")
    train_scripts = sorted(list(method_path.glob("train_*.py")))
    if not train_scripts:
        st.error("No training scripts found.")
    else:
        selected_script = st.selectbox("Select Script", [s.name for s in train_scripts])
        script_full_path = method_path / selected_script
        
        c1, c2 = st.columns(2)
        if c1.button("🚀 Start Training", use_container_width=True, disabled=st.session_state.process_manager.is_running):
            python_path = AI_TRAIN_DIR.parent / "AI_Pedsim-env" / "bin" / "python3"
            if not python_path.exists(): python_path = "python3"
            command = [str(python_path), str(script_full_path), "--config", "config_train.json"]
            st.session_state.process_manager.start_process(command, str(method_path))
            st.rerun()

        if c2.button("🛑 Stop Training", use_container_width=True, disabled=not st.session_state.process_manager.is_running):
            st.session_state.process_manager.stop_process()
            st.rerun()

        # Monitoring
        if st.session_state.process_manager.is_running:
            st.info("🔥 Training in progress...")
            available_runs = get_method_runs(method_path)
            if available_runs:
                latest_run_path = method_path / available_runs[0]
                progress_file = latest_run_path / "progress.json"
                if progress_file.exists():
                    try:
                        with open(progress_file, "r") as f: p_data = json.load(f)
                        st.progress(p_data["percentage"] / 100.0)
                        st.markdown(f"**Epoch:** {p_data['epoch']} / {p_data['total_epochs']} | **Loss:** {p_data.get('loss', 0):.4f}")
                    except: pass
            
            log_container = st.empty()
            if "training_logs" not in st.session_state: st.session_state.training_logs = ""
            new_output = "".join(list(st.session_state.process_manager.get_output()))
            if new_output: st.session_state.training_logs += new_output
            log_container.code(st.session_state.training_logs, language="bash")
            time.sleep(1); st.rerun()
        else:
            if "training_logs" in st.session_state and st.session_state.training_logs:
                st.subheader("Console Output")
                st.code(st.session_state.training_logs)
                if st.button("🧹 Clear Training Logs"): st.session_state.training_logs = ""; st.rerun()

# --- PAGE: Testing model ---
elif navigation == "🔬 Testing model":
    st.header(f"Testing: {selected_method}")

    # 🛠 1. Test Configuration
    st.subheader("🛠 Testing Configuration")
    config_test = load_config(method_path, config_type="test")
    if not config_test: config_test = {"batch_size": 1}
    
    with st.expander("📝 Edit Parameters", expanded=True):
        updated_config_str = st.text_area("JSON Editor (config_test.json)", value=json.dumps(config_test, indent=4), height=200)
        if st.button("💾 Save Testing Configuration", use_container_width=True):
            try:
                new_config = json.loads(updated_config_str)
                save_config(method_path, new_config, config_type="test")
                st.success("✅ Testing configuration saved!")
                st.rerun()
            except Exception as e: st.error(f"❌ Invalid JSON: {e}")

    st.divider()

    # 🔬 2. Manual Evaluation
    st.subheader("🔬 Manual Evaluation")
    available_runs = get_method_runs(method_path)
    if not available_runs:
        st.info("No runs found to test.")
    else:
        selected_run = st.selectbox("Select Run", available_runs)
        run_full_path = method_path / selected_run
        test_scripts = sorted(list(method_path.glob("test_*.py")))
        
        if not test_scripts:
            st.error("No test scripts found.")
        else:
            selected_script = st.selectbox("Select Script", [s.name for s in test_scripts])
            test_script_path = method_path / selected_script
            
            if st.button("🧪 Run Test Evaluation", use_container_width=True, disabled=st.session_state.process_manager.is_running):
                python_path = AI_TRAIN_DIR.parent / "AI_Pedsim-env" / "bin" / "python3"
                if not python_path.exists(): python_path = "python3"
                command = [str(python_path), str(test_script_path), "--config", "config_test.json", "--run_path", str(run_full_path)]
                st.session_state.process_manager.start_process(command, str(method_path))
                st.rerun()

        # Monitoring
        if st.session_state.process_manager.is_running:
            st.info("⌛ Testing in progress...")
            log_container = st.empty()
            if "test_logs" not in st.session_state: st.session_state.test_logs = ""
            new_output = "".join(list(st.session_state.process_manager.get_output()))
            if new_output: st.session_state.test_logs += new_output
            log_container.code(st.session_state.test_logs)
            time.sleep(1); st.rerun()
        else:
            if "test_logs" in st.session_state and st.session_state.test_logs:
                st.subheader("Evaluation Output")
                st.code(st.session_state.test_logs)
                if st.button("🧹 Clear Test Logs"): st.session_state.test_logs = ""; st.rerun()

# --- PAGE: View results ---
elif navigation == "📈 View results":
    st.header(f"Results: {selected_method}")
    available_runs = get_method_runs(method_path)
    if not available_runs:
        st.info("No runs found for this method yet.")
    else:
        selected_run = st.selectbox("Select Run", available_runs)
        run_full_path = method_path / selected_run
        
        t1, t2, t3 = st.tabs(["📊 Loss Curves", "🖼 Training Samples", "🏁 Final Evaluation"])
        with t1:
            csv_path = run_full_path / "logs" / "training_history.csv"
            plot_training_history(csv_path)
        with t2:
            show_sample_images(run_full_path / "samples")
        with t3:
            show_test_evaluation(run_full_path)
