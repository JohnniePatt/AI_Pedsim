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
st.set_page_config(page_title="AI Training Dashboard", layout="wide")

if "process_manager" not in st.session_state:
    st.session_state.process_manager = ProcessManager()

# --- SIDEBAR: Method Selection ---
st.sidebar.title("🚀 AI Pedsim Dashboard")
AI_TRAIN_DIR = pathlib.Path(__file__).parent.parent
available_methods = get_available_methods(AI_TRAIN_DIR)

if not available_methods:
    st.sidebar.error("No 'Method_' folders found in AI_Train/")
    st.stop()

selected_method = st.sidebar.selectbox("Select Training Method", available_methods)
method_path = AI_TRAIN_DIR / selected_method

st.sidebar.markdown(f"**Current Method:** `{selected_method}`")
st.sidebar.divider()

navigation = st.sidebar.radio("Navigation", ["🛠 Configuration", "🏃 Execute Training", "🔬 Test Model", "📈 View Results"])

# --- PAGE: Configuration ---
if navigation == "🛠 Configuration":
    st.header(f"Configuration: {selected_method}")
    config_data = load_config(method_path)
    
    if not config_data:
        st.warning("No `config_active.json` found. Creating default...")
        config_data = {"epochs": 100, "batch_size": 4, "learning_rate": 0.0002}

    # 📝 Separate Note Area (to prevent JSON syntax errors)
    existing_notes = config_data.get("run_notes", "")
    run_notes_input = st.text_area("📝 Run Notes / Description", value=existing_notes, height=150, help="Describe this experiment. Saved inside config_active.json as 'run_notes'")
    
    # Hide notes from the raw JSON editor to keep it clean
    clean_config = config_data.copy()
    if "run_notes" in clean_config: del clean_config["run_notes"]

    with st.expander("📝 Edit Raw JSON Parameters", expanded=True):
        updated_config_str = st.text_area("JSON Editor", value=json.dumps(clean_config, indent=4), height=300)
        
    if st.button("💾 Save Configuration", use_container_width=True):
        try:
            new_config = json.loads(updated_config_str)
            # Re-attach the run notes from the separate input
            new_config["run_notes"] = run_notes_input
            save_config(method_path, new_config)
            st.success("✅ Configuration and notes saved successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Invalid JSON syntax: {e}")

# --- PAGE: Execute Training ---
elif navigation == "🏃 Execute Training":
    st.header(f"Execute: {selected_method}")
    
    # Try to find the training script
    # Look for train_*.py or similar
    # Try to find scripts
    scripts = sorted(list(method_path.glob("train_*.py")) + list(method_path.glob("generate_*.py")) + list(method_path.glob("test_*.py")))
    if not scripts:
        st.error(f"No executable scripts found in {selected_method}")
    else:
        selected_script = st.selectbox("Select Script to Run", [s.name for s in scripts])
        script_full_path = method_path / selected_script
        
        col1, col2 = st.columns(2)
        
        if col1.button("🚀 Start Training", use_container_width=True, disabled=st.session_state.process_manager.is_running):
            # Run using the python interpreter from the env
            # (Assuming standard location relative to project root)
            python_path = AI_TRAIN_DIR.parent / "AI_Pedsim-env" / "bin" / "python3"
            if not python_path.exists():
                python_path = "python3" # Fallback
            
            command = [str(python_path), str(script_full_path), "--config", "config_active.json"]
            st.write(f"Executing: `{' '.join(command)}`")
            
            st.session_state.process_manager.start_process(command, str(method_path))
            st.rerun()

        if col2.button("🛑 Stop Training", use_container_width=True, disabled=not st.session_state.process_manager.is_running):
            st.session_state.process_manager.stop_process()
            st.warning("Training process stopped.")
            st.rerun()

        # Monitoring
        if st.session_state.process_manager.is_running:
            st.info("🔥 Training is in progress...")
            
            # 1. Progress Bar Logic
            available_runs = get_method_runs(method_path)
            if available_runs:
                latest_run_path = method_path / available_runs[0]
                progress_file = latest_run_path / "progress.json"
                if progress_file.exists():
                    try:
                        with open(progress_file, "r") as f:
                            p_data = json.load(f)
                        st.progress(p_data["percentage"] / 100.0)
                        st.markdown(f"**Epoch:** {p_data['epoch']} / {p_data['total_epochs']} | **Loss:** {p_data['loss']:.4f}")
                    except: pass

            # 2. Live Log Streaming
            st.subheader("Console Output")
            log_container = st.empty()
            
            # Persistent logs in session state to avoid clearing on rerun
            if "training_logs" not in st.session_state:
                st.session_state.training_logs = ""
            
            # Poll for new output
            new_output = "".join(list(st.session_state.process_manager.get_output()))
            if new_output:
                st.session_state.training_logs += new_output
            
            log_container.code(st.session_state.training_logs, language="bash")
            
            # Auto-rerun to keep polling logs
            time.sleep(1)
            st.rerun()
        else:
            # Show final logs if any
            if "training_logs" in st.session_state and st.session_state.training_logs:
                st.subheader("Final Console Output")
                st.code(st.session_state.training_logs)
                if st.button("🧹 Clear Logs"):
                    st.session_state.training_logs = ""
                    st.rerun()

# --- PAGE: Test Model (Manual Evaluation) ---
elif navigation == "🔬 Test Model":
    st.header(f"Test Evaluation: {selected_method}")
    
    available_runs = get_method_runs(method_path)
    if not available_runs:
        st.info("No runs found to test.")
    else:
        selected_run = st.selectbox("Select Run to Evaluate", available_runs)
        run_full_path = method_path / selected_run
        
        # Look for test script
        test_scripts = sorted(list(method_path.glob("test_*.py")))
        if not test_scripts:
            st.error("No test script found in this method folder.")
        else:
            selected_test_script = st.selectbox("Select Test Script", [s.name for s in test_scripts])
            test_script_path = method_path / selected_test_script
            
            st.info(f"Checking run: `{selected_run}`")
            
            if st.button("🧪 Run Manual Evaluation", use_container_width=True, disabled=st.session_state.process_manager.is_running):
                python_path = AI_TRAIN_DIR.parent / "AI_Pedsim-env" / "bin" / "python3"
                if not python_path.exists(): python_path = "python3"
                
                command = [str(python_path), str(test_script_path), "--run_path", str(run_full_path)]
                st.session_state.process_manager.start_process(command, str(method_path))
                st.rerun()

        # Monitoring for Test Process
        if st.session_state.process_manager.is_running:
            st.info("⌛ Evaluation in progress...")
            log_container = st.empty()
            if "test_logs" not in st.session_state: st.session_state.test_logs = ""
            new_output = "".join(list(st.session_state.process_manager.get_output()))
            if new_output: st.session_state.test_logs += new_output
            log_container.code(st.session_state.test_logs)
            time.sleep(1)
            st.rerun()
        else:
            if "test_logs" in st.session_state and st.session_state.test_logs:
                st.subheader("Evaluation Output")
                st.code(st.session_state.test_logs)
                if st.button("🧹 Clear Test Logs"):
                    st.session_state.test_logs = ""
                    st.rerun()

# --- PAGE: View Results ---
elif navigation == "📈 View Results":
    st.header(f"Results: {selected_method}")
    
    available_runs = get_method_runs(method_path)
    if not available_runs:
        st.info("No runs found for this method yet.")
    else:
        selected_run = st.selectbox("Select Run Folder", available_runs)
        run_full_path = method_path / selected_run
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Loss Curves", "🖼 Training Samples", "🏁 Final Evaluation"])
        
        with tab1:
            csv_path = run_full_path / "logs" / "training_history.csv"
            plot_training_history(csv_path)
            
        with tab2:
            show_sample_images(run_full_path / "samples")

        with tab3:
            show_test_evaluation(run_full_path)
