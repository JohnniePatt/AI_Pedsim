import streamlit as st
import pathlib
import os
import json
import time

# Import utilities
from utils.config_loader import get_available_methods, load_config, save_config, get_method_runs
from utils.executor import ProcessManager
from utils.visualizer import plot_training_history, show_sample_images, show_test_evaluation, show_housegan_results

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
        
        /* active Item: Use a theme-neutral alpha background */
        div[role="radiogroup"] label[data-selected="true"] {
            background-color: rgba(128, 128, 128, 0.2) !important;
            font-weight: 600 !important;
        }
        
        /* 5. Custom Button Styling (for standalone nav items) */
        .stButton > button.nav-button {
            background-color: transparent;
            border: none;
            padding: 8px 16px !important;
            border-radius: 10px !important;
            text-align: left;
            width: 100%;
            display: flex;
            align-items: center;
            transition: all 0.2s;
            color: inherit;
        }
        
        .stButton > button.nav-button:hover {
            background-color: rgba(128, 128, 128, 0.1) !important;
        }
        
        .stButton > button.nav-button.active {
            background-color: rgba(128, 128, 128, 0.2) !important;
            font-weight: 600 !important;
        }

        /* 6. Tidy up the UI */
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
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
AI_TRAIN_DIR = PROJECT_ROOT / "AI_Train"
AI_RESULT_DIR = PROJECT_ROOT / "AI_Result"

available_methods = get_available_methods(AI_TRAIN_DIR)

if not available_methods:
    st.sidebar.error("No methods found.")
    st.stop()

# Method selection
selected_method = st.sidebar.selectbox("Current AI Method", available_methods, label_visibility="collapsed")
method_path = AI_TRAIN_DIR / selected_method
result_method_path = AI_RESULT_DIR / selected_method
st.sidebar.markdown(f"📍 Method: **{selected_method}**")

# --- NAVIGATION SYNC LOGIC ---
if "current_nav" not in st.session_state:
    if selected_method == "Generate_HouseGAN":
        st.session_state.current_nav = "🏠 Design Floor Plan"
    else:
        st.session_state.current_nav = "🚀 Training model"

def update_nav():
    # Detect which radio was clicked by checking if it matches current_nav
    # If not, update current_nav and reset other radios conceptually
    pass # Managed by direct assignment in this version for simplicity

# ⚙️ Section: Pipeline
st.sidebar.markdown('<p class="sidebar-header">Training Pipeline</p>', unsafe_allow_html=True)
if selected_method == "Generate_HouseGAN":
    pipeline_options = ["🏠 Design Floor Plan", "📈 View Generated AI Layouts", "🏃 Run Pedsim (Architecture)"]
else:
    pipeline_options = ["🚀 Training model", "🔬 Testing model", "📈 View results"]

# Index to keep radio selected if current_nav is in pipeline_options
try:
    pipe_index = pipeline_options.index(st.session_state.current_nav)
except ValueError:
    pipe_index = 0

nav_pipeline = st.sidebar.radio(
    "Pipeline",
    pipeline_options,
    index=pipe_index,
    label_visibility="collapsed",
    key="pipe_radio"
)

# Update session state if this radio is clicked
if nav_pipeline != st.session_state.current_nav:
    # Check if this change was likely a manual click from the user
    # If we are in utility mode, the radio defaults to pipeline_options[0].
    # We only switch back if the user clicks something that is NOT the default 
    # OR if we were previously in a pipeline mode anyway.
    
    in_pipeline = st.session_state.current_nav in pipeline_options
    is_manual_switch = (nav_pipeline != pipeline_options[0]) or in_pipeline
    
    if is_manual_switch:
        st.session_state.current_nav = nav_pipeline
        st.rerun()

# 🛠️ Section: Utilities (Separated Area)
st.sidebar.divider()
st.sidebar.markdown('<p class="sidebar-header">Utilities</p>', unsafe_allow_html=True)

# Use a button with custom class to match radio look
formatter_active_class = "active" if st.session_state.current_nav == "🧹 Data Formatter" else ""
if st.sidebar.button(
    "🧹 Data Formatter", 
    key="btn_formatter", 
    use_container_width=True,
    help="Convert SQLite to Parquet",
    # Note: Streamlit doesn't support custom classes on buttons directly yet, 
    # but we can wrap it in a div or target the key
):
    st.session_state.current_nav = "🧹 Data Formatter"
    st.rerun()
if st.sidebar.button(
    "📊 Split data (Train,Test,Val)", 
    key="btn_split", 
    use_container_width=True,
    help="Split dataset into Train, Test, and Val sets"
):
    st.session_state.current_nav = "📊 Split data (Train,Test,Val)"
    st.rerun()

# Apply the active style via Markdown/CSS hack for the button
if st.session_state.current_nav == "🧹 Data Formatter":
    st.sidebar.markdown("""
    <style>
        div[data-testid="stSidebar"] button[kind="secondary"]:has(div:contains("Data Formatter")) {
            background-color: rgba(128, 128, 128, 0.2) !important;
            font-weight: 600 !important;
        }
        div[data-testid="stSidebar"] button[kind="secondary"]:has(div:contains("Split data")) {
            background-color: rgba(128, 128, 128, 0.2) !important;
            font-weight: 600 !important;
        }
    </style>
    """, unsafe_allow_html=True)

navigation = st.session_state.current_nav

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
            available_runs = get_method_runs(result_method_path)
            if available_runs:
                latest_run_path = result_method_path / available_runs[0]
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
    available_runs = get_method_runs(result_method_path)
    if not available_runs:
        st.info("No runs found to test.")
    else:
        selected_run = st.selectbox("Select Run", available_runs)
        run_full_path = result_method_path / selected_run
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
    available_runs = get_method_runs(result_method_path)
    if not available_runs:
        st.info("No runs found for this method yet.")
    else:
        selected_run = st.selectbox("Select Run", available_runs)
        run_full_path = result_method_path / selected_run
        
        t1, t2, t3 = st.tabs(["📊 Loss Curves", "🖼 Training Samples", "🏁 Final Evaluation"])
        with t1:
            csv_path = run_full_path / "logs" / "training_history.csv"
            plot_training_history(csv_path)
        with t2:
            show_sample_images(run_full_path / "samples")
        with t3:
            show_test_evaluation(run_full_path)

# --- PAGE: HouseGAN Design Floor Plan ---
elif navigation == "🏠 Design Floor Plan":
    st.header("🏠 AI Topology Generator (HouseGAN)")
    st.markdown("Automated generation of diverse topologies using HouseGAN, including dynamic room connection parsing and door carving.")
    
    # 1. Configs
    st.subheader("🛠 Generation Settings")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    num_scenarios = c1.number_input("Total Topologies", min_value=1, max_value=100, value=5)
    num_corridors = c2.number_input("Max Corridors", min_value=1, max_value=10, value=1)
    random_seed = c3.number_input("Base Seed", min_value=0, max_value=999999, value=42)
    door_width = c4.slider("Door Width (m)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    complexity = c5.selectbox("Graph Complexity", ["3-5 Rooms", "5-8 Rooms", "8-15 Rooms", "15-20 Rooms", "20-30 Rooms"], index=1)
    
    config_dict = {
        "num_scenarios": num_scenarios,
        "num_corridors": num_corridors,
        "random_seed": random_seed,
        "door_width": door_width,
        "complexity": complexity
    }
    
    st.divider()
    
    # 2. Execution
    gen_script_path = method_path / "generate_layout.py"
    if not gen_script_path.exists():
        st.warning("HouseGAN Generator script not found yet.")
    else:
        st.subheader("🏃 Execute Generation")
        with st.expander("Show current config", expanded=False):
            st.json(config_dict)
            
        config_file = method_path / "config_housegan.json"
        
        if st.button("✨ Auto-Generate New Topologies", use_container_width=True, disabled=st.session_state.process_manager.is_running):
            # Clear logs before starting
            st.session_state.gen_logs = ""
            
            # Save config for the script to use
            with open(config_file, "w") as f:
                json.dump(config_dict, f, indent=4)
                
            python_path = AI_TRAIN_DIR.parent / "AI_Pedsim-env" / "bin" / "python3"
            if not python_path.exists(): python_path = "python3"
            
            command = [str(python_path), str(gen_script_path), "--config", "config_housegan.json"]
            st.session_state.process_manager.start_process(command, str(method_path))
            st.rerun()

        # Monitoring Loop
        if st.session_state.process_manager.is_running:
            st.info("⌛ AI is actively generating floor plans and carving doors...")
            log_container = st.empty()
            if "gen_logs" not in st.session_state: st.session_state.gen_logs = ""
            new_output = "".join(list(st.session_state.process_manager.get_output()))
            if new_output: st.session_state.gen_logs += new_output
            log_container.code(st.session_state.gen_logs)
            time.sleep(1); st.rerun()
        else:
            if "gen_logs" in st.session_state and st.session_state.gen_logs:
                st.subheader("Generator Output")
                st.code(st.session_state.gen_logs)
                if st.button("🧹 Clear Logs"): st.session_state.gen_logs = ""; st.rerun()

# --- PAGE: HouseGAN View Generated ---
elif navigation == "📈 View Generated AI Layouts":
    st.header("📈 Generated AI Topologies")
    
    runs_dir = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN" / "geo"
    if not runs_dir.exists():
        st.info("No generated layouts found in Geo_scenario/Topo_HouseGAN.")
    else:
        tab1, tab2 = st.tabs(["🖼️ Grid Overview (All Runs)", "🔍 Detailed Run Inspector"])
        
        with tab1:
            # Show Grid Overview
            show_housegan_results(runs_dir)
        
        with tab2:
            # Detailed Inspector
            available_runs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("plan_")], reverse=True)
            if available_runs:
                st.subheader("🔍 Detailed Run Inspector")
                selected_run = st.selectbox("Select Generation Run to Inspect", available_runs)
                run_full_path = runs_dir / selected_run
                
                # Preview + JSON inspect
                st.markdown("### 🗺️ Visualization Preview")
                img_path = run_full_path / "preview.png"
                graph_path = run_full_path / "preview_graph.png"
                
                p_col1, p_col2 = st.columns(2)
                if graph_path.exists():
                    p_col1.image(str(graph_path), caption="Topological Graph (Logic)", use_container_width=True)
                if img_path.exists():
                    p_col2.image(str(img_path), caption="Physical Layout (Space)", use_container_width=True)
                
                # Show Seed Metadata
                meta_path = run_full_path / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, "r") as f: meta = json.load(f)
                    st.info(f"🧬 Generation Seed: **{meta.get('seed', 'N/A')}** | 🚪 Rooms: **{meta.get('rooms', 0)}**")
                    
                st.markdown("### 🗃️ Generated Assets (For Pedsim & Research)")
                c1, c2, c3 = st.columns(3)
                room_json = run_full_path / "geo_room.json"
                corridor_json = run_full_path / "geo_corridor.json"
                graph_json = run_full_path / "topological_graph.json"
                
                if room_json.exists():
                    with c1.expander(f"📄 {room_json.name}"):
                        try:
                            with open(room_json, "r") as f: d = json.load(f)
                            st.json(d)
                        except: pass
                if corridor_json.exists():
                    with c2.expander(f"📄 {corridor_json.name}"):
                        try:
                            with open(corridor_json, "r") as f: d = json.load(f)
                            st.json(d)
                        except: pass
                if graph_json.exists():
                    with c3.expander("📄 topological_graph.json"):
                        try:
                            with open(graph_json, "r") as f: d = json.load(f)
                            st.json(d)
                        except: pass

# --- PAGE: Run Pedsim (Architecture) ---
elif navigation == "🏃 Run Pedsim (Architecture)":
    st.header("🏃 Pedestrian Simulation (Architecture Plan)")
    st.markdown("Run Jupedsim simulation on generated HouseGAN layouts with realistic wall obstacles and carved doors.")
    
    # 1. Selection of Plan
    runs_dir = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN" / "geo"
    if not runs_dir.exists():
        st.info("No generated layouts found in Geo_scenario/Topo_HouseGAN/geo.")
    else:
        available_plans = sorted([d.name for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("plan_")], reverse=True)
        if not available_plans:
            st.info("No architecture plans found yet.")
        else:
            # --- BATCH SIMULATION SECTION ---
            swarm_root = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN" / "dataswarm"
            unsimulated_plans = []
            for p in available_plans:
                p_swarm = swarm_root / p
                if not p_swarm.exists() or not any(p_swarm.glob("*.sqlite")):
                    unsimulated_plans.append(p)
            
            st.subheader("📦 Batch Migration & Simulation (HouseGAN)")
            col_b1, col_b2 = st.columns([3, 1])
            with col_b1:
                if unsimulated_plans:
                    st.warning(f"🔔 Found **{len(unsimulated_plans)}** plans that have **not been simulated** yet.")
                else:
                    st.success("✅ All generated HouseGAN plans have simulation results.")
            
            with col_b2:
                if st.button("🚀 Run All Unsimulated", use_container_width=True, help="Batch simulate all HouseGAN plans that lack results", disabled=st.session_state.process_manager.is_running or not unsimulated_plans):
                    st.session_state.sim_arch_logs = ""
                    python_path = AI_TRAIN_DIR.parent / "AI_Pedsim-env" / "bin" / "python3"
                    if not python_path.exists(): python_path = "python3"
                    
                    command = [
                        str(python_path), str(PROJECT_ROOT / "Prepare_data" / "Architecture_housePlan" / "bottleneck_archhouseplan.py"), 
                        "--batch",
                        "--timeout", str(st.session_state.get("sim_timeout", 5))
                    ]
                    st.session_state.process_manager.start_process(command, str(PROJECT_ROOT / "Prepare_data" / "Architecture_housePlan"))
                    st.rerun()

            st.divider()

            # 2. Config UI
            st.subheader("🛠 Single Plan Settings")
            c1, c2, c3, c4, c5 = st.columns(5)
            selected_plan = c1.selectbox("Select Architecture Plan", available_plans)
            num_agents = c2.number_input("Agents", min_value=1, max_value=200, value=50)
            sim_seed = c3.number_input("Seed", min_value=0, max_value=999, value=42)
            grid_size = c4.number_input("Heatmap Grid", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            sim_timeout = c5.number_input("Timeout (Min)", min_value=1, max_value=60, value=5, key="sim_timeout")
            
            view_options = st.multiselect("View Options", ["Trajectory", "Density Heatmap", "Speed Heatmap"], default=["Trajectory", "Density Heatmap", "Speed Heatmap"])
            st.divider()
            
            # --- PREVIEW SECTION ---
            from utils.visualizer import preview_walkable_area
            if st.button("🔍 Preview Walkable Area & Walls", use_container_width=True):
                with st.spinner("Generating Geometric Preview..."):
                    preview_walkable_area(runs_dir / selected_plan)

            st.divider()

            # 3. Execute
            prepare_script = PROJECT_ROOT / "Prepare_data" / "Architecture_housePlan" / "bottleneck_archhouseplan.py"
            
            cb1, cb2 = st.columns(2)
            
            if cb1.button("🚀 Start New Simulation", use_container_width=True, disabled=st.session_state.process_manager.is_running):
                # Clear session state logs before starting
                st.session_state.sim_arch_logs = ""
                
                python_path = AI_TRAIN_DIR.parent / "AI_Pedsim-env" / "bin" / "python3"
                if not python_path.exists(): python_path = "python3"
                
                command = [
                    str(python_path), str(prepare_script), 
                    "--plan", selected_plan,
                    "--agents", str(num_agents),
                    "--seed", str(sim_seed),
                    "--grid", str(grid_size),
                    "--timeout", str(sim_timeout)
                ]
                st.session_state.process_manager.start_process(command, str(prepare_script.parent))
                st.rerun()

            if cb2.button("🎨 Regenerate Previews (Grid Update)", use_container_width=True, disabled=st.session_state.process_manager.is_running):
                python_path = AI_TRAIN_DIR.parent / "AI_Pedsim-env" / "bin" / "python3"
                if not python_path.exists(): python_path = "python3"
                
                command = [
                    str(python_path), str(prepare_script), 
                    "--plan", selected_plan,
                    "--agents", str(num_agents),
                    "--seed", str(sim_seed),
                    "--grid", str(grid_size),
                    "--preview"
                ]
                st.session_state.process_manager.start_process(command, str(prepare_script.parent))
                st.rerun()

            # Monitoring
            if st.session_state.process_manager.is_running:
                st.info(f"🔥 Simulating Pedestrians on `{selected_plan}`...")
                log_container = st.empty()
                if "sim_arch_logs" not in st.session_state: st.session_state.sim_arch_logs = ""
                new_output = "".join(list(st.session_state.process_manager.get_output()))
                if new_output: st.session_state.sim_arch_logs += new_output
                log_container.code(st.session_state.sim_arch_logs)
                time.sleep(1); st.rerun()
            else:
                if "sim_arch_logs" in st.session_state and st.session_state.sim_arch_logs:
                    st.subheader("Simulation Console Output")
                    st.code(st.session_state.sim_arch_logs)
                    if st.button("🧹 Clear Logs"): st.session_state.sim_arch_logs = ""; st.rerun()
                
                # Show results if exist
                output_base = PROJECT_ROOT / "Prepare_data" / "Architecture_housePlan" / "outputs" / selected_plan
                from utils.visualizer import show_pedsim_arch_results
                show_pedsim_arch_results(output_base, options=view_options)

# --- PAGE: Data Formatter ---
elif navigation == "🧹 Data Formatter":
    st.header("🧹 Data Formatter")
    st.markdown("Convert SQLite simulation datasets into highly-efficient Parquet format for faster AI training and analysis.")
    
    GEO_SCENARIO_ROOT = PROJECT_ROOT / "Geo_scenario"
    
    if not GEO_SCENARIO_ROOT.exists():
        st.error(f"❌ Root directory `Geo_scenario` not found at {GEO_SCENARIO_ROOT}")
    else:
        # 1. Select Topology
        st.subheader("📁 Select Scenario Topology")
        all_topos = sorted([d.name for d in GEO_SCENARIO_ROOT.iterdir() if d.is_dir()])
        
        if not all_topos:
            st.info("No topologies found in `Geo_scenario`.")
        else:
            selected_topo = st.selectbox("Select Directory to Format", all_topos)
            topo_path = GEO_SCENARIO_ROOT / selected_topo
            
            # Check for dataswarm folder
            dataswarm_dir = topo_path / "dataswarm"
            output_parquet_dir = topo_path / "dataswarm_parquet"
            
            c1, c2 = st.columns(2)
            c1.info(f"📂 **Source:** `{dataswarm_dir.relative_to(PROJECT_ROOT)}`")
            c2.info(f"✨ **Output:** `{output_parquet_dir.relative_to(PROJECT_ROOT)}`")
            
            if not dataswarm_dir.exists():
                st.warning(f"⚠️ Source directory `dataswarm` not found inside `{selected_topo}`.")
            else:
                st.divider()
                st.subheader("⚙️ Formatting Settings")
                
                # 1. Filter Input
                table_filter = st.text_input("Table Name Filter", value="trajectory_data", help="Only convert tables containing this text (e.g. 'trajectory_data'). Leave empty to convert all tables.")
                
                # 2. Cleanup Utility
                with st.expander("🗑️ Cleanup Utility"):
                    st.write("(Deletes files in the output folder that do NOT match the filter above.(Typing -> xxxxx_Trajectory_data.parquet so you can imput it only Trajectory_data))")
                    if st.button("🧹 Clean Unwanted Parquet Files", use_container_width=True):
                        if output_parquet_dir.exists():
                            files_deleted = 0
                            for p_file in output_parquet_dir.rglob("*.parquet"):
                                if table_filter.lower() not in p_file.name.lower():
                                    p_file.unlink()
                                    files_deleted += 1
                            st.success(f"✅ Deleted {files_deleted} files that didn't match '{table_filter}'.")
                        else:
                            st.warning("No output directory found to clean.")

                st.subheader("🏃 Execute Conversion")
                
                # Check for existing data
                num_sqlite = len(list(dataswarm_dir.rglob("*.sqlite")))
                st.write(f"🔍 Found **{num_sqlite}** SQLite simulation files.")
                
                if st.button("⚡ Start Formatting (SQLite ➡️ Parquet)", use_container_width=True, disabled=st.session_state.process_manager.is_running):
                    formatter_script = PROJECT_ROOT / "Formater" / "format_to_parquet.py"
                    
                    if not formatter_script.exists():
                        st.error(f"❌ Formatter script not found at {formatter_script}")
                    else:
                        python_path = AI_TRAIN_DIR.parent / "AI_Pedsim-env" / "bin" / "python3"
                        if not python_path.exists(): python_path = "python3"
                        
                        command = [
                            str(python_path), str(formatter_script),
                            "--source", str(dataswarm_dir),
                            "--output", str(output_parquet_dir)
                        ]
                        
                        if table_filter:
                            command.extend(["--filter", table_filter])
                            
                        st.session_state.process_manager.start_process(command, str(PROJECT_ROOT))
                        st.rerun()

                # Monitoring Output
                if st.session_state.process_manager.is_running:
                    st.info(f"🔥 Formatting data for `{selected_topo}`...")
                    log_container = st.empty()
                    if "formatter_logs" not in st.session_state: st.session_state.formatter_logs = ""
                    new_output = "".join(list(st.session_state.process_manager.get_output()))
                    if new_output: st.session_state.formatter_logs += new_output
                    log_container.code(st.session_state.formatter_logs)
                    time.sleep(1); st.rerun()
                else:
                    if "formatter_logs" in st.session_state and st.session_state.formatter_logs:
                        st.subheader("Formatter Output")
                        st.code(st.session_state.formatter_logs)
                        if st.button("🧹 Clear Formatter Logs"): st.session_state.formatter_logs = ""; st.rerun()
                    
                    # Show quick stats if output exists
                    if output_parquet_dir.exists():
                        num_parquet = len(list(output_parquet_dir.rglob("*.parquet")))
                        if num_parquet > 0:
                            st.success(f"📚 Successfully formatted **{num_parquet}** parquet files in `{selected_topo}/dataswarm_parquet`.")

# --- PAGE: Split Data ---
elif navigation == "📊 Split data (Train,Test,Val)":
    st.header("📊 Split Data (Train / Test / Val)")
    st.markdown("Divide your cases into subsets for AI training. This tool randomly shuffles and assigns directories to the correct splits.")
    
    DATASET_TABLE_ROOT = PROJECT_ROOT / "Dataset_Traj_Table"
    
    if not DATASET_TABLE_ROOT.exists():
        st.error(f"❌ Root directory `Dataset_Traj_Table` not found at {DATASET_TABLE_ROOT}")
    else:
        # 1. Select Dataset Directory
        st.subheader("📁 Select Target Dataset")
        all_datasets = sorted([d.name for d in DATASET_TABLE_ROOT.iterdir() if d.is_dir()])
        
        if not all_datasets:
            st.info("No datasets found in `Dataset_Traj_Table`.")
        else:
            selected_ds = st.selectbox("Select Dataset to Split", all_datasets)
            ds_path = DATASET_TABLE_ROOT / selected_ds
            
            # 2. Split Ratio Setup
            st.subheader("⚙️ Split Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1: train_pct = st.number_input("Train Ratio (%)", min_value=0, max_value=100, value=70)
            with col2: test_pct = st.number_input("Test Ratio (%)", min_value=0, max_value=100, value=20)
            with col3: val_pct = st.number_input("Val Ratio (%)", min_value=0, max_value=100, value=10)
            
            total_pct = train_pct + test_pct + val_pct
            
            if total_pct != 100:
                st.warning(f"⚠️ Total percentage is **{total_pct}%**. It must be exactly **100%**.")
            else:
                st.info(f"✅ Split Ratio: **{train_pct}/{test_pct}/{val_pct}**")
                
                # Stats
                # Use a specific glob to count cases
                all_cases = [d for d in ds_path.rglob("case_*") if d.is_dir()]
                st.write(f"🔍 Found **{len(all_cases)}** cases inside this dataset folder.")
                
                # 3. Execution
                st.divider()
                if st.button("🚀 Start Split Data", use_container_width=True, disabled=st.session_state.process_manager.is_running):
                    split_script = PROJECT_ROOT / "Tool_utility" / "split_dataset.py"
                    
                    if not split_script.exists():
                        st.error(f"❌ Split script not found at {split_script}")
                    else:
                        python_path = AI_TRAIN_DIR.parent / "AI_Pedsim-env" / "bin" / "python3"
                        if not python_path.exists(): python_path = "python3"
                        
                        command = [
                            str(python_path), str(split_script),
                            "--source", str(ds_path),
                            "--train", str(train_pct / 100),
                            "--test", str(test_pct / 100),
                            "--val", str(val_pct / 100)
                        ]
                        
                        st.session_state.process_manager.start_process(command, str(PROJECT_ROOT))
                        st.rerun()

    # Monitoring Output
    if st.session_state.process_manager.is_running:
        st.info(f"🔥 Splitting data for `{selected_ds if 'selected_ds' in locals() else ''}`...")
        log_container = st.empty()
        if "split_logs" not in st.session_state: st.session_state.split_logs = ""
        new_output = "".join(list(st.session_state.process_manager.get_output()))
        if new_output: st.session_state.split_logs += new_output
        log_container.code(st.session_state.split_logs)
        time.sleep(1); st.rerun()
    else:
        if "split_logs" in st.session_state and st.session_state.split_logs:
            st.subheader("Split Tool Output")
            st.code(st.session_state.split_logs)
            if st.button("🧹 Clear Split Logs"): st.session_state.split_logs = ""; st.rerun()
