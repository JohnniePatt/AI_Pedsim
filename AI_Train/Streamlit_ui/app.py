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
PROJECT_ROOT = AI_TRAIN_DIR.parent
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
if selected_method == "Generate_HouseGAN":
    navigation = st.sidebar.radio(
        "Pipeline",
        ["🏠 Design Floor Plan", "📈 View Generated AI Layouts", "🏃 Run Pedsim (Architecture)"],
        label_visibility="collapsed"
    )
else:
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

# --- PAGE: HouseGAN Design Floor Plan ---
elif navigation == "🏠 Design Floor Plan":
    st.header("🏠 AI Topology Generator (HouseGAN)")
    st.markdown("Automated generation of diverse topologies using HouseGAN, including dynamic room connection parsing and door carving.")
    
    # 1. Configs
    st.subheader("🛠 Generation Settings")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    num_scenarios = c1.number_input("Total Topologies", min_value=1, max_value=100, value=5)
    num_corridors = c2.number_input("No. Corridors", min_value=1, max_value=10, value=1)
    random_seed = c3.number_input("Base Seed", min_value=0, max_value=999999, value=42)
    door_width = c4.slider("Door Width (m)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    complexity = c5.selectbox("Graph Complexity", ["Low (3-5 Rooms)", "Medium (5-8 Rooms)", "High (8-15 Rooms)"], index=1)
    
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
            # 2. Config UI
            st.subheader("🛠 Simulation Settings")
            c1, c2, c3, c4 = st.columns(4)
            selected_plan = c1.selectbox("Select Architecture Plan", available_plans)
            num_agents = c2.number_input("Agents", min_value=1, max_value=200, value=50)
            sim_seed = c3.number_input("Seed", min_value=0, max_value=999, value=42)
            grid_size = c4.number_input("Heatmap Grid", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            
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
                python_path = AI_TRAIN_DIR.parent / "AI_Pedsim-env" / "bin" / "python3"
                if not python_path.exists(): python_path = "python3"
                
                command = [
                    str(python_path), str(prepare_script), 
                    "--plan", selected_plan,
                    "--agents", str(num_agents),
                    "--seed", str(sim_seed),
                    "--grid", str(grid_size)
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
