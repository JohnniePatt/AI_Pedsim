import streamlit as st
import pathlib
import os
import json
import time
import re
import pandas as pd
import shutil

# Import utilities
from utils.config_loader import get_available_methods, load_config, save_config, get_method_runs
from utils.executor import ProcessManager
from utils.visualizer import (
    plot_training_history,
    show_sample_images,
    show_test_evaluation,
    show_housegan_results,
    show_transformer_val_visuals,
    show_gnn_cvae_val_visuals,
    show_gnn_cvae2_val_visuals,
    show_gpt_knowledge_results,
    show_gpt_knowledge_special_tests,
    show_pix2pix_special_tests,
)

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


def save_uploaded_file(uploaded_file, target_path):
    target = pathlib.Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())


def render_manual_terminal_command(command_parts, key_suffix: str):
    cmd_text = " ".join(str(x) for x in command_parts)
    with st.expander("📝 Manual Terminal Command", expanded=False):
        st.caption("Copy and run this in your terminal (activate your venv first).")
        st.code(cmd_text, language="bash")


def infer_pix2pixhd_variant_from_script(script_name: str) -> str:
    m = re.search(r"_0?([2-4])\.py$", script_name)
    if m:
        return f"0{m.group(1)}"
    return "default"


def resolve_pix2pixhd_config_filename(config_type: str, profile: str) -> str:
    if profile in {"02", "03", "04"}:
        return f"config_{config_type}_{profile}.json"
    return f"config_{config_type}.json"

if "process_manager" not in st.session_state:
    st.session_state.process_manager = ProcessManager()

# --- SIDEBAR: Premium Navigation ---
inject_custom_css()

# Branding
st.sidebar.markdown("# 🚀 AI Pedsim")

# 📁 Section: Workspace
st.sidebar.markdown('<p class="sidebar-header">Workspace</p>', unsafe_allow_html=True)
TRAJECTORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAJECTORY_ROOT.parent
AI_TRAIN_DIR = TRAJECTORY_ROOT / "AI_Train"
AI_RESULT_DIR = TRAJECTORY_ROOT / "AI_Result"
PYTHON_BIN_CANDIDATES = [
    PROJECT_ROOT.parent / "AI_Pedsim-env" / "bin" / "python3",
    PROJECT_ROOT / "AI_Pedsim-env" / "bin" / "python3",
]


def get_python_executable() -> str:
    for python_bin in PYTHON_BIN_CANDIDATES:
        if python_bin.exists():
            return str(python_bin)
    return "python3"

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
    "🌈 Format BW to COLORJET",
    key="btn_bw_to_colorjet",
    use_container_width=True,
    help="Convert BW mask outputs in test_results predictions/targets into COLORJET images while keeping MASK_ backups"
):
    st.session_state.current_nav = "🌈 Format BW to COLORJET"
    st.rerun()

if st.sidebar.button(
    "📊 Split data (Train,Test,Val)", 
    key="btn_split", 
    use_container_width=True,
    help="Split dataset into Train, Test, and Val sets"
):
    st.session_state.current_nav = "📊 Split data (Train,Test,Val)"
    st.rerun()

if st.sidebar.button(
    "🧰 Prepare Data for UNet",
    key="btn_prepare_unet",
    use_container_width=True,
    help="Organize/copy UNet image dataset into Dataset/Data_ImageUNet"
):
    st.session_state.current_nav = "🧰 Prepare Data for UNet"
    st.rerun()

if st.sidebar.button(
    "🏠 HouseGAN Format",
    key="btn_housegan_format",
    use_container_width=True,
    help="Normalize HouseGAN outputs into Dataset/Data_Traj_Table case folders"
):
    st.session_state.current_nav = "🏠 HouseGAN Format"
    st.rerun()

if st.sidebar.button(
    "📚 Method Documentation",
    key="btn_doc",
    use_container_width=True,
    help="Read technical documentation for AI Methods"
):
    st.session_state.current_nav = "📚 Method Documentation"
    st.rerun()

if st.sidebar.button(
    "💡 Method Concept",
    key="btn_concept",
    use_container_width=True,
    help="Read the concept and architecture explanation for AI Methods"
):
    st.session_state.current_nav = "💡 Method Concept"
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
        div[data-testid="stSidebar"] button[kind="secondary"]:has(div:contains("Method Documentation")) {
            background-color: rgba(128, 128, 128, 0.2) !important;
            font-weight: 600 !important;
        }
        div[data-testid="stSidebar"] button[kind="secondary"]:has(div:contains("Method Concept")) {
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
    if selected_method == "Method_GPT_Knowledge":
        st.header("Feed Knowledge: Method_GPT_Knowledge")
        build_config_path = method_path / "config_build.json"
        build_script_path = method_path / "build_knowledge.py"
        config_build = {}
        if build_config_path.exists():
            with open(build_config_path, "r") as f:
                config_build = json.load(f)

        st.subheader("🧠 Knowledge Configuration")
        dataset_table_root = PROJECT_ROOT / "Dataset" / "Data_Traj_Table"
        available_datasets = sorted([d.name for d in dataset_table_root.iterdir() if d.is_dir()]) if dataset_table_root.exists() else []
        selected_datasets = st.multiselect(
            "Select datasets to feed into knowledge",
            options=available_datasets,
            default=[pathlib.Path(p).name for p in config_build.get("dataset_roots", config_build.get("dataset_root", []))] if available_datasets else [],
        )

        split = st.selectbox("Split", ["train", "test", "val"], index=["train", "test", "val"].index(config_build.get("split", "train")))
        max_cases = st.number_input("Max Cases (0 = all)", min_value=0, value=int(config_build.get("max_cases", 0)))
        knowledge_output_default = config_build.get("knowledge_output_dir", "../../AI_Result/Method_GPT_Knowledge/knowledge/topo_bottleneck_v1")
        knowledge_output_dir = st.text_input("Knowledge Output Directory", value=knowledge_output_default)

        with st.expander("📝 Current Build Config", expanded=False):
            st.code(json.dumps(config_build, indent=2), language="json")

        st.divider()
        st.subheader("🧠 Execute Feed Knowledge")
        c1, c2 = st.columns(2)
        manual_feed_cmd = [
            "python",
            "AI_GenerateTrajectory/AI_Train/Method_GPT_Knowledge/build_knowledge.py",
            "--config",
            "AI_GenerateTrajectory/AI_Train/Method_GPT_Knowledge/config_build.json",
        ]
        if c1.button("🚀 Start Feed Knowledge", use_container_width=True, disabled=st.session_state.process_manager.is_running):
            if not selected_datasets:
                st.error("Please select at least one dataset.")
            elif not build_script_path.exists():
                st.error("build_knowledge.py not found.")
            else:
                new_config = {
                    "dataset_roots": [f"../../../Dataset/Data_Traj_Table/{name}" for name in selected_datasets],
                    "split": split,
                    "max_cases": int(max_cases),
                    "knowledge_output_dir": knowledge_output_dir,
                }
                with open(build_config_path, "w") as f:
                    json.dump(new_config, f, indent=2)

                python_path = pathlib.Path(get_python_executable())
                if not python_path.exists(): python_path = "python3"
                command = [str(python_path), str(build_script_path), "--config", "config_build.json"]
                st.session_state.training_logs = ""
                st.session_state.training_task = "gpt_feed_knowledge"
                st.session_state.process_manager.start_process(command, str(method_path))
                st.rerun()

        if c2.button("🛑 Stop Feed Knowledge", use_container_width=True, disabled=not st.session_state.process_manager.is_running):
            st.session_state.process_manager.stop_process()
            st.rerun()
        render_manual_terminal_command(manual_feed_cmd, "gpt_feed_knowledge")

        if st.session_state.process_manager.is_running and st.session_state.get("training_task") == "gpt_feed_knowledge":
            st.info("🔥 Feeding knowledge in progress...")
            log_container = st.empty()
            if "training_logs" not in st.session_state:
                st.session_state.training_logs = ""
            new_output = "".join(list(st.session_state.process_manager.get_output()))
            if new_output:
                st.session_state.training_logs += new_output
            log_container.code(st.session_state.training_logs, language="bash")
            time.sleep(1)
            st.rerun()
        else:
            if "training_logs" in st.session_state and st.session_state.training_logs:
                st.subheader("Console Output")
                st.code(st.session_state.training_logs)
                if st.button("🧹 Clear Feed Knowledge Logs", key="clear_feed_knowledge_logs"):
                    st.session_state.training_logs = ""
                    st.rerun()

            knowledge_root = result_method_path / "knowledge"
            if knowledge_root.exists():
                builds = sorted([d for d in knowledge_root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                if builds:
                    latest_build = builds[0]
                    manifest_path = latest_build / "knowledge_manifest.json"
                    if manifest_path.exists():
                        with open(manifest_path, "r") as f:
                            manifest = json.load(f)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Latest Build", latest_build.name)
                        c2.metric("Cases", manifest.get("num_cases", 0))
                        c3.metric("Datasets", len(manifest.get("dataset_roots", [])))
                        with st.expander("Show latest knowledge manifest", expanded=False):
                            st.json(manifest)
        st.stop()

    st.header(f"Training: {selected_method}")
    
    # 📝 1. Configuration Section (Integrated)
    st.subheader("🛠 Training Configuration")
    train_config_filename = "config_train.json"
    if selected_method == "Method_pix2pixHD":
        train_profile = st.selectbox(
            "Config Profile",
            ["02", "03", "04", "default"],
            index=2,
            key="pix2pixhd_train_profile",
            help="Separate training config per script generation to prevent mixing.",
        )
        train_config_filename = resolve_pix2pixhd_config_filename("train", train_profile)
    config_train = load_config(method_path, config_type="train", config_name=train_config_filename)
    if not config_train:
        config_train = {"epochs": 100, "batch_size": 4, "learning_rate": 0.0002}
    
    with st.expander("📝 Edit Parameters", expanded=True):
        updated_config_str = st.text_area(f"JSON Editor ({train_config_filename})", value=json.dumps(config_train, indent=4), height=250)
        if st.button("💾 Save Training Configuration", use_container_width=True):
            try:
                new_config = json.loads(updated_config_str)
                save_config(method_path, new_config, config_type="train", config_name=train_config_filename)
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
        script_text = script_full_path.read_text(encoding="utf-8", errors="ignore")
        supports_config = "--config" in script_text
        python_path = pathlib.Path(get_python_executable())
        if not python_path.exists():
            python_path = "python3"
        manual_train_cmd = [str(python_path), str(script_full_path)]
        if supports_config:
            manual_train_cmd += ["--config", train_config_filename]

        c1, c2 = st.columns(2)
        if c1.button("🚀 Start Training", use_container_width=True, disabled=st.session_state.process_manager.is_running):
            command = list(manual_train_cmd)
            st.session_state.process_manager.start_process(command, str(method_path))
            st.rerun()

        if c2.button("🛑 Stop Training", use_container_width=True, disabled=not st.session_state.process_manager.is_running):
            st.session_state.process_manager.stop_process()
            st.rerun()

        render_manual_terminal_command(manual_train_cmd, f"train_{selected_method}_{selected_script}")

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
    test_config_filename = "config_test.json"
    if selected_method == "Method_pix2pixHD":
        test_profile = st.selectbox(
            "Config Profile",
            ["02", "03", "04", "default"],
            index=2,
            key="pix2pixhd_test_profile",
            help="Separate testing config per script generation to prevent mixing.",
        )
        test_config_filename = resolve_pix2pixhd_config_filename("test", test_profile)
    config_test = load_config(method_path, config_type="test", config_name=test_config_filename)
    if not config_test: config_test = {"batch_size": 1}
    
    with st.expander("📝 Edit Parameters", expanded=True):
        updated_config_str = st.text_area(f"JSON Editor ({test_config_filename})", value=json.dumps(config_test, indent=4), height=200)
        if st.button("💾 Save Testing Configuration", use_container_width=True):
            try:
                new_config = json.loads(updated_config_str)
                save_config(method_path, new_config, config_type="test", config_name=test_config_filename)
                st.success("✅ Testing configuration saved!")
                st.rerun()
            except Exception as e: st.error(f"❌ Invalid JSON: {e}")

    st.divider()

    if selected_method == "Method_GPT_Knowledge":
        st.subheader("🧪 Special Blind Test")
        st.caption("Upload a new scene, generate AI paths from the knowledge base, then optionally upload the hidden ground truth to compare AI vs GT.")

        special_cfg = config_test or {}
        special_root = result_method_path / "special_tests"
        knowledge_root = result_method_path / "knowledge"
        knowledge_builds = sorted([d for d in knowledge_root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True) if knowledge_root.exists() else []
        configured_build_name = pathlib.Path(str(special_cfg.get("knowledge_dir", "../../AI_Result/Method_GPT_Knowledge/knowledge/topo_bottleneck_v1"))).name
        build_names = [d.name for d in knowledge_builds]

        def parse_plan_key_from_parquet_name(file_name: str) -> str:
            stem = pathlib.Path(file_name).stem.strip()
            stem = re.sub(r"_trajectory_data$", "", stem, flags=re.IGNORECASE)
            stem = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
            return stem or "plan_unknown"

        def next_case_id_for_plan(plan_key: str) -> int:
            start_id = 100001
            used_ids = []
            if not special_root.exists():
                return start_id

            pattern = re.compile(rf"^TEST_{re.escape(plan_key)}_(\d+)$")
            for d in [x for x in special_root.iterdir() if x.is_dir()]:
                manifest_path = d / "special_test_manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                        m_plan = str(manifest.get("plan_key", ""))
                        m_case = str(manifest.get("case_id", ""))
                        if m_plan == plan_key and m_case.isdigit():
                            used_ids.append(int(m_case))
                            continue
                    except Exception:
                        pass

                match = pattern.match(d.name)
                if match:
                    used_ids.append(int(match.group(1)))

            return (max(used_ids) + 1) if used_ids else start_id
        if configured_build_name in build_names:
            default_build_idx = build_names.index(configured_build_name)
        else:
            default_build_idx = 0 if build_names else None

        selected_build_name = st.selectbox(
            "Knowledge Build",
            options=build_names if build_names else [configured_build_name],
            index=default_build_idx if default_build_idx is not None else 0,
            help="Choose which built knowledge index the special test should use.",
        )
        knowledge_dir = knowledge_root / selected_build_name
        scene_index_path = knowledge_dir / "scene_index.parquet"

        kc1, kc2 = st.columns([2, 1])
        with kc1:
            if scene_index_path.exists():
                st.success(f"Knowledge ready: `{knowledge_dir}`")
            else:
                st.warning(f"Knowledge not built yet: `{knowledge_dir}`")
        with kc2:
            if st.button("🧠 Go Build Knowledge", use_container_width=True, key="goto_feed_knowledge"):
                st.session_state.current_nav = "🚀 Training model"
                st.rerun()

        if st.button("💾 Use This Knowledge Build", key="save_selected_knowledge_build"):
            config_test["knowledge_dir"] = f"../../AI_Result/Method_GPT_Knowledge/knowledge/{selected_build_name}"
            save_config(method_path, config_test, config_type="test")
            st.success(f"Saved test config to use `{selected_build_name}`.")
            st.rerun()

        with st.expander("📥 Input Scene Upload", expanded=True):
            c1, c2 = st.columns(2)
            plan_parquet_file = st.file_uploader(
                "Plan Trajectory Parquet (for Session Name + Numeric ID)",
                type=["parquet"],
                key="gpt_special_plan_parquet",
                help="Example: plan_61_58be_42_00_trajectory_data.parquet -> TEST_plan_61_58be_42_00",
            )
            if plan_parquet_file is not None:
                plan_key = parse_plan_key_from_parquet_name(plan_parquet_file.name)
                next_case_id = next_case_id_for_plan(plan_key)
                session_name = f"TEST_{plan_key}_{next_case_id}"
                case_id_input = str(next_case_id)
                c1.text_input("Session Name", value=session_name, disabled=True)
                c2.text_input("Numeric Case ID", value=case_id_input, disabled=True)
            else:
                plan_key = ""
                session_name = ""
                case_id_input = ""
                c1.text_input("Session Name", value="Upload plan parquet to auto-generate", disabled=True)
                c2.text_input("Numeric Case ID", value="Upload plan parquet to auto-generate", disabled=True)
            st.caption("For HouseGAN scenes, upload `Geo_door.json` too so the route respects walls and door openings.")
            geo_room_file = st.file_uploader("Geo_room.json", type=["json"], key="gpt_special_geo_room")
            geo_corridor_file = st.file_uploader("Geo_corridor.json", type=["json"], key="gpt_special_geo_corridor")
            geo_door_file = st.file_uploader("Geo_door.json (Optional, required for HouseGAN)", type=["json"], key="gpt_special_geo_door")
            spawn_location_file = st.file_uploader("Spawn_location.csv", type=["csv"], key="gpt_special_spawn_location")
            spawn_exit_file = st.file_uploader("Spawn_exit.csv", type=["csv"], key="gpt_special_spawn_exit")

            if not scene_index_path.exists():
                st.warning("Knowledge index (`scene_index.parquet`) not found for the selected build. Please build knowledge first.")

            if st.button("🚀 Generate Path", use_container_width=True, key="gpt_special_generate", disabled=st.session_state.process_manager.is_running):
                if not scene_index_path.exists():
                    st.error("Cannot generate yet: selected knowledge build has no `scene_index.parquet`.")
                elif plan_parquet_file is None:
                    st.error("Please upload the plan trajectory parquet first to auto-create Session Name and Numeric Case ID.")
                elif not all([geo_room_file, geo_corridor_file, spawn_location_file, spawn_exit_file]):
                    st.error("Please upload Geo_room, Geo_corridor, Spawn_location, and Spawn_exit before generating.")
                elif not case_id_input.isdigit():
                    st.error("Case ID must be numeric.")
                else:
                    session_dir = special_root / session_name
                    if session_dir.exists():
                        shutil.rmtree(session_dir)
                    case_dir = session_dir / "input_case" / f"case_{case_id_input}"
                    save_uploaded_file(geo_room_file, case_dir / "Geo_room.json")
                    save_uploaded_file(geo_corridor_file, case_dir / "Geo_corridor.json")
                    if geo_door_file is not None:
                        save_uploaded_file(geo_door_file, case_dir / "Geo_door.json")
                    save_uploaded_file(spawn_location_file, case_dir / f"Spawn_location_{case_id_input}.csv")
                    save_uploaded_file(spawn_exit_file, case_dir / f"Spawn_exit_{case_id_input}.csv")
                    save_uploaded_file(plan_parquet_file, case_dir / f"{plan_key}_trajectory_data.parquet")

                    python_path = pathlib.Path(get_python_executable())
                    if not python_path.exists(): python_path = "python3"
                    special_script = method_path / "special_test_gpt_knowledge.py"
                    command = [
                        str(python_path),
                        str(special_script),
                        "--config", test_config_filename,
                        "--input_case_dir", str(case_dir),
                        "--output_dir", str(session_dir),
                        "--plan_key", str(plan_key),
                    ]
                    st.session_state.test_logs = ""
                    st.session_state.test_task = "gpt_special_generate"
                    st.session_state.process_manager.start_process(command, str(method_path))
                    st.rerun()
        with st.expander("🎯 Upload GT Answer And Compare", expanded=False):
            ready_sessions = []
            if special_root.exists():
                for d in sorted([d for d in special_root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True):
                    if (d / "special_test_manifest.json").exists():
                        ready_sessions.append(d)
            if not ready_sessions:
                st.info("Generate at least one successful special test session first.")
            else:
                selected_session = st.selectbox("Select Session To Compare", ready_sessions, format_func=lambda p: p.name, key="gpt_special_compare_session")
                gt_parquet_file = st.file_uploader("Ground Truth Trajectory (.parquet)", type=["parquet"], key="gpt_special_gt")
                if st.button("⚖️ Compare AI vs GT", use_container_width=True, key="gpt_special_compare", disabled=st.session_state.process_manager.is_running):
                    manifest_path = selected_session / "special_test_manifest.json"
                    if not manifest_path.exists():
                        st.error("This session does not have a manifest yet. Generate the AI prediction first.")
                    elif gt_parquet_file is None:
                        st.error("Please upload the GT parquet file before comparing.")
                    else:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                        gt_path = selected_session / "uploaded_gt.parquet"
                        save_uploaded_file(gt_parquet_file, gt_path)
                        python_path = pathlib.Path(get_python_executable())
                        if not python_path.exists(): python_path = "python3"
                        special_script = method_path / "special_test_gpt_knowledge.py"
                        command = [
                            str(python_path),
                            str(special_script),
                            "--config", test_config_filename,
                            "--input_case_dir", manifest["input_case_dir"],
                            "--output_dir", str(selected_session),
                            "--gt_parquet", str(gt_path),
                        ]
                        st.session_state.test_logs = ""
                        st.session_state.test_task = "gpt_special_compare"
                        st.session_state.process_manager.start_process(command, str(method_path))
                        st.rerun()

        if st.session_state.process_manager.is_running and st.session_state.get("test_task") in {"gpt_special_generate", "gpt_special_compare"}:
            status_text = "Generating AI path..." if st.session_state.get("test_task") == "gpt_special_generate" else "Comparing AI vs GT..."
            st.info(f"⌛ {status_text}")
            log_container = st.empty()
            if "test_logs" not in st.session_state:
                st.session_state.test_logs = ""
            new_output = "".join(list(st.session_state.process_manager.get_output()))
            if new_output:
                st.session_state.test_logs += new_output
            log_container.code(st.session_state.test_logs)
            time.sleep(1)
            st.rerun()
        else:
            if "test_logs" in st.session_state and st.session_state.test_logs:
                st.subheader("Special Test Output")
                st.code(st.session_state.test_logs)
                if st.button("🧹 Clear Special Test Logs", key="clear_gpt_special_logs"):
                    st.session_state.test_logs = ""
                    st.rerun()

        show_gpt_knowledge_special_tests(result_method_path)
        st.stop()

    if selected_method in ["Method_Unet-pix2pix", "Method_pix2pixHD"]:
        st.subheader("🧪 Special Blind Test")
        st.caption("Upload one plan parquet, auto-resolve its case folder, then run single-case inference with the selected trained run.")

        special_root = result_method_path / "special_tests"
        special_root.mkdir(parents=True, exist_ok=True)

        def parse_plan_key_from_parquet_name(file_name: str) -> str:
            stem = pathlib.Path(file_name).stem.strip()
            stem = re.sub(r"_trajectory_data$", "", stem, flags=re.IGNORECASE)
            stem = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
            return stem or "plan_unknown"

        def next_case_id_for_plan(plan_key: str) -> int:
            start_id = 100001
            used_ids = []
            pattern = re.compile(rf"^TEST_{re.escape(plan_key)}_(\d+)$")
            for d in [x for x in special_root.iterdir() if x.is_dir()]:
                m = pattern.match(d.name)
                if m:
                    used_ids.append(int(m.group(1)))
            return (max(used_ids) + 1) if used_ids else start_id

        available_runs_for_special = get_method_runs(result_method_path)
        if not available_runs_for_special:
            st.warning("No trained run found in outputs/. Please train first.")
        else:
            selected_special_run = st.selectbox("Run For Special Test", available_runs_for_special, key=f"pix2pix_special_run_{selected_method}")
            run_path_for_special = result_method_path / selected_special_run

            with st.expander("📥 Input Parquet", expanded=True):
                p1, p2 = st.columns(2)
                plan_parquet_file = st.file_uploader(
                    "Plan Trajectory Parquet",
                    type=["parquet"],
                    key=f"pix2pix_special_plan_parquet_{selected_method}",
                    help="Example: plan_61_58be_42_00_trajectory_data.parquet",
                )
                if plan_parquet_file is not None:
                    plan_key = parse_plan_key_from_parquet_name(plan_parquet_file.name)
                    case_name = f"case_{plan_key}"
                    dataset_root = PROJECT_ROOT / "Dataset" / "Data_Traj_Table"
                    matches = list(dataset_root.glob(f"**/{case_name}")) if dataset_root.exists() else []
                    case_dir = matches[0].resolve() if matches else None
                    next_case_id = next_case_id_for_plan(plan_key)
                    session_name = f"TEST_{plan_key}_{next_case_id}"
                    p1.text_input("Session Name", value=session_name, disabled=True, key=f"pix2pix_special_session_{selected_method}")
                    p2.text_input("Numeric Case ID", value=str(next_case_id), disabled=True, key=f"pix2pix_special_caseid_{selected_method}")
                    if case_dir is not None:
                        st.success(f"Resolved case dir: `{case_dir}`")
                    else:
                        st.error(f"Cannot find `{case_name}` under Dataset/Data_Traj_Table.")
                else:
                    plan_key = ""
                    case_dir = None
                    session_name = ""
                    p1.text_input("Session Name", value="Upload parquet to auto-generate", disabled=True, key=f"pix2pix_special_session_empty_{selected_method}")
                    p2.text_input("Numeric Case ID", value="Upload parquet to auto-generate", disabled=True, key=f"pix2pix_special_caseid_empty_{selected_method}")

                if st.button("🚀 Run Special Test", use_container_width=True, key=f"pix2pix_special_generate_{selected_method}", disabled=st.session_state.process_manager.is_running):
                    if plan_parquet_file is None:
                        st.error("Please upload plan trajectory parquet first.")
                    elif case_dir is None:
                        st.error("Case folder could not be auto-resolved from parquet name.")
                    else:
                        session_dir = special_root / session_name
                        if session_dir.exists():
                            shutil.rmtree(session_dir)
                        session_dir.mkdir(parents=True, exist_ok=True)

                        script_name = "special_test_pix2pixHD_trajectory.py" if selected_method == "Method_pix2pixHD" else "special_test_pix2pix_trajectory.py"
                        special_script = method_path / script_name
                        python_path = pathlib.Path(get_python_executable())
                        if not python_path.exists():
                            python_path = "python3"
                        command = [
                            str(python_path),
                            str(special_script),
                            "--run_path", str(run_path_for_special),
                            "--input_case_dir", str(case_dir),
                            "--output_dir", str(session_dir),
                            "--plan_key", str(plan_key),
                        ]
                        st.session_state.test_logs = ""
                        st.session_state.test_task = "pix2pix_special_generate"
                        st.session_state.process_manager.start_process(command, str(method_path))
                        st.rerun()

            show_pix2pix_special_tests(result_method_path)
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
            test_script_text = test_script_path.read_text(encoding="utf-8", errors="ignore")
            supports_config = "--config" in test_script_text
            supports_run_path = "--run_path" in test_script_text
            python_path = pathlib.Path(get_python_executable())
            if not python_path.exists():
                python_path = "python3"
            manual_test_cmd = [str(python_path), str(test_script_path)]
            if supports_config:
                manual_test_cmd += ["--config", test_config_filename]
            if supports_run_path:
                manual_test_cmd += ["--run_path", str(run_full_path)]
            
            if st.button("🧪 Run Test Evaluation", use_container_width=True, disabled=st.session_state.process_manager.is_running):
                command = list(manual_test_cmd)
                st.session_state.process_manager.start_process(command, str(method_path))
                st.rerun()

            render_manual_terminal_command(manual_test_cmd, f"test_{selected_method}_{selected_script}")

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
    if selected_method == "Method_GPT_Knowledge":
        visual_script = method_path / "visual_gpt_knowledge.py"
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("🧭 Generate Visuals", use_container_width=True, key="gen_gpt_knowledge_visuals", disabled=st.session_state.process_manager.is_running or not visual_script.exists()):
                python_path = pathlib.Path(get_python_executable())
                if not python_path.exists(): python_path = "python3"
                command = [str(python_path), str(visual_script), "--result_dir", str(result_method_path)]
                st.session_state.results_logs = ""
                st.session_state.results_task = "gpt_knowledge_visuals"
                st.session_state.process_manager.start_process(command, str(method_path))
                st.rerun()
        with c2:
            if not visual_script.exists():
                st.warning("visual_gpt_knowledge.py not found in this method folder.")
            else:
                st.info("Generate GT vs AI visual comparisons from the saved retrieval outputs and show them below.")

        if st.session_state.process_manager.is_running and st.session_state.get("results_task") == "gpt_knowledge_visuals":
            st.info("🧭 Generating GPT knowledge visuals...")
            if "results_logs" not in st.session_state:
                st.session_state.results_logs = ""
            new_output = "".join(list(st.session_state.process_manager.get_output()))
            if new_output:
                st.session_state.results_logs += new_output
            st.code(st.session_state.results_logs)
            time.sleep(1)
            st.rerun()
        else:
            if st.session_state.get("results_task") == "gpt_knowledge_visuals" and st.session_state.get("results_logs"):
                with st.expander("Visual generation logs", expanded=False):
                    st.code(st.session_state.results_logs)
        show_gpt_knowledge_results(result_method_path)
        st.stop()
    available_runs = get_method_runs(result_method_path)
    if not available_runs:
        st.info("No runs found for this method yet.")
    else:
        selected_run = st.selectbox("Select Run", available_runs)
        run_full_path = result_method_path / selected_run
        
        if selected_method in ["Method_Transformer", "Method_GNN_CVAE", "Method_GNN_CVAE2"]:
            t1, t2, t3, t4 = st.tabs(["📊 Loss Curves", "🖼 Training Samples", "🧭 Val Epoch Visuals", "🏁 Final Evaluation"])
            with t1:
                csv_path = run_full_path / "logs" / "training_history.csv"
                plot_training_history(csv_path)
            with t2:
                show_sample_images(run_full_path / "samples")
            with t3:
                if selected_method == "Method_Transformer":
                    st.caption("Generate and inspect per-epoch validation plots built from the saved transformer samples.")
                    visual_script = method_path / "visual_transformer.py"
                    task_name = "transformer_visuals"
                    viewer_fn = show_transformer_val_visuals
                    missing_msg = "visual_transformer.py not found in this method folder."
                    running_msg = "🧭 Generating transformer validation visuals..."
                    run_path_arg = "--run_dir"
                elif selected_method == "Method_GNN_CVAE2":
                    st.caption("Generate and inspect per-epoch validation plots built from the saved GNN-CVAE2 samples.")
                    visual_script = method_path / "visual_gnn_cvae2.py"
                    task_name = "gnn_cvae2_visuals"
                    viewer_fn = show_gnn_cvae2_val_visuals
                    missing_msg = "visual_gnn_cvae2.py not found in this method folder."
                    running_msg = "🧭 Generating GNN-CVAE2 validation visuals..."
                    run_path_arg = "--run_path"
                else:
                    st.caption("Generate and inspect per-epoch validation plots built from the saved GNN-CVAE samples.")
                    visual_script = method_path / "visual_gnn_cvae.py"
                    task_name = "gnn_cvae_visuals"
                    viewer_fn = show_gnn_cvae_val_visuals
                    missing_msg = "visual_gnn_cvae.py not found in this method folder."
                    running_msg = "🧭 Generating GNN-CVAE validation visuals..."
                    run_path_arg = "--run_dir"

                c1, c2 = st.columns([1, 2])
                with c1:
                    if st.button("🧭 Generate Val Visuals", use_container_width=True, key=f"gen_val_visuals_{selected_method}_{selected_run}", disabled=st.session_state.process_manager.is_running or not visual_script.exists()):
                        python_path = pathlib.Path(get_python_executable())
                        if not python_path.exists(): python_path = "python3"
                        command = [str(python_path), str(visual_script), run_path_arg, str(run_full_path)]
                        st.session_state.results_logs = ""
                        st.session_state.results_task = task_name
                        st.session_state.process_manager.start_process(command, str(method_path))
                        st.rerun()
                with c2:
                    if not visual_script.exists():
                        st.warning(missing_msg)
                    else:
                        st.info("Recommended placement: keep this in View results, because it is a post-training analysis of the saved validation samples for each epoch.")

                if st.session_state.process_manager.is_running and st.session_state.get("results_task") == task_name:
                    st.info(running_msg)
                    if "results_logs" not in st.session_state:
                        st.session_state.results_logs = ""
                    new_output = "".join(list(st.session_state.process_manager.get_output()))
                    if new_output:
                        st.session_state.results_logs += new_output
                    st.code(st.session_state.results_logs)
                    time.sleep(1); st.rerun()
                else:
                    if st.session_state.get("results_task") == task_name and st.session_state.get("results_logs"):
                        with st.expander("Visual generation logs", expanded=False):
                            st.code(st.session_state.results_logs)
                    viewer_fn(run_full_path)
            with t4:
                show_test_evaluation(run_full_path)
        else:
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
                
            python_path = pathlib.Path(get_python_executable())
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
                    python_path = pathlib.Path(get_python_executable())
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
                
                python_path = pathlib.Path(get_python_executable())
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
                python_path = pathlib.Path(get_python_executable())
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
                        python_path = pathlib.Path(get_python_executable())
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

# --- PAGE: Format BW to COLORJET ---
elif navigation == "🌈 Format BW to COLORJET":
    st.header("🌈 Format BW to COLORJET")
    st.markdown(
        "Convert BW mask images inside a run's `test_results/predictions` and `test_results/targets` "
        "into COLORJET images. The original BW image is preserved as `MASK_<filename>`."
    )

    formatter_script = PROJECT_ROOT / "Tool_utility" / "format_bw_to_colorjet.py"
    if not formatter_script.exists():
        st.error(f"❌ Utility script not found: {formatter_script}")
    else:
        st.subheader("📁 Select Run")
        pix2pix_result_method_path = AI_RESULT_DIR / "Method_pix2pixHD"
        runs = get_method_runs(pix2pix_result_method_path)
        if not runs:
            st.warning(f"No runs found under `{pix2pix_result_method_path / 'outputs'}`.")
        else:
            preferred_run = "outputs/run_HD_20260517_133538"
            default_index = runs.index(preferred_run) if preferred_run in runs else 0
            selected_run = st.selectbox("Run Folder", runs, index=default_index)
            run_path = pix2pix_result_method_path / selected_run
            test_results_path = run_path / "test_results"

            c1, c2 = st.columns(2)
            c1.info(f"📂 **Run:** `{run_path.relative_to(PROJECT_ROOT)}`")
            c2.info(f"🧪 **test_results:** `{test_results_path.relative_to(PROJECT_ROOT)}`")

            st.subheader("⚙️ Formatting Settings")
            target_dirs = st.multiselect(
                "Folders under test_results",
                ["predictions", "targets"],
                default=["predictions", "targets"],
                help="Only these folders will be processed. inputs are intentionally not touched.",
            )
            overwrite_colorjet = st.checkbox(
                "Regenerate COLORJET if MASK_ backup already exists",
                value=True,
                help="Use this when you rerun the formatter and want to rebuild the normal filename from MASK_<filename>.",
            )
            dry_run = st.checkbox(
                "Dry run only",
                value=True,
                help="Preview actions without renaming or overwriting files.",
            )

            if test_results_path.exists():
                preview_rows = []
                for dirname in target_dirs:
                    d = test_results_path / dirname
                    if d.exists():
                        normal_count = len([p for p in d.glob("*.png") if not p.name.startswith("MASK_")])
                        mask_count = len(list(d.glob("MASK_*.png")))
                    else:
                        normal_count = 0
                        mask_count = 0
                    preview_rows.append({"folder": dirname, "normal_png": normal_count, "mask_backup": mask_count})
                if preview_rows:
                    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
            else:
                st.warning(f"`test_results` not found in selected run: {test_results_path}")

            render_manual_terminal_command(
                [
                    get_python_executable(),
                    formatter_script,
                    "--run_path",
                    run_path,
                    "--dirs",
                    *target_dirs,
                    *(["--overwrite"] if overwrite_colorjet else []),
                    *(["--dry_run"] if dry_run else []),
                ],
                "bw_to_colorjet",
            )

            if st.button(
                "🚀 Start Format BW to COLORJET",
                use_container_width=True,
                disabled=st.session_state.process_manager.is_running or not target_dirs or not test_results_path.exists(),
            ):
                python_path = pathlib.Path(get_python_executable())
                if not python_path.exists():
                    python_path = "python3"

                command = [
                    str(python_path),
                    str(formatter_script),
                    "--run_path",
                    str(run_path),
                    "--dirs",
                    *target_dirs,
                ]
                if overwrite_colorjet:
                    command.append("--overwrite")
                if dry_run:
                    command.append("--dry_run")

                st.session_state.bw_to_colorjet_logs = ""
                st.session_state.process_manager.start_process(command, str(PROJECT_ROOT))
                st.rerun()

            if st.session_state.process_manager.is_running:
                st.info("🔥 Formatting BW masks to COLORJET...")
                log_container = st.empty()
                if "bw_to_colorjet_logs" not in st.session_state:
                    st.session_state.bw_to_colorjet_logs = ""
                new_output = "".join(list(st.session_state.process_manager.get_output()))
                if new_output:
                    st.session_state.bw_to_colorjet_logs += new_output
                log_container.code(st.session_state.bw_to_colorjet_logs)
                time.sleep(1)
                st.rerun()
            else:
                if "bw_to_colorjet_logs" in st.session_state and st.session_state.bw_to_colorjet_logs:
                    st.subheader("Formatter Output")
                    st.code(st.session_state.bw_to_colorjet_logs)
                    if st.button("🧹 Clear Format BW to COLORJET Logs"):
                        st.session_state.bw_to_colorjet_logs = ""
                        st.rerun()

# --- PAGE: Prepare Data for UNet ---
elif navigation == "🧰 Prepare Data for UNet":
    st.header("🧰 Prepare Data for UNet")
    st.markdown("Organize image-pair dataset format (`A/B + train/test/validation`) into `Dataset/Data_ImageUNet/Trajectory_line_dataset/<Topology>`.")

    source_root = PROJECT_ROOT / "Geo_scenario"
    output_root = PROJECT_ROOT / "Dataset" / "Data_ImageUNet"
    prepare_script = PROJECT_ROOT / "Tool_utility" / "prepare_unet_image_dataset.py"
    dataset_subpath = st.text_input("Dataset Relative Path", value="trajectory_line_dataset/Cleandata_1")
    output_group = st.text_input("Output Group Folder", value="Trajectory_line_dataset")

    c1, c2 = st.columns(2)
    c1.info(f"📂 **Source Root:** `{source_root.relative_to(PROJECT_ROOT)}`")
    c2.info(f"✨ **Output Root:** `{output_root.relative_to(PROJECT_ROOT)}`")

    if not source_root.exists():
        st.error(f"❌ Source root not found: {source_root}")
    elif not prepare_script.exists():
        st.error(f"❌ Utility script not found: {prepare_script}")
    else:
        all_topologies = sorted([d.name for d in source_root.iterdir() if d.is_dir()])
        default_topologies = [t for t in all_topologies if (source_root / t / pathlib.Path(dataset_subpath)).exists()]
        if not default_topologies and all_topologies:
            default_topologies = [all_topologies[0]]

        selected_topologies = st.multiselect(
            "Select Topologies",
            options=all_topologies,
            default=default_topologies,
        )
        overwrite_output = st.checkbox("Overwrite Existing Output", value=False)

        # Advanced options (HouseGAN render parameters)
        with st.expander("⚙️ Advanced Options (HouseGAN render)", expanded=False):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            target_size    = adv_col1.number_input("Target Size (px)", min_value=128, max_value=1024, value=512, step=128,
                                                    help="Square canvas size for A/B images.")
            canvas_padding = adv_col2.number_input("Canvas Padding (m)", min_value=0.0, max_value=5.0, value=1.0, step=0.5,
                                                    help="Padding around walkable area in metres.")
            line_width     = adv_col3.number_input("Trajectory Line Width (px)", min_value=1, max_value=10, value=2, step=1,
                                                    help="Pixel width of drawn trajectory lines.")

        if selected_topologies:
            preview_rows = []
            housegan_hint = False
            for topo in selected_topologies:
                src_dataset = source_root / topo / pathlib.Path(dataset_subpath)
                dst_dataset = output_root / output_group / topo
                if not src_dataset.exists() and topo == "Topo_HouseGAN" and dataset_subpath != "trajectory_line":
                    housegan_hint = True
                preview_rows.append(
                    {
                        "topology": topo,
                        "source_exists": src_dataset.exists(),
                        "source_dataset": str(src_dataset.relative_to(PROJECT_ROOT)) if src_dataset.exists() else str(src_dataset),
                        "output_dataset": str(dst_dataset.relative_to(PROJECT_ROOT)),
                    }
                )
            with st.expander("📋 Prepare Preview", expanded=False):
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)
            if housegan_hint:
                st.warning("For `Topo_HouseGAN`, use `Dataset Relative Path = trajectory_line` to build A/B pairs from `spawn_exit` + `trajectory_line`.")

        st.divider()
        st.subheader("🚀 Execute Prepare")
        python_path = pathlib.Path(get_python_executable())
        if not python_path.exists():
            python_path = "python3"
        manual_prepare_cmd = [
            str(python_path),
            str(prepare_script),
            "--source_root", str(source_root),
            "--output_root", str(output_root),
            "--dataset_subpath", dataset_subpath,
            "--output_group", output_group,
            "--topologies", *selected_topologies,
            "--target_size",    str(int(target_size)),
            "--canvas_padding", str(float(canvas_padding)),
            "--line_width",     str(int(line_width)),
        ]
        if overwrite_output:
            manual_prepare_cmd.append("--overwrite")

        if st.button("🚀 Start Prepare Data for UNet", use_container_width=True, disabled=st.session_state.process_manager.is_running):
            if not selected_topologies:
                st.error("Please select at least one topology.")
            else:
                st.session_state.unet_prepare_logs = ""
                st.session_state.process_manager.start_process(list(manual_prepare_cmd), str(PROJECT_ROOT))
                st.rerun()

        # Manual terminal command — always visible below the button
        st.caption("💻 Manual Terminal Command")
        st.code(" ".join(str(x) for x in manual_prepare_cmd), language="bash")

        if st.session_state.process_manager.is_running:
            st.info("🔥 Preparing Data_ImageUNet...")
            log_container = st.empty()
            if "unet_prepare_logs" not in st.session_state:
                st.session_state.unet_prepare_logs = ""
            new_output = "".join(list(st.session_state.process_manager.get_output()))
            if new_output:
                st.session_state.unet_prepare_logs += new_output
            log_container.code(st.session_state.unet_prepare_logs)
            time.sleep(1)
            st.rerun()
        else:
            if "unet_prepare_logs" in st.session_state and st.session_state.unet_prepare_logs:
                st.subheader("Prepare Output")
                st.code(st.session_state.unet_prepare_logs)
                if st.button("🧹 Clear Prepare Logs", key="clear_prepare_unet_logs"):
                    st.session_state.unet_prepare_logs = ""
                    st.rerun()

            manifest_path = output_root / "manifest_unet_image_dataset.csv"
            if manifest_path.exists():
                try:
                    manifest_df = pd.read_csv(manifest_path)
                    st.subheader("📊 Prepared Dataset Summary")
                    st.dataframe(manifest_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading manifest_unet_image_dataset.csv: {e}")

# --- PAGE: HouseGAN Format ---
elif navigation == "🏠 HouseGAN Format":
    st.header("🏠 HouseGAN Format")
    st.markdown("Normalize HouseGAN simulation outputs into `Dataset/Data_Traj_Table` case folders that match the bottleneck schema.")

    source_root = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN"
    output_root = PROJECT_ROOT / "Dataset" / "Data_Traj_Table" / "Topo_HouseGAN"
    formatter_script = PROJECT_ROOT / "Tool_utility" / "normalize_housegan_dataset.py"

    c1, c2 = st.columns(2)
    c1.info(f"📂 **Source:** `{source_root.relative_to(PROJECT_ROOT)}`")
    c2.info(f"✨ **Output:** `{output_root.relative_to(PROJECT_ROOT)}`")

    if not source_root.exists():
        st.error(f"❌ Source root not found: {source_root}")
    elif not formatter_script.exists():
        st.error(f"❌ Formatter script not found: {formatter_script}")
    else:
        st.subheader("⚙️ Normalize Settings")
        table_filter = st.text_input(
            "SQLite Table Filter",
            value="trajectory_data",
            help="Only import the table whose name contains this text."
        )

        geo_count = len([d for d in (source_root / "geo").iterdir() if d.is_dir()]) if (source_root / "geo").exists() else 0
        sim_count = len(list((source_root / "dataswarm").rglob("*.sqlite"))) if (source_root / "dataswarm").exists() else 0
        m1, m2 = st.columns(2)
        m1.metric("Plan Folders", geo_count)
        m2.metric("SQLite Files", sim_count)

        st.divider()
        if st.button("🚀 Start HouseGAN Format", use_container_width=True, disabled=st.session_state.process_manager.is_running):
            python_path = pathlib.Path(get_python_executable())
            if not python_path.exists(): python_path = "python3"
            command = [
                str(python_path), str(formatter_script),
                "--source_root", str(source_root),
                "--output_root", str(output_root),
                "--filter", table_filter,
            ]
            st.session_state.housegan_format_logs = ""
            st.session_state.process_manager.start_process(command, str(PROJECT_ROOT))
            st.rerun()

        if st.session_state.process_manager.is_running:
            st.info("🔥 Formatting HouseGAN dataset...")
            log_container = st.empty()
            if "housegan_format_logs" not in st.session_state:
                st.session_state.housegan_format_logs = ""
            new_output = "".join(list(st.session_state.process_manager.get_output()))
            if new_output:
                st.session_state.housegan_format_logs += new_output
            log_container.code(st.session_state.housegan_format_logs)
            time.sleep(1)
            st.rerun()
        else:
            if "housegan_format_logs" in st.session_state and st.session_state.housegan_format_logs:
                st.subheader("Formatter Output")
                st.code(st.session_state.housegan_format_logs)
                if st.button("🧹 Clear HouseGAN Format Logs", key="clear_housegan_format_logs"):
                    st.session_state.housegan_format_logs = ""
                    st.rerun()

            manifest_path = output_root / "manifest_housegan_cases.csv"
            if manifest_path.exists():
                try:
                    manifest_df = pd.read_csv(manifest_path)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Normalized Cases", len(manifest_df))
                    c2.metric("Unique Plans", manifest_df["plan_name"].nunique() if "plan_name" in manifest_df.columns else 0)
                    c3.metric("Mean Agents", f"{manifest_df['num_agents'].mean():.1f}" if "num_agents" in manifest_df.columns and len(manifest_df) else "0.0")
                    with st.expander("Show HouseGAN manifest", expanded=False):
                        st.dataframe(manifest_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading manifest_housegan_cases.csv: {e}")

# --- PAGE: Split Data ---
elif navigation == "📊 Split data (Train,Test,Val)":
    st.header("📊 Split Data (Train / Test / Val)")
    st.markdown("Divide your cases into subsets for AI training. This tool randomly shuffles and assigns directories to the correct splits.")
    
    DATASET_TABLE_ROOT = PROJECT_ROOT / "Dataset" / "Data_Traj_Table"
    
    if not DATASET_TABLE_ROOT.exists():
        st.error(f"❌ Root directory `Dataset/Data_Traj_Table` not found at {DATASET_TABLE_ROOT}")
    else:
        # 1. Select Dataset Directory
        st.subheader("📁 Select Target Dataset")
        all_datasets = sorted([d.name for d in DATASET_TABLE_ROOT.iterdir() if d.is_dir()])
        
        if not all_datasets:
            st.info("No datasets found in `Dataset/Data_Traj_Table`.")
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
                        python_path = pathlib.Path(get_python_executable())
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
        if "split_logs" in st.session_state and st.session_state.split_logs:
            st.subheader("Split Tool Output")
            st.code(st.session_state.split_logs)
            if st.button("🧹 Clear Split Logs"): st.session_state.split_logs = ""; st.rerun()

# --- PAGE: Method Documentation ---
elif navigation == "📚 Method Documentation":
    st.header("📚 AI Method Documentation")
    st.markdown("Read the official documentation for the available AI methods in the workspace.")

    if not available_methods:
        st.info("No methods found.")
    else:
        # Default to selected_method from the main sidebar
        default_idx = available_methods.index(selected_method) if selected_method in available_methods else 0
        doc_method = st.selectbox("Select Method to Read", available_methods, index=default_idx)

        doc_path = AI_TRAIN_DIR / doc_method / "DOCUMENT.md"
        st.divider()

        if doc_path.exists():
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content)
        else:
            st.warning(f"No documentation found for `{doc_method}`. (Missing DOCUMENT.md file in the method directory)")

# --- PAGE: Method Concept ---
elif navigation == "💡 Method Concept":
    st.header("💡 AI Method Concept")
    st.markdown("Understand the concept, architecture, and design decisions behind each AI method.")

    if not available_methods:
        st.info("No methods found.")
    else:
        default_idx = available_methods.index(selected_method) if selected_method in available_methods else 0
        concept_method = st.selectbox("Select Method to Read", available_methods, index=default_idx)

        concept_path = AI_TRAIN_DIR / concept_method / "CONCEPT.md"
        st.divider()

        if concept_path.exists():
            with open(concept_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content)
        else:
            st.warning(f"No concept file found for `{concept_method}`. (Missing CONCEPT.md file in the method directory)")



