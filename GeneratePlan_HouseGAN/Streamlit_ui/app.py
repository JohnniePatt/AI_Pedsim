import json
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from utils.executor import ProcessManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "GeneratePlan_HouseGAN"
SCENARIO_ROOT = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN"
GEN_CONFIG = MODULE_ROOT / "Prepare_data" / "config_housegan.json"
SIM_CONFIG = MODULE_ROOT / "Simulation" / "config_density_sim.json"
THUMB_DIR = MODULE_ROOT / "Streamlit_ui" / ".thumbs"
LEGACY_VISUALIZER_PATH = PROJECT_ROOT / "AI_GenerateTrajectory" / "Streamlit_ui" / "utils" / "visualizer.py"
ROUTE_INFO_GENERATOR_PATH = MODULE_ROOT / "Prepare_data" / "generate_route_information.py"


st.set_page_config(page_title="GeneratePlan HouseGAN", page_icon="🏠", layout="wide")

def read_json(path, fallback=None):
    if not Path(path).exists():
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def list_plans():
    geo_root = SCENARIO_ROOT / "geo"
    if not geo_root.exists():
        return []
    return sorted([p.name for p in geo_root.iterdir() if p.is_dir()])


def show_process_output(key):
    manager = st.session_state.get(key)
    if not manager:
        return
    output_box = st.empty()
    logs = st.session_state.setdefault(f"{key}_logs", [])
    for line in manager.get_output():
        logs.append(line.rstrip())
    output_box.code("\n".join(logs[-160:]) if logs else "Waiting for output...")
    if manager.is_running:
        st.info("Process is running. This page can be refreshed; logs will keep streaming in session.")


def ensure_manager(key):
    if key not in st.session_state:
        st.session_state[key] = ProcessManager()
    return st.session_state[key]


def make_thumbnail(path, size=(900, 620)):
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    stat = path.stat()
    thumb_name = f"{path.stem}_{int(stat.st_mtime)}_{stat.st_size}_{size[0]}x{size[1]}.png"
    thumb_path = THUMB_DIR / thumb_name
    if thumb_path.exists():
        return thumb_path

    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (248, 250, 252))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    canvas.save(thumb_path, "PNG", optimize=True)
    return thumb_path


def image_gallery(paths, columns=6):
    paths = [p for p in paths if p.exists()]
    if not paths:
        st.caption("No images yet.")
        return

    for row_start in range(0, len(paths), columns):
        cols = st.columns(columns)
        for col, path in zip(cols, paths[row_start : row_start + columns]):
            with col:
                with st.container(border=True):
                    thumb_path = make_thumbnail(path)
                    st.image(str(thumb_path), width="stretch")
                    st.caption(path.name)


RESULT_VARIANTS = [
    ("full", "Result view (N Agent)"),
    ("half", "Result view (ผลลัพธ์จากที่ได้จาก agent ลดครึ่งนึง)"),
    ("single", "Result view (Agent แค่ตัวเดียว)"),
]


def route_variant(route, variant_id):
    variants = route.get("variants", [])
    if variants:
        return next((variant for variant in variants if variant.get("variant_id") == variant_id), None)
    if variant_id == "full":
        return {
            "variant_id": "full",
            "variant_label": "N Agent",
            "computed_agents": route.get("computed_agents", 0),
            "status": route.get("status", ""),
            "error": route.get("error", ""),
            "agent_count_distributed": route.get("agent_count_distributed", 0),
        }
    return None


def variant_image_paths(folder, selected_plan, variant_id):
    path = SCENARIO_ROOT / folder / selected_plan
    if not path.exists():
        return []
    variant_paths = sorted(path.glob(f"*_{variant_id}.png"))
    if variant_paths:
        return variant_paths
    return sorted(path.glob("*.png")) if variant_id == "full" else []


def legacy_preview_walkable_area(plan_dir):
    spec = importlib.util.spec_from_file_location("legacy_housegan_visualizer", LEGACY_VISUALIZER_PATH)
    if spec is None or spec.loader is None:
        st.error(f"Cannot load legacy visualizer: {LEGACY_VISUALIZER_PATH}")
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.preview_walkable_area(plan_dir)


def plan_geometry_counts(plan_dir):
    room_count = len(read_json(plan_dir / "geo_room.json", []))
    corridor_count = len(read_json(plan_dir / "geo_corridor.json", []))
    door_count = len(read_json(plan_dir / "geo_door.json", []))
    return room_count, corridor_count, door_count


def png_count(folder, plan_name=None):
    path = SCENARIO_ROOT / folder
    if plan_name:
        path = path / plan_name
    if not path.exists():
        return 0
    return len(list(path.rglob("*.png")))


def sqlite_count(plan_name=None):
    path = SCENARIO_ROOT / "dataswarm"
    if plan_name:
        path = path / plan_name
    if not path.exists():
        return 0
    return len(list(path.rglob("*.sqlite")))


def summary_path(plan_name):
    return SCENARIO_ROOT / "metadata" / plan_name / "simulation_summary.json"


def plan_has_simulation(plan_name):
    summary = read_json(summary_path(plan_name), {})
    if summary.get("routes"):
        return True
    return sqlite_count(plan_name) > 0


def build_adjacency_from_plan(plan_dir):
    room_count, corridor_count, _ = plan_geometry_counts(plan_dir)
    nodes = [f"Cor-{idx}" for idx in range(corridor_count)] + [f"Room-{idx}" for idx in range(room_count)]
    adjacency = {node: set() for node in nodes}
    for door in read_json(plan_dir / "geo_door.json", []):
        rooms = door.get("rooms", [])
        if len(rooms) == 2 and rooms[0] in adjacency and rooms[1] in adjacency:
            adjacency[rooms[0]].add(rooms[1])
            adjacency[rooms[1]].add(rooms[0])
    return adjacency


def shortest_paths_from(adjacency, start):
    queue = [start]
    paths = {start: [start]}
    for node in queue:
        for neighbor in sorted(adjacency.get(node, [])):
            if neighbor in paths:
                continue
            paths[neighbor] = paths[node] + [neighbor]
            queue.append(neighbor)
    return paths


def estimated_route_count(plan_name):
    plan_dir = SCENARIO_ROOT / "geo" / plan_name
    adjacency = build_adjacency_from_plan(plan_dir)
    if not adjacency:
        return 0
    first = next(iter(adjacency))
    if len(shortest_paths_from(adjacency, first)) != len(adjacency):
        return 0

    all_paths = {node: shortest_paths_from(adjacency, node) for node in adjacency}
    diameter = max(len(path) - 1 for paths in all_paths.values() for path in paths.values())
    periphery = [
        node
        for node, paths in all_paths.items()
        if max(len(path) - 1 for path in paths.values()) == diameter
    ]
    seen = set()
    count = 0
    for i, start in enumerate(periphery):
        for end in periphery[i + 1 :]:
            if len(all_paths[start][end]) - 1 != diameter:
                continue
            pair = tuple(sorted((start, end)))
            if pair in seen:
                continue
            seen.add(pair)
            count += 1
    return count


def route_count_for_plan(plan_name):
    summary = read_json(summary_path(plan_name), {})
    if summary.get("routes"):
        return len(summary.get("routes", []))
    return estimated_route_count(plan_name)


def dashboard_rows(plans):
    rows = []
    for plan_name in plans:
        plan_dir = SCENARIO_ROOT / "geo" / plan_name
        rooms, corridors, doors = plan_geometry_counts(plan_dir)
        summary = read_json(summary_path(plan_name), {})
        routes = summary.get("routes", [])
        variant_success = sum(
            1
            for route in routes
            for variant in route.get("variants", [{"status": route.get("status")}])
            if variant.get("status") == "success"
        )
        rows.append(
            {
                "plan": plan_name,
                "simulated": plan_has_simulation(plan_name),
                "rooms": rooms,
                "corridors": corridors,
                "doors": doors,
                "routes": len(routes) if routes else estimated_route_count(plan_name),
                "successful_variant_runs": variant_success,
                "sqlite": sqlite_count(plan_name),
                "trajectory_png": png_count("trajectory_line", plan_name),
                "density_png": png_count("heatmap_density", plan_name),
                "speed_png": png_count("heatmap_speed", plan_name),
                "spawn_png": png_count("spawn_exit", plan_name),
                "offset_png": png_count("offset_area", plan_name),
            }
        )
    return rows


def page_dashboard():
    st.header("HouseGAN Dashboard")
    plans = list_plans()
    if not plans:
        st.warning("No generated geometry found yet.")
        return

    rows = dashboard_rows(plans)
    simulated_rows = [row for row in rows if row["simulated"]]
    unsimulated_rows = [row for row in rows if not row["simulated"]]
    total_rooms = sum(row["rooms"] for row in rows)
    total_corridors = sum(row["corridors"] for row in rows)
    total_doors = sum(row["doors"] for row in rows)
    total_routes = sum(row["routes"] for row in rows)
    simulated_routes = sum(row["routes"] for row in simulated_rows)
    unsimulated_routes = sum(row["routes"] for row in unsimulated_rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Geometry plans", len(plans))
    c2.metric("Simulated plans", len(simulated_rows))
    c3.metric("Unsimulated plans", len(unsimulated_rows))
    c4.metric("Total routes", total_routes)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rooms", total_rooms)
    c2.metric("Corridors", total_corridors)
    c3.metric("Doors", total_doors)
    c4.metric("Unsimulated routes", unsimulated_routes)

    st.subheader("Simulation Outputs")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Trajectory PNG", png_count("trajectory_line"))
    c2.metric("Density PNG", png_count("heatmap_density"))
    c3.metric("Speed PNG", png_count("heatmap_speed"))
    c4.metric("Spawn PNG", png_count("spawn_exit"))
    c5.metric("Offset PNG", png_count("offset_area"))
    c6.metric("SQLite files", sqlite_count())

    st.subheader("Plan Status Table")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Final Summary")
    st.info(
        f"ตอนนี้มี geometry ทั้งหมด {len(plans)} plan, simulate แล้ว {len(simulated_rows)} plan "
        f"คิดเป็น {simulated_routes} route และยังไม่ได้ simulate {len(unsimulated_rows)} plan "
        f"คิดเป็นประมาณ {unsimulated_routes} route."
    )

    csv_path = SCENARIO_ROOT / "route_information" / "all_route_information.csv"
    if csv_path.exists():
        try:
            import matplotlib.pyplot as plt
            
            target_keys = set()
            for plan_name in plans:
                summary = read_json(summary_path(plan_name), {})
                routes = summary.get("routes", [])
                if routes:
                    for r in routes:
                        sn, en = r.get("start_node"), r.get("end_node")
                        if sn and en:
                            target_keys.add((plan_name, tuple(sorted([str(sn), str(en)]))))
                else:
                    plan_dir = SCENARIO_ROOT / "geo" / plan_name
                    adjacency = build_adjacency_from_plan(plan_dir)
                    if adjacency:
                        first = next(iter(adjacency))
                        if len(shortest_paths_from(adjacency, first)) == len(adjacency):
                            all_paths = {node: shortest_paths_from(adjacency, node) for node in adjacency}
                            diameter = max(len(path) - 1 for paths in all_paths.values() for path in paths.values())
                            periphery = [
                                node
                                for node, paths in all_paths.items()
                                if max(len(path) - 1 for path in paths.values()) == diameter
                            ]
                            seen = set()
                            for i, start in enumerate(periphery):
                                for end in periphery[i + 1 :]:
                                    if len(all_paths[start][end]) - 1 != diameter:
                                        continue
                                    pair = tuple(sorted([str(start), str(end)]))
                                    if pair not in seen:
                                        seen.add(pair)
                                        target_keys.add((plan_name, pair))

            df_route = pd.read_csv(csv_path, usecols=["plan", "start_node", "end_node", "topology_centerline_distance_m"])
            df_route = df_route.dropna(subset=["topology_centerline_distance_m"])
            
            df_route["pair_key"] = df_route.apply(lambda row: tuple(sorted([str(row["start_node"]), str(row["end_node"])])), axis=1)
            mask = df_route.apply(lambda row: (row["plan"], row["pair_key"]) in target_keys, axis=1)
            df_filtered = df_route[mask].drop_duplicates(subset=["plan", "pair_key"])

            df_dist = df_filtered.sort_values("topology_centerline_distance_m").reset_index(drop=True)
            df_dist["sorted_index"] = df_dist.index + 1

            fig, ax = plt.subplots(figsize=(12, 4.6))
            ax.plot(
                df_dist["sorted_index"],
                df_dist["topology_centerline_distance_m"],
                color="#2563eb",
                linewidth=1.8,
            )
            ax.set_xlabel("Total routes")
            ax.set_ylabel("topology_centerline_distance_m (m)")
            ax.set_title("Distance Profile (Filtered by actual simulated/estimated routes)")
            ax.grid(True, alpha=0.28)
            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
            if not df_dist.empty:
                fig_box, ax_box = plt.subplots(figsize=(12, 2.5))
                ax_box.boxplot(
                    df_dist["topology_centerline_distance_m"],
                    vert=False,
                    patch_artist=True,
                    boxprops=dict(facecolor="#93c5fd", color="#2563eb", linewidth=1.5),
                    medianprops=dict(color="#1e3a8a", linewidth=2),
                    whiskerprops=dict(color="#2563eb", linewidth=1.5),
                    capprops=dict(color="#2563eb", linewidth=1.5),
                    flierprops=dict(marker="o", markerfacecolor="#bfdbfe", markeredgecolor="#2563eb", alpha=0.6)
                )
                ax_box.set_xlabel("topology_centerline_distance_m (m)")
                ax_box.set_title("Distance Distribution (Box Plot)")
                ax_box.set_yticks([])
                ax_box.grid(True, alpha=0.28, axis="x")
                fig_box.tight_layout()
                
                st.pyplot(fig_box, use_container_width=True)
                plt.close(fig_box)

                try:
                    fig_kde, ax_kde = plt.subplots(figsize=(12, 3.5))
                    # Histogram
                    ax_kde.hist(
                        df_dist["topology_centerline_distance_m"], 
                        bins=30, 
                        density=True, 
                        color="#dbeafe", 
                        edgecolor="#93c5fd", 
                        alpha=0.8
                    )
                    # KDE Curve (Bell Curve)
                    df_dist["topology_centerline_distance_m"].plot.kde(
                        ax=ax_kde, 
                        color="#2563eb", 
                        linewidth=2
                    )
                    kde_lines = ax_kde.get_lines()
                    if kde_lines:
                        kde_x, kde_y = kde_lines[0].get_data()
                        ax_kde.fill_between(kde_x, kde_y, color="#2563eb", alpha=0.1)
                        
                        x_min = df_dist["topology_centerline_distance_m"].min()
                        x_max = df_dist["topology_centerline_distance_m"].max()
                        padding = (x_max - x_min) * 0.1
                        ax_kde.set_xlim(max(0, x_min - padding), x_max + padding)
                    
                    ax_kde.set_xlabel("topology_centerline_distance_m (m)")
                    ax_kde.set_ylabel("Density")
                    ax_kde.set_title("Distance Distribution (Density Curve & Histogram)")
                    ax_kde.grid(True, alpha=0.28)
                    fig_kde.tight_layout()
                    
                    st.pyplot(fig_kde, use_container_width=True)
                    plt.close(fig_kde)
                except Exception as kde_err:
                    st.caption(f"Cannot load Density Curve: {kde_err}")

                total_filtered_routes = len(df_dist)
                min_dist = df_dist["topology_centerline_distance_m"].min()
                max_dist = df_dist["topology_centerline_distance_m"].max()
                mean_dist = df_dist["topology_centerline_distance_m"].mean()
                
                st.caption("### Graph Summary")
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Graph Routes Count", f"{total_filtered_routes:,.0f}")
                sc2.metric("Min Distance", f"{min_dist:,.2f} m")
                sc3.metric("Max Distance", f"{max_dist:,.2f} m")
        except Exception as e:
            st.caption(f"Cannot load Distance Profile graph: {e}")

    latest_images = []
    for folder in ["trajectory_line", "heatmap_density", "heatmap_speed", "spawn_exit", "offset_area"]:
        root = SCENARIO_ROOT / folder
        if root.exists():
            latest_images.extend(sorted(root.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:4])
    st.subheader("Latest Simulation Images")
    image_gallery(latest_images[:12], columns=4)

    if unsimulated_rows:
        with st.expander("Unsimulated geometry list"):
            st.dataframe(pd.DataFrame(unsimulated_rows), use_container_width=True)


def sqlite_fps(cursor):
    row = cursor.execute("select value from metadata where key = 'fps'").fetchone()
    if not row:
        return 25.0
    return safe_float(row[0], 25.0)


def collect_trajectory_files(plan_name):
    summary = read_json(summary_path(plan_name), {})
    files = []
    for route in summary.get("routes", []):
        variants = route.get("variants", [])
        if not variants and route.get("trajectory_file"):
            variants = [
                {
                    "variant_id": "full",
                    "variant_label": "N Agent",
                    "computed_agents": route.get("computed_agents", 0),
                    "trajectory_file": route.get("trajectory_file"),
                    "status": route.get("status", ""),
                }
            ]
        for variant in variants:
            trajectory_file = Path(variant.get("trajectory_file", ""))
            if not trajectory_file.exists():
                continue
            files.append(
                {
                    "plan": plan_name,
                    "route_index": route.get("route_index"),
                    "start_node": route.get("start_node"),
                    "end_node": route.get("end_node"),
                    "variant_id": variant.get("variant_id", "full"),
                    "variant_label": variant.get("variant_label", "N Agent"),
                    "computed_agents": variant.get("computed_agents", route.get("computed_agents", 0)),
                    "status": variant.get("status", route.get("status", "")),
                    "trajectory_file": trajectory_file,
                }
            )

    if files:
        return files

    dataswarm_dir = SCENARIO_ROOT / "dataswarm" / plan_name
    if not dataswarm_dir.exists():
        return []
    fallback_files = []
    for trajectory_file in sorted(dataswarm_dir.glob("*.sqlite")):
        stem = trajectory_file.stem
        variant_id = "full"
        for candidate in ["full", "half", "single"]:
            if stem.endswith(f"_{candidate}"):
                variant_id = candidate
                break
        fallback_files.append(
            {
                "plan": plan_name,
                "route_index": None,
                "start_node": "",
                "end_node": "",
                "variant_id": variant_id,
                "variant_label": variant_id,
                "computed_agents": 0,
                "status": "",
                "trajectory_file": trajectory_file,
            }
        )
    return fallback_files


def summarize_trajectory_time(item):
    trajectory_file = item["trajectory_file"]
    con = sqlite3.connect(trajectory_file)
    cur = con.cursor()
    fps = sqlite_fps(cur)
    rows = cur.execute(
        """
        select id, min(frame), max(frame), count(*)
        from trajectory_data
        group by id
        order by id
        """
    ).fetchall()
    max_frame_row = cur.execute("select max(frame) from trajectory_data").fetchone()
    con.close()

    max_frame = int(max_frame_row[0] or 0)
    simulation_duration_s = max_frame / fps if fps > 0 else 0.0
    agent_rows = []
    for agent_id, start_frame, end_frame, total_frames in rows:
        travel_time_s = (int(end_frame) - int(start_frame)) / fps if fps > 0 else 0.0
        agent_rows.append(
            {
                "plan": item["plan"],
                "route_index": item["route_index"],
                "start_node": item["start_node"],
                "end_node": item["end_node"],
                "variant_id": item["variant_id"],
                "variant_label": item["variant_label"],
                "trajectory_file": str(trajectory_file),
                "agent_id": int(agent_id),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "total_frames": int(total_frames),
                "fps": fps,
                "travel_time_s": travel_time_s,
            }
        )

    times = [row["travel_time_s"] for row in agent_rows]
    route_summary = {
        "plan": item["plan"],
        "route_index": item["route_index"],
        "start_node": item["start_node"],
        "end_node": item["end_node"],
        "variant_id": item["variant_id"],
        "variant_label": item["variant_label"],
        "status": item["status"],
        "computed_agents": item["computed_agents"],
        "fps": fps,
        "max_frame": max_frame,
        "simulation_duration_s": simulation_duration_s,
        "mean_agent_time_s": sum(times) / len(times) if times else 0.0,
        "min_agent_time_s": min(times) if times else 0.0,
        "max_agent_time_s": max(times) if times else 0.0,
        "trajectory_file": str(trajectory_file),
    }
    return agent_rows, route_summary


def build_time_summary(plan_names):
    output_root = SCENARIO_ROOT / "time_summary"
    output_root.mkdir(parents=True, exist_ok=True)
    all_agent_rows = []
    all_route_rows = []
    skipped = []

    for plan_name in plan_names:
        plan_items = collect_trajectory_files(plan_name)
        plan_agent_rows = []
        plan_route_rows = []
        for item in plan_items:
            try:
                agent_rows, route_summary = summarize_trajectory_time(item)
            except Exception as exc:
                skipped.append({"plan": plan_name, "trajectory_file": str(item["trajectory_file"]), "error": str(exc)})
                continue
            plan_agent_rows.extend(agent_rows)
            plan_route_rows.append(route_summary)

        if plan_agent_rows or plan_route_rows:
            plan_output = output_root / plan_name
            plan_output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(plan_agent_rows).to_csv(plan_output / "agent_times.csv", index=False)
            pd.DataFrame(plan_route_rows).to_csv(plan_output / "route_time_summary.csv", index=False)
            write_json(
                plan_output / "time_summary.json",
                {
                    "plan": plan_name,
                    "trajectory_files": len(plan_items),
                    "agent_rows": len(plan_agent_rows),
                    "route_rows": len(plan_route_rows),
                    "mean_agent_time_s": (
                        sum(row["travel_time_s"] for row in plan_agent_rows) / len(plan_agent_rows)
                        if plan_agent_rows
                        else 0.0
                    ),
                    "max_simulation_duration_s": (
                        max(row["simulation_duration_s"] for row in plan_route_rows)
                        if plan_route_rows
                        else 0.0
                    ),
                },
            )
            all_agent_rows.extend(plan_agent_rows)
            all_route_rows.extend(plan_route_rows)

    if all_agent_rows:
        pd.DataFrame(all_agent_rows).to_csv(output_root / "all_agent_times.csv", index=False)
    if all_route_rows:
        pd.DataFrame(all_route_rows).to_csv(output_root / "all_route_time_summary.csv", index=False)
    sorted_agent_rows = sorted(all_agent_rows, key=lambda row: row["travel_time_s"])
    min_agent_row = sorted_agent_rows[0] if sorted_agent_rows else {}

    def percentile(values, pct):
        if not values:
            return 0.0
        values = sorted(values)
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * pct)))
        return values[idx]

    agent_times = [row["travel_time_s"] for row in all_agent_rows]
    write_json(
        output_root / "time_summary_manifest.json",
        {
            "plans_requested": len(plan_names),
            "plans_with_time_summary": len({row["plan"] for row in all_route_rows}),
            "trajectory_runs": len(all_route_rows),
            "agent_rows": len(all_agent_rows),
            "min_agent_time_s": (
                min(row["travel_time_s"] for row in all_agent_rows)
                if all_agent_rows
                else 0.0
            ),
            "p05_agent_time_s": percentile(agent_times, 0.05),
            "p50_agent_time_s": percentile(agent_times, 0.50),
            "p95_agent_time_s": percentile(agent_times, 0.95),
            "mean_agent_time_s": (
                sum(row["travel_time_s"] for row in all_agent_rows) / len(all_agent_rows)
                if all_agent_rows
                else 0.0
            ),
            "max_agent_time_s": (
                max(row["travel_time_s"] for row in all_agent_rows)
                if all_agent_rows
                else 0.0
            ),
            "max_simulation_duration_s": (
                max(row["simulation_duration_s"] for row in all_route_rows)
                if all_route_rows
                else 0.0
            ),
            "min_agent_time_source": {
                "plan": min_agent_row.get("plan", ""),
                "route_index": min_agent_row.get("route_index", ""),
                "variant_id": min_agent_row.get("variant_id", ""),
                "agent_id": min_agent_row.get("agent_id", ""),
                "travel_time_s": min_agent_row.get("travel_time_s", 0.0),
                "trajectory_file": min_agent_row.get("trajectory_file", ""),
            },
            "skipped": skipped,
        },
    )
    return output_root, all_agent_rows, all_route_rows, skipped


def readable_route_time_table(route_rows):
    rows = []
    for row in route_rows:
        rows.append(
            {
                "Plan": row.get("plan", ""),
                "Route": row.get("route_index", ""),
                "From": row.get("start_node", ""),
                "To": row.get("end_node", ""),
                "Scenario": row.get("variant_label", row.get("variant_id", "")),
                "Agents expected": row.get("computed_agents", 0),
                "Simulation time (s)": round(safe_float(row.get("simulation_duration_s")), 2),
                "Fastest agent (s)": round(safe_float(row.get("min_agent_time_s")), 2),
                "Average agent (s)": round(safe_float(row.get("mean_agent_time_s")), 2),
                "Slowest agent (s)": round(safe_float(row.get("max_agent_time_s")), 2),
                "Status": row.get("status", ""),
            }
        )
    return pd.DataFrame(rows)


def readable_agent_time_table(agent_rows):
    rows = []
    for row in agent_rows:
        rows.append(
            {
                "Plan": row.get("plan", ""),
                "Route": row.get("route_index", ""),
                "Scenario": row.get("variant_label", row.get("variant_id", "")),
                "Agent ID": row.get("agent_id", ""),
                "Travel time (s)": round(safe_float(row.get("travel_time_s")), 2),
                "Total frames": row.get("total_frames", row.get("observed_frames", "")),
                "Start frame": row.get("start_frame", ""),
                "End frame": row.get("end_frame", ""),
            }
        )
    return pd.DataFrame(rows)


def readable_manifest(manifest):
    labels = {
        "plans_requested": "Plans requested",
        "plans_with_time_summary": "Plans summarized",
        "trajectory_runs": "Trajectory files summarized",
        "agent_rows": "Agent records",
        "min_agent_time_s": "Fastest agent overall (s)",
        "p05_agent_time_s": "Very fast group, P05 (s)",
        "p50_agent_time_s": "Median agent time, P50 (s)",
        "p95_agent_time_s": "Slow group, P95 (s)",
        "mean_agent_time_s": "Average agent time (s)",
        "max_agent_time_s": "Slowest agent overall (s)",
        "max_simulation_duration_s": "Longest simulation duration (s)",
    }
    rows = []
    for key, label in labels.items():
        value = manifest.get(key, 0)
        if isinstance(value, float):
            value = round(value, 2)
        rows.append({"Meaning": label, "Value": value})
    return pd.DataFrame(rows)


def load_route_information_builder():
    spec = importlib.util.spec_from_file_location("housegan_route_information", ROUTE_INFO_GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_route_information


def readable_route_information_manifest(manifest):
    labels = {
        "plans_requested": "Plans requested by latest scope",
        "plans_with_route_information": "Plans with route_information in latest scope",
        "route_rows": "Route rows",
        "output_root": "Output root",
    }
    rows = []
    for key, label in labels.items():
        rows.append({"Meaning": label, "Value": manifest.get(key, "")})
    return pd.DataFrame(rows)


def list_time_summary_plans():
    path = SCENARIO_ROOT / "time_summary" / "all_route_time_summary.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, usecols=["plan"])
    except Exception:
        return []
    return sorted(df["plan"].dropna().astype(str).unique().tolist())


def route_information_plan_names(output_root):
    if not output_root.exists():
        return []
    return sorted(
        path.name
        for path in output_root.iterdir()
        if path.is_dir() and (path / "topology_shortest_distance.json").exists()
    )


def route_json_sample_rows(routes, limit=8):
    sorted_routes = sorted(
        routes,
        key=lambda route: (
            safe_float(route.get("bottleneck_score")),
            safe_float(route.get("topology_hop_distance")),
            safe_float(route.get("topology_shortest_distance_m")),
        ),
        reverse=True,
    )
    rows = []
    for route in sorted_routes[:limit]:
        rows.append(
            {
                "A": route.get("start_node", ""),
                "B": route.get("end_node", ""),
                "path": " -> ".join(str(node) for node in route.get("path", [])),
                "topology_distance_m": route.get("topology_shortest_distance_m", 0.0),
                "hop": route.get("topology_hop_distance", 0),
                "bottleneck_score": route.get("bottleneck_score", 0.0),
            }
        )
    return rows


def route_card_badge(score):
    if score >= 0.85:
        return "Critical", "route-badge route-badge-red"
    if score >= 0.55:
        return "Medium", "route-badge route-badge-amber"
    return "Low", "route-badge route-badge-green"


def render_route_cards(routes):
    rows = route_json_sample_rows(routes, limit=6)
    if not rows:
        st.caption("No route samples in this file.")
        return

    for row_start in range(0, len(rows), 3):
        cols = st.columns(3)
        for index, (col, route) in enumerate(zip(cols, rows[row_start : row_start + 3]), start=row_start + 1):
            score = safe_float(route["bottleneck_score"])
            badge, _ = route_card_badge(score)
            with col:
                with st.container(border=True):
                    top_a, top_b = st.columns([1, 1])
                    top_a.markdown(f"**Route {index:02d}**")
                    top_b.markdown(f"`{badge}`")
                    st.markdown(f"### {route['A']} -> {route['B']}")
                    st.code(route["path"], language="text")
                    stat_a, stat_b, stat_c = st.columns(3)
                    stat_a.metric("Topo dist", f"{safe_float(route['topology_distance_m']):.2f} m")
                    stat_b.metric("Hop", int(route["hop"]))
                    stat_c.metric("Bottleneck", f"{score:.2f}")


def render_topology_shortest_distance_sample(output_root, preferred_plan):
    plan_names = route_information_plan_names(output_root)
    if not plan_names:
        st.info("No topology_shortest_distance.json files found yet. Generate route_information first.")
        return

    default_index = plan_names.index(preferred_plan) if preferred_plan in plan_names else len(plan_names) - 1
    sample_plan = st.selectbox("Sample plan", plan_names, index=default_index, key="route_json_sample_plan")
    payload_path = output_root / sample_plan / "topology_shortest_distance.json"
    payload = read_json(payload_path, {}) or {}
    routes = payload.get("routes", [])
    edge_bottlenecks = payload.get("edge_bottlenecks", [])

    st.subheader("Sample topology_shortest_distance.json")
    st.caption(f"Source: `{payload_path}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plan", sample_plan)
    c2.metric("Routes", payload.get("route_count", len(routes)))
    c3.metric("Edge bottlenecks", len(edge_bottlenecks))
    c4.metric("Top score", f"{max((safe_float(r.get('bottleneck_score')) for r in routes), default=0.0):.2f}")

    route_tab, edge_tab, raw_tab = st.tabs(["Route cards", "Edge bottlenecks", "Raw JSON"])
    with route_tab:
        render_route_cards(routes)
        rows = route_json_sample_rows(routes, limit=12)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with edge_tab:
        if edge_bottlenecks:
            st.dataframe(pd.DataFrame(edge_bottlenecks).head(20), use_container_width=True)
        else:
            st.caption("No edge bottleneck data in this sample.")
    with raw_tab:
        st.json(payload)


def render_time_summary_utility():
    st.subheader("Generate time_summary")

    plans = list_plans()
    simulated_plans = [plan for plan in plans if (SCENARIO_ROOT / "dataswarm" / plan).exists()]
    if not simulated_plans:
        st.warning("No simulation trajectory files found yet.")
        return

    mode = st.radio("Scope", ["Selected plan", "All simulated plans"], horizontal=True)
    selected_plan = st.selectbox("Plan", simulated_plans, index=len(simulated_plans) - 1)
    target_plans = simulated_plans if mode == "All simulated plans" else [selected_plan]
    st.caption(f"Output: `{SCENARIO_ROOT / 'time_summary'}`")
    st.caption("คำอธิบาย: Fastest = agent ที่ถึงเร็วสุด, Median/P50 = ค่ากลางที่ควรดูเป็นหลัก, Slowest = agent ที่ช้าที่สุด")

    if st.button("Generate time_summary", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        output_root, agent_rows, route_rows, skipped = build_time_summary(target_plans)
        progress.progress(1.0)
        status.success(f"Created time summaries for {len(set(row['plan'] for row in route_rows))} plan(s).")
        agent_times = [row["travel_time_s"] for row in agent_rows]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trajectory runs", len(route_rows))
        c2.metric("Agent rows", len(agent_rows))
        c3.metric("Min agent time", f"{min((row['travel_time_s'] for row in agent_rows), default=0.0):.2f}s")
        c4.metric("Median agent time", f"{pd.Series(agent_times).median() if agent_times else 0.0:.2f}s")
        c5.metric("Skipped files", len(skipped))
        st.write(str(output_root))
        if route_rows:
            st.subheader("Route Time Summary")
            st.dataframe(readable_route_time_table(route_rows).head(200), use_container_width=True)
        if agent_rows:
            with st.expander("Agent-level travel times"):
                st.dataframe(readable_agent_time_table(agent_rows).head(500), use_container_width=True)
        if skipped:
            st.warning("Some sqlite files were skipped.")
            st.dataframe(pd.DataFrame(skipped), use_container_width=True)

    manifest = read_json(SCENARIO_ROOT / "time_summary" / "time_summary_manifest.json", {})
    if manifest:
        st.subheader("Latest Time Summary Manifest")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Plans", manifest.get("plans_with_time_summary", 0))
        c2.metric("Trajectory runs", manifest.get("trajectory_runs", 0))
        c3.metric("Agent rows", manifest.get("agent_rows", 0))
        c4.metric("Min agent time", f"{manifest.get('min_agent_time_s', 0.0):.2f}s")
        c5.metric("Median agent time", f"{manifest.get('p50_agent_time_s', 0.0):.2f}s")
        if manifest.get("min_agent_time_source"):
            with st.expander("Min time source"):
                source = manifest.get("min_agent_time_source", {})
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Plan": source.get("plan", ""),
                                "Route": source.get("route_index", ""),
                                "Scenario": source.get("variant_id", ""),
                                "Agent ID": source.get("agent_id", ""),
                                "Travel time (s)": round(safe_float(source.get("travel_time_s")), 2),
                                "Trajectory file": source.get("trajectory_file", ""),
                            }
                        ]
                    ),
                    use_container_width=True,
                )
        st.subheader("Readable Manifest")
        st.dataframe(readable_manifest(manifest), use_container_width=True)
        with st.expander("Raw manifest JSON"):
            st.json(manifest)


def render_route_information_utility():
    st.subheader("Generate route_information")

    plans = list_plans()
    if not plans:
        st.warning("No generated geometry plans found yet.")
        return

    labelled_plans = list_time_summary_plans()
    simulated_plans = [plan for plan in plans if plan_has_simulation(plan)]
    scope_options = [
        "All plans with time_summary labels",
        "Selected plan with time_summary labels",
        "All generated geometry plans (no time labels)",
    ]
    mode = st.selectbox("Scope", scope_options, key="route_info_scope")
    if mode == "All generated geometry plans (no time labels)":
        plan_options = plans
        target_plans = plans
    else:
        plan_options = labelled_plans
        target_plans = labelled_plans
    if not plan_options:
        st.warning("No time_summary labels found yet. Run Generate time_summary before generating route info for training.")
        return
    selected_plan = st.selectbox("Plan", plan_options, index=len(plan_options) - 1, key="route_info_plan")
    if mode == "Selected plan with time_summary labels":
        target_plans = [selected_plan]
    output_root = SCENARIO_ROOT / "route_information"
    st.caption(f"Output: `{output_root}`")
    st.caption(
        "Default scope uses plans that already exist in all_route_time_summary.csv, because AI_Estimate training needs route features matched with time labels."
    )
    st.caption(f"Available: `{len(labelled_plans)}` labelled plans, `{len(simulated_plans)}` simulated plans, `{len(plans)}` generated geometry plans.")
    st.caption(
        "bottleneck_score = max normalized edge load along the route, where edge load comes from all room-to-room shortest paths in this plan."
    )

    if st.button("Generate route_information", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        builder = load_route_information_builder()
        clean_output = mode != "Selected plan with time_summary labels"
        output_root, rows, skipped = builder(SCENARIO_ROOT, target_plans, clean_output=clean_output)
        progress.progress(1.0)
        status.success(f"Created route information for {len(set(row['plan'] for row in rows))} plan(s).")
        c1, c2, c3 = st.columns(3)
        c1.metric("Plans in current scope", len(set(row["plan"] for row in rows)))
        c2.metric("Route rows", len(rows))
        c3.metric("Skipped plans", len(skipped))
        st.write(str(output_root))
        if rows:
            st.subheader("route_information preview")
            st.dataframe(pd.DataFrame(rows).head(300), use_container_width=True)
        if skipped:
            st.warning("Some plans were skipped.")
            st.dataframe(pd.DataFrame(skipped), use_container_width=True)

    manifest = read_json(output_root / "route_information_manifest.json", {})
    if manifest:
        st.subheader("Latest route_information manifest")
        c1, c2, c3 = st.columns(3)
        c1.metric("Plans in latest scope", manifest.get("plans_with_route_information", 0))
        c2.metric("Route rows", manifest.get("route_rows", 0))
        c3.metric("Skipped plans", len(manifest.get("skipped", [])))
        st.dataframe(readable_route_information_manifest(manifest), use_container_width=True)
        with st.expander("Raw manifest JSON"):
            st.json(manifest)

    st.divider()
    render_topology_shortest_distance_sample(output_root, selected_plan)


def page_utilities():
    st.header("Utilities")
    time_tab, route_tab = st.tabs(["Generate time_summary", "Generate route_information"])
    with time_tab:
        render_time_summary_utility()
    with route_tab:
        render_route_information_utility()


def page_generate():
    st.header("Generate HouseGAN Plans")
    cfg = read_json(GEN_CONFIG, {}) or {}
    complexity_options = ["Small (3-5 Rooms)", "Medium (5-8 Rooms)", "Large (8-15 Rooms)", "XL (15-20 Rooms)", "XXL (20-30 Rooms)"]
    complexity_default = cfg.get("complexity", "Large (8-15 Rooms)")
    complexity_index = complexity_options.index(complexity_default) if complexity_default in complexity_options else 2

    def render_area_tab(tab_mode, key_prefix):
        tab_cfg = dict(cfg)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            tab_cfg["num_scenarios"] = st.number_input(
                "Number of plans",
                min_value=1,
                max_value=1000,
                value=int(cfg.get("num_scenarios", 20)),
                key=f"{key_prefix}_num_scenarios",
            )
            tab_cfg["random_seed"] = st.number_input(
                "Start seed",
                min_value=0,
                value=int(cfg.get("random_seed", 42)),
                key=f"{key_prefix}_random_seed",
            )
        with col_b:
            tab_cfg["num_corridors"] = st.number_input(
                "Max corridors",
                min_value=1,
                max_value=20,
                value=int(cfg.get("num_corridors", 3)),
                key=f"{key_prefix}_num_corridors",
            )
            tab_cfg["door_width"] = st.number_input(
                "Door width (m)",
                min_value=0.4,
                max_value=5.0,
                value=float(cfg.get("door_width", 1.5)),
                step=0.1,
                key=f"{key_prefix}_door_width",
            )
        with col_c:
            tab_cfg["complexity"] = st.selectbox(
                "Complexity",
                complexity_options,
                index=complexity_index,
                key=f"{key_prefix}_complexity",
            )
            tab_cfg["output_scenario"] = "Topo_HouseGAN"

        if tab_mode == "big":
            tab_cfg["room_area_mode"] = "big"
            st.info(
                "Big room area mode: room size is fixed to 10.00-25.00 m and corridor profile is scaled "
                "to length 10.00-22.00 m (attached) / 12.00-25.00 m (first) with width 3.00-6.00 m."
            )
        else:
            tab_cfg["room_area_mode"] = "default"
            st.caption("Default room area mode uses room size 2.50-7.00 m and original corridor profile.")

        return tab_cfg

    default_tab, big_tab = st.tabs(["Default room area", "Big room area"])
    with default_tab:
        cfg_default = render_area_tab("default", "default_area")
        trigger_default = st.button("Generate Plans", type="primary", disabled=ensure_manager("generate_manager").is_running, key="generate_default_area")
    with big_tab:
        cfg_big = render_area_tab("big", "big_area")
        trigger_big = st.button("Generate Plans", type="primary", disabled=ensure_manager("generate_manager").is_running, key="generate_big_area")

    st.caption(f"Output: `{SCENARIO_ROOT / 'geo'}`")
    manager = ensure_manager("generate_manager")
    if trigger_default:
        write_json(GEN_CONFIG, cfg_default)
        st.session_state["generate_manager_logs"] = []
        command = [sys.executable, str(MODULE_ROOT / "Prepare_data" / "generate_layout.py"), "--config", str(GEN_CONFIG)]
        manager.start_process(command, str(PROJECT_ROOT))
        st.rerun()

    if trigger_big:
        write_json(GEN_CONFIG, cfg_big)
        st.session_state["generate_manager_logs"] = []
        command = [sys.executable, str(MODULE_ROOT / "Prepare_data" / "generate_layout.py"), "--config", str(GEN_CONFIG)]
        manager.start_process(command, str(PROJECT_ROOT))
        st.rerun()

    if st.button("Stop Generate", disabled=not manager.is_running):
        manager.stop_process()
        st.rerun()
    show_process_output("generate_manager")

    latest = list_plans()
    st.subheader(f"All Plans ({len(latest)} plans)")
    image_gallery([SCENARIO_ROOT / "geo" / p / "preview.png" for p in latest], columns=6)


def page_simulate():
    st.header("Density Simulation")
    plans = list_plans()
    if not plans:
        st.warning("No plans found yet. Generate plans first.")
        return

    cfg = read_json(SIM_CONFIG, {})
    spawn_policy = cfg.setdefault("spawn_policy", {})
    selected_plan = st.selectbox("Plan", plans, index=len(plans) - 1)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        cfg["seed"] = st.number_input("Seed", min_value=0, value=int(cfg.get("seed", 42)))
        cfg["timeout_minutes"] = st.number_input("Timeout / route (minutes)", min_value=1, max_value=120, value=int(cfg.get("timeout_minutes", 5)))
    with col_b:
        spawn_policy["sqm_per_person"] = st.number_input("Density: m² per person", min_value=0.25, max_value=20.0, value=float(spawn_policy.get("sqm_per_person", 2.0)), step=0.25)
        spawn_policy["wall_offset_m"] = st.number_input("Wall offset (m)", min_value=0.0, max_value=5.0, value=float(spawn_policy.get("wall_offset_m", 1.0)), step=0.1)
    with col_c:
        spawn_policy["min_agents"] = st.number_input("Min agents", min_value=1, max_value=500, value=int(spawn_policy.get("min_agents", 1)))
        spawn_policy["max_agents"] = st.number_input("Max agents", min_value=1, max_value=2000, value=int(spawn_policy.get("max_agents", 250)))

    cfg["scenario_name"] = "Topo_HouseGAN"
    cfg["plan"] = selected_plan

    st.info("Agent count is computed from the offset spawn area: `floor(safe_spawn_area / m²_per_person)`, then clamped by min/max.")
    manager = ensure_manager("sim_manager")
    col_run, col_batch, col_stop = st.columns(3)
    with col_run:
        if st.button("Run Selected Plan", type="primary", disabled=manager.is_running):
            write_json(SIM_CONFIG, cfg)
            st.session_state["sim_manager_logs"] = []
            command = [
                sys.executable,
                str(MODULE_ROOT / "Simulation" / "density_housegan_sim.py"),
                "--config",
                str(SIM_CONFIG),
                "--plan",
                selected_plan,
            ]
            manager.start_process(command, str(PROJECT_ROOT))
            st.rerun()
    with col_batch:
        if st.button("Run All Unsimulated", disabled=manager.is_running):
            write_json(SIM_CONFIG, cfg)
            st.session_state["sim_manager_logs"] = []
            command = [sys.executable, str(MODULE_ROOT / "Simulation" / "density_housegan_sim.py"), "--config", str(SIM_CONFIG), "--batch"]
            manager.start_process(command, str(PROJECT_ROOT))
            st.rerun()
    with col_stop:
        if st.button("Stop Simulation", disabled=not manager.is_running):
            manager.stop_process()
            st.rerun()

    show_process_output("sim_manager")

    st.subheader("Selected Plan Preview")
    image_gallery([SCENARIO_ROOT / "geo" / selected_plan / "preview.png", SCENARIO_ROOT / "geo" / selected_plan / "preview_graph.png"], columns=2)


def page_results():
    st.header("View Results")
    plans = list_plans()
    if not plans:
        st.warning("No plans found yet.")
        return

    selected_plan = st.selectbox("Plan", plans, index=len(plans) - 1)
    meta_path = SCENARIO_ROOT / "metadata" / selected_plan / "simulation_summary.json"
    summary = read_json(meta_path, {})

    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Routes", summary.get("route_count", 0))
        c2.metric("Walkable area", f"{summary.get('walkable_area_m2', 0):.2f} m²")
        success_count = sum(
            1
            for route in summary.get("routes", [])
            for variant in route.get("variants", [{"status": route.get("status")}])
            if variant.get("status") == "success"
        )
        c3.metric("Successful routes", success_count)

        rows = [
            {
                "route": r.get("route_index"),
                "from": r.get("start_node"),
                "to": r.get("end_node"),
                "agents": r.get("computed_agents"),
                "raw_area_m2": r.get("raw_spawn_area_m2"),
                "clipped_area_m2": r.get("clipped_spawn_area_m2"),
                "safe_area_m2": r.get("safe_spawn_area_m2"),
                "area_left_percent": (
                    round((safe_float(r.get("safe_spawn_area_m2")) / safe_float(r.get("raw_spawn_area_m2"))) * 100, 2)
                    if safe_float(r.get("raw_spawn_area_m2")) > 0
                    else 0.0
                ),
                "offset_used_m": r.get("wall_offset_used_m"),
                "status": r.get("status"),
                "error": r.get("error", ""),
            }
            for r in summary.get("routes", [])
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No simulation metadata yet for this plan.")

    selected_variant_label = st.radio(
        "Simulation result set",
        [label for _, label in RESULT_VARIANTS],
        horizontal=True,
        key=f"result_variant_{selected_plan}",
    )
    selected_variant_id = next(variant_id for variant_id, label in RESULT_VARIANTS if label == selected_variant_label)

    variant_rows = []
    for route in summary.get("routes", []):
        variant = route_variant(route, selected_variant_id)
        if not variant:
            continue
        variant_rows.append(
            {
                "route": route.get("route_index"),
                "from": route.get("start_node"),
                "to": route.get("end_node"),
                "variant": variant.get("variant_label", selected_variant_id),
                "agents": variant.get("computed_agents", route.get("computed_agents", 0)),
                "distributed": variant.get("agent_count_distributed", 0),
                "status": variant.get("status", ""),
                "error": variant.get("error", ""),
            }
        )
    if variant_rows:
        st.dataframe(pd.DataFrame(variant_rows), use_container_width=True)
    elif summary:
        st.warning("ยังไม่มีผลลัพธ์ของ variant นี้ ให้รัน Density Simulation ใหม่เพื่อสร้าง full / half / single agent outputs.")

    result_view = st.radio(
        "Output type",
        ["Plan", "Offset Area", "Spawn Policy", "Trajectories", "Density", "Speed", "Metadata"],
        horizontal=True,
        key=f"result_view_{selected_plan}_{selected_variant_id}",
    )

    if result_view == "Plan":
        image_gallery([SCENARIO_ROOT / "geo" / selected_plan / "preview.png", SCENARIO_ROOT / "geo" / selected_plan / "preview_graph.png"], columns=2)
    elif result_view == "Offset Area":
        offset_rows = []
        for r in summary.get("routes", []):
            variant = route_variant(r, selected_variant_id)
            if not variant:
                continue
            offset_rows.append(
                {
                    "route": r.get("route_index"),
                    "path": " -> ".join(str(n) for n in r.get("topological_path", [])),
                    "raw_area_m2": r.get("raw_spawn_area_m2", 0.0),
                    "clipped_area_m2": r.get("clipped_spawn_area_m2", 0.0),
                    "offset_area_m2": r.get("safe_spawn_area_m2", 0.0),
                    "offset_requested_m": r.get("wall_offset_requested_m", 0.0),
                    "offset_used_m": r.get("wall_offset_used_m", 0.0),
                    "area_left_percent": (
                        round((safe_float(r.get("safe_spawn_area_m2")) / safe_float(r.get("raw_spawn_area_m2"))) * 100, 2)
                        if safe_float(r.get("raw_spawn_area_m2")) > 0
                        else 0.0
                    ),
                    "agents": variant.get("computed_agents", r.get("computed_agents", 0)),
                    "status": variant.get("status", ""),
                }
            )
        if offset_rows:
            st.dataframe(pd.DataFrame(offset_rows), use_container_width=True)
        offset_images = variant_image_paths("offset_area", selected_plan, selected_variant_id)
        if not offset_images:
            st.info("ยังไม่มีรูป Offset Area สำหรับ plan นี้ ให้รัน Density Simulation ใหม่ 1 รอบเพื่อสร้างภาพ offset_area.")
        image_gallery(offset_images, columns=2)
    elif result_view == "Spawn Policy":
        image_gallery(variant_image_paths("spawn_exit", selected_plan, selected_variant_id), columns=2)
    elif result_view == "Trajectories":
        image_gallery(variant_image_paths("trajectory_line", selected_plan, selected_variant_id), columns=2)
    elif result_view == "Density":
        image_gallery(variant_image_paths("heatmap_density", selected_plan, selected_variant_id), columns=2)
    elif result_view == "Speed":
        image_gallery(variant_image_paths("heatmap_speed", selected_plan, selected_variant_id), columns=2)
    else:
        if summary:
            st.json(summary)
        else:
            st.caption(f"Missing `{meta_path}`")


def page_preview_generate_result():
    st.header("Preview Result Generate Plan")
    plans = list_plans()
    if not plans:
        st.warning("No generated plans found yet.")
        return

    selected_plan = st.selectbox("Generated plan", plans, index=len(plans) - 1)
    plan_dir = SCENARIO_ROOT / "geo" / selected_plan

    meta = read_json(plan_dir / "metadata.json", {})
    room_count, corridor_count, door_count = plan_geometry_counts(plan_dir)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rooms", room_count)
    c2.metric("Corridors", corridor_count)
    c3.metric("Doors", door_count)

    tabs = st.tabs(["Plan Preview", "Legacy Walkable Preview"])
    with tabs[0]:
        image_gallery([plan_dir / "preview.png", plan_dir / "preview_graph.png"], columns=2)
    with tabs[1]:
        st.caption("This tab reuses the original Streamlit `preview_walkable_area()` implementation.")
        legacy_preview_walkable_area(plan_dir)

    with st.expander("Geometry files"):
        st.write(str(plan_dir))
        st.json(
            {
                "geo_room": (plan_dir / "geo_room.json").exists(),
                "geo_corridor": (plan_dir / "geo_corridor.json").exists(),
                "geo_door": (plan_dir / "geo_door.json").exists(),
                "metadata": (plan_dir / "metadata.json").exists(),
            }
        )


st.sidebar.title("GeneratePlan HouseGAN")
page = st.sidebar.radio("Page", ["Dashboard", "Generate Plans", "Preview Result Generate Plan", "Density Simulation", "View Results", "Utilities"])
st.sidebar.caption("Outputs are isolated under `Geo_scenario/Topo_HouseGAN`.")

if page == "Dashboard":
    page_dashboard()
elif page == "Generate Plans":
    page_generate()
elif page == "Preview Result Generate Plan":
    page_preview_generate_result()
elif page == "Density Simulation":
    page_simulate()
elif page == "Utilities":
    page_utilities()
else:
    page_results()
