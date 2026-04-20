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
        cols = [c for c in df.columns if any(w in c.lower() for w in ['loss', 'mae', 'l1', 'g_adv', 'fm', 'm'])]
        if not cols:
            cols = [c for c in df.columns if c != 'epoch']
            
        # Input fields for custom labels
        col_x, col_y = st.columns(2)
        x_label = col_x.text_input("X-Axis Label", value="Epoch")
        y_label = col_y.text_input("Y-Axis Label", value="Loss / Metric Value")
        
        selected_cols = st.multiselect("Select Metrics to Plot", options=cols, default=cols)
        
        if selected_cols:
            st.line_chart(
                df.set_index('epoch')[selected_cols], 
                x_label=x_label, 
                y_label=y_label
            )
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
    Displays the test evaluation metrics and a compact comparison grid for Input, AI Prediction, and Ground Truth.
    """
    run_path = pathlib.Path(run_dir)
    # Search for summary in run root or test_results
    score_path = run_path / "test_evaluation_summary.csv"
    txt_score_path = run_path / "evaluation.txt"
    
    if not score_path.exists(): score_path = run_path / "test_results" / "test_evaluation_summary.csv"
    
    test_results_dir = run_path / "test_results"

    st.markdown("### 🏁 Final Test Evaluation")

    # 1. Dashboard Metrics Summary
    if score_path.exists():
        try:
            df_score = pd.read_csv(score_path)
            # Find common error metrics (Added ADE/FDE)
            m_data = df_score[df_score['metric'].str.contains('MAE|MSE|RMSE|L1|mse|mae|ade|fde', case=False, na=False)]
            if not m_data.empty:
                # Use st.columns for metric cards
                m_cols = st.columns(len(m_data))
                for idx, row in enumerate(m_data.itertuples()):
                    m_cols[idx].metric(label=row.metric.upper(), value=f"{float(row.value):.4f}")
            else:
                st.dataframe(df_score)
        except Exception as e:
            st.error(f"Error loading score file: {e}")
    elif txt_score_path.exists():
        try:
            with open(txt_score_path, "r") as f:
                content = f.read()
            st.code(content, language="markdown")
            # Proactively try to parse some metrics for the cards if possible
            import re
            metrics = re.findall(r"(ADE|FDE|Average Displacement Error|Final Displacement Error):\s+([\d\.]+)", content, re.IGNORECASE)
            if metrics:
                m_cols = st.columns(len(metrics))
                for idx, (m_name, m_val) in enumerate(metrics):
                    m_cols[idx].metric(label=m_name.upper(), value=m_val)
        except: pass

    
    st.divider()

    # 2. Comparison Grid (Inputs vs Predictions vs Targets - Pix2Pix Style)
    pred_dir = test_results_dir / "predictions"
    input_dir = test_results_dir / "inputs"
    target_dir = test_results_dir / "targets"

    # 3. Trajectory Plot (Recursive Generation - LSTM style)
    traj_dir = test_results_dir / "trajectories"

    # --- Display Logic ---
    if pred_dir.exists() and input_dir.exists() and target_dir.exists():
        st.subheader("🖼 Result Comparison")
        pred_images = sorted(list(pred_dir.glob("*.png")), key=lambda x: x.name)
        if pred_images:
            # ... (Rest of existing image grid logic)
            max_samples = 50 
            samples_to_show = pred_images[:max_samples]
            with st.container():
                h_col1, h_col2, h_col3 = st.columns(3)
                h_col1.caption("⬅️ INPUT (Scenario)")
                h_col2.caption("🤖 PREDICTION (AI)")
                h_col3.caption("🎯 TARGET (Reality)")
                for p_path in samples_to_show:
                    stem = p_path.name.split('_')[-1]
                    i_path = input_dir / f"input_{stem}"
                    t_path = target_dir / f"target_{stem}"
                    if i_path.exists() and t_path.exists():
                        r_col1, r_col2, r_col3 = st.columns(3)
                        r_col1.image(str(i_path), use_container_width=True)
                        r_col2.image(str(p_path), use_container_width=True)
                        r_col3.image(str(t_path), use_container_width=True)
                        st.divider()
        else:
            st.info("No prediction images found in subdirectory.")

    elif traj_dir.exists():
        # Already handled below, but prevents fallback 'No samples' from appearing first
        pass
    
    # Fallback to old flat/combined logic (only if not an LSTM/Traj run)
    elif test_results_dir.exists() and not traj_dir.exists():
        images = sorted(list(test_results_dir.glob("*.png")), key=lambda x: x.name)
        if images:
            st.subheader("🖼 Result Comparison")
            for img_path in images:
                try:
                    img = Image.open(str(img_path)); w, h = img.size
                    if w >= (h * 2.5):
                        # ... (existing crop logic) ...
                        unit_w = w // 3
                        input_img = img.crop((0, 0, unit_w, h))
                        gt_img = img.crop((unit_w, 0, unit_w * 2, h))
                        ai_img = img.crop((unit_w * 2, 0, w, h))
                        r_col1, r_col2, r_col3 = st.columns(3)
                        r_col1.caption("INPUT"); r_col1.image(input_img, use_container_width=True)
                        r_col2.caption("TARGET"); r_col2.image(gt_img, use_container_width=True)
                        r_col3.caption("PREDICTION"); r_col3.image(ai_img, use_container_width=True)
                        st.divider()
                    else:
                        st.image(img, caption=img_path.name, use_container_width=True)
                except: pass
        else:
            st.info("No test evaluation samples found.")
    else:
        if not traj_dir.exists():
            st.info("Results directory or trajectories not found. Please run 'Test Evaluation' first.")

    # 3. Final display for Trajectories (Bottom section)
    if traj_dir.exists():
        st.subheader("🏁 Trajectory Result Comparison (AI predict vs Reality)")
        traj_images = sorted(list(traj_dir.glob("*.png")), key=lambda x: x.name)
        if traj_images:
            cols_per_row = 3
            for r_idx in range((len(traj_images) + cols_per_row - 1) // cols_per_row):
                st_cols = st.columns(cols_per_row)
                for c_idx in range(cols_per_row):
                    img_idx = r_idx * cols_per_row + c_idx
                    if img_idx < len(traj_images):
                        img_path = traj_images[img_idx]
                        st_cols[c_idx].image(str(img_path), caption=img_path.name, use_container_width=True)
            st.divider()
        else:
            st.info("Trajectory samples folder is empty.")

def show_housegan_results(outputs_dir):
    """
    Specifically for HouseGAN - Lists all generation runs and shows previews in a grid.
    Supports dual-preview (Graph + Layout).
    """
    path = pathlib.Path(outputs_dir)
    if not path.exists():
        st.info("Outputs directory does not exist yet.")
        return

    runs = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("plan_")], key=lambda x: x.name, reverse=True)
    if not runs:
        st.info("No generated layouts found.")
        return

    st.subheader(f"Generated Layouts ({len(runs)})")
    
    # Show in a 2-column grid (each item has dual preview)
    cols = st.columns(4)
    for idx, run_dir in enumerate(runs):
        layout_img = run_dir / "preview.png"
        graph_img = run_dir / "preview_graph.png"
        
        col_idx = idx % 4
        with cols[col_idx]:
            st.markdown(f"**Run:** `{run_dir.name}`")
            # Side-by-side sub-columns for Graph and Layout
            sub1, sub2 = st.columns(2)
            if graph_img.exists():
                sub1.image(str(graph_img), caption="Abstract Graph", use_container_width=True)
            if layout_img.exists():
                sub2.image(str(layout_img), caption="Physical Layout", use_container_width=True)
            st.divider()

def show_pedsim_arch_results(output_dir, options=None):
    """
    Shows results (Trajectory, Heatmaps) for a specific Architecture Plan simulation run.
    """
    path = pathlib.Path(output_dir)
    if not path.exists():
        return

    if options is None:
        options = ["Trajectory", "Density Heatmap", "Speed Heatmap"]

    st.subheader("📊 Simulation Results")
    
    # คำนวณจำนวนคอลัมน์ที่ต้องใช้จริงตามที่ User เลือก
    num_cols = len(options)
    if num_cols == 0:
        st.info("Please select at least one view option.")
        return
        
    cols = st.columns(num_cols)
    current_col_idx = 0
    
    if "Trajectory" in options:
        with cols[current_col_idx]:
            st.write("**🛣️ Trajectory**")
            traj_dir = path / "trajectory_line"
            if traj_dir.exists():
                images = sorted(list(traj_dir.glob("*.png")))
                for img in images:
                    st.image(str(img), caption=img.name, use_container_width=True)
                if not images: st.info("No trajectory image found.")
        current_col_idx += 1
    
    if "Density Heatmap" in options:
        with cols[current_col_idx]:
            st.write("**🔥 Density Heatmap**")
            den_dir = path / "heatmap_density"
            if den_dir.exists():
                images = sorted(list(den_dir.glob("*.png")))
                for img in images:
                    st.image(str(img), caption=img.name, use_container_width=True)
                if not images: st.info("No density heatmap found.")
        current_col_idx += 1

    if "Speed Heatmap" in options:
        with cols[current_col_idx]:
            st.write("**⚡ Speed Heatmap**")
            spd_dir = path / "heatmap_speed"
            if spd_dir.exists():
                images = sorted(list(spd_dir.glob("*.png")))
                for img in images:
                    st.image(str(img), caption=img.name, use_container_width=True)
                if not images: st.info("No speed heatmap found.")

def preview_walkable_area(plan_dir):
    """
    Simulates the union and wall carving logic to show a preview of the walkable area.
    """
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    import matplotlib.patches as patches

    plan_path = pathlib.Path(plan_dir)
    room_file = plan_path / "geo_room.json"
    corridor_file = plan_path / "geo_corridor.json"
    door_file = plan_path / "geo_door.json"

    if not (room_file.exists() and corridor_file.exists() and door_file.exists()):
        st.error("Missing geometry files for preview.")
        return

    # Load data
    with open(room_file, 'r') as f: rooms_coords = json.load(f)
    with open(corridor_file, 'r') as f: corridors_coords = json.load(f)
    with open(door_file, 'r') as f: doors_data = json.load(f)

    room_polys = [Polygon(c) for c in rooms_coords]
    corridor_polys = [Polygon(c) for c in corridors_coords]
    all_geoms = room_polys + corridor_polys
    
    # Final Refined Logic:
    footprint = unary_union(all_geoms)
    WALL_THICKNESS = 0.2
    DOOR_WIDTH = 1.5
    
    # 1. สร้างกำแพงดิบ (Raw Walls) จากขอบห้อง
    raw_walls = unary_union([poly.exterior.buffer(WALL_THICKNESS/2, join_style=2) for poly in all_geoms])
    
    # 2. ปรับปรุงการเจาะประตู (Door Cutouts)
    door_cutouts = []
    for d in doors_data:
        px, py = d["pos"]
        dw, dt = DOOR_WIDTH, WALL_THICKNESS * 4.0 # เจาะลึกเพื่อให้เห็นผลลัพธ์ชัดเจน
        if d["horizontal"]: cutout = box(px - dw/2, py - dt/2, px + dw/2, py + dt/2)
        else: cutout = box(px - dt/2, py - dw/2, px + dt/2, py + dw/2)
        door_cutouts.append(cutout)

    if door_cutouts:
        doors_poly = unary_union(door_cutouts)
        raw_walls = raw_walls.difference(doors_poly) # ตัดอุปสรรคตรงช่องประตูออก
    
    # 3. [Goal] Walkable Area : เอาตึกทั้งหมด ลบด้วยกำแพงดิบที่เหลืออยู่
    walkable_area = footprint.difference(raw_walls)
    
    # 4. [New Logic] Obstacles : เอาตึกทั้งหมด (Footprint) ลบด้วย Walkable Area 
    # วิธีนี้จะทำให้ได้ "เส้นกำแพง" ที่แท้จริง และไม่มีส่วนเกินในพื้นที่ห้อง
    walls = footprint.difference(walkable_area)

    # --- Plotting Helpers ---
    def plot_polygon(ax, poly, color, alpha, bg_color=None, label=None):
        if poly.is_empty: return
        artist = None
        if hasattr(poly, 'geoms'):
            for part in poly.geoms: 
                artist = plot_polygon(ax, part, color, alpha, bg_color)
        elif isinstance(poly, Polygon):
            artist = patches.Polygon(np.array(poly.exterior.coords), color=color, alpha=alpha, lw=0, label=label)
            ax.add_patch(artist)
            if bg_color:
                for interior in poly.interiors:
                    ax.add_patch(patches.Polygon(np.array(interior.coords), color=bg_color, alpha=1.0, lw=0))
        return artist

    # --- Sequential Labeling Logic ---
    # ตามตรรกะเดิม คืออ่าน Corridor ก่อนแล้วค่อย Room
    node_labels = {} 
    
    # 1. เรียงลำดับ Corridor ก่อน (เริ่มจาก 0)
    current_idx = 0
    for i in range(len(corridor_polys)):
        node_id = f"Cor-{i}"
        node_labels[node_id] = str(current_idx)
        current_idx += 1
        
    # 2. เรียงลำดับ Room ต่อจาก Corridor
    for i in range(len(room_polys)):
        node_id = f"Room-{i}"
        node_labels[node_id] = str(current_idx)
        current_idx += 1

    def add_geom_labels(ax, room_polys, corridor_polys, labels_dict):
        # วาดเลข Corridor (เปลี่ยนเป็นสีดำตามคำขอ)
        for i, p in enumerate(corridor_polys):
            node_id = f"Cor-{i}"
            c = p.centroid
            ax.text(c.x, c.y, labels_dict[node_id], fontsize=10, fontweight='bold', ha='center', va='center', color='black',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        # วาดเลข Room (สิิิีดำอยู่แล้ว)
        for i, p in enumerate(room_polys):
            node_id = f"Room-{i}"
            c = p.centroid
            ax.text(c.x, c.y, labels_dict[node_id], fontsize=10, fontweight='bold', ha='center', va='center', color='black', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    st.markdown("#### Geometric & Topological Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    
    minx, miny, maxx, maxy = footprint.bounds

    # 1. Combined View
    with col1:
        st.caption("Combined Area")
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        plot_polygon(ax1, walkable_area, 'skyblue', 0.6, label='Walkable')
        plot_polygon(ax1, walls, 'black', 0.8, bg_color='#ecf0f1', label='Obstacles')
        add_geom_labels(ax1, room_polys, corridor_polys, node_labels)
        ax1.set_xlim(minx - 1, maxx + 1); ax1.set_ylim(miny - 1, maxy + 1); ax1.set_aspect('equal')
        ax1.legend(loc='upper right', fontsize='xx-small'); ax1.axis('off')
        st.pyplot(fig1)

    # 2. Walkable Only
    with col2:
        st.caption("Walkable (Pure)")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        plot_polygon(ax2, walkable_area, '#ecf0f1', 0.8, label='Walkable')
        add_geom_labels(ax2, room_polys, corridor_polys, node_labels)
        ax2.set_xlim(minx - 1, maxx + 1); ax2.set_ylim(miny - 1, maxy + 1); ax2.set_aspect('equal')
        ax2.legend(loc='upper right', fontsize='xx-small'); ax2.axis('off')
        st.pyplot(fig2)

    # 3. Obstacles Only
    with col3:
        st.caption("Obstacles (Walls)")
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        plot_polygon(ax3, walls, 'black', 0.8, bg_color='white', label='Obstacles')
        ax3.set_xlim(minx - 1, maxx + 1); ax3.set_ylim(miny - 1, maxy + 1); ax3.set_aspect('equal')
        ax3.legend(loc='upper right', fontsize='xx-small'); ax3.axis('off')
        st.pyplot(fig3)

    # 4. Adjacency Graph (Topological View)
    with col4:
        st.caption("Adjacency Graph")
        try:
            import networkx as nx
            G = nx.Graph()
            
            # Nodes: ใช้ IDs จริงจาก JSON แต่ป้ายชื่อเป็นลำดับที่ไล่มา (labels_dict)
            for node_id, _ in node_labels.items():
                node_type = 'Corridor' if "Cor-" in node_id else 'Room'
                G.add_node(node_id, type=node_type)
            
            # Edges
            for d in doors_data:
                if 'rooms' in d and len(d['rooms']) == 2:
                    u, v = d['rooms']
                    if u in G.nodes and v in G.nodes:
                        G.add_edge(u, v)
            
            fig4, ax4 = plt.subplots(figsize=(5, 5))
            pos = nx.spring_layout(G, k=1.0, seed=42)
            # สีโหนด: เขียว=Corridor, ขาวเทา=Room
            node_colors = ['#27ae60' if G.nodes[n].get('type') == 'Corridor' else '#ecf0f1' for n in G.nodes]
            
            # เปลี่ยน font_color เป็น black ตามคำขอ
            nx.draw(G, pos, ax=ax4, labels=node_labels, with_labels=True, 
                    node_color=node_colors, edge_color='#34495e', 
                    node_size=700, font_size=10, font_weight='bold', font_color='black', width=2)
            
            ax4.axis('off')
            st.pyplot(fig4)
            st.info("⚪ = Room | 🟢 = Corridor", icon="ℹ️")
        except Exception as e:
            st.warning(f"Graph error: {e}")

    st.divider()
    st.markdown("### Topology Details and Movement Paths")
    d_col1, d_col2, d_col3, d_col4 = st.columns([0.8, 1.4, 1.4, 1.4])
    
    with d_col1:
        st.write("**Summary Counts**")
        num_cor = len(corridor_polys)
        num_room = len(room_polys)
        st.metric("Total Corridors", num_cor)
        st.metric("Total Rooms", num_room)
        st.metric("Total Nodes", num_cor + num_room)
        st.metric("Total Doors", len(doors_data))

    with d_col2:
        st.write("**All Paths (Sorted by Distance)**")
        path_data = [] 
        diameter = 0
        if 'G' in locals() and len(G.nodes) > 1:
            try:
                import networkx as nx
                all_paths = dict(nx.all_pairs_shortest_path(G))
                processed_pairs = set()
                is_connected = nx.is_connected(G)
                diameter = nx.diameter(G) if is_connected else 0
                nodes_list = sorted(list(G.nodes))
                for start_node in nodes_list:
                    for end_node in nodes_list:
                        if start_node == end_node: continue
                        pair = tuple(sorted((start_node, end_node)))
                        if pair in processed_pairs: continue
                        processed_pairs.add(pair)
                        if end_node in all_paths[start_node]:
                            path = all_paths[start_node][end_node]
                            length = len(path) - 1
                            labeled_path = [f"**{node_labels[n]}**" for n in path]
                            path_str = " → ".join(labeled_path)
                            path_data.append({'len': length, 'str': path_str, 'nodes': path})
                path_data.sort(key=lambda x: x['len'], reverse=True)
                for item in path_data:
                    if item['len'] == diameter: st.success(item['str'])
                    else: st.markdown(item['str'])
            except Exception as e:
                st.write(f"Error: {e}")

    with d_col3:
        st.write("**Longest Path Visualizer**")
        if 'G' in locals() and diameter > 0:
            try:
                longest_paths = [p for p in path_data if p['len'] == diameter]
                for idx, p_info in enumerate(longest_paths):
                    st.caption(f"Path Vis #{idx+1}")
                    fig_p, ax_p = plt.subplots(figsize=(4, 3))
                    pos_p = nx.spring_layout(G, seed=42)
                    path_nodes = p_info['nodes']
                    path_edges = list(zip(path_nodes, path_nodes[1:]))
                    edge_colors = ['#e74c3c' if (u, v) in path_edges or (v, u) in path_edges else '#bdc3c7' for u, v in G.edges()]
                    edge_widths = [4.0 if (u, v) in path_edges or (v, u) in path_edges else 1.5 for u, v in G.edges()]
                    node_colors_p = ['#27ae60' if G.nodes[n].get('type') == 'Corridor' else '#ecf0f1' for n in G.nodes]
                    nx.draw(G, pos_p, ax=ax_p, labels=node_labels, with_labels=True, node_color=node_colors_p, 
                            edge_color=edge_colors, width=edge_widths, node_size=500, font_size=8, font_color='black')
                    ax_p.axis('off'); st.pyplot(fig_p); plt.close(fig_p)
            except Exception as e:
                st.write(f"Viz error: {e}")

    with d_col4:
        st.write("**Start & Exit Area (Geom)**")
        if 'G' in locals() and diameter > 0:
            try:
                longest_paths = [p for p in path_data if p['len'] == diameter]
                # ใช้พื้นที่ขอบ (diameter endpoints) เป็น Start / Exit
                for idx, p_info in enumerate(longest_paths):
                    start_node_id = p_info['nodes'][0]
                    exit_node_id = p_info['nodes'][-1]
                    st.caption(f"Scenario #{idx+1}: {node_labels[start_node_id]} (Start) → {node_labels[exit_node_id]} (Exit)")
                    
                    fig_g, ax_g = plt.subplots(figsize=(4, 4))
                    # วาดพื้นที่พื้นหลังทั้งหมดเป็นสีเทาอ่อน
                    plot_polygon(ax_g, walkable_area, '#ecf0f1', 0.5)
                    
                    # ไฮไลต์ Start (แดง) และ Exit (เขียว) ในแปลน
                    for nid, color in [(start_node_id, '#e74c3c'), (exit_node_id, '#2ecc71')]:
                        # แยกคำอธิบาย (Room-0 -> 0)
                        if "Room-" in nid:
                            ridx = int(nid.split("-")[1])
                            plot_polygon(ax_g, room_polys[ridx], color, 0.8)
                        elif "Cor-" in nid:
                            cidx = int(nid.split("-")[1])
                            plot_polygon(ax_g, corridor_polys[cidx], color, 0.8)
                    
                    add_geom_labels(ax_g, room_polys, corridor_polys, node_labels)
                    ax_g.set_xlim(minx - 1, maxx + 1); ax_g.set_ylim(miny - 1, maxy + 1); ax_g.set_aspect('equal')
                    ax_g.axis('off'); st.pyplot(fig_g); plt.close(fig_g)
            except Exception as e:
                st.write(f"Geom preview error: {e}")
        else:
            st.write("Not enough nodes to calculate movement.")

def show_epoch_visuals(run_dir, title="Val Epoch Visuals", key_prefix="epoch_visual"):
    """
    Displays per-epoch validation visuals generated by a method-specific visual script.
    """
    run_path = pathlib.Path(run_dir)
    vis_dir = run_path / "visuals" / "val_epochs"
    metrics_path = vis_dir / "epoch_metrics.csv"

    st.markdown(f"### {title}")

    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)
            if not df.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("Epoch Visuals", len(df))
                c2.metric("Best ADE (m)", f"{df['ade_m'].min():.3f}")
                c3.metric("Best FDE (m)", f"{df['fde_m'].min():.3f}")

                chart_cols = [c for c in ["ade_m", "fde_m"] if c in df.columns]
                if chart_cols and "epoch" in df.columns:
                    chart_df = df.copy()
                    chart_df["epoch_idx"] = chart_df["epoch"].str.extract(r"(\d+)").astype(float)
                    chart_df = chart_df.sort_values("epoch_idx")
                    st.line_chart(chart_df.set_index("epoch")[chart_cols], x_label="Epoch", y_label="Meters")

                with st.expander("Show epoch metrics table", expanded=False):
                    st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading epoch metrics: {e}")

    if not vis_dir.exists():
        st.info("No validation visualisations found yet. Click 'Generate Val Visuals' to create them.")
        return

    images = sorted(list(vis_dir.glob("epoch_*.png")), key=lambda x: x.name)
    if not images:
        st.info("No epoch visual images found yet.")
        return

    selected_image = st.selectbox(
        "Select epoch visual",
        images,
        index=len(images) - 1,
        format_func=lambda p: p.name,
        key=f"{key_prefix}_{run_path.name}",
    )
    st.image(str(selected_image), caption=selected_image.name, use_container_width=True)

    with st.expander("Show all epoch visuals", expanded=False):
        cols_per_row = 2
        for row_idx in range((len(images) + cols_per_row - 1) // cols_per_row):
            st_cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                img_idx = row_idx * cols_per_row + col_idx
                if img_idx < len(images):
                    img_path = images[img_idx]
                    st_cols[col_idx].image(str(img_path), caption=img_path.name, use_container_width=True)


def show_transformer_val_visuals(run_dir):
    """
    Backward-compatible wrapper for transformer epoch visuals.
    """
    show_epoch_visuals(run_dir, title="Val Epoch Visuals", key_prefix="transformer_epoch_visual")


def show_gnn_cvae_val_visuals(run_dir):
    """
    Viewer for GNN-CVAE per-epoch validation visuals.
    """
    show_epoch_visuals(run_dir, title="Val Epoch Visuals", key_prefix="gnn_cvae_epoch_visual")


def show_gpt_knowledge_results(method_result_dir):
    """
    Viewer for Method_GPT_Knowledge outputs.
    """
    import json

    result_root = pathlib.Path(method_result_dir)
    knowledge_root = result_root / "knowledge"
    outputs_root = result_root / "outputs"
    eval_root = result_root / "evaluation"
    vis_root = result_root / "visuals" / "generated_samples"

    t1, t2, t3, t4 = st.tabs(["Knowledge Index", "Generated Visuals", "Generated Samples", "Evaluation"])

    with t1:
        st.markdown("### Knowledge Index")
        if not knowledge_root.exists():
            st.info("No knowledge directory found yet.")
        else:
            knowledge_dirs = sorted([d for d in knowledge_root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
            if not knowledge_dirs:
                st.info("No knowledge builds found yet.")
            else:
                selected_knowledge = st.selectbox("Select Knowledge Build", knowledge_dirs, format_func=lambda p: p.name)
                manifest_path = selected_knowledge / "knowledge_manifest.json"
                scene_index_csv = selected_knowledge / "scene_index.csv"

                if manifest_path.exists():
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    c1, c2 = st.columns(2)
                    c1.metric("Split", manifest.get("split", "-"))
                    c2.metric("Cases", manifest.get("num_cases", 0))
                    with st.expander("Show knowledge manifest", expanded=False):
                        st.json(manifest)

                if scene_index_csv.exists():
                    df = pd.read_csv(scene_index_csv)
                    if not df.empty:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Rows", len(df))
                        c2.metric("Mean Agents", f"{df['n_agents'].mean():.1f}" if "n_agents" in df.columns else "-")
                        c3.metric("Mean Goal Dist", f"{df['goal_distance'].mean():.2f}" if "goal_distance" in df.columns else "-")
                        with st.expander("Show scene index table", expanded=False):
                            st.dataframe(df, use_container_width=True)

    with t2:
        st.markdown("### Generated Visuals")
        metrics_path = vis_root / "sample_metrics.csv"
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                if not df.empty:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Visuals", len(df))
                    c2.metric("Best ADE (m)", f"{df['ade_m'].min():.3f}")
                    c3.metric("Best FDE (m)", f"{df['fde_m'].min():.3f}")
                    chart_cols = [c for c in ["ade_m", "fde_m", "out_of_bounds_rate"] if c in df.columns]
                    if chart_cols and "sample" in df.columns:
                        st.line_chart(df.set_index("sample")[chart_cols], x_label="Sample", y_label="Metric")
                    with st.expander("Show visual metrics table", expanded=False):
                        st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading GPT knowledge visual metrics: {e}")

        if not vis_root.exists():
            st.info("No generated visuals found yet. Click 'Generate Visuals' to create them.")
        else:
            images = sorted(list(vis_root.glob("sample_case_*.png")), key=lambda x: x.name)
            if not images:
                st.info("No generated visual images found yet.")
            else:
                selected_image = st.selectbox("Select generated visual", images, index=len(images) - 1, format_func=lambda p: p.name)
                st.image(str(selected_image), caption=selected_image.name, use_container_width=True)
                with st.expander("Show all generated visuals", expanded=False):
                    cols_per_row = 2
                    for row_idx in range((len(images) + cols_per_row - 1) // cols_per_row):
                        st_cols = st.columns(cols_per_row)
                        for col_idx in range(cols_per_row):
                            img_idx = row_idx * cols_per_row + col_idx
                            if img_idx < len(images):
                                img_path = images[img_idx]
                                st_cols[col_idx].image(str(img_path), caption=img_path.name, use_container_width=True)

    with t3:
        st.markdown("### Generated Samples")
        if not outputs_root.exists():
            st.info("No generated sample outputs found yet.")
        else:
            sample_dirs = sorted([d for d in outputs_root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
            if not sample_dirs:
                st.info("No generated sample outputs found yet.")
            else:
                selected_sample = st.selectbox("Select Generated Sample", sample_dirs, format_func=lambda p: p.name)
                summary_json = selected_sample / "generation_summary.json"
                retrieved_json = selected_sample / "retrieved_cases.json"
                prompt_txt = selected_sample / "planner_prompt.txt"
                route_skeleton_png = selected_sample / "route_skeleton.png"
                route_skeleton_json = selected_sample / "route_skeleton.json"
                pred_files = sorted(selected_sample.glob("AI_pred_*.parquet"))

                if summary_json.exists():
                    with open(summary_json, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Case ID", summary.get("case_id", "-"))
                    c2.metric("OOB Rate", f"{summary.get('out_of_bounds_rate', 0.0):.4f}")
                    c3.metric("Retrieved Cases", len(summary.get("retrieved_cases", [])))
                    with st.expander("Show generation summary", expanded=False):
                        st.json(summary)
                    attempts = summary.get("generation_attempts", [])
                    if attempts:
                        with st.expander("Show generation attempts", expanded=False):
                            st.dataframe(pd.DataFrame(attempts), use_container_width=True)

                if retrieved_json.exists():
                    retrieved = json.loads(retrieved_json.read_text(encoding="utf-8"))
                    if retrieved:
                        st.markdown("#### Retrieved Cases")
                        st.dataframe(pd.DataFrame(retrieved), use_container_width=True)

                if prompt_txt.exists():
                    with st.expander("Show planner prompt", expanded=False):
                        st.code(prompt_txt.read_text(encoding="utf-8"))

                if route_skeleton_png.exists():
                    st.markdown("#### Route Skeleton Preview")
                    st.image(str(route_skeleton_png), caption=route_skeleton_png.name, use_container_width=True)
                if route_skeleton_json.exists():
                    with st.expander("Show route skeleton JSON", expanded=False):
                        st.json(json.loads(route_skeleton_json.read_text(encoding="utf-8")))

                if pred_files:
                    selected_pred = st.selectbox("Select Prediction File", pred_files, format_func=lambda p: p.name)
                    pred_df = pd.read_parquet(selected_pred)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rows", len(pred_df))
                    c2.metric("Agents", pred_df["id"].nunique() if "id" in pred_df.columns else 0)
                    c3.metric("Frames", pred_df["frame"].nunique() if "frame" in pred_df.columns else 0)
                    with st.expander("Show prediction table", expanded=False):
                        st.dataframe(pred_df, use_container_width=True)

    with t4:
        st.markdown("### Evaluation")
        if not eval_root.exists():
            st.info("No evaluation outputs found yet.")
        else:
            eval_dirs = sorted([d for d in eval_root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
            if not eval_dirs:
                st.info("No evaluation outputs found yet.")
            else:
                selected_eval = st.selectbox("Select Evaluation Output", eval_dirs, format_func=lambda p: p.name)
                summary_jsons = sorted(selected_eval.glob("*_evaluation_summary.json"))
                scene_csvs = sorted(selected_eval.glob("*_scene_metrics.csv"))
                agent_csvs = sorted(selected_eval.glob("*_agent_metrics.csv"))

                if summary_jsons:
                    with open(summary_jsons[0], "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Cases", summary.get("num_cases", 0))
                    c2.metric("Mean Path ADE", f"{summary.get('mean_path_ade_m', 0.0):.3f}")
                    c3.metric("Mean Path FDE", f"{summary.get('mean_path_fde_m', 0.0):.3f}")
                    c4, c5 = st.columns(2)
                    c4.metric("Mean Collision", f"{summary.get('mean_collision_rate', 0.0):.4f}")
                    c5.metric("Mean OOB", f"{summary.get('mean_out_of_bounds_rate', 0.0):.4f}")
                    with st.expander("Show evaluation summary JSON", expanded=False):
                        st.json(summary)

                if scene_csvs:
                    scene_df = pd.read_csv(scene_csvs[0])
                    with st.expander("Show scene metrics", expanded=False):
                        st.dataframe(scene_df, use_container_width=True)

                if agent_csvs:
                    agent_df = pd.read_csv(agent_csvs[0])
                    with st.expander("Show agent metrics", expanded=False):
                        st.dataframe(agent_df, use_container_width=True)


def show_gpt_knowledge_special_tests(method_result_dir):
    """
    Viewer for Method_GPT_Knowledge special blind-test sessions.
    """
    import json

    result_root = pathlib.Path(method_result_dir)
    special_root = result_root / "special_tests"
    st.markdown("### Special Test Results")

    if not special_root.exists():
        st.info("No special test sessions found yet.")
        return

    session_dirs = []
    compared_sessions = []
    for d in sorted([d for d in special_root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True):
        if (d / "special_test_manifest.json").exists():
            session_dirs.append(d)
            compare_dir = d / "compare"
            has_compare = (
                (compare_dir / "compare_summary.json").exists()
                or (compare_dir / "special_evaluation_summary.json").exists()
                or (compare_dir / "special_scene_metrics.csv").exists()
            )
            if has_compare:
                compared_sessions.append(d)
    if not session_dirs:
        st.info("No completed special test sessions found yet.")
        return

    default_session = compared_sessions[0] if compared_sessions else session_dirs[0]
    default_index = session_dirs.index(default_session)
    selected_session = st.selectbox("Select Special Test Session", session_dirs, index=default_index, format_func=lambda p: p.name)
    manifest_path = selected_session / "special_test_manifest.json"
    ai_image = selected_session / "ai_prediction.png"
    generation_dir = selected_session / "generation"
    compare_dir = selected_session / "compare"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Case ID", manifest.get("case_id", "-"))
        retrieved = manifest.get("retrieved_cases", [])
        c2.metric("Retrieved Cases", len(retrieved))
        c3.metric("Top Case", retrieved[0]["case_id"] if retrieved else "-")
        with st.expander("Show special test manifest", expanded=False):
            st.json(manifest)

    if generation_dir.exists():
        retrieved_json = generation_dir / "retrieved_cases.json"
        prompt_txt = generation_dir / "planner_prompt.txt"
        route_skeleton_png = generation_dir / "route_skeleton.png"
        route_skeleton_json = generation_dir / "route_skeleton.json"
        pred_files = sorted(generation_dir.glob("AI_pred_*.parquet"))
        if retrieved_json.exists():
            retrieved = json.loads(retrieved_json.read_text(encoding="utf-8"))
            if retrieved:
                st.markdown("#### Retrieved Reference Cases")
                st.dataframe(pd.DataFrame(retrieved), use_container_width=True)
        if prompt_txt.exists():
            with st.expander("Show planner prompt", expanded=False):
                st.code(prompt_txt.read_text(encoding="utf-8"))
        if route_skeleton_png.exists():
            st.markdown("#### Route Skeleton Preview")
            st.image(str(route_skeleton_png), caption=route_skeleton_png.name, use_container_width=True)
        if route_skeleton_json.exists():
            with st.expander("Show route skeleton JSON", expanded=False):
                st.json(json.loads(route_skeleton_json.read_text(encoding="utf-8")))
        if pred_files:
            pred_df = pd.read_parquet(pred_files[0])
            with st.expander("Show AI prediction table", expanded=False):
                st.dataframe(pred_df, use_container_width=True)
        summary_json = generation_dir / "generation_summary.json"
        if summary_json.exists():
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            attempts = summary.get("generation_attempts", [])
            if attempts:
                with st.expander("Show generation attempts", expanded=False):
                    st.dataframe(pd.DataFrame(attempts), use_container_width=True)

    if ai_image.exists():
        st.markdown("#### AI Visual")
        st.image(str(ai_image), caption=ai_image.name, use_container_width=True)

    compare_summary = compare_dir / "compare_summary.json"
    special_overview = compare_dir / "special_evaluation_summary.json"
    special_scene_csv = compare_dir / "special_scene_metrics.csv"
    compare_image = compare_dir / "compare_ai_vs_gt.png"
    compare_agents = compare_dir / "compare_agent_metrics.csv"
    special_agents = compare_dir / "special_agent_metrics.csv"

    if compare_summary.exists():
        summary = json.loads(compare_summary.read_text(encoding="utf-8"))
        st.markdown("#### Compare AI vs GT")
        c1, c2, c3 = st.columns(3)
        c1.metric("Path ADE", f"{summary.get('path_ade_m', 0.0):.3f}")
        c2.metric("Path FDE", f"{summary.get('path_fde_m', 0.0):.3f}")
        c3.metric("Duration Error", f"{summary.get('duration_abs_error_frames', 0.0):.1f}")
        c4, c5 = st.columns(2)
        c4.metric("Collision", f"{summary.get('collision_rate', 0.0):.4f}")
        c5.metric("OOB", f"{summary.get('out_of_bounds_rate', 0.0):.4f}")
        if compare_image.exists():
            st.image(str(compare_image), caption=compare_image.name, use_container_width=True)
        if compare_agents.exists():
            agent_df = pd.read_csv(compare_agents)
            with st.expander("Show compare agent metrics", expanded=False):
                st.dataframe(agent_df, use_container_width=True)
    elif special_overview.exists() or special_scene_csv.exists():
        st.markdown("#### Compare AI vs GT")
        summary_row = {}
        if special_scene_csv.exists():
            try:
                scene_df = pd.read_csv(special_scene_csv)
                if not scene_df.empty:
                    summary_row = scene_df.iloc[0].to_dict()
            except Exception:
                summary_row = {}

        if not summary_row and special_overview.exists():
            overview = json.loads(special_overview.read_text(encoding="utf-8"))
            summary_row = {
                "path_ade_m": overview.get("mean_path_ade_m", 0.0),
                "path_fde_m": overview.get("mean_path_fde_m", 0.0),
                "duration_abs_error_frames": overview.get("mean_duration_abs_error_frames", 0.0),
                "collision_rate": overview.get("mean_collision_rate", 0.0),
                "out_of_bounds_rate": overview.get("mean_out_of_bounds_rate", 0.0),
            }

        c1, c2, c3 = st.columns(3)
        c1.metric("Path ADE", f"{float(summary_row.get('path_ade_m', 0.0)):.3f}")
        c2.metric("Path FDE", f"{float(summary_row.get('path_fde_m', 0.0)):.3f}")
        c3.metric("Duration Error", f"{float(summary_row.get('duration_abs_error_frames', 0.0)):.1f}")
        c4, c5 = st.columns(2)
        c4.metric("Collision", f"{float(summary_row.get('collision_rate', 0.0)):.4f}")
        c5.metric("OOB", f"{float(summary_row.get('out_of_bounds_rate', 0.0)):.4f}")
        if compare_image.exists():
            st.image(str(compare_image), caption=compare_image.name, use_container_width=True)
        if special_agents.exists():
            agent_df = pd.read_csv(special_agents)
            with st.expander("Show compare agent metrics", expanded=False):
                st.dataframe(agent_df, use_container_width=True)
    else:
        st.info("No GT comparison uploaded for this special test yet.")


def show_pix2pix_special_tests(method_result_dir):
    """
    Viewer for pix2pix / pix2pixHD single-case special test sessions.
    """
    import json

    result_root = pathlib.Path(method_result_dir)
    special_root = result_root / "special_tests"
    st.markdown("### Special Test Results")

    if not special_root.exists():
        st.info("No special test sessions found yet.")
        return

    session_dirs = sorted([d for d in special_root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
    if not session_dirs:
        st.info("No special test sessions found yet.")
        return

    selected_session = st.selectbox("Select Special Test Session", session_dirs, format_func=lambda p: p.name, key=f"pix2pix_special_{result_root.name}")
    summary_path = selected_session / "special_summary.json"
    metrics_path = selected_session / "special_metrics.csv"
    input_img = selected_session / "special_input.png"
    target_img = selected_session / "special_target.png"
    pred_img = selected_session / "special_prediction.png"
    compare_img = selected_session / "special_compare.png"

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            c1, c2, c3 = st.columns(3)
            c1.metric("Case", summary.get("case_id", "-"))
            c2.metric("MAE (L1)", f"{float(summary.get('mae_l1', 0.0)):.6f}")
            c3.metric("RMSE", f"{float(summary.get('rmse', 0.0)):.6f}")
            with st.expander("Show special summary", expanded=False):
                st.json(summary)
        except Exception as e:
            st.error(f"Failed to load special summary: {e}")

    if compare_img.exists():
        st.image(str(compare_img), caption=compare_img.name, use_container_width=True)
    else:
        cols = st.columns(3)
        if input_img.exists():
            cols[0].image(str(input_img), caption="Input", use_container_width=True)
        if target_img.exists():
            cols[1].image(str(target_img), caption="Target", use_container_width=True)
        if pred_img.exists():
            cols[2].image(str(pred_img), caption="Prediction", use_container_width=True)

    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)
            with st.expander("Show metrics table", expanded=False):
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load metrics CSV: {e}")

