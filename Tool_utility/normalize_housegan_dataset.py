"""
normalize_housegan_dataset.py
-----------------------------
Convert Topo_HouseGAN outputs into Dataset_Traj_Table-style case folders.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sqlite3

import networkx as nx
import pandas as pd
from shapely.geometry import Polygon


def load_polygons_from_json(file_path: pathlib.Path) -> list[Polygon]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Polygon(coords) for coords in data]


def sqlite_connect_readonly(sqlite_path: pathlib.Path):
    uri = f"file:{sqlite_path}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=1)


def load_trajectory_table(sqlite_path: pathlib.Path, table_filter: str = "trajectory_data") -> pd.DataFrame:
    conn = sqlite_connect_readonly(sqlite_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if row[0] != "sqlite_sequence"]
        target = None
        for table_name in tables:
            if table_filter.lower() in table_name.lower():
                target = table_name
                break
        if target is None:
            raise RuntimeError(f"No table matching '{table_filter}' found in {sqlite_path.name}. Tables={tables}")
        df = pd.read_sql_query(f"SELECT * FROM {target}", conn)
        return normalize_trajectory_schema(df)
    finally:
        conn.close()


def normalize_trajectory_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    f_col = cols.get("frame")
    id_col = cols.get("id") or cols.get("agentid")
    x_col = cols.get("posx") or cols.get("positionx") or cols.get("x")
    y_col = cols.get("posy") or cols.get("positiony") or cols.get("y")
    ox_col = cols.get("orix") or cols.get("orientationx")
    oy_col = cols.get("oriy") or cols.get("orientationy")
    if not all([f_col, id_col, x_col, y_col]):
        raise ValueError(f"Unable to map trajectory columns from {list(df.columns)}")

    out = df[[f_col, id_col, x_col, y_col]].copy()
    out.columns = ["frame", "id", "pos_x", "pos_y"]
    if ox_col and oy_col:
        out["ori_x"] = df[ox_col].to_numpy()
        out["ori_y"] = df[oy_col].to_numpy()
    return out.sort_values(["id", "frame"]).reset_index(drop=True)


def polygon_to_wkt_row(poly: Polygon, label: str) -> dict:
    return {"type": label, "area": poly.wkt}


def generate_spawn_location(df: pd.DataFrame, output_path: pathlib.Path):
    frame_0 = df[df["frame"] == df["frame"].min()][["id", "pos_x", "pos_y"]].copy()
    frame_0.to_csv(output_path, index=False)


def build_routes_for_plan(geo_dir: pathlib.Path) -> tuple[list[tuple[str, str]], dict[str, Polygon]]:
    corridor_polys = load_polygons_from_json(geo_dir / "geo_corridor.json")
    room_polys = load_polygons_from_json(geo_dir / "geo_room.json")
    with open(geo_dir / "geo_door.json", "r", encoding="utf-8") as f:
        doors_data = json.load(f)

    G = nx.Graph()
    node_polys: dict[str, Polygon] = {}
    for i, p in enumerate(corridor_polys):
        nid = f"Cor-{i}"
        G.add_node(nid, type="Corridor")
        node_polys[nid] = p
    for i, p in enumerate(room_polys):
        nid = f"Room-{i}"
        G.add_node(nid, type="Room")
        node_polys[nid] = p
    for d in doors_data:
        if "rooms" in d and len(d["rooms"]) == 2:
            u, v = d["rooms"]
            if u in G.nodes and v in G.nodes:
                G.add_edge(u, v)

    if not nx.is_connected(G):
        raise RuntimeError(f"Graph for {geo_dir.name} is not connected.")

    diameter = nx.diameter(G)
    periphery = nx.periphery(G)
    routes_to_sim: list[tuple[str, str]] = []
    processed_pairs = set()
    for i, start_nid in enumerate(periphery):
        for end_nid in periphery[i + 1 :]:
            path = nx.shortest_path(G, start_nid, end_nid)
            if len(path) - 1 == diameter:
                pair = tuple(sorted((start_nid, end_nid)))
                if pair not in processed_pairs:
                    processed_pairs.add(pair)
                    routes_to_sim.append((start_nid, end_nid))
    return routes_to_sim, node_polys


def parse_sqlite_name(sqlite_path: pathlib.Path) -> tuple[str, str]:
    # plan_sim_42_00.sqlite -> seed='42', suffix='00'
    parts = sqlite_path.stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected sqlite filename: {sqlite_path.name}")
    return parts[-2], parts[-1]


def normalize_housegan(source_root: pathlib.Path, output_root: pathlib.Path, table_filter: str = "trajectory_data"):
    geo_root = source_root / "geo"
    dataswarm_root = source_root / "dataswarm"
    if not geo_root.exists() or not dataswarm_root.exists():
        raise FileNotFoundError(f"Expected geo and dataswarm folders inside {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    plan_dirs = sorted([p for p in dataswarm_root.iterdir() if p.is_dir() and p.name.startswith("plan_")], key=lambda p: p.name)
    print(f"[HouseGAN] Plans found: {len(plan_dirs)}")

    for plan_dir in plan_dirs:
        geo_dir = geo_root / plan_dir.name
        if not geo_dir.exists():
            print(f"[HouseGAN] Skip {plan_dir.name}: missing geo directory")
            continue

        try:
            routes_to_sim, node_polys = build_routes_for_plan(geo_dir)
        except Exception as e:
            print(f"[HouseGAN] Skip {plan_dir.name}: route build failed -> {e}")
            continue

        sqlite_files = sorted(plan_dir.glob("*.sqlite"), key=lambda p: p.name)
        route_map = {}
        for sqlite_path in sqlite_files:
            _, suffix = parse_sqlite_name(sqlite_path)
            route_idx = int(suffix)
            if route_idx >= len(routes_to_sim):
                print(f"[HouseGAN] Skip {sqlite_path.name}: route index {route_idx} out of range")
                continue
            route_map[sqlite_path] = routes_to_sim[route_idx]

        for sqlite_path, (start_nid, end_nid) in route_map.items():
            seed_str, suffix = parse_sqlite_name(sqlite_path)
            case_id = f"{plan_dir.name}_{seed_str}_{suffix}"
            case_dir = output_root / f"case_{case_id}"
            case_dir.mkdir(parents=True, exist_ok=True)

            try:
                traj_df = load_trajectory_table(sqlite_path, table_filter=table_filter)
            except Exception as e:
                print(f"[HouseGAN] Failed {sqlite_path.name}: {e}")
                continue

            parquet_name = f"{plan_dir.name}_{seed_str}_{suffix}_trajectory_data.parquet"
            traj_df.to_parquet(case_dir / parquet_name, index=False)
            shutil.copy2(geo_dir / "geo_room.json", case_dir / "Geo_room.json")
            shutil.copy2(geo_dir / "geo_corridor.json", case_dir / "Geo_corridor.json")

            spawn_exit_df = pd.DataFrame(
                [
                    polygon_to_wkt_row(node_polys[start_nid], "spawning_area"),
                    polygon_to_wkt_row(node_polys[end_nid], "exit_area"),
                ]
            )
            spawn_exit_df.to_csv(case_dir / f"Spawn_exit_{case_id}.csv", index=False)
            generate_spawn_location(traj_df, case_dir / f"Spawn_location_{case_id}.csv")

            manifest_rows.append(
                {
                    "case_id": case_id,
                    "plan_name": plan_dir.name,
                    "seed": int(seed_str),
                    "route_suffix": suffix,
                    "start_node": start_nid,
                    "end_node": end_nid,
                    "sqlite_path": str(sqlite_path),
                    "geo_dir": str(geo_dir),
                    "num_agents": int(traj_df["id"].nunique()),
                    "num_frames": int(traj_df["frame"].nunique()),
                }
            )
            print(f"[HouseGAN] case_{case_id} ready")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_root / "manifest_housegan_cases.csv", index=False)
    print(f"[HouseGAN] Done. Cases={len(manifest_df)} -> {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize Topo_HouseGAN into Dataset_Traj_Table case folders.")
    parser.add_argument("--source_root", type=str, required=True, help="Path to Geo_scenario/Topo_HouseGAN")
    parser.add_argument("--output_root", type=str, required=True, help="Path to Dataset_Traj_Table/Topo_HouseGAN")
    parser.add_argument("--filter", type=str, default="trajectory_data", help="SQLite table filter, default: trajectory_data")
    args = parser.parse_args()

    normalize_housegan(pathlib.Path(args.source_root).resolve(), pathlib.Path(args.output_root).resolve(), table_filter=args.filter)
