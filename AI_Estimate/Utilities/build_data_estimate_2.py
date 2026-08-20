import argparse
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTE_UTILITY_ROOT = PROJECT_ROOT / "GeneratePlan_HouseGAN" / "Prepare_data"
for import_root in (ROUTE_UTILITY_ROOT,):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from generate_route_information import (  # noqa: E402
    compute_edge_bottlenecks,
    load_plan_graph,
    natural_node_key,
    route_bottleneck_score,
    route_row,
    shortest_paths_from,
)


SPLIT_DIRECTORIES = {"train": "train", "val": "validation", "test": "test"}
OUTPUT_DIRECTORIES = {"train": "Train", "val": "Val", "test": "Test"}
TIME_COLUMNS = [
    "plan",
    "route_index",
    "start_node",
    "end_node",
    "variant_id",
    "variant_label",
    "status",
    "computed_agents",
    "fps",
    "max_frame",
    "simulation_duration_s",
    "mean_agent_time_s",
    "min_agent_time_s",
    "max_agent_time_s",
    "trajectory_file",
]


def safe_divide(a, b, default=0.0):
    a = float(a) if pd.notna(a) else 0.0
    b = float(b) if pd.notna(b) else 0.0
    if abs(b) < 1e-9:
        return default
    return a / b


def add_derived_features(df):
    df["computed_agents"] = pd.to_numeric(df.get("computed_agents", 0), errors="coerce").fillna(0)
    df["topology_centerline_distance_m"] = pd.to_numeric(
        df["topology_centerline_distance_m"], errors="coerce"
    ).fillna(0)
    df["straight_distance_m"] = pd.to_numeric(df["straight_distance_m"], errors="coerce").fillna(0)
    df["walkable_area_near_path"] = pd.to_numeric(df["walkable_area_near_path"], errors="coerce").fillna(0)
    df["door_count_between_A_B"] = pd.to_numeric(df["door_count_between_A_B"], errors="coerce").fillna(0)
    df["min_door_width_between_A_B"] = pd.to_numeric(
        df["min_door_width_between_A_B"], errors="coerce"
    ).fillna(1.5)
    df["detour_ratio"] = df.apply(
        lambda row: safe_divide(
            row["topology_centerline_distance_m"], row["straight_distance_m"], default=1.0
        ),
        axis=1,
    )
    df["distance_gap_m"] = (df["topology_centerline_distance_m"] - df["straight_distance_m"]).clip(lower=0)
    df["agent_density_near_path"] = df.apply(
        lambda row: safe_divide(row["computed_agents"], row["walkable_area_near_path"], default=0.0),
        axis=1,
    )
    df["area_per_agent"] = df.apply(
        lambda row: safe_divide(row["walkable_area_near_path"], max(row["computed_agents"], 1), default=0.0),
        axis=1,
    )
    df["door_pressure_per_agent"] = df.apply(
        lambda row: safe_divide(
            row["computed_agents"] * row["door_count_between_A_B"],
            max(row["min_door_width_between_A_B"], 0.1),
            default=0.0,
        ),
        axis=1,
    )


def add_variant_columns(df):
    for variant in ("full", "half", "single"):
        df[f"variant_{variant}"] = (df["variant_id"].astype(str) == variant).astype(float)


def feature_columns_from_config(df, config):
    numeric_columns = list(config["features"].get("numeric", []))
    categorical_columns = []
    for name in config["features"].get("categorical", []):
        if name == "variant_id":
            categorical_columns.extend(["variant_full", "variant_half", "variant_single"])
    columns = numeric_columns + categorical_columns
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return columns


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenario_id_from_trajectory(plan_name, trajectory_file):
    stem = Path(str(trajectory_file)).stem
    suffix = stem[len("plan_sim_") :] if stem.startswith("plan_sim_") else stem
    return f"{plan_name}__{suffix}"


def canonical_inventory(canonical_root):
    inventory = {}
    scenario_to_split = {}
    plan_to_split = {}
    for split_name, directory_name in SPLIT_DIRECTORIES.items():
        image_paths = sorted((canonical_root / directory_name).glob("*.png"))
        scenario_ids = [path.stem for path in image_paths]
        plan_names = sorted({scenario_id.split("__", 1)[0] for scenario_id in scenario_ids})
        inventory[split_name] = {
            "directory": directory_name,
            "scenario_ids": scenario_ids,
            "plan_names": plan_names,
        }
        for scenario_id in scenario_ids:
            previous = scenario_to_split.setdefault(scenario_id, split_name)
            if previous != split_name:
                raise ValueError(f"Scenario appears in multiple canonical splits: {scenario_id}")
        for plan_name in plan_names:
            previous = plan_to_split.setdefault(plan_name, split_name)
            if previous != split_name:
                raise ValueError(f"Plan appears in multiple canonical splits: {plan_name}")
    return inventory, scenario_to_split, plan_to_split


def canonical_inventory_hash(inventory):
    lines = []
    for split_name in ("train", "val", "test"):
        lines.extend(f"{split_name},{scenario_id}" for scenario_id in inventory[split_name]["scenario_ids"])
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def metadata_items(metadata_root, dataswarm_root, canonical_scenarios, plan_names):
    by_scenario = {}
    metadata_paths = []
    for plan_name in sorted(plan_names):
        summary_path = metadata_root / plan_name / "simulation_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing simulation metadata: {summary_path}")
        metadata_paths.append(summary_path)
        summary = read_json(summary_path)
        for route in summary.get("routes", []):
            variants = route.get("variants", [])
            if not variants and route.get("trajectory_file"):
                variants = [route]
            for variant in variants:
                old_trajectory_path = variant.get("trajectory_file", route.get("trajectory_file", ""))
                scenario_id = scenario_id_from_trajectory(plan_name, old_trajectory_path)
                if scenario_id not in canonical_scenarios:
                    continue
                if scenario_id in by_scenario:
                    raise ValueError(f"Duplicate metadata for canonical scenario: {scenario_id}")
                local_path = dataswarm_root / plan_name / Path(str(old_trajectory_path)).name
                if not local_path.exists():
                    raise FileNotFoundError(f"Missing local trajectory for {scenario_id}: {local_path}")
                by_scenario[scenario_id] = {
                    "scenario_id": scenario_id,
                    "plan": plan_name,
                    "route_index": route.get("route_index"),
                    "start_node": route.get("start_node"),
                    "end_node": route.get("end_node"),
                    "variant_id": variant.get("variant_id", "full"),
                    "variant_label": variant.get("variant_label", "N Agent"),
                    "computed_agents": variant.get("computed_agents", route.get("computed_agents", 0)),
                    "status": variant.get("status", route.get("status", "")),
                    "trajectory_file": local_path,
                }

    missing = sorted(canonical_scenarios - set(by_scenario))
    extra = sorted(set(by_scenario) - canonical_scenarios)
    if missing or extra:
        raise ValueError(
            f"Canonical/metadata mismatch: missing={len(missing)} extra={len(extra)} "
            f"missing_sample={missing[:5]} extra_sample={extra[:5]}"
        )
    return by_scenario, metadata_paths


def sqlite_fps(cursor):
    row = cursor.execute("select value from metadata where key = 'fps'").fetchone()
    return float(row[0]) if row else 25.0


def summarize_trajectory(item):
    with sqlite3.connect(item["trajectory_file"]) as connection:
        cursor = connection.cursor()
        fps = sqlite_fps(cursor)
        rows = cursor.execute(
            """
            select id, min(frame), max(frame), count(*)
            from trajectory_data
            group by id
            order by id
            """
        ).fetchall()
        max_frame_row = cursor.execute("select max(frame) from trajectory_data").fetchone()
    if not rows:
        raise ValueError(f"No trajectory_data rows: {item['trajectory_file']}")
    times = [(int(end_frame) - int(start_frame)) / fps for _, start_frame, end_frame, _ in rows]
    max_frame = int(max_frame_row[0] or 0)
    try:
        portable_path = item["trajectory_file"].resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        portable_path = str(item["trajectory_file"].resolve())
    return {
        "scenario_id": item["scenario_id"],
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
        "simulation_duration_s": max_frame / fps,
        "mean_agent_time_s": sum(times) / len(times),
        "min_agent_time_s": min(times),
        "max_agent_time_s": max(times),
        "trajectory_file": portable_path,
    }


def required_route_rows(geo_root, items_by_scenario):
    required_pairs = defaultdict(set)
    for item in items_by_scenario.values():
        required_pairs[item["plan"]].add((item["start_node"], item["end_node"]))

    rows = []
    for plan_name in sorted(required_pairs):
        plan_dir = geo_root / plan_name
        if not plan_dir.exists():
            raise FileNotFoundError(f"Missing plan geometry: {plan_dir}")
        adjacency, node_polys, edge_doors = load_plan_graph(plan_dir)
        path_records = []
        path_lookup = {}
        for start_node in sorted(node_polys, key=natural_node_key):
            paths = shortest_paths_from(adjacency, start_node)
            for end_node in sorted(node_polys, key=natural_node_key):
                if start_node == end_node or end_node not in paths:
                    continue
                path = paths[end_node]
                path_records.append(path)
                path_lookup[(start_node, end_node)] = path
        edge_scores, _ = compute_edge_bottlenecks(path_records, edge_doors)
        for start_node, end_node in sorted(required_pairs[plan_name]):
            path = path_lookup.get((start_node, end_node))
            if path is None:
                raise ValueError(f"No topology path for {plan_name}: {start_node} -> {end_node}")
            rows.append(
                route_row(
                    plan_name,
                    start_node,
                    end_node,
                    path,
                    node_polys,
                    edge_doors,
                    route_bottleneck_score(path, edge_scores),
                )
            )
    return rows


def load_config(config_path):
    config = read_json(config_path)
    if not config.get("features", {}).get("target"):
        raise ValueError(f"Config has no target columns: {config_path}")
    return config


def build_dataset(config_path, canonical_root, scenario_root, output_root):
    config = load_config(config_path)
    inventory, scenario_to_split, plan_to_split = canonical_inventory(canonical_root)
    canonical_scenarios = set(scenario_to_split)
    canonical_plans = set(plan_to_split)
    items_by_scenario, metadata_paths = metadata_items(
        scenario_root / "metadata",
        scenario_root / "dataswarm",
        canonical_scenarios,
        canonical_plans,
    )

    non_success = sorted(
        scenario_id
        for scenario_id, item in items_by_scenario.items()
        if str(item["status"]).lower() != "success"
    )
    if non_success:
        raise ValueError(f"Canonical scenarios not marked success: {non_success[:10]}")

    print(f"[Data_Estimate_2] Summarizing {len(items_by_scenario)} SQLite trajectories...")
    time_rows = [summarize_trajectory(items_by_scenario[key]) for key in sorted(items_by_scenario)]
    print(f"[Data_Estimate_2] Computing route features for {len(canonical_plans)} plans...")
    route_rows = required_route_rows(scenario_root / "geo", items_by_scenario)

    time_df = pd.DataFrame(time_rows)
    route_df = pd.DataFrame(route_rows)
    key_columns = ["plan", "start_node", "end_node"]
    merged = time_df.merge(route_df, on=key_columns, how="left", validate="many_to_one")
    missing_route = merged[merged["topology_path"].isna()]["scenario_id"].tolist()
    if missing_route:
        raise ValueError(f"Missing route features for {len(missing_route)} scenarios: {missing_route[:10]}")
    add_derived_features(merged)
    add_variant_columns(merged)
    feature_columns = feature_columns_from_config(merged, config)
    target_columns = list(config["features"]["target"])

    merged.insert(0, "split", merged["scenario_id"].map(scenario_to_split))
    if merged["split"].isna().any():
        raise ValueError("One or more rows have no canonical split")
    merged = merged.sort_values(["split", "plan", "route_index", "variant_id"]).reset_index(drop=True)
    dataset_columns = ["split"] + TIME_COLUMNS + [
        "topology_path",
        "topology_hop_distance",
        "topology_centerline_distance_m",
        "straight_distance_m",
        "number_of_rooms_between_A_B",
        "door_count_between_A_B",
        "min_door_width_between_A_B",
        "walkable_area_near_path",
        "bottleneck_score",
        "detour_ratio",
        "distance_gap_m",
        "agent_density_near_path",
        "area_per_agent",
        "door_pressure_per_agent",
        "variant_full",
        "variant_half",
        "variant_single",
    ]
    merged = merged[dataset_columns]

    output_root.mkdir(parents=True, exist_ok=False)
    split_manifest = {}
    for split_name in ("train", "val", "test"):
        frame = merged[merged["split"] == split_name].reset_index(drop=True)
        split_path = output_root / OUTPUT_DIRECTORIES[split_name] / "data_estimate.csv"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(split_path, index=False)
        split_manifest[split_name] = {
            "directory": OUTPUT_DIRECTORIES[split_name],
            "csv": str(split_path.resolve()),
            "rows": int(len(frame)),
            "plans": int(frame["plan"].nunique()),
            "canonical_image_scenarios": len(inventory[split_name]["scenario_ids"]),
            "canonical_image_plans": len(inventory[split_name]["plan_names"]),
            "sha256": sha256_file(split_path),
        }

    combined_path = output_root / "all_data_estimate.csv"
    merged.to_csv(combined_path, index=False)
    manifest = {
        "dataset_id": "data_estimate_2_housegan_canonical_imagebase_split_v1",
        "canonical_dataset_id": "housegan_canonical_imagebase_split_v1",
        "output_root": str(output_root.resolve()),
        "all_data_estimate_csv": str(combined_path.resolve()),
        "all_data_estimate_sha256": sha256_file(combined_path),
        "rows": int(len(merged)),
        "plans": int(merged["plan"].nunique()),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "split_source": str(canonical_root.resolve()),
        "split_source_inventory_sha256": canonical_inventory_hash(inventory),
        "sources": {
            "scenario_root": str(scenario_root.resolve()),
            "config": str(config_path.resolve()),
            "metadata_files": len(metadata_paths),
            "trajectory_sqlite_files": len(items_by_scenario),
            "geometry_plan_directories": len(canonical_plans),
        },
        "integrity": {
            "canonical_scenarios_matched": len(items_by_scenario),
            "canonical_scenarios_missing": 0,
            "canonical_scenarios_non_success": 0,
            "rows_missing_route_features": 0,
            "plan_split_overlap": 0,
            "scenario_split_overlap": 0,
        },
        "splits": split_manifest,
        "split_plan_names": {
            split_name: inventory[split_name]["plan_names"] for split_name in ("train", "val", "test")
        },
    }
    write_json(output_root / "data_estimate_manifest.json", manifest)
    print(f"[Data_Estimate_2] output={output_root.resolve()}")
    for split_name in ("train", "val", "test"):
        values = split_manifest[split_name]
        print(f"[Data_Estimate_2] {split_name}: rows={values['rows']} plans={values['plans']}")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Data_Estimate_2 using the canonical Image-based HouseGAN split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "AI_Estimate" / "AI_Train" / "Method_MLP_PyTorch" / "config_train.json",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=PROJECT_ROOT / "Dataset" / "Data_ImageUNet" / "DensityMap_dataset" / "Topo_HouseGAN" / "B",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "Dataset" / "Data_Estimate_2",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_dataset(
        args.config.resolve(),
        args.canonical_root.resolve(),
        args.scenario_root.resolve(),
        args.output_root.resolve(),
    )


if __name__ == "__main__":
    main()
