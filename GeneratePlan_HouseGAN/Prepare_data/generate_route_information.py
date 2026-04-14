import argparse
import csv
import json
import math
import shutil
from collections import deque
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union


DEFAULT_DOOR_WIDTH_M = 1.5

CSV_COLUMNS = [
    "plan",
    "start_node",
    "end_node",
    "topology_path",
    "topology_hop_distance",
    "topology_centerline_distance_m",
    "straight_distance_m",
    "number_of_rooms_between_A_B",
    "door_count_between_A_B",
    "min_door_width_between_A_B",
    "walkable_area_near_path",
    "bottleneck_score",
]


def read_json(path, fallback=None):
    if not path.exists():
        return fallback
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def natural_node_key(node_name):
    prefix, _, value = str(node_name).partition("-")
    try:
        index = int(value)
    except ValueError:
        index = 10**9
    return (prefix, index, str(node_name))


def polygon_from_coords(coords):
    polygon = Polygon(coords)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def load_polygons(path):
    return [polygon_from_coords(coords) for coords in read_json(path, [])]


def edge_key(node_a, node_b):
    return tuple(sorted((str(node_a), str(node_b))))


def node_index_from_name(node_name):
    _, _, value = str(node_name).partition("-")
    return int(value)


def load_node_polygons(plan_dir, graph_nodes):
    room_polys = load_polygons(plan_dir / "geo_room.json")
    corridor_polys = load_polygons(plan_dir / "geo_corridor.json")
    node_polys = {}

    if graph_nodes:
        for node in graph_nodes:
            name = node.get("name")
            if not name:
                continue
            try:
                index = node_index_from_name(name)
            except ValueError:
                continue
            if name.startswith("Room-") and index < len(room_polys):
                node_polys[name] = room_polys[index]
            elif name.startswith("Cor-") and index < len(corridor_polys):
                node_polys[name] = corridor_polys[index]
        return node_polys

    for index, polygon in enumerate(corridor_polys):
        node_polys[f"Cor-{index}"] = polygon
    for index, polygon in enumerate(room_polys):
        node_polys[f"Room-{index}"] = polygon
    return node_polys


def load_plan_graph(plan_dir):
    graph = read_json(plan_dir / "topological_graph.json", {}) or {}
    graph_nodes = graph.get("nodes", [])
    node_id_to_name = {
        node.get("id"): node.get("name")
        for node in graph_nodes
        if node.get("id") is not None and node.get("name")
    }
    node_polys = load_node_polygons(plan_dir, graph_nodes)
    adjacency = {name: set() for name in node_polys}

    for edge in graph.get("edges", []):
        node_a = node_id_to_name.get(edge.get("from"), edge.get("from"))
        node_b = node_id_to_name.get(edge.get("to"), edge.get("to"))
        if node_a in adjacency and node_b in adjacency:
            adjacency[node_a].add(node_b)
            adjacency[node_b].add(node_a)

    doors = read_json(plan_dir / "geo_door.json", []) or []
    edge_doors = {}
    for door in doors:
        rooms = door.get("rooms", [])
        if len(rooms) != 2:
            continue
        node_a, node_b = rooms
        if node_a not in adjacency:
            adjacency[node_a] = set()
        if node_b not in adjacency:
            adjacency[node_b] = set()
        adjacency[node_a].add(node_b)
        adjacency[node_b].add(node_a)
        edge_doors.setdefault(edge_key(node_a, node_b), []).append(door)

    return adjacency, node_polys, edge_doors


def shortest_paths_from(adjacency, start_node):
    queue = deque([start_node])
    paths = {start_node: [start_node]}
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency.get(node, []), key=natural_node_key):
            if neighbor in paths:
                continue
            paths[neighbor] = paths[node] + [neighbor]
            queue.append(neighbor)
    return paths


def centroid_xy(polygon):
    centroid = polygon.centroid
    return float(centroid.x), float(centroid.y)


def distance_between(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def door_width(door):
    for key in ("door_width", "width", "width_m"):
        if key in door:
            try:
                return float(door[key])
            except (TypeError, ValueError):
                pass
    return DEFAULT_DOOR_WIDTH_M


def path_centerline_distance(path, node_polys):
    distance = 0.0
    for node_a, node_b in zip(path, path[1:]):
        if node_a not in node_polys or node_b not in node_polys:
            continue
        distance += distance_between(centroid_xy(node_polys[node_a]), centroid_xy(node_polys[node_b]))
    return distance


def edge_capacity_width(edge, edge_doors):
    doors = edge_doors.get(edge, [])
    widths = [door_width(door) for door in doors]
    if widths:
        return sum(widths) / len(widths)
    return DEFAULT_DOOR_WIDTH_M


def compute_edge_bottlenecks(paths, edge_doors):
    edge_loads = {}
    for path in paths:
        for node_a, node_b in zip(path, path[1:]):
            key = edge_key(node_a, node_b)
            edge_loads[key] = edge_loads.get(key, 0) + 1

    edge_pressures = {}
    for edge, load in edge_loads.items():
        capacity_width = edge_capacity_width(edge, edge_doors)
        edge_pressures[edge] = load / max(capacity_width, 0.1)

    max_pressure = max(edge_pressures.values(), default=0.0)
    edge_scores = {
        edge: (pressure / max_pressure if max_pressure > 0 else 0.0)
        for edge, pressure in edge_pressures.items()
    }
    edge_details = []
    for edge in sorted(edge_loads, key=lambda item: (edge_scores[item], item), reverse=True):
        node_a, node_b = edge
        edge_details.append(
            {
                "edge": [node_a, node_b],
                "shortest_path_load": edge_loads[edge],
                "capacity_width_m": round(edge_capacity_width(edge, edge_doors), 4),
                "load_per_width": round(edge_pressures[edge], 6),
                "bottleneck_score": round(edge_scores[edge], 6),
            }
        )
    return edge_scores, edge_details


def route_bottleneck_score(path, edge_scores):
    scores = [edge_scores.get(edge_key(node_a, node_b), 0.0) for node_a, node_b in zip(path, path[1:])]
    return max(scores, default=0.0)


def route_row(plan_name, start_node, end_node, path, node_polys, edge_doors, bottleneck_score):
    start_center = centroid_xy(node_polys[start_node])
    end_center = centroid_xy(node_polys[end_node])
    door_count = 0
    door_widths = []

    for node_a, node_b in zip(path, path[1:]):
        doors = edge_doors.get(edge_key(node_a, node_b), [])
        if doors:
            door_count += len(doors)
            door_widths.extend(door_width(door) for door in doors)
        else:
            door_count += 1
            door_widths.append(DEFAULT_DOOR_WIDTH_M)

    path_polygons = [node_polys[node] for node in path if node in node_polys]
    path_area = float(unary_union(path_polygons).area) if path_polygons else 0.0
    min_width = min(door_widths) if door_widths else 0.0
    centerline_distance = path_centerline_distance(path, node_polys)

    return {
        "plan": plan_name,
        "start_node": start_node,
        "end_node": end_node,
        "topology_path": " -> ".join(path),
        "topology_hop_distance": max(0, len(path) - 1),
        "topology_centerline_distance_m": round(centerline_distance, 4),
        "straight_distance_m": round(distance_between(start_center, end_center), 4),
        "number_of_rooms_between_A_B": sum(1 for node in path[1:-1] if str(node).startswith("Room-")),
        "door_count_between_A_B": door_count,
        "min_door_width_between_A_B": round(min_width, 4),
        "walkable_area_near_path": round(path_area, 4),
        "bottleneck_score": round(bottleneck_score, 6),
    }


def build_plan_route_information(plan_dir, output_root):
    plan_name = plan_dir.name
    adjacency, node_polys, edge_doors = load_plan_graph(plan_dir)
    route_nodes = sorted(node_polys, key=natural_node_key)
    rows = []
    topology_routes = []
    path_records = []

    for start_node in route_nodes:
        paths = shortest_paths_from(adjacency, start_node)
        for end_node in route_nodes:
            if start_node == end_node or end_node not in paths:
                continue
            path_records.append((start_node, end_node, paths[end_node]))

    edge_scores, edge_bottlenecks = compute_edge_bottlenecks([path for _, _, path in path_records], edge_doors)

    for start_node, end_node, path in path_records:
        bottleneck_score = route_bottleneck_score(path, edge_scores)
        row = route_row(plan_name, start_node, end_node, path, node_polys, edge_doors, bottleneck_score)
        rows.append(row)
        topology_routes.append(
            {
                "start_node": start_node,
                "end_node": end_node,
                "path": path,
                "topology_hop_distance": row["topology_hop_distance"],
                "topology_shortest_distance_m": row["topology_centerline_distance_m"],
                "topology_centerline_distance_m": row["topology_centerline_distance_m"],
                "straight_distance_m": row["straight_distance_m"],
                "bottleneck_score": row["bottleneck_score"],
            }
        )

    plan_output = output_root / plan_name
    plan_output.mkdir(parents=True, exist_ok=True)
    write_csv(plan_output / "route_information.csv", rows)
    write_json(
        plan_output / "topology_shortest_distance.json",
        {
            "plan": plan_name,
            "route_count": len(rows),
            "edge_bottlenecks": edge_bottlenecks,
            "routes": topology_routes,
        },
    )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def list_plan_dirs(scenario_root):
    geo_root = scenario_root / "geo"
    if not geo_root.exists():
        return []
    return sorted([path for path in geo_root.iterdir() if path.is_dir()], key=lambda path: path.name)


def clean_output_root(output_root, scenario_root):
    output_root = Path(output_root).resolve()
    scenario_root = Path(scenario_root).resolve()
    if output_root.name != "route_information" or output_root.parent != scenario_root:
        raise ValueError(f"Refusing to clean unexpected output directory: {output_root}")
    if output_root.exists():
        shutil.rmtree(output_root)


def build_route_information(scenario_root, plan_names=None, output_root=None, clean_output=False):
    scenario_root = Path(scenario_root)
    output_root = Path(output_root) if output_root else scenario_root / "route_information"
    if clean_output:
        clean_output_root(output_root, scenario_root)
    selected = set(plan_names or [])
    plan_dirs = [path for path in list_plan_dirs(scenario_root) if not selected or path.name in selected]
    all_rows = []
    skipped = []

    for plan_dir in plan_dirs:
        try:
            rows = build_plan_route_information(plan_dir, output_root)
        except Exception as exc:
            skipped.append({"plan": plan_dir.name, "error": str(exc)})
            print(f"[RouteInformation][Skip] {plan_dir}: {type(exc).__name__}: {exc}")
            continue
        all_rows.extend(rows)

    write_csv(output_root / "all_route_information.csv", all_rows)
    write_json(
        output_root / "route_information_manifest.json",
        {
            "plans_requested": len(plan_dirs),
            "plans_with_route_information": len({row["plan"] for row in all_rows}),
            "route_rows": len(all_rows),
            "output_root": str(output_root),
            "bottleneck_score_formula": "max_normalized(edge_shortest_path_load / edge_capacity_width_m) along the route",
            "bottleneck_score_inputs": [
                "all room-to-room shortest topology paths in the same plan",
                "door_width from geo_door.json",
            ],
            "skipped": skipped,
        },
    )
    return output_root, all_rows, skipped


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    default_scenario_root = project_root / "Geo_scenario" / "Topo_HouseGAN"
    parser = argparse.ArgumentParser(description="Generate route-level topology features for HouseGAN plans.")
    parser.add_argument("--scenario-root", type=Path, default=default_scenario_root)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--plan", action="append", default=None, help="Plan name. Repeat to build multiple plans.")
    parser.add_argument("--clean", action="store_true", help="Clean route_information before writing this run.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root, rows, skipped = build_route_information(args.scenario_root, args.plan, args.output_root, args.clean)
    print(f"[RouteInformation] Built {len(rows)} route rows into {output_root}")
    if skipped:
        print(f"[RouteInformation] Skipped {len(skipped)} plans")


if __name__ == "__main__":
    main()
