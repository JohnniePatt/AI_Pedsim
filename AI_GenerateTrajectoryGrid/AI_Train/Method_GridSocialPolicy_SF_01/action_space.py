from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

import pandas as pd

from path_utils import resolve_manifest_path


def load_json(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def default_offsets(limit: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for radius in range(1, 4):
        ring: list[tuple[int, int]] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if max(abs(dx), abs(dy)) == radius:
                    ring.append((dx, dy))
        ring.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], abs(p[0]) + abs(p[1]), p[0], p[1]))
        offsets.extend(ring)
        if len(offsets) >= limit:
            break
    return offsets[:limit]


def collect_grid_deltas(dataset_root: pathlib.Path, split: str, max_cases: int | None = None, frame_stride: int = 1) -> Counter:
    manifest_path = dataset_root / "manifest_trajectory_grid.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["split"] == split].reset_index(drop=True)
    if max_cases is not None:
        manifest = manifest.head(max_cases)

    counter: Counter = Counter()
    for _, row in manifest.iterrows():
        traj_path = resolve_manifest_path(dataset_root, row["target_dir"]) / "trajectory.parquet"
        if not traj_path.exists():
            continue
        traj = pd.read_parquet(traj_path, columns=["frame", "agent_id", "grid_x", "grid_y"])
        for _, agent_df in traj.groupby("agent_id", sort=False):
            agent_df = agent_df.sort_values("frame")
            stride = max(int(frame_stride), 1)
            if len(agent_df) <= stride:
                continue
            dx = agent_df["grid_x"].to_numpy()[stride:] - agent_df["grid_x"].to_numpy()[:-stride]
            dy = agent_df["grid_y"].to_numpy()[stride:] - agent_df["grid_y"].to_numpy()[:-stride]
            for x, y in zip(dx, dy):
                counter[(int(x), int(y))] += 1
    return counter


def build_action_space(
    dataset_root: pathlib.Path,
    output_path: pathlib.Path,
    movement_count: int = 20,
    split: str = "train",
    max_cases: int | None = None,
    frame_stride: int = 1,
) -> dict:
    counter = collect_grid_deltas(dataset_root, split=split, max_cases=max_cases, frame_stride=frame_stride)
    movement_offsets = [delta for delta, _ in counter.most_common() if delta != (0, 0)]
    if len(movement_offsets) < movement_count:
        seen = set(movement_offsets)
        for offset in default_offsets(movement_count):
            if offset not in seen:
                movement_offsets.append(offset)
                seen.add(offset)
            if len(movement_offsets) >= movement_count:
                break
    movement_offsets = movement_offsets[:movement_count]

    actions = []
    for idx, (dx, dy) in enumerate(movement_offsets):
        actions.append({"id": idx, "name": f"move_{dx}_{dy}", "dx": int(dx), "dy": int(dy), "kind": "move", "count": int(counter.get((dx, dy), 0))})

    wait_id = len(actions)
    actions.append({"id": wait_id, "name": "wait", "dx": 0, "dy": 0, "kind": "wait", "count": int(counter.get((0, 0), 0))})

    payload = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "source_split": split,
        "action_frame_stride": int(frame_stride),
        "movement_count": len(movement_offsets),
        "wait_action_id": wait_id,
        "stop_is_separate_head": True,
        "actions": actions,
    }
    write_json(output_path, payload)
    return payload


class ActionSpace:
    def __init__(self, payload: dict):
        self.payload = payload
        self.actions = list(payload["actions"])
        self.wait_action_id = int(payload["wait_action_id"])
        self.offset_to_id = {(int(a["dx"]), int(a["dy"])): int(a["id"]) for a in self.actions}
        self.offsets = [(int(a["dx"]), int(a["dy"])) for a in self.actions]

    @classmethod
    def load(cls, path: pathlib.Path) -> "ActionSpace":
        return cls(load_json(path))

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def action_id_for_delta(self, dx: int, dy: int) -> int:
        delta = (int(dx), int(dy))
        if delta in self.offset_to_id:
            return self.offset_to_id[delta]
        if delta == (0, 0):
            return self.wait_action_id
        best_id = self.wait_action_id
        best_dist = float("inf")
        for action in self.actions:
            if action["kind"] == "wait":
                continue
            adx, ady = int(action["dx"]), int(action["dy"])
            dist = (adx - dx) ** 2 + (ady - dy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_id = int(action["id"])
        return best_id

    def delta_for_action_id(self, action_id: int) -> tuple[int, int]:
        action = self.actions[int(action_id)]
        return int(action["dx"]), int(action["dy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grid movement action space from trajectory grid data.")
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--movement-count", type=int, default=20)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()

    payload = build_action_space(
        dataset_root=args.dataset_root.resolve(),
        output_path=args.output.resolve(),
        movement_count=args.movement_count,
        split=args.split,
        max_cases=args.max_cases,
        frame_stride=args.frame_stride,
    )
    print(f"[ActionSpace] wrote {args.output}")
    for action in payload["actions"]:
        print(f"  {action['id']:02d}: {action['name']} count={action['count']}")


if __name__ == "__main__":
    main()
