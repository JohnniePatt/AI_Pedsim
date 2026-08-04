"""Shared normalized rollout renderer used by the time-series gallery.

This module only renders trajectories supplied by a method.  It never creates,
repairs, or substitutes trajectories, which keeps model provenance separate
from the visual normalization step.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon
from shapely.ops import unary_union


AXES_COLOR = "#101820"
WALKABLE_COLOR = "#f3f6f8"


def load_walkable(case_dir: pathlib.Path):
    polygons = []
    for name in ("Geo_room.json", "Geo_corridor.json"):
        path = case_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as file:
            polygons.extend(Polygon(coords) for coords in json.load(file) if len(coords) >= 3)
    if not polygons:
        raise FileNotFoundError(f"No room/corridor geometry in {case_dir}")
    return unary_union(polygons)


def plot_normalized_rollout(
    case_dir: str | pathlib.Path,
    trajectories: list[np.ndarray],
    output_path: str | pathlib.Path,
    title: str,
):
    """Render predicted trajectories only, using the shared gallery contract."""
    case_dir = pathlib.Path(case_dir)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    room_json = case_dir / "Geo_room.json"
    corridor_json = case_dir / "Geo_corridor.json"
    door_json = case_dir / "Geo_door.json"

    walkable = load_walkable(case_dir)
    min_x, min_y, max_x, max_y = walkable.bounds
    width, height = max(max_x - min_x, 1e-6), max(max_y - min_y, 1e-6)
    aspect = max(width / height, 0.25)
    fig_width = min(18.0, max(7.0, 7.0 * aspect))
    fig_height = min(12.0, max(4.5, fig_width / aspect))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    layout_x: list[float] = []
    layout_y: list[float] = []

    def draw_polygons(path: pathlib.Path):
        if not path.exists():
            return
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        for poly_coords in data:
            if not isinstance(poly_coords, list) or len(poly_coords) < 3:
                continue
            coords = np.asarray(poly_coords, dtype=np.float64)
            layout_x.extend(coords[:, 0].tolist())
            layout_y.extend(coords[:, 1].tolist())
            ax.add_patch(
                MplPolygon(
                    coords,
                    closed=True,
                    facecolor=WALKABLE_COLOR,
                    edgecolor=AXES_COLOR,
                    linewidth=2.0,
                    zorder=1,
                )
            )

    draw_polygons(room_json)
    draw_polygons(corridor_json)

    # Erase the wall stroke at each doorway so doors are true visual voids.
    if door_json.exists():
        with door_json.open(encoding="utf-8") as file:
            doors = json.load(file)
        for door in doors:
            pos = door["pos"]
            door_width = float(door.get("door_width", 1.5))
            dw, dh = (door_width, 0.18) if door.get("horizontal", False) else (0.18, door_width)
            ax.add_patch(
                Rectangle(
                    (pos[0] - dw / 2, pos[1] - dh / 2),
                    dw,
                    dh,
                    facecolor=WALKABLE_COLOR,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=2,
                )
            )

    exit_files = sorted(case_dir.glob("Spawn_exit_*.csv"))
    if exit_files:
        exit_df = pd.read_csv(exit_files[0])
        exit_rows = exit_df[exit_df["type"] == "exit_area"]
        if not exit_rows.empty:
            exit_poly = shapely_wkt.loads(exit_rows.iloc[0]["area"])
            if exit_poly.geom_type == "Polygon":
                coords = np.asarray(exit_poly.exterior.coords)
                ax.add_patch(
                    MplPolygon(
                        coords,
                        closed=True,
                        facecolor="#f59e0b",
                        edgecolor="#f97316",
                        linewidth=1.3,
                        alpha=0.35,
                        zorder=3,
                        label="exit room",
                    )
                )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    spawn_label_pending = True
    for idx, raw_points in enumerate(trajectories):
        points = np.asarray(raw_points, dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)] if points.ndim == 2 else np.empty((0, 2))
        if len(points) == 0:
            continue
        ax.scatter(
            points[0, 0],
            points[0, 1],
            s=18,
            c="#22c55e",
            edgecolors="#052e16",
            linewidths=0.4,
            zorder=5,
            label="spawn" if spawn_label_pending else None,
        )
        spawn_label_pending = False
        ax.plot(points[:, 0], points[:, 1], color=colors[idx % len(colors)], linewidth=1.2, alpha=0.78, zorder=4)

    ax.set_aspect("equal", adjustable="box")
    dx, dy = max(width * 0.04, 0.5), max(height * 0.04, 0.5)
    ax.set_xlim(min_x - dx, max_x + dx)
    ax.set_ylim(min_y - dy, max_y + dy)
    bg_x0, bg_x1 = ax.get_xlim()
    bg_y0, bg_y1 = ax.get_ylim()
    ax.add_patch(
        Rectangle(
            (bg_x0, bg_y0),
            bg_x1 - bg_x0,
            bg_y1 - bg_y0,
            facecolor=AXES_COLOR,
            edgecolor="none",
            zorder=-10,
        )
    )
    ax.set_xlim(bg_x0, bg_x1)
    ax.set_ylim(bg_y0, bg_y1)
    ax.set_title(title)
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

