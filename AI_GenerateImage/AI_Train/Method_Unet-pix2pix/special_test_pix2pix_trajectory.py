"""
special_test_pix2pix_trajectory.py
----------------------------------
Run a single-case special test for Method_Unet-pix2pix.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from test_pix2pix_trajectoryLine import GeneratorNetwork

GPT_METHOD_DIR = pathlib.Path(__file__).resolve().parents[1] / "Method_GPT_Knowledge"
if str(GPT_METHOD_DIR) not in sys.path:
    sys.path.append(str(GPT_METHOD_DIR))

from prepare_geometry_gpt_knowledge import load_scene, load_trajectory  # noqa: E402


def _iter_polygons(geom) -> Iterable:
    if geom is None:
        return []
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if not g.is_empty]
    return [geom]


def _world_to_px(x: float, y: float, minx: float, miny: float, height: int, grid_size: float) -> tuple[int, int]:
    px = int(round((x - minx) / grid_size))
    py = height - 1 - int(round((y - miny) / grid_size))
    return px, py


def _draw_geom(draw: ImageDraw.ImageDraw, geom, fill_rgb: tuple[int, int, int], minx: float, miny: float, height: int, grid_size: float):
    for poly in _iter_polygons(geom):
        ext = [_world_to_px(x, y, minx, miny, height, grid_size) for x, y in poly.exterior.coords]
        draw.polygon(ext, fill=fill_rgb)
        for interior in poly.interiors:
            hole = [_world_to_px(x, y, minx, miny, height, grid_size) for x, y in interior.coords]
            draw.polygon(hole, fill=(0, 0, 0))


def render_input_target(case_dir: pathlib.Path, grid_size: float = 0.5) -> tuple[Image.Image, Image.Image]:
    scene = load_scene(case_dir)
    gt_df = load_trajectory(case_dir)
    walkable = scene["walkable"]
    minx, miny, maxx, maxy = walkable.bounds
    width = max(8, int(np.ceil((maxx - minx) / grid_size)) + 1)
    height = max(8, int(np.ceil((maxy - miny) / grid_size)) + 1)

    input_img = Image.new("RGB", (width, height), (0, 0, 0))
    input_draw = ImageDraw.Draw(input_img)
    _draw_geom(input_draw, walkable, (255, 0, 0), minx, miny, height, grid_size)
    _draw_geom(input_draw, scene["spawn_polygon"], (0, 255, 0), minx, miny, height, grid_size)
    _draw_geom(input_draw, scene["exit_polygon"], (0, 0, 255), minx, miny, height, grid_size)

    target_img = Image.new("RGB", (width, height), (0, 0, 0))
    target_draw = ImageDraw.Draw(target_img)
    _draw_geom(target_draw, walkable, (220, 220, 220), minx, miny, height, grid_size)

    line_color = (255, 150, 180)
    for _, g in gt_df.groupby("id"):
        pts = [_world_to_px(float(r.pos_x), float(r.pos_y), minx, miny, height, grid_size) for r in g.itertuples(index=False)]
        if len(pts) >= 2:
            target_draw.line(pts, fill=line_color, width=1)

    return input_img, target_img


def _nearest_multiple_of_32(v: int) -> int:
    return max(256, ((int(v) + 31) // 32) * 32)


def load_generator(run_path: pathlib.Path, device: torch.device) -> tuple[GeneratorNetwork, pathlib.Path]:
    run_path = run_path.resolve()
    checkpoints = []
    if run_path.is_file() and run_path.suffix == ".pth":
        checkpoints = [run_path]
    else:
        checkpoints.extend([
            run_path / "checkpoints" / "generator_best.pth",
            run_path / "generator_best.pth",
        ])
        checkpoints.extend(sorted((run_path / "checkpoints").glob("*.pth")) if (run_path / "checkpoints").exists() else [])

    ckpt = next((p for p in checkpoints if p.exists()), None)
    if ckpt is None:
        raise FileNotFoundError(f"No pix2pix checkpoint found in {run_path}")

    net = GeneratorNetwork(3, 3).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    return net, ckpt


def infer_one(generator: GeneratorNetwork, input_img: Image.Image, target_img: Image.Image, device: torch.device) -> tuple[np.ndarray, dict]:
    ow, oh = input_img.size
    tw = _nearest_multiple_of_32(ow)
    th = _nearest_multiple_of_32(oh)

    input_resized = input_img.resize((tw, th), Image.BICUBIC)
    target_resized = target_img.resize((tw, th), Image.BICUBIC)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    x = to_tensor(input_resized).unsqueeze(0).to(device)
    y = to_tensor(target_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        y_hat = generator(x)

    mae = torch.mean(torch.abs(y_hat - y)).item()
    mse = torch.mean((y_hat - y) ** 2).item()
    rmse = float(np.sqrt(mse))

    pred = ((y_hat[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    pred_img = Image.fromarray(pred).resize((ow, oh), Image.BICUBIC)
    pred_np = np.asarray(pred_img, dtype=np.uint8)
    return pred_np, {"mae_l1": float(mae), "mse": float(mse), "rmse": float(rmse)}


def main(run_path: str, input_case_dir: str, output_dir: str, plan_key: str | None = None):
    case_dir = pathlib.Path(input_case_dir).resolve()
    out_dir = pathlib.Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_img, target_img = render_input_target(case_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, ckpt_path = load_generator(pathlib.Path(run_path), device)
    pred_np, metrics = infer_one(generator, input_img, target_img, device)
    pred_img = Image.fromarray(pred_np)

    input_img.save(out_dir / "special_input.png")
    target_img.save(out_dir / "special_target.png")
    pred_img.save(out_dir / "special_prediction.png")

    comp = Image.new("RGB", (input_img.width * 3, input_img.height))
    comp.paste(input_img, (0, 0))
    comp.paste(target_img, (input_img.width, 0))
    comp.paste(pred_img, (input_img.width * 2, 0))
    comp.save(out_dir / "special_compare.png")

    case_name = case_dir.name
    case_id = re.sub(r"^case_", "", case_name)
    summary = {
        "method": "Method_Unet-pix2pix",
        "case_id": case_id,
        "case_dir": str(case_dir),
        "plan_key": str(plan_key or ""),
        "checkpoint": str(ckpt_path),
        "device": str(device),
        **metrics,
    }
    with open(out_dir / "special_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "special_metrics.csv", "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k in ["mae_l1", "mse", "rmse"]:
            f.write(f"{k},{summary[k]:.8f}\n")

    print(f"[SpecialTest] Method_Unet-pix2pix case={case_id}")
    print(f"[SpecialTest] Output dir: {out_dir}")
    print(f"[SpecialTest] Checkpoint: {ckpt_path}")
    print(f"[SpecialTest] MAE={summary['mae_l1']:.6f} MSE={summary['mse']:.6f} RMSE={summary['rmse']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--input_case_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--plan_key", type=str, default=None)
    args = parser.parse_args()
    main(args.run_path, args.input_case_dir, args.output_dir, args.plan_key)

