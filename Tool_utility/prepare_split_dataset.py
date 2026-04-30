import argparse
import csv
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


GROUP_RE = re.compile(r"^(plan_[^_]+_[^_]+)(?:__|_)")


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float
    test_ratio: float
    val_ratio: float
    seed: int
    mode: str


def extract_group_key(filename: str) -> str:
    stem = Path(filename).stem
    match = GROUP_RE.match(stem)
    if match:
        return match.group(1)
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "plan":
        return "_".join(parts[:3])
    return stem


def normalize_ratios(train: float, test: float, val: float) -> tuple[float, float, float]:
    total = train + test + val
    if total <= 0:
        raise ValueError("train/test/validation ratios must sum to a positive value.")
    return train / total, test / total, val / total


def ensure_clean_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists (use --overwrite): {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def collect_side_files(side_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in side_dir.rglob("*.png"):
        if p.is_file():
            if p.name in files and files[p.name] != p:
                raise RuntimeError(
                    f"Duplicate filename detected in {side_dir}: '{p.name}'\n"
                    f"- {files[p.name]}\n- {p}\n"
                    "Please ensure unique filenames before splitting."
                )
            files[p.name] = p
    return files


def collect_common_files(a_dir: Path, b_dir: Path) -> tuple[dict[str, Path], dict[str, Path], list[str]]:
    a_files = collect_side_files(a_dir)
    b_files = collect_side_files(b_dir)

    only_a = sorted(set(a_files) - set(b_files))
    only_b = sorted(set(b_files) - set(a_files))
    if only_a:
        print(f"[WARN] {len(only_a)} files exist only in A. They will be skipped.")
    if only_b:
        print(f"[WARN] {len(only_b)} files exist only in B. They will be skipped.")

    common = sorted(set(a_files) & set(b_files))
    if not common:
        raise RuntimeError("No paired PNG filenames found in A and B.")
    return a_files, b_files, common


def build_group_map(filenames: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name in filenames:
        key = extract_group_key(name)
        groups.setdefault(key, []).append(name)
    return groups


def split_group_keys(
    group_keys: list[str],
    train_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    shuffled = list(group_keys)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)
    n_val = n - n_train - n_test

    train_keys = shuffled[:n_train]
    test_keys = shuffled[n_train:n_train + n_test]
    validation_keys = shuffled[n_train + n_test:n_train + n_test + n_val]

    return {"train": train_keys, "test": test_keys, "validation": validation_keys}


def copy_or_move(src: Path, dst: Path, mode: str) -> None:
    if src.resolve() == dst.resolve():
        return
    if mode == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def prepare_split_dataset(source_root: Path, output_root: Path, config: SplitConfig, overwrite: bool) -> None:
    a_dir = source_root / "A"
    b_dir = source_root / "B"
    if not a_dir.exists() or not b_dir.exists():
        raise FileNotFoundError(f"Expected A and B folders under: {source_root}")

    train_r, test_r, val_r = normalize_ratios(config.train_ratio, config.test_ratio, config.val_ratio)
    inplace = source_root.resolve() == output_root.resolve()

    if inplace:
        output_root.mkdir(parents=True, exist_ok=True)
        if config.mode != "move":
            raise ValueError("In-place split requires --mode move.")
    else:
        ensure_clean_output(output_root, overwrite=overwrite)

    for side in ("A", "B"):
        for split in ("train", "test", "validation"):
            (output_root / side / split).mkdir(parents=True, exist_ok=True)

    a_files, b_files, common_names = collect_common_files(a_dir, b_dir)
    group_map = build_group_map(common_names)
    split_keys = split_group_keys(list(group_map.keys()), train_r, test_r, config.seed)

    manifest_rows: list[dict[str, str]] = []
    split_file_counts = {"train": 0, "test": 0, "validation": 0}

    for split_name in ("train", "test", "validation"):
        keys = split_keys[split_name]
        for key in keys:
            filenames = sorted(group_map[key])
            for fname in filenames:
                src_a = a_files[fname]
                src_b = b_files[fname]
                dst_a = output_root / "A" / split_name / fname
                dst_b = output_root / "B" / split_name / fname
                copy_or_move(src_a, dst_a, config.mode)
                copy_or_move(src_b, dst_b, config.mode)
                split_file_counts[split_name] += 1
                manifest_rows.append(
                    {
                        "filename": fname,
                        "group_key": key,
                        "split": split_name,
                    }
                )

    manifest_path = output_root / "split_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "group_key", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("=" * 70)
    print(f"source_root : {source_root}")
    print(f"output_root : {output_root}")
    print(f"mode        : {config.mode}")
    print(f"inplace     : {inplace}")
    print(f"seed        : {config.seed}")
    print(f"ratios      : train={train_r:.4f}, test={test_r:.4f}, val={val_r:.4f}")
    print("-" * 70)
    print(f"total_pairs : {len(common_names)}")
    print(f"groups      : {len(group_map)}")
    print(
        "group_split : "
        f"train={len(split_keys['train'])}, "
        f"test={len(split_keys['test'])}, "
        f"validation={len(split_keys['validation'])}"
    )
    print(
        "file_split  : "
        f"train={split_file_counts['train']}, "
        f"test={split_file_counts['test']}, "
        f"validation={split_file_counts['validation']}"
    )
    print(f"manifest    : {manifest_path}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split flat paired PNG dataset (A/B) into train/test/validation by group key "
            "'plan_<code1>_<uniqueID>' with group-level shuffling."
        )
    )
    parser.add_argument("--source_root", type=Path, required=True, help="Directory containing A/ and B/ flat PNGs.")
    parser.add_argument("--output_root", type=Path, required=True, help="Output directory for split dataset.")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio.")
    parser.add_argument("--test", type=float, default=0.2, help="Test ratio.")
    parser.add_argument("--val", type=float, default=0.1, help="Val ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed at group level.")
    parser.add_argument("--mode", choices=["copy", "move"], default="copy", help="Copy or move files to output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_root if it exists.")
    args = parser.parse_args()

    cfg = SplitConfig(
        train_ratio=args.train,
        test_ratio=args.test,
        val_ratio=args.val,
        seed=args.seed,
        mode=args.mode,
    )
    prepare_split_dataset(
        source_root=args.source_root.resolve(),
        output_root=args.output_root.resolve(),
        config=cfg,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
