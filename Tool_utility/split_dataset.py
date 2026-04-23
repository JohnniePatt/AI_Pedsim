import os
import shutil
import random
import argparse
import pathlib

def get_group_key(folder_name):
    """
    Heuristic to extract a grouping key from case folder names.
    Goal: Group different variants/seeds of the same plan to avoid data leakage.
    For HouseGAN: case_plan_100_8ec0_42_00_full -> plan_100_8ec0
    """
    if not folder_name.startswith("case_"):
        return folder_name
    
    name = folder_name[5:] # remove 'case_'
    
    # HouseGAN pattern: plan_{id}_{hash}_{seed}_{route}_{variant}
    if name.startswith("plan_"):
        parts = name.split("_")
        # plan_100_8ec0 -> parts=['plan', '100', '8ec0']
        if len(parts) >= 3:
            return "_".join(parts[:3])
            
    return folder_name


def split_dataset(source_dir, train_ratio, test_ratio, val_ratio, seed=42):
    """
    Split case_ folders within a dataset directory into train, test, and val subfolders.
    Supports re-splitting by collecting cases from existing split folders.
    Groups cases by plan name to prevent data leakage.
    """
    source_path = pathlib.Path(source_dir)
    if not source_path.exists():
        print(f"Error: {source_dir} does not exist.")
        return

    # 1. Collect all case directories
    potential_locations = [source_path] + [source_path / s for s in ["train", "test", "val", "validation"]]
    
    all_cases = []
    case_names = set()
    
    for loc in potential_locations:
        if loc.exists() and loc.is_dir():
            for d in loc.iterdir():
                if d.is_dir() and d.name.startswith("case_") and d.name not in case_names:
                    all_cases.append(d)
                    case_names.add(d.name)
    
    if not all_cases:
        print(f"No case_ directories found in {source_dir} or its subfolders.")
        return

    print(f"Found {len(all_cases)} cases in total.")

    # 2. Group cases by plan to avoid leakage
    groups = {}
    for case_path in all_cases:
        key = get_group_key(case_path.name)
        if key not in groups:
            groups[key] = []
        groups[key].append(case_path)
    
    group_keys = list(groups.keys())
    print(f"Grouped into {len(group_keys)} unique plans/groups.")

    # 3. Shuffle and split GROUPS
    random.seed(seed)
    random.shuffle(group_keys)

    total_groups = len(group_keys)
    train_count = int(total_groups * train_ratio)
    test_count = int(total_groups * test_ratio)
    # Ensure at least some in each if enough groups exist
    if train_count == 0 and total_groups > 0: train_count = 1
    
    train_keys = group_keys[:train_count]
    test_keys = group_keys[train_count:train_count + test_count]
    val_keys = group_keys[train_count + test_count:]

    def flatten_groups(keys):
        return [case for k in keys for case in groups[k]]

    splits = {
        "train": flatten_groups(train_keys),
        "test": flatten_groups(test_keys),
        "val": flatten_groups(val_keys)
    }


    # 4. Create target folders and move
    for split_name, cases in splits.items():
        target_dir = source_path / split_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {split_name} ({len(cases)} cases)...")
        for case_path in cases:
            dest_path = target_dir / case_path.name
            
            # Skip if it's already in the right place
            if case_path.resolve() == dest_path.resolve():
                continue
                
            # Move
            try:
                # If target exists, remove it first to avoid collision
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.move(str(case_path), str(dest_path))
            except Exception as e:
                print(f"Error moving {case_path.name} to {split_name}: {e}")

    # 5. Cleanup
    old_val_path = source_path / "validation"
    if old_val_path.exists() and old_val_path.is_dir():
        if not any(old_val_path.iterdir()):
            old_val_path.rmdir()
            print("Removed empty 'validation' folder.")

    print("\n--- Split Summary ---")
    total_all = len(all_cases)
    print(f"Total Cases:  {total_all}")
    print(f"Total Groups: {len(group_keys)}")
    print(f"Train: {len(splits['train'])} cases ({len(train_keys)} groups)")
    print(f"Test:  {len(splits['test'])} cases ({len(test_keys)} groups)")
    print(f"Val:   {len(splits['val'])} cases ({len(val_keys)} groups)")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset cases into train, test, and val.")
    parser.add_argument("--source", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--train", type=float, default=0.7, help="Ratio for train set (0-1)")
    parser.add_argument("--test", type=float, default=0.2, help="Ratio for test set (0-1)")
    parser.add_argument("--val", type=float, default=0.1, help="Ratio for val set (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Simple normalization
    total = args.train + args.test + args.val
    if abs(total - 1.0) > 0.001:
        args.train /= total
        args.test /= total
        args.val /= total

    split_dataset(args.source, args.train, args.test, args.val, args.seed)
