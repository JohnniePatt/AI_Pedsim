import os
import shutil
import random
import argparse
import pathlib

def split_dataset(source_dir, train_ratio, test_ratio, val_ratio, seed=42):
    """
    Split case_ folders within a dataset directory into train, test, and val subfolders.
    Supports re-splitting by collecting cases from existing split folders.
    """
    source_path = pathlib.Path(source_dir)
    if not source_path.exists():
        print(f"Error: {source_dir} does not exist.")
        return

    # 1. Collect all case directories
    # We look in the root, and in existing train/test/val folders
    # Note: Use standard 'val' instead of 'validation' as per user request
    potential_locations = [source_path] + [source_path / s for s in ["train", "test", "val", "validation"]]
    
    all_cases = []
    case_names = set() # To avoid duplicates if same named folder exists twice (shouldn't)
    
    for loc in potential_locations:
        if loc.exists() and loc.is_dir():
            # Find all directories that start with 'case_'
            for d in loc.iterdir():
                if d.is_dir() and d.name.startswith("case_") and d.name not in case_names:
                    all_cases.append(d)
                    case_names.add(d.name)
    
    if not all_cases:
        print(f"No case_ directories found in {source_dir} or its subfolders.")
        return

    print(f"Found {len(all_cases)} cases in total. Preparing to split...")

    # 2. Shuffle
    random.seed(seed)
    random.shuffle(all_cases)

    # 3. Calculate splits
    total_count = len(all_cases)
    train_count = int(total_count * train_ratio)
    test_count = int(total_count * test_ratio)
    val_count = total_count - train_count - test_count

    splits = {
        "train": all_cases[:train_count],
        "test": all_cases[train_count:train_count + test_count],
        "val": all_cases[train_count + test_count:]
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
    print(f"Total: {total_count}")
    print(f"Train: {train_count} ({train_ratio*100:.1f}%)")
    print(f"Test:  {test_count} ({test_ratio*100:.1f}%)")
    print(f"Val:   {val_count} ({val_ratio*100:.1f}%)")
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
