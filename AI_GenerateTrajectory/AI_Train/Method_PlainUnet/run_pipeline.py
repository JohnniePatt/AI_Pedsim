import argparse
import subprocess
import pathlib
import os

def run_train(script_dir, config):
    print("\n========================================")
    print("      STEP 1: Starting Training         ")
    print("========================================")
    train_cmd = ["python", str(script_dir / "train_PlainUnet_densitymap.py"), "--config", config]
    
    try:
        subprocess.run(train_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False

def run_test(script_dir, run_path):
    print("\n========================================")
    print(f"      STEP 2: Starting Testing          ")
    print(f"      Target: {run_path.name}")
    print("========================================")
    
    test_cmd = [
        "python", 
        str(script_dir / "test_PlainUnet_densitymap.py"), 
        "--run_path", str(run_path),
        "--checkpoint", "best_loss.pt"
    ]
    
    try:
        subprocess.run(test_cmd, check=True)
        print("\n[SUCCESS] Testing completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Testing failed with error: {e}")

def get_run_directories(script_dir):
    project_root = script_dir.parent.parent
    outputs_dir = project_root / "AI_Result" / "Method_PlainUnet" / "outputs"
    
    if not outputs_dir.exists():
        return []
        
    run_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_PlainUNet_")]
    # Sort descending by modification time (newest first)
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return run_dirs

def main():
    parser = argparse.ArgumentParser(description="Run PlainUNet Pipeline")
    parser.add_argument("--config", type=str, default="config_train.json")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    
    print("\n=== AI Pedsim PlainUNet Pipeline ===")
    print("1. Full run script (Train -> Test latest)")
    print("2. Test only (Select from existing runs)")
    
    try:
        choice = input("Please select an option (1 or 2): ").strip()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    if choice == "1":
        # Full run
        success = run_train(script_dir, args.config)
        if not success:
            return
            
        run_dirs = get_run_directories(script_dir)
        if not run_dirs:
            print("Error: No run directories found after training.")
            return
            
        latest_run = run_dirs[0] # Because it's sorted descending
        run_test(script_dir, latest_run)
        
    elif choice == "2":
        # Test only
        run_dirs = get_run_directories(script_dir)
        if not run_dirs:
            print("Error: No run directories found. Please train a model first.")
            return
            
        print("\n--- Available Runs (Newest first) ---")
        for i, d in enumerate(run_dirs, start=1):
            print(f"[{i}] {d.name}")
            
        try:
            run_choice = input(f"\nSelect a run to test (1-{len(run_dirs)}): ").strip()
            idx = int(run_choice) - 1
            if 0 <= idx < len(run_dirs):
                selected_run = run_dirs[idx]
                run_test(script_dir, selected_run)
            else:
                print("Invalid selection. Exiting.")
        except ValueError:
            print("Invalid input. Please enter a number. Exiting.")
        except KeyboardInterrupt:
            print("\nCancelled.")
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.")

if __name__ == "__main__":
    main()
