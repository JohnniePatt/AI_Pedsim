import argparse
import subprocess
import pathlib
import os
import json

def run_train(script_dir, train_script, config):
    print("\n========================================")
    print(f"      STEP 1: Starting Training ({train_script})")
    print("========================================")
    train_cmd = ["python", str(script_dir / train_script), "--config", config]
    
    try:
        subprocess.run(train_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False

def run_test(script_dir, test_script, run_path, test_config=None):
    print("\n========================================")
    print(f"      STEP 2: Starting Testing          ")
    print(f"      Script: {test_script}")
    print(f"      Target: {run_path.name}")
    print("========================================")
    
    test_cmd = [
        "python", 
        str(script_dir / test_script), 
        "--run_path", str(run_path)
    ]
    if test_config:
        test_cmd += ["--config", test_config]
        
    try:
        subprocess.run(test_cmd, check=True)
        print("\n[SUCCESS] Testing completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Testing failed with error: {e}")

def get_run_directories(script_dir):
    project_root = script_dir.parent.parent
    outputs_dir = project_root / "AI_Result" / "Method_pix2pixhd_No_D" / "outputs"
    
    if not outputs_dir.exists():
        return []
        
    run_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_HD_NoD_")]
    # Sort descending by modification time (newest first)
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return run_dirs

def guess_test_script(run_path):
    # Find any config files copied in run_path
    config_names = [p.name for p in run_path.glob("config_train*.json")]
    if config_names:
        cfg_name = config_names[0]
        if "bw" in cfg_name:
            return "test_pix2pixhd_NoD_densitymap_bw.py"
        else:
            return "test_pix2pixhd_NoD_densitymap_bw.py"
            
    return None

def main():
    script_dir = pathlib.Path(__file__).parent.resolve()
    
    print("\n=== AI Pedsim pix2pixhd_No_D Pipeline ===")
    print("1. Full run script (Train -> Test latest)")
    print("2. Test only (Select from existing runs)")
    
    try:
        choice = input("Please select an option (1 or 2): ").strip()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    # Gather train scripts
    train_scripts = sorted([p.name for p in script_dir.glob("train_pix2pixhd_NoD_*.py")])
    configs = sorted([p.name for p in script_dir.glob("config_train*.json")])
    if "config_train.json" in [p.name for p in script_dir.glob("config_train.json")]:
        configs.append("config_train.json")
    configs = sorted(list(set(configs)))

    if choice == "1":
        if not train_scripts:
            print("Error: No training scripts found.")
            return
            
        print("\n--- Available Training Scripts ---")
        for i, s in enumerate(train_scripts, start=1):
            print(f"[{i}] {s}")
        try:
            script_choice = input(f"Select a script (1-{len(train_scripts)}): ").strip()
            script_idx = int(script_choice) - 1
            if not (0 <= script_idx < len(train_scripts)):
                print("Invalid selection. Exiting.")
                return
            selected_script = train_scripts[script_idx]
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            return

        print("\n--- Available Configs ---")
        for i, c in enumerate(configs, start=1):
            print(f"[{i}] {c}")
        try:
            config_choice = input(f"Select a config (1-{len(configs)} or Enter to use default): ").strip()
            if config_choice == "":
                # Extract default config from script name or standard
                selected_config = "config_train.json"
            else:
                cfg_idx = int(config_choice) - 1
                if not (0 <= cfg_idx < len(configs)):
                    print("Invalid selection. Exiting.")
                    return
                selected_config = configs[cfg_idx]
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            return

        success = run_train(script_dir, selected_script, selected_config)
        if not success:
            return
            
        run_dirs = get_run_directories(script_dir)
        if not run_dirs:
            print("Error: No run directories found after training.")
            return
            
        latest_run = run_dirs[0]
        guessed_test = guess_test_script(latest_run)
        if not guessed_test:
            guessed_test = "test_pix2pixhd_NoD_densitymap_bw.py"
            
        # Match config_test for guessed script
        guessed_test_config = "config_test.json"
            
        run_test(script_dir, guessed_test, latest_run, guessed_test_config)
        
    elif choice == "2":
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
            if not (0 <= idx < len(run_dirs)):
                print("Invalid selection. Exiting.")
                return
            selected_run = run_dirs[idx]
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            return

        guessed_test = guess_test_script(selected_run)
        test_scripts = sorted([p.name for p in script_dir.glob("test_pix2pixhd_NoD_*.py")])

        if guessed_test and guessed_test in test_scripts:
            print(f"\nDetected matching test script: {guessed_test}")
            use_guessed = input("Use this script? (Y/n): ").strip().lower()
            if use_guessed == "n":
                guessed_test = None

        if not guessed_test:
            print("\n--- Available Test Scripts ---")
            for i, s in enumerate(test_scripts, start=1):
                print(f"[{i}] {s}")
            try:
                test_choice = input(f"Select a test script (1-{len(test_scripts)}): ").strip()
                t_idx = int(test_choice) - 1
                if 0 <= t_idx < len(test_scripts):
                    guessed_test = test_scripts[t_idx]
                else:
                    print("Invalid selection. Exiting.")
                    return
            except (ValueError, KeyboardInterrupt):
                print("\nCancelled.")
                return

        # Select test config
        test_configs = sorted([p.name for p in script_dir.glob("config_test*.json")])
        if "config_test.json" in [p.name for p in script_dir.glob("config_test.json")]:
            test_configs.append("config_test.json")
        test_configs = sorted(list(set(test_configs)))

        print("\n--- Available Test Configs ---")
        for i, c in enumerate(test_configs, start=1):
            print(f"[{i}] {c}")
        try:
            cfg_choice = input(f"Select a test config (1-{len(test_configs)} or Enter to skip/default): ").strip()
            if cfg_choice == "":
                selected_cfg = None
            else:
                cfg_idx = int(cfg_choice) - 1
                if 0 <= cfg_idx < len(test_configs):
                    selected_cfg = test_configs[cfg_idx]
                else:
                    print("Invalid selection. Exiting.")
                    return
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            return

        run_test(script_dir, guessed_test, selected_run, selected_cfg)
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.")

if __name__ == "__main__":
    main()
