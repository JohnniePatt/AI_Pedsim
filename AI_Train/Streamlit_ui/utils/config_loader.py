import os
import json
import pathlib

def get_available_methods(ai_train_dir):
    """
    Scans the AI_Train directory for subfolders starting with 'Method_'.
    Returns a list of method names.
    """
    path = pathlib.Path(ai_train_dir)
    methods = [d.name for d in path.iterdir() if d.is_dir() and d.name.startswith("Method_")]
    return sorted(methods)

def get_method_config_path(method_path, config_type="train"):
    """
    Returns the full path to the config file based on type.
    """
    filename = "config_train.json" if config_type == "train" else "config_test.json"
    full_path = pathlib.Path(method_path) / filename
    if not full_path.exists():
        # Fallback for older folders that haven't been split yet
        fallback = pathlib.Path(method_path) / "config_active.json"
        if fallback.exists(): return fallback
    return full_path

def load_config(method_path, config_type="train"):
    """
    Loads requested config from the given method directory.
    Returns a dictionary of parameters.
    """
    config_file = get_method_config_path(method_path, config_type)
    if config_file.exists():
        with open(config_file, "r") as f:
            return json.load(f)
    return {}

def save_config(method_path, config_data, config_type="train"):
    """
    Saves the given config_data as config_train.json or config_test.json.
    """
    filename = "config_train.json" if config_type == "train" else "config_test.json"
    config_file = pathlib.Path(method_path) / filename
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=4)

def get_method_runs(method_path):
    """
    Scans for 'runs_' or 'runs' folder within the method directory 
    and returns a list of individual run folders.
    """
    path = pathlib.Path(method_path)
    # Search for directories that might contain runs
    run_containers = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("runs")]
    
    all_runs = []
    for container in run_containers:
        runs = [r.name for r in container.iterdir() if r.is_dir()]
        all_runs.extend([f"{container.name}/{r}" for r in runs])
    
    return sorted(all_runs, reverse=True)
