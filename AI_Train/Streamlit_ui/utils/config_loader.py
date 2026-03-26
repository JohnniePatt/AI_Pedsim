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

def load_config(method_path):
    """
    Loads config_active.json from the given method directory.
    Returns a dictionary of parameters.
    """
    config_file = pathlib.Path(method_path) / "config_active.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            return json.load(f)
    return {}

def save_config(method_path, config_data):
    """
    Saves the given config_data as config_active.json in the method directory.
    """
    config_file = pathlib.Path(method_path) / "config_active.json"
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
