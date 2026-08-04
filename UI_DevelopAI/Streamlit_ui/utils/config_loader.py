import os
import json
import pathlib

def get_available_methods(ai_train_dir):
    """
    Scans the AI_Train directory for subfolders starting with 'Method_'.
    Returns a list of method names.
    """
    path = pathlib.Path(ai_train_dir)
    if not path.exists():
        return []
    methods = [d.name for d in path.iterdir() if d.is_dir() and (d.name.startswith("Method_") or d.name == "Generate_HouseGAN")]
    return sorted(methods)

def get_method_config_path(method_path, config_type="train", config_name=None):
    """
    Returns the full path to the config file based on type.
    """
    filename = config_name if config_name else ("config_train.json" if config_type == "train" else "config_test.json")
    full_path = pathlib.Path(method_path) / filename
    if not full_path.exists():
        if config_type == "train":
            fallback = pathlib.Path(method_path) / "config_active.json"
            if fallback.exists():
                return fallback
    return full_path

def load_config(method_path, config_type="train", config_name=None):
    """
    Loads requested config from the given method directory.
    Returns a dictionary of parameters.
    """
    config_file = get_method_config_path(method_path, config_type, config_name=config_name)
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {}

def save_config(method_path, config_data, config_type="train", config_name=None):
    """
    Saves the given config_data as config_train.json or config_test.json.
    """
    filename = config_name if config_name else ("config_train.json" if config_type == "train" else "config_test.json")
    config_file = pathlib.Path(method_path) / filename
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=4)

def get_method_runs(method_path):
    """
    Scans for run folders within method directory (outputs/run_* and direct run_* folders).
    Returns a list of individual run folder relative paths.
    """
    path = pathlib.Path(method_path)
    if not path.exists():
        return []

    runs = []
    outputs_dir = path / "outputs"
    if outputs_dir.exists() and outputs_dir.is_dir():
        for r in outputs_dir.iterdir():
            if r.is_dir() and (r.name.startswith("run") or r.name.endswith("_evaluate")):
                runs.append(f"outputs/{r.name}")

    for r in path.iterdir():
        if r.is_dir() and r.name != "outputs" and (r.name.startswith("run") or r.name.endswith("_evaluate")):
            runs.append(r.name)

    return sorted(list(set(runs)), reverse=True)
