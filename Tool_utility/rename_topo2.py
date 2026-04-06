import os
import shutil

def rename_topo2():
    root_dir = "."
    old_name = "Topo_2"
    new_name = "Topo_bottleneck"

    # Step 1: Content replacement (source code, configs)
    exclude_dirs = {".git", "Geo_scenario", ".gemini"}
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith((".py", ".json", ".txt", ".md", ".ipynb")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if old_name in content:
                        new_content = content.replace(old_name, new_name)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Updated content: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Step 2: Filename renaming
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if "Topo2" in file:
                old_file_path = os.path.join(root, file)
                new_file = file.replace("Topo2", new_name)
                new_file_path = os.path.join(root, new_file)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed file: {old_file_path} -> {new_file_path}")

    # Step 3: Directory renaming (Topo_bottleneck in Geo_scenario)
    old_dir = os.path.join("Geo_scenario", old_name)
    new_dir = os.path.join("Geo_scenario", new_name)
    if os.path.exists(old_dir):
        os.rename(old_dir, new_dir)
        print(f"Renamed directory: {old_dir} -> {new_dir}")
    else:
        print(f"Directory {old_dir} not found or already renamed.")

if __name__ == "__main__":
    rename_topo2()
