import os
import shutil
import pathlib

PROJECT_ROOT = pathlib.Path("/home/johnfaqpc/programming/AI_Pedsim")
BASE_ROOT = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN"

def remigrate():
    # Get all plans from the BASE_ROOT (excluding known category folders if they exist as folders)
    know_categories = ["geo", "dataswarm", "heatmap_density", "heatmap_speed", "spawn_exit", "trajectory_line", "dataswarm_parquet"]
    
    plans = [d.name for d in BASE_ROOT.iterdir() if d.is_dir() and d.name not in know_categories]
    
    print(f"Found {len(plans)} plans to remigrate.")
    
    for plan in sorted(plans):
        plan_path = BASE_ROOT / plan
        print(f"Processing {plan}...")
        
        # For each subfolder in the plan_path, move its content to BASE_ROOT / subfolder / plan
        for item in plan_path.iterdir():
            if item.is_dir():
                dest_cat_dir = BASE_ROOT / item.name / plan
                dest_cat_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"  Moving {item.name} contents...")
                for f in item.iterdir():
                    shutil.move(str(f), str(dest_cat_dir / f.name))
                
                # Remove the empty item folder
                item.rmdir()
            else:
                # If there are files directly in plan_path (though unexpected), handle them
                # For safety, let's just leave them or move to a generic folder if needed.
                # In our case, we mostly expect subfolders.
                pass
        
        # Remove the now empty plan folder
        plan_path.rmdir()

if __name__ == "__main__":
    remigrate()
    print("Remigration completed successfully.")
