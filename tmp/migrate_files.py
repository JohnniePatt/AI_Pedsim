import os
import shutil
import pathlib

PROJECT_ROOT = pathlib.Path("/home/johnfaqpc/programming/AI_Pedsim")
OLD_GEO_ROOT = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN" / "geo"
OLD_OUTPUTS_ROOT = PROJECT_ROOT / "Prepare_data" / "Architecture_housePlan" / "outputs"
NEW_ROOT = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN"

def migrate():
    # Get all plans from both sources
    plans = set()
    if OLD_GEO_ROOT.exists():
        plans.update([d.name for d in OLD_GEO_ROOT.iterdir() if d.is_dir()])
    if OLD_OUTPUTS_ROOT.exists():
        plans.update([d.name for d in OLD_OUTPUTS_ROOT.iterdir() if d.is_dir()])
        
    print(f"Found {len(plans)} plans to migrate.")
    
    for plan in sorted(plans):
        plan_dir = NEW_ROOT / plan
        plan_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Migrate Geometry
        old_geo_plan_dir = OLD_GEO_ROOT / plan
        if old_geo_plan_dir.exists():
            new_geo_dir = plan_dir / "geo"
            new_geo_dir.mkdir(parents=True, exist_ok=True)
            print(f"Moving geometry for {plan}...")
            for f in old_geo_plan_dir.iterdir():
                shutil.move(str(f), str(new_geo_dir / f.name))
            old_geo_plan_dir.rmdir()
            
        # 2. Migrate Outputs
        old_output_plan_dir = OLD_OUTPUTS_ROOT / plan
        if old_output_plan_dir.exists():
            print(f"Moving outputs for {plan}...")
            for d in old_output_plan_dir.iterdir():
                if d.is_dir():
                    shutil.move(str(d), str(plan_dir / d.name))
                else:
                    shutil.move(str(d), str(plan_dir / d.name))
            old_output_plan_dir.rmdir()
            
    # Cleanup OLD_GEO_ROOT and OLD_OUTPUTS_ROOT if empty
    if OLD_GEO_ROOT.exists() and not any(OLD_GEO_ROOT.iterdir()):
        OLD_GEO_ROOT.rmdir()
        print(f"Removed empty directory: {OLD_GEO_ROOT}")
    if OLD_OUTPUTS_ROOT.exists() and not any(OLD_OUTPUTS_ROOT.iterdir()):
        OLD_OUTPUTS_ROOT.rmdir()
        print(f"Removed empty directory: {OLD_OUTPUTS_ROOT}")

if __name__ == "__main__":
    migrate()
    print("Migration completed successfully.")
