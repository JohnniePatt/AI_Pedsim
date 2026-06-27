import pandas as pd
import os

plan_conditions_path = "/home/johnnie/programming/AI_Pedsim/AI_Pedsim/Document_Research/Output_FrameworkResearch/Synthetic_Dataset/plan_conditions.csv"
time_summary_path = "/home/johnnie/programming/AI_Pedsim/AI_Pedsim/Geo_scenario/Topo_HouseGAN/time_summary/all_route_time_summary.csv"
output_csv_path = "/home/johnnie/programming/AI_Pedsim/AI_Pedsim/Document_Research/Output_FrameworkResearch/Synthetic_Dataset/plan_sample_conditions.csv"

# Load the plan conditions
df_cond = pd.read_csv(plan_conditions_path)

# Load the time summary
df_time = pd.read_csv(time_summary_path)

# Calculate total simulation time for each plan (sum of simulation_duration_s)
plan_times = df_time.groupby("plan")["simulation_duration_s"].sum().reset_index()
plan_times.rename(columns={"plan": "plan_name", "simulation_duration_s": "sample_simulation_time_s"}, inplace=True)

# Merge condition and time data
df_merged = pd.merge(df_cond, plan_times, on="plan_name", how="inner")

# Get counts per condition
condition_counts = df_cond["condition"].value_counts().to_dict()

# Select 3 samples per condition (sorted alphabetically to be deterministic)
df_merged = df_merged.sort_values(by=["condition", "plan_name"])

samples = []
for cond, count in condition_counts.items():
    cond_df = df_merged[df_merged["condition"] == cond]
    # Take the first 3 samples
    cond_samples = cond_df.head(3).copy()
    cond_samples["total_layouts_in_condition"] = count
    cond_samples["Time of simulation with traditional method"] = cond_samples["sample_simulation_time_s"] * count
    samples.append(cond_samples)

# Combine all samples
df_output = pd.concat(samples, ignore_index=True)

# Save to CSV
df_output.to_csv(output_csv_path, index=False)
print(f"Generated CSV with {len(df_output)} rows.")
print(df_output)
