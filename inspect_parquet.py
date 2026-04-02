import pandas as pd
import sys

file = 'Geo_scenario/Topo_2/dataswarm_parquet/test/double-botteleneck_100801_trajectory_data.parquet'
try:
    df = pd.read_parquet(file)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Frames range: {df['frame_id'].min()} - {df['frame_id'].max()}")
    print(f"Unique frames: {len(df['frame_id'].unique())}")
    print(f"Unique agents: {len(df['id'].unique())}")
except Exception as e:
    print(f"Error: {e}")
