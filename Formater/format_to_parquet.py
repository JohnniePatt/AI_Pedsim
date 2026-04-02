import os
import sqlite3
import pandas as pd
import pathlib
import argparse
from tqdm import tqdm

def format_sqlite_to_parquet(source_dir, output_dir, table_filter=None):
    """
    Converts SQLite files in dataswarm layout (train/test/validation) to Parquet.
    """
    source_path = pathlib.Path(source_dir)
    output_path = pathlib.Path(output_dir)
    
    # Check for subdirectories (train, test, validation)
    splits = ["train", "test", "validation"]
    
    for split in splits:
        split_src = source_path / split
        if not split_src.exists():
            continue
            
        print(f"📊 Processing split: {split}")
        split_out = output_path / split
        split_out.mkdir(parents=True, exist_ok=True)
        
        sqlite_files = list(split_src.glob("*.sqlite"))
        if not sqlite_files:
            continue
            
        for sqlite_file in tqdm(sqlite_files, desc=f"Converting {split}"):
            try:
                # Connect to SQLite
                conn = sqlite3.connect(sqlite_file)
                
                # Check for tables
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall() if row[0] != "sqlite_sequence"]
                
                if not tables:
                    conn.close()
                    continue
                    
                for table_name in tables:
                    # Table filtering logic
                    if table_filter and table_filter.lower() not in table_name.lower():
                        # If a filter is provided, skip tables that don't match
                        continue
                        
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    
                    # If multiple tables exist or we are filtering, we include table name in filename
                    parquet_name = f"{sqlite_file.stem}_{table_name}.parquet"
                        
                    df.to_parquet(split_out / parquet_name, engine='pyarrow', index=False)
                    
                conn.close()
            except Exception as e:
                print(f"❌ Error converting {sqlite_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pedsim SQLite files to Parquet")
    parser.add_argument("--source", type=str, required=True, help="Source dataswarm directory")
    parser.add_argument("--output", type=str, required=True, help="Output dataswarm_parquet directory")
    parser.add_argument("--filter", type=str, default=None, help="Only convert tables containing this string (e.g. trajectory_data)")
    
    args = parser.parse_args()
    
    format_sqlite_to_parquet(args.source, args.output, args.filter)
    print("✅ Formatting complete!")
