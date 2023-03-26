from pathlib import Path
import pandas as pd


def combine_parquet(data_path: str = "data/"):
    data_dir = Path(data_path)
    full_df = pd.concat(
        pd.read_parquet(parquet_file) for parquet_file in data_dir.glob("*.parquet")
    )
    return full_df
