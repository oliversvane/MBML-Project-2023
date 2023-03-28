from pathlib import Path
import pandas as pd
from tqdm import tqdm


def combine_parquet(data_path: str = "data/") -> pd.DataFrame:
    """
    Reads all parquet files from a single directory, and returns a combined dataframe.

    Parameters:
    -----------------
    data_path: str [Optional]
        directory of data folder containing parquet files
    
        
    Returns:
    -----------------
    full_df: pd.DataFrame
        A combined dataframe of all the parquet files
    
    """
    data_dir = Path(data_path)
    full_df = pd.concat(
        pd.read_parquet(parquet_file) for parquet_file in data_dir.glob("*.parquet")
    )
    return full_df

