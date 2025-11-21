import pandas as pd
from pathlib import Path

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Loading raw csv data into pandas dataframe.
    
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at:{path}")
    
    dataframe = pd.read_csv(path)
    return dataframe
