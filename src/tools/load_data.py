import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "condor_real_data.csv"

def load_condor_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("Year").reset_index(drop=True)
    return df
