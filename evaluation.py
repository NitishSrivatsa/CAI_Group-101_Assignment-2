from typing import List, Dict
import pandas as pd
from utils_finance import softmax

def evaluate_rows(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "raw_score" in df.columns:
        df["confidence"] = softmax(df["raw_score"].fillna(0).tolist())
    if "time_s" in df.columns:
        df["time_s"] = df["time_s"].round(3)
    cols = ["question","method","answer","confidence","time_s","correct"]
    return df[[c for c in cols if c in df.columns]]
