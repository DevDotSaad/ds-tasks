import pandas as pd
import numpy as np
from typing import Tuple

REQUIRED_COLUMNS = [
    "year",
    "headcount_ratio_national",
    "unemployment_rate_pct",
    "population_millions",
    "gdp_growth_rate_pct",
    "inflation_rate_pct",
]

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    present = [c for c in REQUIRED_COLUMNS if c in df.columns]
    df = df[present].copy()
    df = df.drop_duplicates(subset=["year"]).sort_values("year").reset_index(drop=True)
    for c in df.columns:
        if c != "year":
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].interpolate().bfill().ffill()
    df["year"] = df["year"].astype(int)
    return df

def add_features(df: pd.DataFrame, target_col: str = "headcount_ratio_national", nlags: int = 2) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["t"] = out["year"] - out["year"].min()
    for L in range(1, nlags+1):
        out[f"{target_col}_lag{L}"] = out[target_col].shift(L)
    if "gdp_growth_rate_pct" in out.columns and "inflation_rate_pct" in out.columns:
        out["real_growth_proxy"] = out["gdp_growth_rate_pct"] - out["inflation_rate_pct"]
    for c in out.columns:
        if c not in [target_col]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def train_test_split_by_year(df: pd.DataFrame, test_years: int = 4):
    df = df.sort_values("year").reset_index(drop=True)
    cutoff = df["year"].max() - test_years
    train = df[df["year"] <= cutoff].copy().reset_index(drop=True)
    test  = df[df["year"] > cutoff].copy().reset_index(drop=True)
    return train, test

def design_matrix(df_feat: pd.DataFrame, target_col: str = "headcount_ratio_national", nlags: int = 2):
    cols = []
    base = ["t"]
    for c in ["unemployment_rate_pct", "population_millions", "gdp_growth_rate_pct", "inflation_rate_pct", "real_growth_proxy"]:
        if c in df_feat.columns:
            base.append(c)
    lag_cols = [f"{target_col}_lag{L}" for L in range(1, nlags+1) if f"{target_col}_lag{L}" in df_feat.columns]
    cols = base + lag_cols
    X = df_feat[cols].copy()
    y = df_feat[target_col].copy()
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
