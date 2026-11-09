import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw load data into feature-rich tabular data.
    Applied identically across training, validation, and December datasets.
    """
    feat_df = df.copy()
    
    # 1. Parse dates and extract temporal features
    feat_df["date"] = pd.to_datetime(feat_df["date"], errors="coerce")
    feat_df["year"] = feat_df["date"].dt.year
    feat_df["month"] = feat_df["date"].dt.month
    feat_df["day"] = feat_df["date"].dt.day
    feat_df["day_of_week"] = feat_df["date"].dt.dayofweek
    feat_df["day_of_year"] = feat_df["date"].dt.dayofyear
    feat_df["is_weekend"] = feat_df["day_of_week"].isin([5, 6]).astype(int)
    
    # 2. Geographic / Coordinate Differences
    if {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"}.issubset(feat_df.columns):
        feat_df["lat_diff"] = feat_df["delivery_lat"] - feat_df["pickup_lat"]
        feat_df["lon_diff"] = feat_df["delivery_lon"] - feat_df["pickup_lon"]
    else:
        feat_df["lat_diff"] = 0.0
        feat_df["lon_diff"] = 0.0

    # 3. Handle missing values for numerical columns
    numerical_cols = ["distance", "weight", "market_index", "quote_signal", "lat_diff", "lon_diff"]
    for col in numerical_cols:
        if col in feat_df.columns:
            feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")
            feat_df[col] = feat_df[col].fillna(feat_df[col].median() if not feat_df[col].isna().all() else 0)

    # 4. Standardize categorical columns
    categorical_cols = ["pickup", "delivery", "equipment"]
    for col in categorical_cols:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].fillna("Unknown").astype(str)

    return feat_df