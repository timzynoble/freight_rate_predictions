import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import engineer_features

# Load training data
train_df = pd.read_csv("data/train-test.csv")
train_processed = engineer_features(train_df)

features_to_use = [
    "pickup", "delivery", "distance", "equipment", "weight",
    "market_index", "quote_signal", "month", "day", "day_of_week",
    "day_of_year", "is_weekend", "lat_diff", "lon_diff"
]
target_col = "posted_rate"
cat_features = ["pickup", "delivery", "equipment"]

# Chronological Split for Validation
train_mask = train_processed["date"] < "2025-09-01"
val_mask = train_processed["date"] >= "2025-09-01"

X_tr = train_processed.loc[train_mask, features_to_use]
y_tr = train_processed.loc[train_mask, target_col]
X_va = train_processed.loc[val_mask, features_to_use]
y_va = train_processed.loc[val_mask, target_col]

# Train Model
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    verbose=200
)

model.fit(
    X_tr, y_tr,
    eval_set=(X_va, y_va),
    cat_features=cat_features,
    early_stopping_rounds=50,
    use_best_model=True
)

val_preds = model.predict(X_va)
print(f"Validation MAE: ${mean_absolute_error(y_va, val_preds):.2f}")
print(f"Validation R²: {r2_score(y_va, val_preds):.4f}")