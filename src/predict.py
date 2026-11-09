import pandas as pd
from catboost import CatBoostRegressor
from preprocess import engineer_features

# 1. Load data
train_df = pd.read_csv("data/train-test.csv")
val_df = pd.read_csv("data/validation.csv")
december_raw = pd.read_csv("data/december-chart-inputs.csv")

# 2. Process data
train_processed = engineer_features(train_df)
val_processed = engineer_features(val_df)
december_processed = engineer_features(december_raw)

features_to_use = [
    "pickup", "delivery", "distance", "equipment", "weight",
    "market_index", "quote_signal", "month", "day", "day_of_week",
    "day_of_year", "is_weekend", "lat_diff", "lon_diff"
]
cat_features = ["pickup", "delivery", "equipment"]

# 3. Retrain on 100% of historical data
X_full = train_processed[features_to_use]
y_full = train_processed["posted_rate"]

final_model = CatBoostRegressor(
    iterations=330,  # Optimal iteration count from validation run
    learning_rate=0.05,
    depth=6,
    loss_function="MAE",
    random_seed=42,
    verbose=False
)
final_model.fit(X_full, y_full, cat_features=cat_features)

# 4. Generate Validation Predictions
val_predictions = final_model.predict(val_processed[features_to_use])
validation_output = pd.DataFrame({
    "load_id": val_df["load_id"],
    "predicted_rate": val_predictions
})
validation_output.to_csv("outputs/validation_predictions.csv", index=False)
print("Saved outputs/validation_predictions.csv successfully!")

# 5. Generate December Predictions
for feature in features_to_use:
    if feature not in december_processed.columns:
        december_processed[feature] = 0.0

december_predictions = final_model.predict(december_processed[features_to_use])
december_raw["predicted_rate"] = december_predictions
december_raw.to_csv("outputs/december-chart-inputs.csv", index=False)
print("Saved outputs/december-chart-inputs.csv successfully!")