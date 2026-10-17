
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

rng = np.random.default_rng(42)
n = 7 * 24 * 60
t = np.arange(n)
minute = t % 1440

daily = 30 + 15 * np.sin(
    2 * np.pi * (minute - 360) / 1440
)
cpu = np.clip(daily + rng.normal(0, 4, n), 5, 95)

for start in [500, 1800, 3100, 5200, 7200, 9000]:
    cpu[start:min(start + 20, n)] += 20

cpu = np.clip(cpu, 5, 98)
requests = np.clip(cpu * 8 + rng.normal(0, 10, n), 0, None)
memory = np.clip(20 + 0.55 * cpu + rng.normal(0, 3, n), 5, 95)

df = pd.DataFrame({
    "minute": t,
    "cpu": cpu,
    "requests_per_minute": requests,
    "memory": memory,
})

for lag in [1, 2, 5]:
    df[f"cpu_lag_{lag}"] = df["cpu"].shift(lag)

df["cpu_mean_5"] = df["cpu"].shift(1).rolling(5).mean()
df["minute_of_day"] = minute
df["target_cpu"] = df["cpu"].shift(-5)
df = df.dropna().reset_index(drop=True)

features = [
    "cpu", "cpu_lag_1", "cpu_lag_2", "cpu_lag_5",
    "cpu_mean_5", "requests_per_minute", "memory",
    "minute_of_day",
]

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(train[features], train["target_cpu"])
pred = np.clip(model.predict(test[features]), 0, 100)
actual = test["target_cpu"].to_numpy()

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

joblib.dump(
    {"model": model, "features": features},
    "models/workload_model.joblib",
)

pd.DataFrame({
    "actual_cpu": actual,
    "predicted_cpu": pred,
}).to_csv("results/predictions.csv", index=False)

print("Training complete.")
