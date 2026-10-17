import os
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

MODEL_PATH = "models/workload_model.joblib"
MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "3"))
TARGET_CPU = float(os.getenv("TARGET_CPU", "60"))

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]


def desired_replicas(predicted_cpu):
    count = math.ceil(max(0.0, predicted_cpu) / TARGET_CPU)
    return max(MIN_REPLICAS, min(MAX_REPLICAS, count))


def predict_from_demo_metrics():
    """
    Demo only: these are simulated current metrics, not Azure telemetry.
    Replace this function with real metrics before claiming live prediction.
    """
    now = datetime.now()
    minute_of_day = now.hour * 60 + now.minute

    # Keep recent CPU samples to provide the lag features.
    rng = np.random.default_rng()
    cpu_samples = np.clip(
        30 + 15 * np.sin(
            2 * np.pi * (minute_of_day - 360) / 1440
        ) + rng.normal(0, 4, 6),
        5,
        98,
    )

    current_cpu = float(cpu_samples[-1])
    row = {
        "cpu": current_cpu,
        "cpu_lag_1": float(cpu_samples[-2]),
        "cpu_lag_2": float(cpu_samples[-3]),
        "cpu_lag_5": float(cpu_samples[-6]),
        "cpu_mean_5": float(np.mean(cpu_samples[-6:-1])),
        "requests_per_minute": max(0.0, current_cpu * 8),
        "memory": float(np.clip(20 + 0.55 * current_cpu, 5, 95)),
        "minute_of_day": minute_of_day,
    }

    X = pd.DataFrame([row])[features]
    prediction = float(np.clip(model.predict(X)[0], 0, 100))
    replicas = desired_replicas(prediction)

    return {
        "input_type": "SIMULATED_DEMO_METRICS",
        "predicted_cpu_5min_ahead": round(prediction, 2),
        "recommended_replicas": replicas,
        "limits": {
            "min_replicas": MIN_REPLICAS,
            "max_replicas": MAX_REPLICAS,
        },
    }


if __name__ == "__main__":
    print(predict_from_demo_metrics())
