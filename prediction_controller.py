
import os
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from azure.identity import DefaultAzureCredential

MODEL_PATH = "models/workload_model.joblib"

MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "3"))
TARGET_CPU = float(os.getenv("TARGET_CPU", "60"))

# Load the trained model.
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]


def desired_replicas(predicted_cpu):
    """Recommend replicas based on predicted CPU utilization."""
    count = math.ceil(max(0.0, predicted_cpu) / TARGET_CPU)
    return max(MIN_REPLICAS, min(MAX_REPLICAS, count))


def predict_from_demo_metrics():
    """
    Demo only: uses simulated metrics, not live Azure telemetry.
    Predict CPU utilization five minutes ahead.
    """
    now = datetime.now()
    minute_of_day = now.hour * 60 + now.minute

    rng = np.random.default_rng()

    cpu_samples = np.clip(
        30
        + 15 * np.sin(
            2 * np.pi * (minute_of_day - 360) / 1440
        )
        + rng.normal(0, 4, 6),
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


def apply_replica_count(replicas: int) -> dict:
    """
    Update the Container App's configured replica limits.

    Requires Azure environment variables and a working Azure identity.
    This changes configuration; it does not guarantee an exact live
    replica count.
    """
    from azure.mgmt.appcontainers import ContainerAppsAPIClient

    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_RESOURCE_GROUP"]
    app_name = os.environ["AZURE_CONTAINER_APP_NAME"]

    max_allowed = max(1, MAX_REPLICAS)
    recommended = max(1, min(max_allowed, int(replicas)))

    credential = DefaultAzureCredential()
    client = ContainerAppsAPIClient(
        credential,
        subscription_id,
    )

    app = client.container_apps.get(resource_group, app_name)

    app.template.scale.min_replicas = 1
    app.template.scale.max_replicas = recommended

    client.container_apps.begin_update(
        resource_group,
        app_name,
        app,
    ).result()

    return {
        "app_name": app_name,
        "min_replicas": 1,
        "max_replicas": recommended,
        "status": "updated",
    }


if __name__ == "__main__":
    print(predict_from_demo_metrics())
