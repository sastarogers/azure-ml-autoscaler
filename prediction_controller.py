import os
import math
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from azure.identity import DefaultAzureCredential

MODEL_PATH = "models/workload_model.joblib"

MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "3"))
TARGET_CPU = float(os.getenv("TARGET_CPU", "60"))

RESOURCE_MANAGER_API = "https://management.azure.com"
CONTAINER_APPS_API_VERSION = "2025-07-01"

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
    Update only the Container App's template scale settings
    using Azure Resource Manager's REST API.
    """
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_RESOURCE_GROUP"]
    app_name = os.environ["AZURE_CONTAINER_APP_NAME"]

    min_replicas = max(0, MIN_REPLICAS)
    max_allowed = max(1, MAX_REPLICAS)
    recommended = max(min_replicas, min(max_allowed, int(replicas)))

    credential = DefaultAzureCredential()
    token = credential.get_token(
        "https://management.azure.com/.default"
    ).token

    url = (
        f"{RESOURCE_MANAGER_API}/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.App/containerApps/{app_name}"
    )

    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params={"api-version": CONTAINER_APPS_API_VERSION},
        timeout=30,
    )
    response.raise_for_status()
    app_data = response.json()

    properties = app_data.get("properties", {})
    template = properties.get("template", {})
    scale = template.get("scale", {})

    scale["minReplicas"] = min_replicas
    scale["maxReplicas"] = recommended
    template["scale"] = scale

    patch_response = requests.patch(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/merge-patch+json",
        },
        params={"api-version": CONTAINER_APPS_API_VERSION},
        json={
    "location": app_data["location"],
    "properties": {
        "template": {
            "scale": scale,
        }
    },
    },
        timeout=60,
    )
    patch_response.raise_for_status()

    return {
        "app_name": app_name,
        "min_replicas": min_replicas,
        "max_replicas": recommended,
        "status": "updated",
    }


if __name__ == "__main__":
    print(predict_from_demo_metrics())
