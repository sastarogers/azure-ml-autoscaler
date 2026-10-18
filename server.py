
import hashlib
import hmac
import os
import time

from flask import Flask, jsonify, request

from prediction_controller import (
    predict_from_demo_metrics,
    apply_replica_count,
)

app = Flask(__name__)


def valid_scaling_token():
    expected_token = os.getenv("SCALING_API_TOKEN")
    provided_token = request.headers.get("X-Scaling-Token", "")

    return bool(expected_token) and hmac.compare_digest(
        provided_token, expected_token
    )


@app.get("/")
def home():
    return jsonify({
        "project": "Azure Predictive Autoscaler",
        "status": "running",
        "endpoints": [
            "/health",
            "/work",
            "/predict",
            "/scale",
            "/auto-scale",
        ],
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.get("/work")
def work():
    iterations = request.args.get(
        "iterations", default=10000, type=int
    )
    iterations = max(0, min(iterations or 10000, 100000))

    start = time.perf_counter()
    value = b"azure-workload"

    for _ in range(iterations):
        value = hashlib.sha256(value).digest()

    return jsonify({
        "latency_ms": round(
            (time.perf_counter() - start) * 1000, 2
        ),
        "iterations": iterations,
    })


@app.get("/predict")
def predict():
    try:
        return jsonify(predict_from_demo_metrics())
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check application logs."
        }), 500


@app.post("/scale")
def scale():
    if not valid_scaling_token():
        if not os.getenv("SCALING_API_TOKEN"):
            return jsonify({
                "error": "Scaling endpoint is not configured"
            }), 503
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json(silent=True) or {}
        replicas = data.get("replicas")

        if isinstance(replicas, bool) or not isinstance(replicas, int):
            return jsonify({
                "error": "replicas must be an integer"
            }), 400

        max_allowed = max(1, int(os.getenv("MAX_REPLICAS", "3")))

        if not 1 <= replicas <= max_allowed:
            return jsonify({
                "error": f"replicas must be between 1 and {max_allowed}"
            }), 400

        result = apply_replica_count(replicas)
        return jsonify(result), 200

    except (ValueError, TypeError):
        return jsonify({"error": "Invalid replica configuration"}), 400
    except Exception:
        app.logger.exception("Scaling request failed")
        return jsonify({
            "error": "Scaling request failed. Check application logs."
        }), 500


@app.post("/auto-scale")
def auto_scale():
    if not valid_scaling_token():
        if not os.getenv("SCALING_API_TOKEN"):
            return jsonify({
                "error": "Scaling endpoint is not configured"
            }), 503
        return jsonify({"error": "Unauthorized"}), 401

    try:
        prediction = predict_from_demo_metrics()
        replicas = prediction["recommended_replicas"]
        scaling = apply_replica_count(replicas)

        return jsonify({
            "prediction": prediction,
            "scaling": scaling,
            "status": "success",
        }), 200

    except Exception:
        app.logger.exception("Automatic scaling failed")
        return jsonify({
            "error": "Automatic scaling failed. Check application logs."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
