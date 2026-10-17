import hashlib
import os
import time
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({
        "project": "Azure Predictive Autoscaler",
        "status": "running",
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


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
