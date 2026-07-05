from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import joblib
import pandas as pd
import time
import psutil

app = Flask(__name__)

model = joblib.load("model.pkl")

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

PREDICTION_POSITIVE = Counter(
    "prediction_positive_total",
    "Total positive predictions"
)

CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "CPU Usage"
)

MEMORY_USAGE = Gauge(
    "memory_usage_percent",
    "Memory Usage"
)


@app.route("/predict", methods=["POST"])
def predict():

    start = time.time()

    data = pd.DataFrame([request.json])

    pred = model.predict(data)[0]

    REQUEST_COUNT.inc()

    if pred == 1:
        PREDICTION_POSITIVE.inc()

    REQUEST_LATENCY.observe(time.time() - start)

    CPU_USAGE.set(psutil.cpu_percent())

    MEMORY_USAGE.set(psutil.virtual_memory().percent)

    return jsonify({"prediction": int(pred)})


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": "text/plain"
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)