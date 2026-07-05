from flask import Flask, request, jsonify
import joblib
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest
import time

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

# Metrics Prometheus
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

PREDICTION_TIME = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

@app.route("/")
def home():
    return "Stroke Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.inc()

    start = time.time()

    data = request.get_json()

    df = pd.DataFrame([data])

    prediction = int(model.predict(df)[0])

    PREDICTION_TIME.observe(time.time() - start)

    return jsonify({
        "prediction": prediction
    })

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": "text/plain"
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)