import mlflow.pyfunc
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

model = mlflow.pyfunc.load_model(
    "runs:/1cf0cf4760624bdeb0c403cd8e198a6c/model"
)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame(data)

    pred = model.predict(df)

    return jsonify(pred.tolist())

app.run(port=5001)