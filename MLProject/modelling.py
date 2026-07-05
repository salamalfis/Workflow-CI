import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

mlflow.sklearn.autolog()

train = pd.read_csv("dataset_preprocessing/train.csv")
test = pd.read_csv("dataset_preprocessing/test.csv")

X_train = train.drop("stroke", axis=1)
y_train = train["stroke"]

X_test = test.drop("stroke", axis=1)
y_test = test["stroke"]

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    print(train.head())
    print(train["stroke"].isna().sum())

    print(test.head())
    print(test["stroke"].isna().sum())

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("=" * 50)
    print("HASIL EVALUASI MODEL")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred, zero_division=0))

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )
joblib.dump(model, "model.pkl")
