import os

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "iris-classification"
REGISTERED_MODEL_NAME = "iris-rf-classifier"


def train() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    params = {
        "n_estimators": 100,
        "random_state": 42,
    }

    with mlflow.start_run():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")

        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test F1-score: {f1:.4f}")

        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

        artifact = {
            "model": model,
            "target_names": iris.target_names.tolist(),
        }
        joblib.dump(artifact, "model.joblib")
        print("Model saved to model.joblib")

        mlflow.log_artifact("model.joblib")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        print(f"Model registered as '{REGISTERED_MODEL_NAME}' in the MLflow Model Registry")


if __name__ == "__main__":
    train()
