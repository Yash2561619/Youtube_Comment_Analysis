import mlflow
import random
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "Yash2561619"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8317bf6a6fd7950f6097e966791ba44c9524117b"
# Set the MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Yash2561619/Youtube_Comment_Analysis.mlflow")

# Start an MLflow run
with mlflow.start_run():
    # Log some random parameters
    mlflow.log_param("param1", random.randint(1, 100))
    mlflow.log_param("param2", random.random())

    # Log some random metrics
    mlflow.log_metric("metric1", random.random())
    mlflow.log_metric("metric2", random.uniform(0.5, 1.5))

    print("Logged random parameters and metrics.")