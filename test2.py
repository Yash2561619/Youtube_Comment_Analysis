import mlflow
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "Yash2561619"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8317bf6a6fd7950f6097e966791ba44c9524117b"

mlflow.set_tracking_uri(
    "https://dagshub.com/Yash2561619/Youtube_Comment_Analysis.mlflow"
)

model = mlflow.pyfunc.load_model("models:/yt_chrome_plugin_model/2")

print(model.metadata.signature)