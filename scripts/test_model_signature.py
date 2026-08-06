import os
import pickle
import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

# DagsHub Credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = "Yash2561619"
os.environ["MLFLOW_TRACKING_PASSWORD"] = (
    "8317bf6a6fd7950f6097e966791ba44c9524117b"
)

# Use DagsHub Tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/Yash2561619/Youtube_Comment_Analysis.mlflow"
)


@pytest.mark.parametrize(
    "model_name, stage, vectorizer_path",
    [("lgbm_model", "Staging", "tfidf_vectorizer.pkl")],
)
def test_model_with_vectorizer(model_name, stage, vectorizer_path):
  client = MlflowClient()

  # Fetch versions for target stage
  versions = client.search_model_versions(
      f"name='{model_name}' and current_stage='{stage}'"
  )
  assert (
      len(versions) > 0
  ), f"No model found in the '{stage}' stage for '{model_name}'"

  latest_version = versions[0].version

  try:
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    with open(vectorizer_path, "rb") as file:
      vectorizer = pickle.load(file)

    input_text = "hi how are you"
    input_data = vectorizer.transform([input_text])
    input_df = pd.DataFrame(
        input_data.toarray(), columns=vectorizer.get_feature_names_out()
    )

    prediction = model.predict(input_df)

    assert input_df.shape[1] == len(
        vectorizer.get_feature_names_out()
    ), "Input feature count mismatch"
    assert len(prediction) == input_df.shape[0], "Output row count mismatch"

  except Exception as e:
    pytest.fail(f"Model test failed with error: {e}")