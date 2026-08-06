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

# Set DagsHub Tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/Yash2561619/Youtube_Comment_Analysis.mlflow"
)


@pytest.mark.parametrize(
    "model_name, vectorizer_path",
    [("lgbm_model", "tfidf_vectorizer.pkl")],
)
def test_model_with_vectorizer(model_name, vectorizer_path):
  client = MlflowClient()

  # 1. Fetch all registered versions of the model
  all_versions = client.search_model_versions(f"name='{model_name}'")

  assert (
      len(all_versions) > 0
  ), f"No registered versions found for model '{model_name}'"

  # 2. Get the latest version number automatically
  latest_version = max([int(mv.version) for mv in all_versions])

  try:
    # 3. Load model using latest version number
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # 4. Load vectorizer
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../")
    )
    full_vectorizer_path = os.path.join(root_dir, vectorizer_path)

    with open(full_vectorizer_path, "rb") as file:
      vectorizer = pickle.load(file)

    # 5. Transform sample text input
    input_text = "hi how are you"
    input_data = vectorizer.transform([input_text])
    input_df = pd.DataFrame(
        input_data.toarray(), columns=vectorizer.get_feature_names_out()
    )

    # 6. Predict using loaded pyfunc model
    prediction = model.predict(input_df)

    # 7. Assertions
    assert input_df.shape[1] == len(
        vectorizer.get_feature_names_out()
    ), "Input feature count mismatch"
    assert len(prediction) == input_df.shape[0], "Output row count mismatch"

  except Exception as e:
    pytest.fail(f"Model test failed with error: {e}")