import os
import pickle
import mlflow
import pandas as pd
import pytest

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
    "model_artifact_path, vectorizer_path",
    [("lgbm_model", "tfidf_vectorizer.pkl")],
)
def test_model_with_vectorizer(model_artifact_path, vectorizer_path):
  # 1. Search for the latest run in the MLflow experiment
  experiment = mlflow.get_experiment_by_name("dvc-pipeline-runs")
  assert (
      experiment is not None
  ), "Experiment 'dvc-pipeline-runs' not found in MLflow."

  runs = mlflow.search_runs(
      experiment_ids=[experiment.experiment_id],
      order_by=["start_time DESC"],
      max_results=1,
  )

  assert not runs.empty, "No runs found in MLflow experiment."
  latest_run_id = runs.iloc[0]["run_id"]

  try:
    # 2. Load the model directly from the latest run's artifact path
    model_uri = f"runs:/{latest_run_id}/{model_artifact_path}"
    model = mlflow.pyfunc.load_model(model_uri)

    # 3. Load vectorizer from root directory
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../")
    )
    full_vectorizer_path = os.path.join(root_dir, vectorizer_path)

    with open(full_vectorizer_path, "rb") as file:
      vectorizer = pickle.load(file)

    # 4. Transform sample input text
    input_text = "hi how are you"
    input_data = vectorizer.transform([input_text])
    input_df = pd.DataFrame(
        input_data.toarray(), columns=vectorizer.get_feature_names_out()
    )

    # 5. Predict using loaded pyfunc model
    prediction = model.predict(input_df)

    # 6. Assertions
    assert input_df.shape[1] == len(
        vectorizer.get_feature_names_out()
    ), "Input feature count mismatch"
    assert len(prediction) == input_df.shape[0], "Output row count mismatch"

  except Exception as e:
    pytest.fail(f"Model test failed with error: {e}")