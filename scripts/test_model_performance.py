import os
import pickle
import mlflow
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Set DagsHub Tracking Credentials & URI
os.environ["MLFLOW_TRACKING_USERNAME"] = "Yash2561619"
os.environ["MLFLOW_TRACKING_PASSWORD"] = (
    "8317bf6a6fd7950f6097e966791ba44c9524117b"
)
mlflow.set_tracking_uri(
    "https://dagshub.com/Yash2561619/Youtube_Comment_Analysis.mlflow"
)


@pytest.mark.parametrize(
    "model_artifact_path, holdout_data_path, vectorizer_path",
    [(
        "lgbm_model",
        "data/interim/test_processed.csv",
        "tfidf_vectorizer.pkl",
    )],
)
def test_model_performance(
    model_artifact_path, holdout_data_path, vectorizer_path
):
  try:
    # 1. Resolve Project Root Directory
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../")
    )
    full_data_path = os.path.join(root_dir, holdout_data_path)
    full_vectorizer_path = os.path.join(root_dir, vectorizer_path)

    # 2. Get latest successful MLflow run
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

    # 3. Load model from run artifacts
    model_uri = f"runs:/{latest_run_id}/{model_artifact_path}"
    model = mlflow.pyfunc.load_model(model_uri)

    # 4. Load vectorizer
    with open(full_vectorizer_path, "rb") as file:
      vectorizer = pickle.load(file)

    # 5. Load and process test dataset
    holdout_data = pd.read_csv(full_data_path)
    holdout_data.fillna("", inplace=True)

    X_text = holdout_data["clean_comment"].values
    y_holdout = holdout_data["category"].values

    # 6. Transform input features using TF-IDF
    X_tfidf = vectorizer.transform(X_text)
    X_tfidf_df = pd.DataFrame(
        X_tfidf.toarray(), columns=vectorizer.get_feature_names_out()
    )

    # 7. Generate predictions
    y_pred = model.predict(X_tfidf_df)

    # 8. Calculate evaluation metrics
    accuracy = accuracy_score(y_holdout, y_pred)
    precision = precision_score(
        y_holdout, y_pred, average="weighted", zero_division=1
    )
    recall = recall_score(
        y_holdout, y_pred, average="weighted", zero_division=1
    )
    f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=1)

    # 9. Performance Threshold Assertions
    expected_threshold = 0.40

    assert (
        accuracy >= expected_threshold
    ), f"Accuracy below threshold: got {accuracy:.4f}, expected >= {expected_threshold}"
    assert (
        precision >= expected_threshold
    ), f"Precision below threshold: got {precision:.4f}, expected >= {expected_threshold}"
    assert (
        recall >= expected_threshold
    ), f"Recall below threshold: got {recall:.4f}, expected >= {expected_threshold}"
    assert (
        f1 >= expected_threshold
    ), f"F1 score below threshold: got {f1:.4f}, expected >= {expected_threshold}"

    print(
        f"Performance test passed successfully for run ID {latest_run_id}!"
    )

  except Exception as e:
    pytest.fail(f"Model performance test failed with error: {e}")