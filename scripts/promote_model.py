import os
import mlflow
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


def promote_model():
  client = MlflowClient()
  registered_model_name = "lgbm_model"

  # 1. Get the latest successful run from MLflow experiment
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

  # 2. Register the model artifact into MLflow Model Registry if not already registered
  model_src = f"runs:/{latest_run_id}/lgbm_model"
  model_version = mlflow.register_model(
      model_uri=model_src, name=registered_model_name
  )

  # 3. Archive any existing model versions currently in Production
  try:
    existing_versions = client.search_model_versions(
        f"name='{registered_model_name}'"
    )
    for mv in existing_versions:
      if mv.current_stage.lower() == "production":
        client.transition_model_version_stage(
            name=registered_model_name,
            version=mv.version,
            stage="Archived",
        )
        print(f"Archived previous Production version {mv.version}")
  except Exception as e:
    print(f"No previous Production models to archive: {e}")

  # 4. Promote the new model version to Production
  client.transition_model_version_stage(
      name=registered_model_name,
      version=model_version.version,
      stage="Production",
  )

  print(
      f"Successfully promoted model '{registered_model_name}' version"
      f" {model_version.version} to Production!"
  )


if __name__ == "__main__":
  promote_model()