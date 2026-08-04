import os
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('model_evaluation')

import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "Yash2561619"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8317bf6a6fd7950f6097e966791ba44c9524117b"

def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV and fill missing values."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load serialized model using pickle."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except FileNotFoundError:
        logger.error('Model file not found: %s', model_path)
        raise
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load serialized TF-IDF vectorizer using pickle."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except FileNotFoundError:
        logger.error('Vectorizer file not found: %s', vectorizer_path)
        raise
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and return classification report metrics and confusion matrix."""
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug('Model evaluation completed')
        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm: np.ndarray, dataset_name: str):
    """Log confusion matrix as an MLflow artifact plot."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_file_path = f'confusion_matrix_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()


def main():
    mlflow.set_tracking_uri("https://dagshub.com/Yash2561619/Youtube_Comment_Analysis.mlflow")

    mlflow.set_experiment('dvc-pipeline-runs')
    
    with mlflow.start_run():
        try:
            # Resolve directory paths
            root_dir = get_root_directory()

            # Load parameters
            params = load_params(os.path.join(root_dir, 'params.yaml'))
            if isinstance(params, dict):
                for key, value in params.items():
                    mlflow.log_param(key, value)

            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Log model parameters (if available)
            if hasattr(model, 'get_params'):
                for param_name, param_value in model.get_params().items():
                    mlflow.log_param(param_name, param_value)

            # Log model and vectorizer artifacts
           
            mlflow.lightgbm.log_model(
                 lgb_model=model,
                 name="lgbm_model",
                 
)
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Load test data
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Prepare test data features and target
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Evaluate model
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log classification metrics to MLflow
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log confusion matrix plot artifact
            log_confusion_matrix(cm, "Test Data")

            # Add MLflow metadata tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")


if __name__ == '__main__':
    main()