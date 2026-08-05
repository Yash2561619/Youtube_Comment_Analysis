import joblib
import logging
import mlflow
import os
import re
import traceback
from flask import Flask, jsonify, request
from flask_cors import CORS
from mlflow.tracking import MlflowClient
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

app = Flask(__name__)
CORS(app)

# ------------------------------
# DagsHub Credentials
# ------------------------------
os.environ["MLFLOW_TRACKING_USERNAME"] = "Yash2561619"
os.environ["MLFLOW_TRACKING_PASSWORD"] = (
    "8317bf6a6fd7950f6097e966791ba44c9524117b"
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Yash2561619/Youtube_Comment_Analysis.mlflow"
)


# ------------------------------
# Text Preprocessing
# ------------------------------
def preprocess_comment(comment):
  comment = comment.lower()
  comment = comment.strip()
  comment = re.sub(r"\n", " ", comment)
  comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

  stop_words = set(stopwords.words("english")) - {
      "not",
      "no",
      "but",
      "however",
      "yet",
  }

  comment = " ".join(
      [word for word in comment.split() if word not in stop_words]
  )

  lemmatizer = WordNetLemmatizer()

  comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])

  return comment


# ------------------------------
# Load Model + Vectorizer
# ------------------------------
def load_model_and_vectorizer():

  model_uri = "models:/yt_chrome_plugin_model/2"

  print("\nLoading MLflow model...")
  model = mlflow.pyfunc.load_model(model_uri)

  print("\n" + "=" * 100)
  print("MODEL SIGNATURE")
  print("=" * 100)

  signature = model.metadata.signature
  print(signature)

  # -----------------------------
  # Model Schema
  # -----------------------------
  try:
    model_columns = signature.inputs.input_names()

    print("\nModel expects:", len(model_columns), "features")

    print("\nFirst 20 model columns:")
    print(model_columns[:20])

    print("\nLast 20 model columns:")
    print(model_columns[-20:])

  except Exception as e:
    print("Could not read model signature:", e)
    model_columns = []

  print("=" * 100)

  # -----------------------------
  # Load Vectorizer
  # -----------------------------
  print("\nLoading TF-IDF Vectorizer...")
  vectorizer = joblib.load("tfidf_vectorizer.pkl")

  vectorizer_columns = list(vectorizer.get_feature_names_out())

  print("\n" + "=" * 100)
  print("VECTORIZER INFORMATION")
  print("=" * 100)

  print("Vocabulary Size :", len(vectorizer_columns))

  print("\nFirst 20 Features:")
  print(vectorizer_columns[:20])

  print("\nLast 20 Features:")
  print(vectorizer_columns[-20:])

  print("=" * 100)

  # -----------------------------
  # Compare Model vs Vectorizer
  # -----------------------------
  if model_columns:

    print("\n" + "=" * 100)
    print("COMPARING MODEL SCHEMA WITH VECTORIZER")
    print("=" * 100)

    if model_columns == vectorizer_columns:
      print("\nSUCCESS")
      print("Model schema EXACTLY matches vectorizer vocabulary.")
    else:

      print("\nERROR")
      print("Model schema DOES NOT match vectorizer.")

      missing = list(set(model_columns) - set(vectorizer_columns))
      extra = list(set(vectorizer_columns) - set(model_columns))

      print("\nMissing Features :", len(missing))
      print("Extra Features   :", len(extra))

      if len(missing):
        print("\nFirst 20 Missing Features:")
        print(missing[:20])

      if len(extra):
        print("\nFirst 20 Extra Features:")
        print(extra[:20])

      # Check order mismatch
      if len(model_columns) == len(vectorizer_columns):
        mismatch = []

        for i, (m, v) in enumerate(zip(model_columns, vectorizer_columns)):
          if m != v:
            mismatch.append((i, m, v))

          if len(mismatch) == 20:
            break

        if mismatch:
          print("\nFirst 20 Order Mismatches:")

          for idx, expected, found in mismatch:
            print(f"Index {idx}")
            print(f"Expected : {expected}")
            print(f"Found    : {found}")
            print("-" * 60)

  print("\nModel loaded successfully!")

  return model, vectorizer


print("Starting application...")
model, vectorizer = load_model_and_vectorizer()


# ------------------------------
# Prediction API
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():

  data = request.get_json()
  comments = data.get("comments", [])

  if len(comments) == 0:
    return jsonify({"error": "No comments provided"}), 400

  try:

    print("\n" + "=" * 100)
    print("NEW PREDICTION REQUEST")
    print("=" * 100)

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------
    processed_comments = [preprocess_comment(comment) for comment in comments]

    print("\nProcessed Comments:")
    for i, c in enumerate(processed_comments):
      print(f"{i+1}. {c}")

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------
    X_sparse = vectorizer.transform(processed_comments)

    print("\nTF-IDF Matrix Shape :", X_sparse.shape)

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------
    X = pd.DataFrame(
        X_sparse.toarray(), columns=vectorizer.get_feature_names_out()
    )

    # --------------------------------------------------------
    # SCHEMA ALIGNMENT FIX FOR MLFLOW
    # --------------------------------------------------------
    signature = model.metadata.signature
    if signature and signature.inputs:
      model_columns = signature.inputs.input_names()
      X = X.reindex(columns=model_columns, fill_value=0.0)

    print("\n" + "=" * 100)
    print("INPUT DATAFRAME")
    print("=" * 100)

    print("Type :", type(X))
    print("Shape:", X.shape)

    print("\nVocabulary Size :", len(X.columns))

    print("\nFirst 20 Columns:")
    print(X.columns[:20].tolist())

    print("\nLast 20 Columns:")
    print(X.columns[-20:].tolist())

    print("\nData Types:")
    print(X.dtypes.head())

    print("\nNull Values :", X.isnull().values.any())

    print("\nFirst Row (first 20 values)")
    print(X.iloc[0, :20])

    print("=" * 100)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    print("\nCalling model.predict()...\n")

    predictions = model.predict(X)

    print("Prediction Successful")

    print("\nPredictions:")
    print(predictions)

    predictions = predictions.tolist()

    # --------------------------------------------------------
    # Build Response
    # --------------------------------------------------------
    results = []

    for comment, prediction in zip(comments, predictions):
      results.append({"comment": comment, "sentiment": prediction})

    print("\nRequest Completed Successfully.")
    print("=" * 100)

    return jsonify(results)

  except Exception as e:

    print("\n" + "=" * 100)
    print("FULL EXCEPTION")
    print("=" * 100)

    traceback.print_exc()

    print("=" * 100)

    return (
        jsonify({"error": str(e), "traceback": traceback.format_exc()}),
        500,
    )


@app.route("/")
def home():
  return jsonify({"message": "YouTube Comment Sentiment API Running"})


# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)