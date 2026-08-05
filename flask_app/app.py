import io
import joblib
import logging
import os
import re
import traceback
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server chart generation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from mlflow.tracking import MlflowClient
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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
  try:
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
  except Exception as e:
    print(f"Error preprocessing comment: {e}")
    return comment


# ------------------------------
# Load Model + Vectorizer
# ------------------------------
def load_model_and_vectorizer():
  model_uri = "models:/yt_chrome_plugin_model/2"

  print("\nLoading MLflow model...")
  model = mlflow.pyfunc.load_model(model_uri)

  print("\nLoading TF-IDF Vectorizer...")
  vectorizer = joblib.load("tfidf_vectorizer.pkl")

  print("\nModel & Vectorizer loaded successfully!")
  return model, vectorizer


print("Starting application...")
model, vectorizer = load_model_and_vectorizer()


# ------------------------------
# Helper Functions
# ------------------------------
def get_aligned_dataframe(preprocessed_comments):
  X_sparse = vectorizer.transform(preprocessed_comments)
  X = pd.DataFrame(
      X_sparse.toarray(), columns=vectorizer.get_feature_names_out()
  )

  signature = model.metadata.signature
  if signature and signature.inputs:
    model_columns = signature.inputs.input_names()
    X = X.reindex(columns=model_columns, fill_value=0.0)

  return X


@app.route("/")
def home():
  return jsonify({"message": "YouTube Comment Sentiment API Running"})


# ------------------------------
# Prediction API Endpoints
# ------------------------------
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
  if request.method == "OPTIONS":
    return jsonify({"status": "ok"}), 200

  data = request.get_json()
  if not data:
    return jsonify({"error": "Invalid or missing JSON payload"}), 400

  comments = data.get("comments", [])
  if len(comments) == 0:
    return jsonify({"error": "No comments provided"}), 400

  try:
    print("\n" + "=" * 100)
    print(f"NEW PREDICTION REQUEST ({len(comments)} comments)")
    print("=" * 100)

    processed_comments = [preprocess_comment(comment) for comment in comments]
    X = get_aligned_dataframe(processed_comments)

    predictions = model.predict(X)

    if isinstance(predictions, (np.ndarray, pd.Series)):
      predictions = predictions.astype(int).tolist()
    else:
      predictions = [int(p) for p in predictions]

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


@app.route("/predict_with_timestamps", methods=["POST", "OPTIONS"])
def predict_with_timestamps():
  if request.method == "OPTIONS":
    return jsonify({"status": "ok"}), 200

  data = request.get_json()
  if not data:
    return jsonify({"error": "Invalid JSON request body"}), 400

  comments_data = data.get("comments", [])
  if not comments_data:
    return jsonify({"error": "No comments provided"}), 400

  try:
    comments = [item["text"] for item in comments_data]
    timestamps = [item["timestamp"] for item in comments_data]

    preprocessed_comments = [preprocess_comment(c) for c in comments]
    X = get_aligned_dataframe(preprocessed_comments)

    predictions = model.predict(X)

    if isinstance(predictions, (np.ndarray, pd.Series)):
      predictions = predictions.astype(int).tolist()
    else:
      predictions = [int(p) for p in predictions]

    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(
            comments, predictions, timestamps
        )
    ]
    return jsonify(response)

  except Exception as e:
    return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ------------------------------
# Visualization Endpoints
# ------------------------------
@app.route("/generate_chart", methods=["POST", "OPTIONS"])
def generate_chart():
  if request.method == "OPTIONS":
    return jsonify({"status": "ok"}), 200

  try:
    data = request.get_json()
    sentiment_counts = data.get("sentiment_counts")

    if not sentiment_counts:
      return jsonify({"error": "No sentiment counts provided"}), 400

    labels = ["Positive", "Neutral", "Negative"]
    sizes = [
        int(sentiment_counts.get("1", 0)),
        int(sentiment_counts.get("0", 0)),
        int(sentiment_counts.get("-1", 0)),
    ]
    if sum(sizes) == 0:
      raise ValueError("Sentiment counts sum to zero")

    colors = ["#36A2EB", "#C9CBCF", "#FF6384"]

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"color": "w"},
    )
    plt.axis("equal")

    img_io = io.BytesIO()
    plt.savefig(img_io, format="PNG", transparent=True)
    img_io.seek(0)
    plt.close()

    return send_file(img_io, mimetype="image/png")
  except Exception as e:
    app.logger.error(f"Error in /generate_chart: {e}")
    return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route("/generate_wordcloud", methods=["POST", "OPTIONS"])
def generate_wordcloud():
  if request.method == "OPTIONS":
    return jsonify({"status": "ok"}), 200

  try:
    data = request.get_json()
    comments = data.get("comments")

    if not comments:
      return jsonify({"error": "No comments provided"}), 400

    preprocessed_comments = [preprocess_comment(c) for c in comments]
    text = " ".join(preprocessed_comments)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="Blues",
        stopwords=set(stopwords.words("english")),
        collocations=False,
    ).generate(text)

    img_io = io.BytesIO()
    wordcloud.to_image().save(img_io, format="PNG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/png")
  except Exception as e:
    app.logger.error(f"Error in /generate_wordcloud: {e}")
    return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route("/generate_trend_graph", methods=["POST", "OPTIONS"])
def generate_trend_graph():
  if request.method == "OPTIONS":
    return jsonify({"status": "ok"}), 200

  try:
    data = request.get_json()
    sentiment_data = data.get("sentiment_data")

    if not sentiment_data:
      return jsonify({"error": "No sentiment data provided"}), 400

    df = pd.DataFrame(sentiment_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df["sentiment"] = df["sentiment"].astype(int)

    sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    monthly_counts = (
        df.resample("ME")["sentiment"].value_counts().unstack(fill_value=0)
    )
    monthly_totals = monthly_counts.sum(axis=1)

    monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

    for sentiment_value in [-1, 0, 1]:
      if sentiment_value not in monthly_percentages.columns:
        monthly_percentages[sentiment_value] = 0

    monthly_percentages = monthly_percentages[[-1, 0, 1]]

    plt.figure(figsize=(12, 6))
    colors = {-1: "red", 0: "gray", 1: "green"}

    for sentiment_value in [-1, 0, 1]:
      plt.plot(
          monthly_percentages.index,
          monthly_percentages[sentiment_value],
          marker="o",
          linestyle="-",
          label=sentiment_labels[sentiment_value],
          color=colors[sentiment_value],
      )

    plt.title("Monthly Sentiment Percentage Over Time")
    plt.xlabel("Month")
    plt.ylabel("Percentage of Comments (%)")
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

    plt.legend()
    plt.tight_layout()

    img_io = io.BytesIO()
    plt.savefig(img_io, format="PNG")
    img_io.seek(0)
    plt.close()

    return send_file(img_io, mimetype="image/png")
  except Exception as e:
    app.logger.error(f"Error in /generate_trend_graph: {e}")
    return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


# ------------------------------
# Run Server
# ------------------------------
if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=False)