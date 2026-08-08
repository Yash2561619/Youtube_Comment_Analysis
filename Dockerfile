FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies & clean apt cache in the same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Leverage Docker cache layer for dependencies
COPY flask_app/requirements.txt /app/requirements.txt

# 3. Install Python dependencies without pip cache
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader stopwords wordnet

# 4. Copy application source code and model artifacts
COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

EXPOSE 5000

CMD ["python", "app.py"]