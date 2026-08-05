import joblib

vec = joblib.load("tfidf_vectorizer.pkl")

features = vec.get_feature_names_out()

print("Vocabulary size:", len(features))

print("\nFirst 20:")
print(features[:20])

print("\nLast 20:")
print(features[-20:])