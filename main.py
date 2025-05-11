from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask_cors import CORS  # Import CORS
import os

# Initialize Flask app and enable CORS for all routes
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load all assets
model = pickle.load(open("model.pkl", "rb"))
embeddings = np.load("embeddings.npy")
df = pd.read_csv("cve_ttp_data.csv")
model_bert = SentenceTransformer("bert_model")  # Load local dir

@app.route('/')
def home():
    return "Welcome to the CVE TTP Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    description = data.get("description")

    if not description:
        return jsonify({"error": "Missing 'description' in request"}), 400

    # Embed the new input
    desc_embedding = model_bert.encode([description])

    # Find cosine similarities
    similarities = cosine_similarity(desc_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:5]

    # Get top TTPs
    top_ttps = df.iloc[top_indices]["TTP_ID"].unique().tolist()

    return jsonify({"predicted_ttps": top_ttps})

# Ensure the app runs correctly in the Render environment
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Get the port from the environment
    app.run(host="0.0.0.0", port=port)  # Ensure the app listens on the correct port
