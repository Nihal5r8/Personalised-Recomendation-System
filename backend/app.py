
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
import scipy.sparse
import joblib
import pickle
import os
from database import init_db, get_db_connection, save_user, get_user, get_category_data

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Explicitly allow your frontend

# Initialize database
init_db()

# Load recommendation model for books
MODEL_DIR = "MODEL"
BOOKS_DIR = os.path.join(MODEL_DIR, "BOOKS")
MLB_MODEL_PATH = os.path.join(BOOKS_DIR, "mlb_model.pkl")
TFIDF_MODEL_PATH = os.path.join(BOOKS_DIR, "tfidf_model.pkl")
KNN_MODEL_PATH = os.path.join(BOOKS_DIR, "knn_model.pkl")
COMBINED_FEATURES_PATH = os.path.join(BOOKS_DIR, "combined_features.pkl")
BOOKS_DF_PATH = os.path.join(BOOKS_DIR, "books_df.pkl")

books_combined_features = joblib.load(COMBINED_FEATURES_PATH)
books_df = joblib.load(BOOKS_DF_PATH)
books_knn = joblib.load(KNN_MODEL_PATH)
books_mlb = joblib.load(MLB_MODEL_PATH)
books_tfidf = joblib.load(TFIDF_MODEL_PATH)

# Recommendation function
def recommend_content_based(title, initial_n=20, final_n=10):
    try:
        idx = books_df[books_df['title'] == title].index[0]
        query = books_combined_features[idx].reshape(1, -1)
        distances, indices = books_knn.kneighbors(query, n_neighbors=initial_n + 1)
        content_based_recs = books_df[['title', 'genre', 'desc', 'rating', 'totalratings', 'img', 'link', 'author']].iloc[indices[0][1:]]
        
        # Calculate Bayesian rating for the initial recommendations
        C = content_based_recs['rating'].mean()  # Mean rating of recommended books
        m = content_based_recs['totalratings'].quantile(0.9)  # 90th percentile of total ratings
        
        def bayesian_rating(row, C, m):
            v = row['totalratings']
            R = row['rating']
            return (v / (v + m) * R) + (m / (v + m) * C)
        
        # Add Bayesian score to the recommendations
        content_based_recs["bayesian_score"] = content_based_recs.apply(lambda row: bayesian_rating(row, C, m), axis=1)
        
        # Sort by Bayesian score and take top final_n
        content_based_recs = content_based_recs.sort_values('bayesian_score', ascending=False).head(final_n)
        
        return content_based_recs.to_dict('records')
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return []

# User data routes
@app.route('/user/<user_id>', methods=['GET'])
def get_user_data(user_id):
    user = get_user(user_id)
    if user:
        return jsonify(user)
    return jsonify({"error": "User not found"}), 404

@app.route('/user', methods=['POST'])
def save_user_data():
    data = request.get_json()
    user_id = data.get('user_id')
    email = data.get('email')
    username = data.get('username')
    
    if not user_id or not email:
        return jsonify({"error": "user_id and email are required"}), 400
    
    save_user(user_id, email, username)
    return jsonify({"message": "User data saved successfully"}), 201

# Category data route
@app.route('/category/<user_id>/<category>', methods=['GET'])
def get_category(user_id, category):
    data = get_category_data(user_id, category)
    return jsonify(data)

# Recommendation route
@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Title is required"}), 400
    recommendations = recommend_content_based(title)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, port=5000)