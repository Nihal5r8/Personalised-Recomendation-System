from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
import uuid

# Flask setup
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:1234@localhost:5432/infinity_recs'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# DB setup
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Movie(db.Model):
    __tablename__ = 'movies'
    movie_id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, nullable=False, unique=True)
    title = db.Column(db.String(255), nullable=False)
    genres = db.Column(db.String(255))
    tags = db.Column(db.Text)
    release_date = db.Column(db.String(20))
    rating = db.Column(db.Float)
    poster_url = db.Column(db.String(255))
    tmdb_link = db.Column(db.String(255))

class UserMovieInteraction(db.Model):
    __tablename__ = 'user_movie_interactions'
    interaction_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.movie_id'))
    watched = db.Column(db.Boolean, default=False)
    rating = db.Column(db.Float)
    liked = db.Column(db.Boolean, default=False)
    clicked = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Load dataset
tmdb_movies = pd.read_csv("C:/Users/nihal/PycharmProjects/Mini/INFINITY_RECS/backend/Datasets/tmdb_movies.csv",
                          usecols=['tmdb_id', 'title', 'genres', 'overview', 'release_date', 'popularity'])
tmdb_movies.fillna('', inplace=True)
tmdb_movies.rename(columns={'overview': 'tags', 'popularity': 'rating'}, inplace=True)
tmdb_movies['combined_features'] = tmdb_movies['title'] + ' ' + tmdb_movies['genres'] + ' ' + tmdb_movies['tags']

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(tmdb_movies['combined_features'])
tmdb_id_to_index = {row['tmdb_id']: idx for idx, row in tmdb_movies.iterrows()}

# PPO Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)

state_dim = tfidf_matrix.shape[1]
num_movies = len(tmdb_movies)
policy_net = PolicyNetwork(state_dim, num_movies)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

# Reward Function
def compute_reward(interaction):
    reward = 0
    if interaction.rating:
        reward += interaction.rating / 5.0
    if interaction.liked:
        reward += 1
    else:
        reward -= 1
    if interaction.clicked:
        reward += 0.5
    if interaction.watched:
        reward += 0.75
    return reward

# PPO Training
def train_ppo():
    interactions = UserMovieInteraction.query.all()
    states, actions, rewards = [], [], []

    print(f"Found {len(interactions)} interactions in UserMovieInteraction table")

    for inter in interactions:
        movie = Movie.query.get(inter.movie_id)
        if not movie:
            print(f"Movie with movie_id {inter.movie_id} not found in movies table")
            continue
        movie_row = tmdb_movies[tmdb_movies['tmdb_id'] == movie.tmdb_id]
        if movie_row.empty:
            print(f"No movie found in tmdb_movies for tmdb_id {movie.tmdb_id}")
            continue
        idx = movie_row.index[0]
        state = tfidf_matrix[idx].toarray()[0]
        states.append(state)
        actions.append(idx)
        rewards.append(compute_reward(inter))

    if not states:
        print("No valid interactions found for PPO training. Skipping training.")
        return

    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)

    print(f"states_tensor shape: {states_tensor.shape}")
    print(f"actions_tensor shape: {actions_tensor.shape}")
    print(f"rewards_tensor shape: {rewards_tensor.shape}")

    for _ in range(10):
        logits = policy_net(states_tensor)
        log_probs = torch.log(logits.gather(1, actions_tensor.view(-1, 1)).squeeze())
        loss = -(log_probs * rewards_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(policy_net.state_dict(), "Movie_ppo.pt")
    print("PPO model trained and saved successfully")

# Fetch from TMDB API
TMDB_API_KEY = 'de850f8ae8571a6ff56908a90cc7d9ac'
TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'
TMDB_API_URL = 'https://api.themoviedb.org/3/movie/'

movie_cache = {}

async def fetch_movie_details(session, tmdb_id):
    if tmdb_id in movie_cache:
        return movie_cache[tmdb_id]
    url = f"{TMDB_API_URL}{tmdb_id}?api_key={TMDB_API_KEY}"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                poster_path = data.get('poster_path')
                poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None
                tmdb_link = f"https://www.themoviedb.org/movie/{tmdb_id}"
                movie_cache[tmdb_id] = (poster_url, tmdb_link)
                return poster_url, tmdb_link
    except Exception as e:
        print(f"Error fetching TMDB details for tmdb_id {tmdb_id}: {str(e)}")
    return None, None

# General Recommendations
def get_general_recommendations(top_k=16):
    general_recs = tmdb_movies.sort_values(by='rating', ascending=False).head(top_k)
    return general_recs.reset_index(drop=True)

# User Recommendations
async def get_user_recommendations(username, top_k=16):
    # Create or get user
    user = get_or_create_user(username)
    print(f"User '{username}' (user_id: {user.user_id}) retrieved or created")

    # Try training the PPO model
    try:
        train_ppo()
    except Exception as e:
        print(f"PPO training failed: {str(e)}. Falling back to general recommendations.")

    # Get all interactions for the user (not just watched)
    interactions = UserMovieInteraction.query.filter_by(user_id=user.user_id).all()

    # Collect all interacted tmdb_ids
    interacted_tmdb_ids = set()
    for interaction in interactions:
        movie = Movie.query.filter_by(movie_id=interaction.movie_id).first()
        if movie:
            interacted_tmdb_ids.add(movie.tmdb_id)

    print(f"User has interacted with {len(interacted_tmdb_ids)} movies")

    # Fallback to general recommendations if interaction history is small
    if len(interactions) < 5:
        general_recs = get_general_recommendations(top_k)
        # Filter out interacted movies
        general_recs = general_recs[~general_recs['tmdb_id'].isin(interacted_tmdb_ids)]
        # If not enough recommendations, add more
        if len(general_recs) < top_k:
            additional_recs = tmdb_movies[~tmdb_movies['tmdb_id'].isin(interacted_tmdb_ids)].head(top_k - len(general_recs))
            general_recs = pd.concat([general_recs, additional_recs]).reset_index(drop=True)

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_movie_details(session, row['tmdb_id']) for _, row in general_recs.iterrows()]
            poster_info = await asyncio.gather(*tasks)

        result = []
        for i, (_, row) in enumerate(general_recs.iterrows()):
            poster_url, tmdb_link = poster_info[i]
            result.append({
                'tmdb_id': int(row['tmdb_id']),
                'title': row['title'],
                'genres': row['genres'],
                'poster_url': poster_url,
                'tmdb_link': tmdb_link
            })
        return result[:top_k]

    # PPO-based recommendations
    state = tfidf_matrix.mean(axis=0)
    state_tensor = torch.FloatTensor(state.A)

    try:
        policy_net.load_state_dict(torch.load("Movie_ppo.pt", map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("PPO model file not found. Using untrained model.")
    policy_net.eval()

    action_probs = policy_net(state_tensor)
    top_indices = torch.topk(action_probs, top_k * 2, dim=1).indices.numpy()[0]  # Get more indices to account for filtering
    recommended_movies = tmdb_movies.iloc[top_indices].reset_index(drop=True)

    # Filter out interacted movies
    recommended_movies = recommended_movies[~recommended_movies['tmdb_id'].isin(interacted_tmdb_ids)]

    # If not enough recommendations, add more non-interacted movies
    if len(recommended_movies) < top_k:
        remaining_movies = tmdb_movies[~tmdb_movies['tmdb_id'].isin(interacted_tmdb_ids)]
        additional_recommendations = remaining_movies.head(top_k - len(recommended_movies))
        recommended_movies = pd.concat([recommended_movies, additional_recommendations]).reset_index(drop=True)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_movie_details(session, row['tmdb_id']) for _, row in recommended_movies.iterrows()]
        poster_info = await asyncio.gather(*tasks)

    result = []
    for i, (_, row) in enumerate(recommended_movies.iterrows()):
        poster_url, tmdb_link = poster_info[i]
        result.append({
            'tmdb_id': int(row['tmdb_id']),
            'title': row['title'],
            'genres': row['genres'],
            'poster_url': poster_url,
            'tmdb_link': tmdb_link
        })

    return result[:top_k]

@app.route('/api/recommend', methods=['GET'])
def recommend():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "Missing username"}), 400

    recommendations = asyncio.run(get_user_recommendations(username))
    return jsonify(recommendations)

# Utility: Get/Create user
def get_or_create_user(username):
    user = User.query.filter_by(username=username).first()
    if not user:
        user = User(username=username)
        db.session.add(user)
        db.session.commit()
        print(f"Created new user: {username}")
    return user

# Utility: Get/Create movie (only called for interactions)
def get_or_create_movie(tmdb_id):
    movie = Movie.query.filter_by(tmdb_id=int(tmdb_id)).first()
    if not movie:
        movie_info = tmdb_movies[tmdb_movies['tmdb_id'] == int(tmdb_id)]
        if movie_info.empty:
            print(f"Movie with tmdb_id {tmdb_id} not found in tmdb_movies")
            return None
        movie_info = movie_info.iloc[0]
        movie = Movie(
            tmdb_id=int(movie_info['tmdb_id']),
            title=str(movie_info['title']),
            genres=str(movie_info['genres']),
            tags=str(movie_info['tags']),
            release_date=str(movie_info['release_date']),
            rating=float(movie_info['rating']) if pd.notna(movie_info['rating']) else None
        )
        db.session.add(movie)
        db.session.commit()
        print(f"Created new movie: {movie.title} (tmdb_id: {movie.tmdb_id})")
    return movie

@app.route('/api/rate', methods=['POST'])
def rate_movie():
    data = request.get_json()
    username = data.get('username')
    tmdb_id = data.get('tmdb_id')
    rating = data.get('rating')
    watched = data.get('watched')
    liked = data.get('liked')

    if not all([username, tmdb_id]):
        return jsonify({"error": "Missing data"}), 400

    try:
        user = get_or_create_user(username)
        movie = get_or_create_movie(tmdb_id)
        if not movie:
            return jsonify({"error": "Movie not found"}), 404

        interaction = UserMovieInteraction.query.filter_by(user_id=user.user_id, movie_id=movie.movie_id).first()

        if not interaction:
            interaction = UserMovieInteraction(user_id=user.user_id, movie_id=movie.movie_id)
            db.session.add(interaction)

        if watched is not None:
            interaction.watched = watched
        if rating is not None:
            interaction.rating = rating
        if liked is not None:
            interaction.liked = liked

        db.session.commit()
        return jsonify({"message": "Rating & interaction saved"}), 200

    except Exception:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/watched', methods=['POST'])
def update_watched_status():
    data = request.get_json()
    username = data.get('username')
    tmdb_id = data.get('tmdb_id')
    watched = data.get('watched')

    if not username or not tmdb_id or watched is None:
        return jsonify({'error': 'Missing required data'}), 400

    try:
        user = get_or_create_user(username)
        movie = get_or_create_movie(tmdb_id)
        if not movie:
            return jsonify({'error': 'Movie not found'}), 404

        interaction = UserMovieInteraction.query.filter_by(user_id=user.user_id, movie_id=movie.movie_id).first()
        if interaction:
            interaction.watched = watched
        else:
            interaction = UserMovieInteraction(user_id=user.user_id, movie_id=movie.movie_id, watched=watched)
            db.session.add(interaction)

        db.session.commit()
        return jsonify({'message': 'Watched status updated successfully'}), 200

    except Exception:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/like', methods=['POST'])
def like_movie():
    data = request.get_json()
    username = data.get('username')
    tmdb_id = data.get('tmdb_id')

    if not username or not tmdb_id:
        return jsonify({"error": "Missing data"}), 400

    try:
        user = get_or_create_user(username)
        movie = get_or_create_movie(tmdb_id)
        if not movie:
            return jsonify({"error": "Movie not found"}), 404

        interaction = UserMovieInteraction.query.filter_by(user_id=user.user_id, movie_id=movie.movie_id).first()
        if not interaction:
            interaction = UserMovieInteraction(user_id=user.user_id, movie_id=movie.movie_id)
            db.session.add(interaction)

        interaction.liked = True
        db.session.commit()
        return jsonify({"message": "Like recorded"}), 200

    except Exception:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/dislike', methods=['POST'])
def dislike_movie():
    data = request.get_json()
    username = data.get('username')
    tmdb_id = data.get('tmdb_id')

    if not username or not tmdb_id:
        return jsonify({"error": "Missing data"}), 400

    try:
        user = get_or_create_user(username)
        movie = get_or_create_movie(tmdb_id)
        if not movie:
            return jsonify({"error": "Movie not found"}), 404

        interaction = UserMovieInteraction.query.filter_by(user_id=user.user_id, movie_id=movie.movie_id).first()
        if not interaction:
            interaction = UserMovieInteraction(user_id=user.user_id, movie_id=movie.movie_id)
            db.session.add(interaction)

        interaction.liked = False
        db.session.commit()
        return jsonify({"message": "Dislike recorded"}), 200

    except Exception:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/click', methods=['POST'])
def click_movie():
    data = request.get_json()
    username = data.get('username')
    tmdb_id = data.get('tmdb_id')

    if not username or not tmdb_id:
        return jsonify({"error": "Missing data"}), 400

    try:
        user = get_or_create_user(username)
        movie = get_or_create_movie(tmdb_id)
        if not movie:
            return jsonify({"error": "Movie not found"}), 404

        interaction = UserMovieInteraction.query.filter_by(user_id=user.user_id, movie_id=movie.movie_id).first()
        if interaction:
            interaction.clicked = True
        else:
            interaction = UserMovieInteraction(
                user_id=user.user_id,
                movie_id=movie.movie_id,
                clicked=True
            )
            db.session.add(interaction)

        db.session.commit()
        return jsonify({"message": "Click recorded"}), 200

    except Exception:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5003)
