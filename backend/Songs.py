from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import joblib
import os
from sqlalchemy import Index

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:3000"]}})

# --- Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:1234@localhost:5432/infinity_recs'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Spotify API credentials ---
CLIENT_ID = '5f692e70fbab409798ea792e8cb6b721'
CLIENT_SECRET = 'cdc232c03db74bbeab62d3a832ba027f'
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

# --- SQLAlchemy Models ---
class Song(db.Model):
    __tablename__ = 'songs'
    song_id = db.Column(db.Integer, primary_key=True)
    track_id = db.Column(db.String(100), unique=True, nullable=False)
    track_name = db.Column(db.String(255), nullable=False)
    artist_name = db.Column(db.String(255))
    genre = db.Column(db.String(255))
    popularity = db.Column(db.Integer)
    year = db.Column(db.Integer)
    spotify_url = db.Column(db.String(255))
    preview_url = db.Column(db.String(255))
    album_image_url = db.Column(db.String(255))

class User(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserSongInteraction(db.Model):
    __tablename__ = 'user_song_interactions'
    interaction_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    track_id = db.Column(db.String(100), db.ForeignKey('songs.track_id'), nullable=False)
    rating = db.Column(db.Float, nullable=True)
    liked = db.Column(db.Boolean, default=False)
    watched = db.Column(db.Boolean, default=False)
    clicked = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())

# Create indexes for faster queries
Index('idx_user_song_interactions_user_id', UserSongInteraction.user_id)
Index('idx_songs_track_id', Song.track_id)

# --- Load and preprocess dataset ---
DATASET_PATH = 'C:\\Users\\nihal\\PycharmProjects\\Mini\\INFINITY_RECS\\backend\\Datasets\\spotify_data_processed.csv'
CACHE_PATH = 'C:\\Users\\nihal\\PycharmProjects\\Mini\\INFINITY_RECS\\backend\\cache'

os.makedirs(CACHE_PATH, exist_ok=True)

# Cache file paths
COMBINED_FEATURES_CACHE = os.path.join(CACHE_PATH, 'combined_features.pkl')
TFIDF_ARTIST_CACHE = os.path.join(CACHE_PATH, 'tfidf_artist.pkl')
TFIDF_GENRE_CACHE = os.path.join(CACHE_PATH, 'tfidf_genre.pkl')
KNN_CACHE = os.path.join(CACHE_PATH, 'knn_model.pkl')
PPO_MODEL_PATH = os.path.join(CACHE_PATH, 'song_ppo.pt')

# Load dataset with optimizations
necessary_columns = ['track_id', 'track_name', 'artist_name', 'genre', 'popularity', 'year'] + [
    'danceability', 'energy', 'loudness', 'tempo', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness'
]
try:
    df = pd.read_csv(DATASET_PATH, usecols=necessary_columns, low_memory=True)
    logging.info("Dataset loaded successfully")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    raise

df.fillna('', inplace=True)
df['popularity'] = df['popularity'].apply(lambda x: int(x / 5))

# Load or compute features with reduced dimensions
key_audio_features = ['danceability', 'energy', 'loudness', 'tempo', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness']
if os.path.exists(COMBINED_FEATURES_CACHE):
    combined_features = joblib.load(COMBINED_FEATURES_CACHE)
    tfidf_artist = joblib.load(TFIDF_ARTIST_CACHE)
    tfidf_genre = joblib.load(TFIDF_GENRE_CACHE)
    knn = joblib.load(KNN_CACHE)
    logging.info("Loaded cached features and KNN model")
else:
    scaler = StandardScaler()
    scaled_audio_features = scaler.fit_transform(df[key_audio_features])

    tfidf_artist = TfidfVectorizer(max_features=1000)
    artist_matrix = tfidf_artist.fit_transform(df['artist_name']) * 5

    tfidf_genre = TfidfVectorizer(max_features=250)
    genre_matrix = tfidf_genre.fit_transform(df['genre']) * 3

    combined_features = hstack([scaled_audio_features, artist_matrix, genre_matrix]).tocsr()
    combined_features = normalize(combined_features, norm='l2')

    knn = NearestNeighbors(n_neighbors=11, metric='cosine')
    knn.fit(combined_features)

    joblib.dump(combined_features, COMBINED_FEATURES_CACHE)
    joblib.dump(tfidf_artist, TFIDF_ARTIST_CACHE)
    joblib.dump(tfidf_genre, TFIDF_GENRE_CACHE)
    joblib.dump(knn, KNN_CACHE)
    logging.info("Computed and cached features and KNN model")

# Create a mapping from track_id to DataFrame index
track_id_to_index = {row['track_id']: idx for idx, row in df.iterrows()}

# --- PPO Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.softmax(x, dim=-1)

# Initialize PPO model
state_dim = combined_features.shape[1]
num_songs = len(df)
policy_net = PolicyNetwork(state_dim, num_songs)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

# --- Reward Function ---
def compute_reward(interaction):
    reward = 0
    if interaction.rating:
        reward += float(interaction.rating) / 5.0
    if interaction.liked:
        reward += 1
    else:
        reward -= 1  # Treat liked: False as dislike
    if interaction.clicked:
        reward += 0.5
    if interaction.watched:
        reward += 0.75
    return reward

# --- PPO Training ---
def train_ppo(first_n=500):
    interactions = UserSongInteraction.query.limit(first_n).all()
    states, actions, rewards = [], [], []

    for inter in interactions:
        song = Song.query.filter_by(track_id=inter.track_id).first()
        if not song:
            logging.debug(f"No song found for track_id: {inter.track_id}")
            continue
        song_row = df[df['track_id'] == song.track_id]
        if song_row.empty:
            logging.debug(f"Track_id {inter.track_id} not in dataset")
            continue
        idx = song_row.index[0]
        state = combined_features[idx].toarray()[0]
        states.append(state)
        actions.append(idx)
        reward = compute_reward(inter)
        rewards.append(reward)

    if not states:
        logging.warning("No valid states for PPO training")
        return

    logging.info(f"Training PPO with {len(states)} interactions")
    try:
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
    except Exception as e:
        logging.error(f"Error creating tensors for PPO: {e}")
        return

    for epoch in range(10):
        logits = policy_net(states_tensor)
        log_probs = torch.log(logits.gather(1, actions_tensor.view(-1, 1)).squeeze())
        loss = -(log_probs * rewards_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(policy_net.state_dict(), PPO_MODEL_PATH)
    logging.info("PPO model trained and saved")

# --- Helper Functions ---
def sort_by_popularity_year(recommendations, pop_weight=0.5, year_weight=0.5):
    if recommendations.empty:
        return recommendations
    max_popularity = recommendations['popularity'].max()
    min_popularity = recommendations['popularity'].min()
    max_year = recommendations['year'].max()
    min_year = recommendations['year'].min()

    recommendations['normalized_popularity'] = (
        (recommendations['popularity'] - min_popularity) / (max_popularity - min_popularity)
        if max_popularity != min_popularity else 1.0
    )
    recommendations['normalized_year'] = (
        (recommendations['year'] - min_year) / (max_year - min_year)
        if max_year != min_year else 1.0
    )
    recommendations['score'] = (
        pop_weight * recommendations['normalized_popularity'] +
        year_weight * recommendations['normalized_year']
    )
    return recommendations.sort_values('score', ascending=False)

def get_or_create_user(username):
    user = User.query.filter_by(username=username).first()
    if not user:
        user = User(username=username)
        db.session.add(user)
        db.session.commit()
        logging.info(f"Created new user: {username}, user_id: {user.user_id}")
    return user

def get_spotify_track_details(track_id, include_preview=False):
    song = Song.query.filter_by(track_id=track_id).first()
    if song and song.album_image_url and song.spotify_url:
        return song.album_image_url, song.spotify_url if not include_preview else song.preview_url
    try:
        track = sp.track(track_id)
        image_url = track['album']['images'][0]['url'] if track['album']['images'] else None
        song_url = track['preview_url'] if include_preview else track['external_urls']['spotify']
        if song:
            song.album_image_url = image_url
            song.spotify_url = song_url if not include_preview else song.preview_url
            db.session.commit()
        return image_url, song_url
    except Exception as e:
        logging.error(f"Error fetching track details for {track_id}: {e}")
        return None, None

def recommend_songs_by_name(track_name, df, knn_model, combined_features, top_n=10, interacted_track_ids=None):
    song_idx = df.index[df['track_name'] == track_name].tolist()
    if not song_idx:
        logging.info(f"Song not found: {track_name}")
        return pd.DataFrame()

    song_idx = song_idx[0]
    song_features = combined_features[song_idx].toarray()
    distances, indices = knn_model.kneighbors(song_features, n_neighbors=top_n + 1)
    recommended_indices = indices[0][1:]

    recommendations = df.iloc[recommended_indices][
        ['track_name', 'artist_name', 'genre', 'popularity', 'year', 'track_id']
    ].copy()
    recommendations['similarity'] = 1 - distances[0][1:]
    recommendations = sort_by_popularity_year(recommendations)

    if interacted_track_ids:
        recommendations = recommendations[~recommendations['track_id'].isin(interacted_track_ids)]

    recommendations[['album_image_url', 'spotify_url']] = recommendations['track_id'].apply(
        lambda x: pd.Series(get_spotify_track_details(x))
    )

    return recommendations.reset_index(drop=True)

# --- User-Based PPO Recommendations ---
def get_user_recommendations(username, top_k=16):
    user = User.query.filter_by(username=username).first()
    if not user:
        logging.info(f"User not found: {username}")
        return []

    # Train PPO only if significant new interactions (e.g., 20+ new)
    last_interaction = UserSongInteraction.query.order_by(UserSongInteraction.timestamp.desc()).first()
    interaction_count = UserSongInteraction.query.filter_by(user_id=user.user_id).count()
    model_exists = os.path.exists(PPO_MODEL_PATH)
    should_train = not model_exists or (last_interaction and interaction_count > 20 and os.path.getmtime(PPO_MODEL_PATH) < last_interaction.timestamp.timestamp())
    if should_train:
        train_ppo()

    all_interactions = UserSongInteraction.query.filter_by(user_id=user.user_id).all()
    interacted_songs = [
        Song.query.filter_by(track_id=interaction.track_id).first()
        for interaction in all_interactions
        if Song.query.filter_by(track_id=interaction.track_id).first()
    ]
    interacted_track_ids = [song.track_id for song in interacted_songs if song]

    watched_songs = [inter for inter in all_interactions if inter.watched]
    logging.info(f"User {username} has {len(watched_songs)} watched interactions")

    if len(watched_songs) < 5:
        logging.info(f"Insufficient watched interactions, falling back to KNN-based")
        default_track = df.iloc[0]['track_name']
        recommendations = recommend_songs_by_name(default_track, df, knn, combined_features, top_n=top_k, interacted_track_ids=interacted_track_ids)
        return recommendations.to_dict(orient='records') if not recommendations.empty else []

    state = combined_features.mean(axis=0).A
    watched_track_ids = [song.track_id for song in interacted_songs if song and any(inter.watched for inter in all_interactions if inter.track_id == song.track_id)]
    if watched_track_ids:
        indices = [track_id_to_index.get(track_id) for track_id in watched_track_ids if track_id in track_id_to_index]
        if indices:
            weights = np.linspace(0.5, 1.0, len(indices))
            weights = weights / weights.sum()
            weighted_states = np.array([combined_features[idx].toarray()[0] * w for idx, w in zip(indices, weights)])
            state = weighted_states.sum(axis=0)

    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    try:
        policy_net.load_state_dict(torch.load(PPO_MODEL_PATH, map_location=torch.device('cpu')))
    except FileNotFoundError:
        logging.info(f"PPO model not found, falling back to KNN-based for user: {username}")
        default_track = df.iloc[0]['track_name']
        recommendations = recommend_songs_by_name(default_track, df, knn, combined_features, top_n=top_k, interacted_track_ids=interacted_track_ids)
        return recommendations.to_dict(orient='records') if not recommendations.empty else []

    policy_net.eval()
    with torch.no_grad():
        action_probs = policy_net(state_tensor)

    top_indices = torch.topk(action_probs, top_k * 2, dim=1).indices.numpy()[0]
    recommendations = df.iloc[top_indices][
        ['track_name', 'artist_name', 'genre', 'popularity', 'year', 'track_id']
    ].copy()

    recommendations = recommendations[~recommendations['track_id'].isin(interacted_track_ids)]

    if len(recommendations) < top_k:
        remaining_songs = df[~df['track_id'].isin(interacted_track_ids)]
        additional_recs = remaining_songs.head(top_k - len(recommendations))
        recommendations = pd.concat([recommendations, additional_recs]).reset_index(drop=True)

    recommendations = sort_by_popularity_year(recommendations)
    recommendations[['album_image_url', 'spotify_url']] = recommendations['track_id'].apply(
        lambda x: pd.Series(get_spotify_track_details(x))
    )

    recs_with_ids = []
    recommended_track_ids = []
    for idx, row in recommendations.iloc[:top_k].iterrows():
        recs_with_ids.append({
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "artist_name": row["artist_name"],
            "genre": row["genre"],
            "popularity": int(row["popularity"]),
            "year": int(row["year"]),
            "album_image_url": row["album_image_url"],
            "spotify_url": row["spotify_url"]
        })
        recommended_track_ids.append(row["track_id"])

    logging.info(f"PPO-based recommendations generated for user: {username}, count: {len(recs_with_ids)}")
    return recs_with_ids

# --- Routes ---
@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    username = request.args.get('username')

    if not title and not username:
        return jsonify({"error": "Username or title is required"}), 400

    logging.debug(f"Recommend request: title={title}, username={username}")
    if username:
        recommendations = get_user_recommendations(username)
    else:
        recommendations = recommend_songs_by_name(title, df, knn, combined_features)
        recommendations = recommendations.to_dict(orient='records') if not recommendations.empty else []

    return jsonify(recommendations)

@app.route('/save_interaction', methods=['POST'])
def save_interaction():
    data = request.json
    username = data.get('username')
    track_id = data.get('track_id')
    rating = data.get('rating')
    liked = data.get('liked', False)
    watched = data.get('watched', False)
    clicked = data.get('clicked', False)

    logging.debug(f"Save interaction: username={username}, track_id={track_id}, rating={rating}, liked={liked}, watched={watched}, clicked={clicked}")

    if not username or not track_id:
        logging.error("Missing username or track_id")
        return jsonify({"error": "Username and Track ID are required"}), 400

    try:
        user = get_or_create_user(username)
        logging.debug(f"User: {user.username}, user_id: {user.user_id}")
        song = Song.query.filter_by(track_id=track_id).first()
        if not song:
            logging.debug(f"No song found for track_id: {track_id}, fetching from Spotify")
            image_url, preview_url = get_spotify_track_details(track_id, include_preview=True)
            if not image_url:
                logging.error("Failed to fetch song details from Spotify")
                return jsonify({"error": "Failed to fetch song details from Spotify"}), 500

            song_row = df[df['track_id'] == track_id]
            if song_row.empty:
                logging.error(f"Track_id {track_id} not found in dataset")
                return jsonify({"error": "Track ID not found in dataset"}), 400

            song = Song(
                track_id=track_id,
                track_name=song_row.iloc[0]['track_name'],
                artist_name=song_row.iloc[0]['artist_name'],
                genre=song_row.iloc[0]['genre'],
                popularity=int(song_row.iloc[0]['popularity']),
                year=int(song_row.iloc[0]['year']),
                spotify_url=image_url,
                preview_url=preview_url,
                album_image_url=image_url
            )
            db.session.add(song)
            db.session.commit()
            logging.debug(f"Added new song: {song.track_name}")

        existing_interaction = UserSongInteraction.query.filter_by(user_id=user.user_id, track_id=track_id).first()
        if existing_interaction:
            existing_interaction.rating = rating
            existing_interaction.liked = liked
            existing_interaction.watched = watched
            existing_interaction.clicked = clicked
            logging.debug(f"Updated interaction: interaction_id={existing_interaction.interaction_id}")
        else:
            interaction = UserSongInteraction(
                user_id=user.user_id,
                track_id=track_id,
                rating=rating,
                liked=liked,
                watched=watched,
                clicked=clicked
            )
            db.session.add(interaction)
            logging.debug("Created new interaction")
        db.session.commit()
        logging.info(f"Interaction saved for user: {username}, track_id: {track_id}")
        return jsonify({"message": "Interaction saved successfully"}), 200
    except Exception as e:
        db.session.rollback()
        logging.error(f"Failed to save interaction: {str(e)}")
        return jsonify({"error": f"Failed to save interaction: {str(e)}"}), 500

@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('query')
    if not query:
        return jsonify([]), 200
    try:
        matches = df[df['track_name'].str.contains(query, case=False, na=False)]['track_name'].head(10).tolist()
        logging.debug(f"Suggestions for query '{query}': {matches}")
        return jsonify(matches), 200
    except Exception as e:
        logging.error(f"Error in suggest endpoint: {str(e)}")
        return jsonify([]), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000, threads=4)