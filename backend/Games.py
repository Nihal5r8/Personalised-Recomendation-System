import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import joblib
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# App setup
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:1234@localhost:5432/infinity_recs'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database models
class User(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Game(db.Model):
    __tablename__ = 'games'
    game_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    genres = db.Column(db.String(255))
    platforms = db.Column(db.String(255))
    rating = db.Column(db.Numeric(3, 2))
    released = db.Column(db.Date)
    cover_image = db.Column(db.String(255))
    game_link = db.Column(db.String(255))
    release_year = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserGameInteraction(db.Model):
    __tablename__ = 'user_game_interactions'
    interaction_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    game_id = db.Column(db.Integer, db.ForeignKey('games.game_id'), nullable=False)
    rating = db.Column(db.Numeric(3, 2))
    liked = db.Column(db.Boolean, default=False)
    clicked = db.Column(db.Boolean, default=False)
    watched = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Load data and models
try:
    df = pd.read_csv('C:\\Users\\nihal\\PycharmProjects\\Mini\\INFINITY_RECS\\backend\\Datasets\\games.csv', encoding='latin1')
    logging.info("Dataset loaded successfully")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    raise

# Clean dataset
df.fillna('', inplace=True)
df["combined"] = df["genres"] + " " + df["platforms"]

# Convert rating to numeric, replacing invalid values with NaN
def clean_rating(value):
    try:
        return float(value) if value != '' else np.nan
    except (ValueError, TypeError):
        logging.warning(f"Invalid rating value: {value}")
        return np.nan

df["rating"] = df["rating"].apply(clean_rating)
logging.info(f"Rating column cleaned. Non-null count: {df['rating'].notnull().sum()}")

# Load TF-IDF and similarity matrix
try:
    games_tfidf = joblib.load('C:\\Users\\nihal\\PycharmProjects\\Mini\\INFINITY_RECS\\backend\\MODEL\\GAMES\\tfidf_vectorizer.pkl')
    games_tfidf_matrix = games_tfidf.transform(df["combined"])
    games_similarity_matrix = joblib.load('C:\\Users\\nihal\\PycharmProjects\Mini\\INFINITY_RECS\\backend\\MODEL\\GAMES\\similarity_matrix.pkl')
    logging.info("TF-IDF and similarity matrix loaded successfully")
except Exception as e:
    logging.error(f"Failed to load TF-IDF or similarity matrix: {e}")
    raise

# Create a mapping from game title to DataFrame index
game_title_to_index = {row['name']: idx for idx, row in df.iterrows()}

# Verify dataset columns
expected_columns = ['name', 'genres', 'platforms', 'rating', 'released', 'cover_image', 'game_link']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    logging.error(f"Missing columns in dataset: {missing_columns}")
    raise ValueError(f"Dataset missing required columns: {missing_columns}")

# Robustly extract release year
def extract_year(date_str):
    if not date_str or pd.isna(date_str):
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").year
    except:
        try:
            return datetime.strptime(date_str, "%d-%m-%Y").year
        except:
            logging.warning(f"Invalid date format: {date_str}")
            return None

# Create release_year column
df["release_year"] = df["released"].apply(extract_year)
logging.info(f"Release year column created. Non-null count: {df['release_year'].notnull().sum()}")

# PPO Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.softmax(x, dim=-1)

# Initialize PPO model
state_dim = games_tfidf_matrix.shape[1]
num_games = len(df)
policy_net = PolicyNetwork(state_dim, num_games)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

# Reward Function
def compute_reward(interaction):
    reward = 0
    if interaction.rating:
        reward += float(interaction.rating) / 5.0
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
    interactions = UserGameInteraction.query.all()
    states, actions, rewards = [], [], []
    logging.debug(f"Total interactions found: {len(interactions)}")

    for inter in interactions:
        game = Game.query.get(inter.game_id)
        if not game:
            logging.warning(f"Game not found for interaction ID: {inter.interaction_id}")
            continue
        game_row = df[df['name'] == game.title]
        if game_row.empty:
            logging.warning(f"Game title not in dataset: {game.title}")
            continue
        idx = game_row.index[0]
        state = games_tfidf_matrix[idx].toarray()[0]
        states.append(state)
        actions.append(idx)
        reward = compute_reward(inter)
        rewards.append(reward)
        logging.debug(f"Interaction ID: {inter.interaction_id}, Game: {game.title}, Reward: {reward}")

    if not states:
        logging.info("No valid interactions for PPO training")
        return

    logging.info(f"Training PPO with {len(states)} interactions")
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)

    for epoch in range(50):
        logits = policy_net(states_tensor)
        log_probs = torch.log(logits.gather(1, actions_tensor.view(-1, 1)).squeeze())
        loss = -(log_probs * rewards_tensor).mean()
        logging.debug(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(policy_net.state_dict(), r"C:\Users\nihal\PycharmProjects\Mini\INFINITY_RECS\backend\MODEL\GAMES\game_ppo.pt")
    logging.info("PPO model trained and saved")

# Utility functions
def parse_date(date_str):
    if not date_str or pd.isna(date_str):
        return None
    try:
        return datetime.strptime(date_str, "%d-%m-%Y")
    except:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            logging.warning(f"Invalid date format for parsing: {date_str}")
            return None

def find_closest_game(query, titles):
    scores = [(title, fuzz.token_sort_ratio(query.lower(), title.lower())) for title in titles]
    best_match, score = max(scores, key=lambda x: x[1])
    return best_match if score >= 50 else None

# Content-based recommendation
def recommend_content_based_games(query, initial_n=20, final_n=20, date_weight=0.4):
    closest_title = find_closest_game(query, df["name"])
    if not closest_title:
        logging.info(f"No matching game found for query: {query}")
        return []

    game_idx = df[df["name"] == closest_title].index[0]
    matched_year = df.loc[game_idx, "release_year"]
    if pd.isna(matched_year):
        logging.warning(f"No valid release year for game: {closest_title}")
        matched_year = None

    sim_scores = []
    for idx, sim in enumerate(games_similarity_matrix[game_idx]):
        if idx == game_idx:
            continue
        release_year = df.loc[idx, "release_year"]
        if matched_year is None or pd.isna(release_year):
            year_score = 0.0
        else:
            year_score = 1.0 if release_year >= matched_year else 0.0
        combined_score = (1 - date_weight) * sim + date_weight * year_score
        sim_scores.append((idx, combined_score))

    top_indices = [idx for idx, _ in sorted(sim_scores, key=lambda x: x[1], reverse=True)[:final_n]]
    recommendations = df.iloc[top_indices][["name", "genres", "platforms", "rating", "released", "cover_image", "game_link"]].copy()

    recs_with_ids = []
    for idx, row in recommendations.iterrows():
        try:
            rating = float(row["rating"]) if pd.notna(row["rating"]) and row["rating"] != '' else None
        except (ValueError, TypeError):
            logging.warning(f"Invalid rating for game {row['name']}: {row['rating']}")
            rating = None
        recs_with_ids.append({
            "game_id": str(idx),
            "title": row["name"],
            "genres": row["genres"],
            "platforms": row["platforms"],
            "rating": rating,
            "released": row["released"],
            "image_url": row["cover_image"],
            "link": row["game_link"]
        })

    logging.info(f"Content-based recommendations generated for query: {query}, count: {len(recs_with_ids)}")
    return recs_with_ids

# PPO-based recommendation
def get_user_recommendations(username, top_k=20):
    user = User.query.filter_by(username=username).first()
    if not user:
        logging.info(f"User not found: {username}")
        return []

    train_ppo()

    # Get all interacted games
    all_interactions = UserGameInteraction.query.filter_by(user_id=user.user_id).all()
    interacted_games = [
        Game.query.filter_by(game_id=interaction.game_id).first()
        for interaction in all_interactions
        if Game.query.filter_by(game_id=interaction.game_id).first() is not None
    ]
    interacted_titles = [game.title for game in interacted_games if game]
    logging.debug(f"Interacted titles: {interacted_titles}")

    watched_games = [inter for inter in all_interactions if inter.watched]
    logging.info(f"User {username} has {len(watched_games)} watched interactions")

    if len(watched_games) < 5:
        logging.info(f"Insufficient watched interactions, falling back to content-based")
        default_title = df.iloc[0]["name"]
        recommendations = recommend_content_based_games(default_title, final_n=top_k)
        recommendations = [rec for rec in recommendations if rec["title"] not in interacted_titles]
        logging.info(f"Content-based recommendations after filtering: {len(recommendations)}")
        return recommendations

    # Weighted state computation
    state = games_tfidf_matrix.mean(axis=0).A
    watched_titles = [game.title for game in interacted_games if game and any(inter.watched for inter in all_interactions if inter.game_id == game.game_id)]
    if watched_titles:
        indices = [game_title_to_index[title] for title in watched_titles if title in game_title_to_index]
        logging.debug(f"Watched titles: {watched_titles}, Valid indices: {indices}")
        if indices:
            weights = np.linspace(0.5, 1.0, len(indices))
            weights = weights / weights.sum()
            weighted_states = np.array([games_tfidf_matrix[idx].toarray()[0] * w for idx, w in zip(indices, weights)])
            state = weighted_states.sum(axis=0)
        else:
            logging.warning("No valid indices for watched titles, using default state")
    logging.debug(f"State vector shape: {state.shape}, Sample: {state[:5]}")

    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    try:
        policy_net.load_state_dict(torch.load("game_ppo.pt", map_location=torch.device('cpu')))
    except FileNotFoundError:
        logging.warning(f"PPO model not found, falling back to content-based for user: {username}")
        default_title = df.iloc[0]["name"]
        recommendations = recommend_content_based_games(default_title, final_n=top_k)
        recommendations = [rec for rec in recommendations if rec["title"] not in interacted_titles]
        logging.info(f"Content-based recommendations after filtering: {len(recommendations)}")
        return recommendations

    policy_net.eval()
    with torch.no_grad():
        action_probs = policy_net(state_tensor)
    logging.debug(f"Action probabilities sample: {action_probs[0, :5].detach().numpy()}")

    top_indices = torch.topk(action_probs, top_k * 2, dim=1).indices.numpy()[0]
    recommended_games = df.iloc[top_indices][["name", "genres", "platforms", "rating", "released", "cover_image", "game_link"]].copy()

    # Filter out all interacted games
    recommended_games = recommended_games[~recommended_games["name"].isin(interacted_titles)]
    logging.debug(f"PPO recommendations after filtering: {len(recommended_games)} games")

    if len(recommended_games) < top_k:
        logging.info(f"Supplementing with {top_k - len(recommended_games)} additional games")
        remaining_games = df[~df["name"].isin(interacted_titles)]
        additional_recs = remaining_games.head(top_k - len(recommended_games))
        recommended_games = pd.concat([recommended_games, additional_recs]).reset_index(drop=True)

    recs_with_ids = []
    recommended_titles = []
    for idx, row in recommended_games.iloc[:top_k].iterrows():  # Slice DataFrame before iterating
        try:
            rating = float(row["rating"]) if pd.notna(row["rating"]) and row["rating"] != '' else None
        except (ValueError, TypeError):
            logging.warning(f"Invalid rating for game {row['name']}: {row['rating']}")
            rating = None
        recs_with_ids.append({
            "game_id": str(df[df["name"] == row["name"]].index[0]),
            "title": row["name"],
            "genres": row["genres"],
            "platforms": row["platforms"],
            "rating": rating,
            "released": row["released"],
            "image_url": row["cover_image"],
            "link": row["game_link"]
        })
        recommended_titles.append(row["name"])

    logging.info(f"PPO-based recommendations generated for user: {username}, count: {len(recs_with_ids)}")
    logging.debug(f"Recommended titles: {recommended_titles}")
    # Check for overlaps with interacted titles
    overlaps = set(recommended_titles).intersection(set(interacted_titles))
    if overlaps:
        logging.warning(f"Overlaps detected in recommendations: {overlaps}")
    return recs_with_ids

# API Endpoints
@app.route('/recommend', methods=['GET'])
def recommend():
    username = request.args.get('username')
    title = request.args.get('title')

    if not username and not title:
        logging.error("Missing username or title in /recommend request")
        return jsonify({"error": "Username or title is required"}), 400

    if username:
        recommendations = get_user_recommendations(username)
    else:
        recommendations = recommend_content_based_games(title)

    return jsonify(recommendations)

@app.route('/interact', methods=['POST'])
def interact():
    data = request.get_json()
    username = data.get('username')
    game_data = data.get('game')
    interaction_data = data.get('interaction')

    if not username or not game_data or not interaction_data:
        logging.error("Missing required fields in /interact request")
        return jsonify({'error': 'Missing required fields'}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        user = User(username=username)
        db.session.add(user)
        db.session.commit()
        logging.info(f"Created new user: {username}")

    game = Game.query.filter_by(title=game_data['title']).first()
    if not game:
        try:
            rating = float(game_data['rating']) if game_data.get('rating') and game_data['rating'] != '' else None
        except (ValueError, TypeError):
            logging.warning(f"Invalid rating in game data: {game_data.get('rating')}")
            rating = None
        game = Game(
            title=game_data['title'],
            genres=game_data.get('genres'),
            platforms=game_data.get('platforms'),
            rating=rating,
            released=parse_date(game_data['released']) if game_data.get('released') else None,
            cover_image=game_data.get('image_url'),
            game_link=game_data.get('link'),
            release_year=extract_year(game_data['released']) if game_data.get('released') else None
        )
        db.session.add(game)
        db.session.commit()
        logging.info(f"Created new game: {game_data['title']}")

    liked_value = interaction_data.get('liked', False)

    interaction = UserGameInteraction.query.filter_by(user_id=user.user_id, game_id=game.game_id).first()

    if interaction:
        interaction.rating = interaction_data.get('rating')
        interaction.liked = liked_value
        interaction.clicked = interaction_data.get('clicked', False)
        interaction.watched = interaction_data.get('watched', False)
    else:
        interaction = UserGameInteraction(
            user_id=user.user_id,
            game_id=game.game_id,
            rating=interaction_data.get('rating'),
            liked=liked_value,
            clicked=interaction_data.get('clicked', False),
            watched=interaction_data.get('watched', False)
        )
        db.session.add(interaction)

    db.session.commit()
    logging.info(f"Interaction recorded for user: {username}, game: {game_data['title']}")
    return jsonify({'message': 'Interaction recorded successfully'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5002)