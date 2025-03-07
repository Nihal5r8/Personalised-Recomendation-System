from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from sqlalchemy import Index
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Set up the database URI for PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:1234@localhost:5432/infinity_recs'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load recommendation models
MODEL_DIR = "C:\\Users\\nihal\\PycharmProjects\\Mini\\INFINITY_RECS\\backend\\MODEL"
BOOKS_DIR = f"{MODEL_DIR}\\BOOKS"
MLB_MODEL_PATH = f"{BOOKS_DIR}\\mlb_model.pkl"
TFIDF_MODEL_PATH = f"{BOOKS_DIR}\\tfidf_model.pkl"
KNN_MODEL_PATH = f"{BOOKS_DIR}\\knn_model.pkl"
COMBINED_FEATURES_PATH = f"{BOOKS_DIR}\\combined_features.pkl"
BOOKS_DF_PATH = f"{BOOKS_DIR}\\books_df.pkl"

try:
    books_combined_features = joblib.load(COMBINED_FEATURES_PATH)
    books_df = joblib.load(BOOKS_DF_PATH)
    books_knn = joblib.load(KNN_MODEL_PATH)
    books_mlb = joblib.load(MLB_MODEL_PATH)
    books_tfidf = joblib.load(TFIDF_MODEL_PATH)
    logging.info("Loaded cached book models and features")
except Exception as e:
    logging.error(f"Failed to load models: {e}")
    raise

# Ensure books_df has a book_id column
if 'book_id' not in books_df.columns:
    logging.warning("book_id column missing in books_df; using index as book_id")
    books_df['book_id'] = books_df.index
    # Save updated DataFrame to pickle
    joblib.dump(books_df, BOOKS_DF_PATH)
    logging.info("Added book_id column and saved updated books_df.pkl")

# Create a mapping from book_id to DataFrame index
try:
    book_id_to_index = {row['book_id']: idx for idx, row in books_df.iterrows()}
    logging.info("Created book_id_to_index mapping")
except Exception as e:
    logging.error(f"Failed to create book_id_to_index: {e}")
    raise

# Database models
class User(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    created_at = db.Column(db.TIMESTAMP, default=db.func.now())

class Book(db.Model):
    __tablename__ = 'books'
    book_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    author = db.Column(db.String(255))
    genre = db.Column(db.String(255))
    description = db.Column(db.Text)
    rating = db.Column(db.Numeric(3, 2))
    totalratings = db.Column(db.Integer)
    created_at = db.Column(db.TIMESTAMP, default=db.func.now())

class UserBookInteraction(db.Model):
    __tablename__ = 'user_book_interactions'
    interaction_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    book_id = db.Column(db.Integer, db.ForeignKey('books.book_id'), nullable=False)
    rating = db.Column(db.Numeric(3, 2))
    liked = db.Column(db.Boolean, default=False)
    clicked = db.Column(db.Boolean, default=False)
    watched = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.TIMESTAMP, default=db.func.now())

# Create indexes for faster queries
Index('idx_user_book_interactions_user_id', UserBookInteraction.user_id)
Index('idx_books_book_id', Book.book_id)

# --- PPO Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, output_dim)

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
state_dim = books_combined_features.shape[1]
num_books = len(books_df)
policy_net = PolicyNetwork(state_dim, num_books)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

# --- Reward Function ---
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

# --- PPO Training ---
def train_ppo(first_n=1000):
    interactions = UserBookInteraction.query.limit(first_n).all()
    states, actions, rewards = [], [], []
    logging.debug(f"Total interactions found: {len(interactions)}")

    for inter in interactions:
        book = Book.query.filter_by(book_id=inter.book_id).first()
        if not book:
            logging.warning(f"Book not found for interaction ID: {inter.interaction_id}")
            continue
        book_row = books_df[books_df['book_id'] == book.book_id]
        if book_row.empty:
            logging.warning(f"Book ID not in dataset: {book.book_id}")
            continue
        idx = book_row.index[0]
        state = books_combined_features[idx].toarray()[0]
        states.append(state)
        actions.append(idx)
        reward = compute_reward(inter)
        rewards.append(reward)

    if not states:
        logging.info("No valid interactions for PPO training")
        return

    logging.info(f"Training PPO with {len(states)} interactions")
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)

    for epoch in range(20):
        logits = policy_net(states_tensor)
        log_probs = torch.log(logits.gather(1, actions_tensor.view(-1, 1)).squeeze())
        loss = -(log_probs * rewards_tensor).mean()
        logging.debug(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model_path = os.path.join(BOOKS_DIR, "book_ppo.pt")
    torch.save(policy_net.state_dict(), model_path)
    logging.info("PPO model trained and saved")

# --- Helper Functions ---
def get_or_create_user(username):
    user = User.query.filter_by(username=username).first()
    if not user:
        user = User(username=username)
        db.session.add(user)
        db.session.commit()
    return user

def recommend_content_based(title, initial_n=20, final_n=10, interacted_book_ids=None):
    try:
        idx = books_df[books_df['title'] == title].index[0]
        query = books_combined_features[idx].reshape(1, -1)
        distances, indices = books_knn.kneighbors(query, n_neighbors=initial_n + 1)
        recs = books_df.iloc[indices[0][1:]][['book_id', 'title', 'genre', 'desc', 'rating', 'totalratings', 'author']].copy()

        # Filter out interacted books
        if interacted_book_ids:
            recs = recs[~recs['book_id'].isin(interacted_book_ids)]
            logging.debug(f"KNN recommendations after filtering: {len(recs)} books")

        # Add image and link
        recs['img'] = recs['title'].apply(lambda t: books_df[books_df['title'] == t]['img'].values[0] if not books_df[books_df['title'] == t].empty else None)
        recs['link'] = recs['title'].apply(lambda t: books_df[books_df['title'] == t]['link'].values[0] if not books_df[books_df['title'] == t].empty else None)

        # Bayesian scoring
        C = recs['rating'].mean() if not recs.empty else 0
        m = recs['totalratings'].quantile(0.9) if not recs.empty else 0

        def bayesian_rating(row, C, m):
            v = row['totalratings']
            R = row['rating']
            return (v / (v + m) * R) + (m / (v + m) * C) if v + m > 0 else R

        recs['bayesian_score'] = recs.apply(lambda row: bayesian_rating(row, C, m), axis=1)
        recs = recs.sort_values('bayesian_score', ascending=False).head(final_n)

        return recs.to_dict('records')
    except Exception as e:
        logging.error(f"Recommendation error: {e}")
        return []

# --- User-Based PPO Recommendations ---
def get_user_recommendations(username, top_k=10):
    user = User.query.filter_by(username=username).first()
    if not user:
        logging.info(f"User not found: {username}")
        return []

    # Train PPO only if model doesn't exist or new interactions were added
    model_path = os.path.join(BOOKS_DIR, "book_ppo.pt")
    last_interaction = UserBookInteraction.query.order_by(UserBookInteraction.created_at.desc()).first()
    model_exists = os.path.exists(model_path)
    should_train = not model_exists or (last_interaction and os.path.getmtime(model_path) < last_interaction.created_at.timestamp())
    if should_train:
        train_ppo()

    # Get all interacted books
    all_interactions = UserBookInteraction.query.filter_by(user_id=user.user_id).all()
    interacted_books = [
        Book.query.filter_by(book_id=interaction.book_id).first()
        for interaction in all_interactions
        if Book.query.filter_by(book_id=interaction.book_id).first() is not None
    ]
    interacted_book_ids = [book.book_id for book in interacted_books if book]
    logging.debug(f"Interacted book_ids: {interacted_book_ids}")

    watched_books = [inter for inter in all_interactions if inter.watched]
    logging.info(f"User {username} has {len(watched_books)} watched interactions")

    if len(watched_books) < 5:
        logging.info(f"Insufficient watched interactions, falling back to KNN-based")
        default_title = books_df.iloc[0]['title']
        recommendations = recommend_content_based(default_title, initial_n=20, final_n=top_k, interacted_book_ids=interacted_book_ids)
        return recommendations

    # Weighted state computation
    state = books_combined_features.mean(axis=0).A
    watched_book_ids = [book.book_id for book in interacted_books if book and any(inter.watched for inter in all_interactions if inter.book_id == book.book_id)]
    if watched_book_ids:
        indices = [book_id_to_index.get(book_id) for book_id in watched_book_ids if book_id in book_id_to_index]
        indices = [idx for idx in indices if idx is not None]
        logging.debug(f"Watched book_ids: {watched_book_ids}, Valid indices: {indices}")
        if indices:
            weights = np.linspace(0.5, 1.0, len(indices))
            weights = weights / weights.sum()
            weighted_states = np.array([books_combined_features[idx].toarray()[0] * w for idx, w in zip(indices, weights)])
            state = weighted_states.sum(axis=0)
        else:
            logging.warning("No valid indices for watched book_ids, using default state")
    logging.debug(f"State vector shape: {state.shape}, Sample: {state[:5]}")

    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        logging.warning(f"PPO model not found, falling back to KNN-based for user: {username}")
        default_title = books_df.iloc[0]['title']
        recommendations = recommend_content_based(default_title, initial_n=20, final_n=top_k, interacted_book_ids=interacted_book_ids)
        return recommendations

    policy_net.eval()
    with torch.no_grad():
        action_probs = policy_net(state_tensor)
    logging.debug(f"Action probabilities sample: {action_probs[0, :5].detach().numpy()}")

    top_indices = torch.topk(action_probs, top_k * 2, dim=1).indices.numpy()[0]
    recommendations = books_df.iloc[top_indices][
        ['book_id', 'title', 'genre', 'desc', 'rating', 'totalratings', 'author']
    ].copy()

    # Filter out interacted books
    recommendations = recommendations[~recommendations['book_id'].isin(interacted_book_ids)]
    logging.debug(f"PPO recommendations after filtering: {len(recommendations)} books")

    if len(recommendations) < top_k:
        logging.info(f"Supplementing with {top_k - len(recommendations)} additional books")
        remaining_books = books_df[~books_df['book_id'].isin(interacted_book_ids)]
        additional_recs = remaining_books.head(top_k - len(recommendations))
        recommendations = pd.concat([recommendations, additional_recs]).reset_index(drop=True)

    # Add image and link
    recommendations['img'] = recommendations['title'].apply(lambda t: books_df[books_df['title'] == t]['img'].values[0] if not books_df[books_df['title'] == t].empty else None)
    recommendations['link'] = recommendations['title'].apply(lambda t: books_df[books_df['title'] == t]['link'].values[0] if not books_df[books_df['title'] == t].empty else None)

    # Bayesian scoring
    C = recommendations['rating'].mean() if not recommendations.empty else 0
    m = recommendations['totalratings'].quantile(0.9) if not recommendations.empty else 0

    def bayesian_rating(row, C, m):
        v = row['totalratings']
        R = row['rating']
        return (v / (v + m) * R) + (m / (v + m) * C) if v + m > 0 else R

    recommendations['bayesian_score'] = recommendations.apply(lambda row: bayesian_rating(row, C, m), axis=1)
    recommendations = recommendations.sort_values('bayesian_score', ascending=False).head(top_k)

    recs_with_ids = []
    recommended_book_ids = []
    for _, row in recommendations.iterrows():
        recs_with_ids.append({
            "book_id": int(row["book_id"]),
            "title": row["title"],
            "author": row["author"],
            "genre": row["genre"],
            "desc": row["desc"],
            "rating": float(row["rating"]),
            "totalratings": int(row["totalratings"]),
            "img": row["img"],
            "link": row["link"],
            "bayesian_score": float(row["bayesian_score"])
        })
        recommended_book_ids.append(row["book_id"])

    logging.info(f"PPO-based recommendations generated for user: {username}, count: {len(recs_with_ids)}")
    logging.debug(f"Recommended book_ids: {recommended_book_ids}")
    overlaps = set(recommended_book_ids).intersection(set(interacted_book_ids))
    if overlaps:
        logging.warning(f"Overlaps detected in recommendations: {overlaps}")
    return recs_with_ids

# Save user interaction
@app.route('/save_interaction', methods=['POST'])
def save_interaction():
    data = request.get_json()
    username = data.get('username')
    title = data.get('title')
    rating = data.get('rating')
    liked = data.get('liked', False)
    watched = data.get('watched', False)
    clicked = data.get('clicked', False)

    if not username or not title:
        return jsonify({'error': 'Username and title are required'}), 400

    try:
        user = get_or_create_user(username)

        # Get or create the book
        book = Book.query.filter_by(title=title).first()
        if not book:
            book_data = books_df[books_df['title'] == title]
            if not book_data.empty:
                book_data = book_data.iloc[0]
                book_id = int(book_data['book_id'])  # Use book_id from books_df
                book = Book(
                    book_id=book_id,
                    title=str(book_data.get('title', title)),
                    author=str(book_data.get('author', 'Unknown')),
                    genre=str(book_data.get('genre', 'Unknown')),
                    description=str(book_data.get('desc', 'No description available')),
                    rating=float(book_data.get('rating', 0.0)),
                    totalratings=int(book_data.get('totalratings', 0))
                )
                db.session.add(book)
                db.session.commit()
            else:
                return jsonify({'error': 'Book details not found in dataset'}), 404

        # Check if interaction exists
        interaction = UserBookInteraction.query.filter_by(user_id=user.user_id, book_id=book.book_id).first()
        if not interaction:
            interaction = UserBookInteraction(user_id=user.user_id, book_id=book.book_id)

        # Update interaction details
        interaction.rating = rating if rating is not None else interaction.rating
        interaction.liked = liked
        interaction.watched = watched
        interaction.clicked = clicked

        db.session.add(interaction)
        db.session.commit()
        logging.info(f"Interaction saved for user: {username}, book_id: {book.book_id}")
        return jsonify({'message': 'Interaction saved successfully'})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Failed to save interaction: {str(e)}")
        return jsonify({'error': f"Failed to save interaction: {str(e)}"}), 500

# Recommend route
@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    username = request.args.get('username')

    if not title and not username:
        return jsonify({"error": "Username or title is required"}), 400

    if username:
        recommendations = get_user_recommendations(username)
    else:
        recommendations = recommend_content_based(title)
    return jsonify(recommendations)

# Suggest route for autocomplete
@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('query')
    if not query:
        return jsonify([]), 200
    try:
        matches = books_df[books_df['title'].str.contains(query, case=False, na=False)]['title'].head(10).tolist()
        return jsonify(matches), 200
    except Exception as e:
        logging.error(f"Error in suggest endpoint: {str(e)}")
        return jsonify([]), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False, port=5001)