# import sqlite3

# def init_db():
#     conn = sqlite3.connect('infinity_recs.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS users
#                  (user_id TEXT UNIQUE NOT NULL,
#                   email TEXT NOT NULL,
#                   username TEXT)''')
#     conn.commit()
#     conn.close()

# def get_db_connection():
#     conn = sqlite3.connect("infinity_recs.db")
#     conn.row_factory = sqlite3.Row  # Allows accessing columns by name
#     return conn

# def save_user(user_id, email, username=None):
#     conn = get_db_connection()
#     c = conn.cursor()
#     c.execute('''INSERT OR REPLACE INTO users (user_id, email, username)
#                  VALUES (?, ?, ?)''', (user_id, email, username))
#     conn.commit()
#     conn.close()

# def get_user(user_id):
#     conn = get_db_connection()
#     c = conn.cursor()
#     c.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
#     user = c.fetchone()
#     conn.close()
#     return dict(user) if user else None
import sqlite3

def init_db():
    conn = sqlite3.connect('infinity_recs.db')
    c = conn.cursor()
    
    # Existing users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY,
                  user_id TEXT UNIQUE NOT NULL,
                  email TEXT NOT NULL,
                  username TEXT)''')
    
    # New interactions table
    c.execute('''CREATE TABLE IF NOT EXISTS interactions
                 (id INTEGER PRIMARY KEY,
                  user_id TEXT NOT NULL,
                  category TEXT NOT NULL,
                  category_id TEXT NOT NULL,
                  interactions INTEGER DEFAULT 0,
                  FOREIGN KEY (user_id) REFERENCES users(user_id))''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect("infinity_recs.db")
    conn.row_factory = sqlite3.Row
    return conn

def save_user(user_id, email, username=None):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO users (user_id, email, username)
                 VALUES (?, ?, ?)''', (user_id, email, username))
    conn.commit()
    conn.close()

def get_user(user_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

# Placeholder for retrieving category data (to be used later with recommender system)
def get_category_data(user_id, category):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM interactions WHERE user_id = ? AND category = ?', (user_id, category))
    data = c.fetchall()
    conn.close()
    return [dict(row) for row in data]