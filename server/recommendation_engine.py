from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS for CORS support
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load data
users_df = pd.read_csv('Users.csv')
clicks_df = pd.read_csv('Clicks.csv')
purchases_df = pd.read_csv('Purchases.csv')
items_df = pd.read_csv('Items.csv')

# Preprocessing data
def preprocess_data(users, clicks, purchases, items):
    clicks_items = clicks.merge(items, on='item_id', how='left')
    purchases_items = purchases.merge(items, on='item_id', how='left')
    user_clicks = clicks_items.groupby('user_id')['item_name'].apply(list).reset_index()
    user_purchases = purchases_items.groupby('user_id')['item_name'].apply(list).reset_index()
    user_data = pd.merge(user_clicks, user_purchases, on='user_id', how='left', suffixes=('_clicks', '_purchases'))
    return user_data

# Collaborative filtering with probability scores
def collaborative_filtering(user_data, user_id):
    user_data['item_name_purchases'] = user_data['item_name_purchases'].apply(lambda x: x if isinstance(x, list) else [])
    user_data['all_items'] = user_data['item_name_clicks'] + user_data['item_name_purchases']
    user_item_matrix = user_data.set_index('user_id')['all_items'].apply(lambda x: ' '.join(x))
    
    # Apply TF-IDF to vectorize user-item interactions
    tfidf = TfidfVectorizer(stop_words='english')
    user_item_matrix_tfidf = tfidf.fit_transform(user_item_matrix)
    
    # Compute cosine similarity between users
    user_similarities = cosine_similarity(user_item_matrix_tfidf, user_item_matrix_tfidf)
    user_idx = user_data[user_data['user_id'] == user_id].index[0]
    
    # Find top 5 similar users
    similar_users_idx = user_similarities[user_idx].argsort()[::-1][1:6]
    similar_users_items = user_data.iloc[similar_users_idx]['all_items'].values
    
    # Get the similarity scores (proxy for probability)
    probabilities = user_similarities[user_idx][similar_users_idx]
    
    # Get recommended items with their corresponding probability scores
    recommended_items = {}
    for i, items in enumerate(similar_users_items):
        for item in items:
            if item not in recommended_items:
                recommended_items[item] = probabilities[i]
    
    return recommended_items

# Content-based filtering with probability scores
def content_based_filtering(items, user_clicks):
    item_matrix = items[['item_name', 'category', 'price']].copy()
    item_matrix['combined_features'] = item_matrix['category'] + ' ' + item_matrix['price'].astype(str)
    
    # Apply TF-IDF to vectorize the combined features
    tfidf = TfidfVectorizer(stop_words='english')
    item_matrix_tfidf = tfidf.fit_transform(item_matrix['combined_features'])
    
    clicked_item_names = user_clicks['item_name_clicks'].values[0]
    clicked_items_tfidf = item_matrix_tfidf[item_matrix['item_name'].isin(clicked_item_names)]
    
    # Calculate similarity with all other items
    item_similarities = cosine_similarity(clicked_items_tfidf, item_matrix_tfidf)
    
    # Get top 5 recommended items and their corresponding similarity scores
    top_items_idx = item_similarities.argsort()[0, ::-1][:5]
    recommended_items = {}
    for idx in top_items_idx:
        recommended_items[item_matrix.iloc[idx]['item_name']] = item_similarities[0, idx]
    
    return recommended_items

# Hybrid approach to combine collaborative and content-based filtering
def hybrid_recommendation(users, clicks, purchases, items, user_id):
    user_data = preprocess_data(users, clicks, purchases, items)
    
    # Get collaborative filtering recommendations with probabilities
    collaborative_recommendations = collaborative_filtering(user_data, user_id)
    
    # Get content-based filtering recommendations with probabilities
    user_clicks = user_data[user_data['user_id'] == user_id]
    content_recommendations = content_based_filtering(items, user_clicks)
    
    # Merge both sets of recommendations, taking the higher probability where applicable
    final_recommendations = {**collaborative_recommendations, **content_recommendations}
    
    return final_recommendations

# Flask route to get recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_name = data.get('user_name', '').strip().title()
    try:
        user_id = users_df.loc[users_df['name'] == user_name, 'user_id'].values[0]
        recommended_items = hybrid_recommendation(users_df, clicks_df, purchases_df, items_df, user_id)
        print({
            "user": user_name,
            "recommendations": [
                {"item": item, "probability": round(probability, 2)}
                for item, probability in recommended_items.items()
            ]
        })
        return jsonify({
            "user": user_name,
            "recommendations": [
                {"item": item, "probability": round(probability, 2)}
                for item, probability in recommended_items.items()
            ]
        })
    except IndexError:
        return jsonify({"error": "Invalid user"}), 400

if __name__ == "__main__":
    app.run(debug=True)
