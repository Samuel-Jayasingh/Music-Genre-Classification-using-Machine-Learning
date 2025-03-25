import sys
import os
# Ensure we can import from one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.similarity import compute_llm_similarity
from src.preprocessing import preprocess_data

class FeedbackManager:
    def __init__(self, data_path='data/user_interactions.csv'):
        """
        FeedbackManager handles user feedback (like/dislike) and 
        maintains a user profile for personalized recommendations.
        """
        # 1. Load or create user interactions
        data_file_abs = os.path.join(os.path.dirname(__file__), '..', data_path)
        self.feedback_log = pd.read_csv(data_file_abs) if os.path.exists(data_file_abs) else pd.DataFrame()
        self.user_profiles = {}
        
        # 2. Build absolute paths for the main CSV and embeddings
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        tracks_csv_path = os.path.join(root_path, 'data', 'tracks_records.csv')
        genre_embed_path = os.path.join(root_path, 'data', 'genre_embeddings.npy')
        artist_embed_path = os.path.join(root_path, 'data', 'artist_embeddings.npy')

        # 3. Load preprocessed data
        self.tracks, self.text_matrix, self.numeric_matrix, _, _ = preprocess_data(tracks_csv_path)
        
        # 4. Load precomputed embeddings
        self.genre_embeddings = np.load(genre_embed_path)
        self.artist_embeddings = np.load(artist_embed_path)

        # 5. If the number of tracks > embedding size, reduce the data
        n_embeddings = self.genre_embeddings.shape[0]
        if self.tracks.shape[0] > n_embeddings:
            print(f"Warning: Reducing tracks from {self.tracks.shape[0]} to {n_embeddings} to match embeddings.")
            self.tracks = self.tracks.iloc[:n_embeddings].reset_index(drop=True)
            self.text_matrix = self.text_matrix[:n_embeddings]
            self.numeric_matrix = self.numeric_matrix[:n_embeddings]

    def log_feedback(self, user_id, song_name, feedback_type):
        """Log user interaction (like/dislike)."""
        print(f"Logging feedback: {user_id} - {song_name} - {feedback_type}")
        new_entry = {
            'user_id': user_id,
            'song': song_name,
            'feedback': feedback_type,
            'timestamp': pd.Timestamp.now()
        }
        self.feedback_log = pd.concat([self.feedback_log, pd.DataFrame([new_entry])], ignore_index=True)
        self.update_user_profile(user_id, song_name, feedback_type)
    
    def update_user_profile(self, user_id, song_name, feedback_type):
        """Update the user's preference vector based on feedback."""
        # Initialize user profile if it doesn't exist
        if user_id not in self.user_profiles:
            embed_dim = self.genre_embeddings.shape[1]
            self.user_profiles[user_id] = {
                'preference_vector': np.zeros(embed_dim),
                'interaction_count': 0
            }
        
        # Find the track index by track_name
        song_idx = self.tracks[self.tracks['track_name'] == song_name].index
        if len(song_idx) == 0:
            print(f"Warning: Song '{song_name}' not found in dataset.")
            return
        
        song_idx = song_idx[0]
        weight = 1.0 if feedback_type == 'like' else -0.5
        
        # Combine genre & artist embeddings
        llm_vector = self.genre_embeddings[song_idx] * 0.6 + self.artist_embeddings[song_idx] * 0.4
        
        self.user_profiles[user_id]['preference_vector'] += weight * llm_vector
        self.user_profiles[user_id]['interaction_count'] += 1

        print(f"Updated user profile for {user_id}: {self.user_profiles[user_id]['preference_vector'][:5]}")
    
    def get_personalized_recommendations(self, user_id, top_n=10):
        """Generate personalized recommendations for a user."""
        # If no profile or no interactions, return random songs
        if user_id not in self.user_profiles or self.user_profiles[user_id]['interaction_count'] == 0:
            print("User has no interactions. Returning random songs.")
            return self.tracks.sample(top_n)
        
        user_vector = self.user_profiles[user_id]['preference_vector']
        similarities = []
        for idx in range(len(self.tracks)):
            # Merge genre & artist embeddings for each song
            song_vector = self.genre_embeddings[idx] * 0.6 + self.artist_embeddings[idx] * 0.4
            sim = cosine_similarity([user_vector], [song_vector])[0][0]
            similarities.append(sim)
        
        print(f"Similarities calculated. Example values: {similarities[:5]}")
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return self.tracks.iloc[top_indices][['track_name', 'artists', 'popularity']]
    
    def retrain_model_with_feedback(self):
        """Periodically retrain the recommendation model with new feedback."""
        positive_feedback = self.feedback_log[self.feedback_log['feedback'] == 'like']
        print(f"Retraining model with {len(positive_feedback)} positive feedback entries.")
        # Implementation for actual retraining would go here

# ========== Example Test ==========
if __name__ == "__main__":
    feedback_manager = FeedbackManager()

    # Example user interactions (ensure these track names exist in your CSV)
    feedback_manager.log_feedback("user_123", "Shape of You", "like")
    feedback_manager.log_feedback("user_123", "Melody", "dislike")
    feedback_manager.log_feedback("user_123", "Rap", "like")

    # Get personalized recommendations for the user
    recommendations = feedback_manager.get_personalized_recommendations("user_123", top_n=5)
    print("\nPersonalized Recommendations:")
    print(recommendations)
