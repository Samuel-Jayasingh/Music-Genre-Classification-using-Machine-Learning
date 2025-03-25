import sys
import os
# Add the project root (one level up) to sys.path so modules can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.similarity import compute_traditional_similarity, compute_llm_similarity
from src.preprocessing import preprocess_data
from src.feedback import FeedbackManager
import joblib

class MusicRecommender:
    def __init__(self):
        """
        Initializes the MusicRecommender by loading:
          - tracks from tracks_records.csv
          - clustering model from clustering_model.pkl
          - embeddings from genre_embeddings.npy & artist_embeddings.npy
          - a FeedbackManager instance
        """

        # Build absolute paths from the location of this file (recommender.py)
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_path = os.path.join(root_path, 'data', 'tracks_records.csv')
        model_path = os.path.join(root_path, 'models', 'clustering_model.pkl')
        genre_embed_path = os.path.join(root_path, 'data', 'genre_embeddings.npy')
        artist_embed_path = os.path.join(root_path, 'data', 'artist_embeddings.npy')

        # 1. Load preprocessed data
        self.tracks, self.text_matrix, self.numeric_features, _, _ = preprocess_data(data_path)

        # 2. Load clustering model
        self.clustering_model = joblib.load(model_path)
        self.clusters = self.clustering_model.predict(self.numeric_features)

        # 3. Load LLM embeddings
        self.genre_embeddings = np.load(genre_embed_path)
        self.artist_embeddings = np.load(artist_embed_path)

        # 4. Ensure the number of tracks does not exceed the embedding count
        n_embeddings = self.genre_embeddings.shape[0]
        if self.tracks.shape[0] > n_embeddings:
            print(f"Warning: Reducing tracks from {self.tracks.shape[0]} to {n_embeddings} to match embeddings.")
            self.tracks = self.tracks.iloc[:n_embeddings].reset_index(drop=True)
            self.text_matrix = self.text_matrix[:n_embeddings]
            self.numeric_features = self.numeric_features[:n_embeddings]
            self.clusters = self.clustering_model.predict(self.numeric_features)

        # 5. Initialize FeedbackManager
        self.feedback_manager = FeedbackManager()

    def recommend(self, song_name, top_n=10, method='hybrid', alpha=0.5):
        """
        Generate song recommendations using the specified method.

        Args:
            song_name (str): Reference song for recommendations.
            top_n (int): Number of recommendations to return.
            method (str): One of 'traditional', 'cluster', 'llm', or 'hybrid'.
            alpha (float): Weighting for hybrid methods.

        Returns:
            pd.DataFrame: Recommended songs with metadata.
        """
        song_idx = self._get_song_index(song_name)
        if song_idx is None:
            return pd.DataFrame()

        # Compute similarity scores based on the chosen method
        if method == 'traditional':
            sim_scores = compute_traditional_similarity(
                song_idx,
                self.text_matrix,
                self.numeric_features,
                alpha=alpha
            )
        elif method == 'cluster':
            sim_scores = self._cluster_based_similarity(song_idx)
        elif method == 'llm':
            sim_scores = compute_llm_similarity(
                song_idx,
                self.text_matrix,
                self.numeric_features,
                self.genre_embeddings,
                self.artist_embeddings,
                alpha=alpha
            )
        elif method == 'hybrid':
            traditional = compute_traditional_similarity(
                song_idx,
                self.text_matrix,
                self.numeric_features,
                alpha=alpha
            )
            llm = compute_llm_similarity(
                song_idx,
                self.text_matrix,
                self.numeric_features,
                self.genre_embeddings,
                self.artist_embeddings,
                alpha=alpha
            )
            sim_scores = (traditional + llm) / 2
        else:
            raise ValueError("Invalid method. Choose from 'traditional', 'cluster', 'llm', 'hybrid'")

        # Get top indices and return those rows
        top_indices = self._get_top_indices(sim_scores, song_idx, top_n)
        return self.tracks.iloc[top_indices][['track_name', 'artists', 'popularity']]

    def recommend_for_user(self, user_id, top_n=10, method='llm'):
        """
        Generate personalized recommendations for a user.

        Args:
            user_id (str): User identifier.
            top_n (int): Number of recommendations.
            method (str): 'llm' or 'traditional'.

        Returns:
            pd.DataFrame: Personalized recommendations.
        """
        if method == 'llm':
            return self.feedback_manager.get_personalized_recommendations(user_id, top_n)
        else:
            # Fallback to popularity-based
            return self.tracks.sort_values('popularity', ascending=False).head(top_n)

    def log_user_feedback(self, user_id, song_name, feedback_type):
        """
        Logs user feedback and retrains the model with new feedback.
        """
        self.feedback_manager.log_feedback(user_id, song_name, feedback_type)
        self.feedback_manager.retrain_model_with_feedback()

    def _get_song_index(self, song_name):
        """Helper to get the index of a song by its name (using 'track_name')."""
        matches = self.tracks[self.tracks['track_name'].str.contains(song_name, case=False)]
        if len(matches) == 0:
            print(f"Song '{song_name}' not found")
            return None
        return matches.index[0]

    def _cluster_based_similarity(self, song_idx):
        """
        Calculate similarity scores within the same cluster.
        """
        cluster_id = self.clusters[song_idx]
        cluster_indices = np.where(self.clusters == cluster_id)[0]

        sim_scores = compute_traditional_similarity(
            song_idx,
            self.text_matrix,
            self.numeric_features
        )

        # Mask out songs not in the same cluster
        mask = np.zeros(len(self.tracks))
        mask[cluster_indices] = 1
        masked_scores = sim_scores * mask

        return masked_scores

    def _get_top_indices(self, sim_scores, song_idx, top_n):
        """Return the top indices excluding the input song itself."""
        sorted_indices = np.argsort(sim_scores)[::-1]
        filtered = [i for i in sorted_indices if i != song_idx]
        return filtered[:top_n]

# ========== Test Example ==========
if __name__ == "__main__":
    # Initialize the recommender system
    recommender = MusicRecommender()

    # Example: Song-based recommendation
    test_song = "Shape of You"
    print(f"\nRecommendations for '{test_song}':")
    recs = recommender.recommend(test_song, top_n=5, method='hybrid')
    print(recs)

    # Example: User-based recommendation
    test_user = "user_123"
    print(f"\nPersonalized recommendations for {test_user}:")
    user_recs = recommender.recommend_for_user(test_user, top_n=5, method='llm')
    print(user_recs)

    # Log some feedback
    recommender.log_user_feedback(test_user, "Shape of You", "like")
    recommender.log_user_feedback(test_user, "Blinding Lights", "dislike")

    # Retrieve updated personalized recommendations
    print("\nUpdated personalized recommendations for the user after feedback:")
    updated_recs = recommender.recommend_for_user(test_user, top_n=5, method='llm')
    print(updated_recs)
