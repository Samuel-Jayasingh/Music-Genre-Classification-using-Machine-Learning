# clustering.py
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

class MusicClustering:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, features, n_clusters=10, random_state=42):
        """
        Train K-Means clustering model
        
        Args:
            features (np.array): Input features for clustering
            n_clusters (int): Number of clusters
            random_state (int): Random seed for reproducibility
        
        Returns:
            cluster_labels (np.array): Assigned cluster labels
        """
        # Feature scaling
        scaled_features = self.scaler.fit_transform(features)
        
        # Initialize and train model
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        cluster_labels = self.model.fit_predict(scaled_features)
        
        return cluster_labels
    
    def evaluate(self, features):
        """
        Evaluate clustering performance
        
        Args:
            features (np.array): Scaled features used for clustering
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return {
            'inertia': self.model.inertia_,
            'silhouette_score': silhouette_score(features, self.model.labels_)
        }
    
    def predict(self, new_features):
        """
        Predict clusters for new data
        
        Args:
            new_features (np.array): New input features
            
        Returns:
            np.array: Predicted cluster labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        scaled_new = self.scaler.transform(new_features)
        return self.model.predict(scaled_new)
    
    def save_model(self, path='models/clustering_model.pkl'):
        """Save trained model and scaler"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)
    
    @staticmethod
    def load_model(path='models/clustering_model.pkl'):
        """Load pre-trained model and scaler"""
        loaded = joblib.load(path)
        clustering = MusicClustering()
        clustering.model = loaded['model']
        clustering.scaler = loaded['scaler']
        return clustering