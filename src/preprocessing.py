import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path='data/tracks_records.csv'):
    """
    Perform a full preprocessing pipeline on music data.

    Args:
        input_path (str): Path to the input CSV file containing music track data.

    Returns:
        tuple: A tuple containing:
            - processed DataFrame,
            - TF-IDF matrix for the 'track_genre' column,
            - Scaled numeric features array,
            - StandardScaler object,
            - TfidfVectorizer object.
    """
    # Check if file exists; if not, try an alternate path relative to current working directory.
    if not os.path.exists(input_path):
        alt_path = os.path.join(os.getcwd(), 'data', 'tracks_records.csv')
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            raise FileNotFoundError(f"File not found at {input_path}")
    
    # Load dataset
    tracks = pd.read_csv(input_path)
    
    # Data cleaning: remove missing values and duplicate song entries using 'track_name'
    tracks = tracks.drop_duplicates(subset=['track_name'], keep='first')
    tracks = tracks.dropna()
    
    # Limit to top 10,000 songs based on popularity
    tracks = tracks.sort_values(by='popularity', ascending=False).head(10000)
    
    # Reset index so that it matches the feature matrices
    tracks = tracks.reset_index(drop=True)
    
    # Extract text features from the 'track_genre' column using TF-IDF.
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    genre_matrix = tfidf.fit_transform(tracks['track_genre'].fillna(''))
    
    # Identify numeric columns and scale their features.
    numeric_cols = tracks.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = tracks[numeric_cols].values
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features)
    
    return tracks, genre_matrix, numeric_features_scaled, scaler, tfidf
