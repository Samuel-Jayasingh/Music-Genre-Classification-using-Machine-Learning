import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_traditional_similarity(song_index, text_matrix, numeric_matrix, alpha=0.5):
    """
    Compute hybrid similarity combining text and numeric features
    
    Args:
        song_index (int): Index of target song
        text_matrix (scipy.sparse.csr_matrix): TF-IDF vectorized text features
        numeric_matrix (np.array): Scaled numeric features
        alpha (float): Weight for text similarity (0 <= alpha <= 1)
    
    Returns:
        np.array: Similarity scores for all songs
    """
    # Text similarity
    text_sim = cosine_similarity(text_matrix[song_index], text_matrix).flatten()
    
    # Numeric similarity
    numeric_sim = cosine_similarity(
        numeric_matrix[song_index].reshape(1, -1),
        numeric_matrix
    ).flatten()
    
    return alpha * text_sim + (1 - alpha) * numeric_sim

def compute_llm_similarity(song_index, text_matrix, numeric_matrix,
                         genre_embeddings, artist_embeddings,
                         alpha=0.4, beta=0.3, gamma=0.3):
    """
    Compute similarity combining traditional features with LLM embeddings
    
    Args:
        song_index (int): Index of target song
        text_matrix (scipy.sparse.csr_matrix): TF-IDF vectorized text features
        numeric_matrix (np.array): Scaled numeric features
        genre_embeddings (np.array): LLM-generated genre embeddings
        artist_embeddings (np.array): LLM-generated artist embeddings
        alpha (float): Weight for traditional similarity
        beta (float): Weight for genre similarity
        gamma (float): Weight for artist similarity
    
    Returns:
        np.array: Combined similarity scores
    """
    # Traditional similarity component
    traditional = compute_traditional_similarity(
        song_index,
        text_matrix,
        numeric_matrix,
        alpha=0.5  # Maintain original balance within traditional features
    )
    
    # LLM-based similarities
    genre_sim = cosine_similarity(
        genre_embeddings[song_index].reshape(1, -1),
        genre_embeddings
    ).flatten()
    
    artist_sim = cosine_similarity(
        artist_embeddings[song_index].reshape(1, -1),
        artist_embeddings
    ).flatten()
    
    return (alpha * traditional) + (beta * genre_sim) + (gamma * artist_sim)