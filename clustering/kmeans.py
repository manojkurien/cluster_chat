"""
K-means clustering module for conversation analysis.

This module provides functionality to perform K-means clustering 
on conversation embeddings and find the optimal number of clusters.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, List
from tqdm import tqdm


def find_optimal_clusters(embeddings: np.ndarray, min_clusters: int, 
                         max_clusters: int, random_seed: int) -> Tuple[int, List[float]]:
    """
    Find the optimal number of clusters using silhouette scores, respecting a minimum cluster constraint.
    
    Args:
        embeddings: Text embeddings
        min_clusters: Minimum number of clusters to consider
        max_clusters: Maximum number of clusters to try
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (optimal number of clusters, list of silhouette scores)
    """
    silhouette_scores = []
    cluster_range = range(min_clusters, max_clusters + 1)
    
    print(f"Calculating silhouette scores for {min_clusters} to {max_clusters} clusters...")
    for k in tqdm(cluster_range):
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(score)
    
    # Find the best silhouette score
    best_idx = np.argmax(silhouette_scores)
    optimal_k = cluster_range[best_idx]
    
    print(f"Best silhouette score: {silhouette_scores[best_idx]:.4f} with {optimal_k} clusters")
    
    return optimal_k, silhouette_scores


def perform_clustering(embeddings: np.ndarray, num_clusters: int, random_seed: int) -> KMeans:
    """
    Perform K-means clustering on the embeddings.
    
    Args:
        embeddings: Text embeddings
        num_clusters: Number of clusters
        random_seed: Random seed for reproducibility
        
    Returns:
        Fitted KMeans model
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init="auto")
    kmeans.fit(embeddings)
    return kmeans 