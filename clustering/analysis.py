"""
Cluster analysis module for conversation clustering.

This module provides functionality to analyze clusters, extract representative
keywords, and generate descriptive titles for clusters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_cluster_keywords(df: pd.DataFrame, n_keywords: int = 10) -> Dict[int, List[str]]:
    """
    Extract the most representative keywords for each cluster.
    
    This function uses TF-IDF vectorization to identify the most important terms
    for each cluster.
    
    Args:
        df: DataFrame with processed conversation data and cluster assignments
        n_keywords: Number of keywords to extract for each cluster (default: 10)
        
    Returns:
        Dictionary mapping cluster IDs to lists of representative keywords
    """
    # Prepare corpus for TF-IDF
    texts = df["text"].tolist()
    
    # Create and fit TF-IDF vectorizer
    # We filter out very common and very rare words
    tfidf = TfidfVectorizer(
        min_df=2,         # Ignore terms that appear in fewer than 2 documents
        max_df=0.7,       # Ignore terms that appear in more than 70% of documents
        stop_words='english',  # Remove English stopwords
        ngram_range=(1, 2)     # Consider both unigrams and bigrams
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    
    # Get feature names (words)
    feature_names = np.array(tfidf.get_feature_names_out())
    
    # Extract keywords for each cluster
    keywords_by_cluster = {}
    
    for cluster_id in sorted(df["cluster"].unique()):
        # Get indices of conversations in this cluster
        cluster_indices = df[df["cluster"] == cluster_id].index
        
        if len(cluster_indices) == 0:
            keywords_by_cluster[cluster_id] = []
            continue
            
        # Get TF-IDF vectors for documents in this cluster
        cluster_tfidf = tfidf_matrix[cluster_indices]
        
        # Calculate average TF-IDF scores for each term in this cluster
        avg_tfidf = cluster_tfidf.mean(axis=0).A1
        
        # Get the top N keywords based on average TF-IDF score
        top_indices = avg_tfidf.argsort()[-n_keywords:][::-1]
        top_keywords = feature_names[top_indices]
        
        # Store keywords for this cluster
        keywords_by_cluster[cluster_id] = list(top_keywords)
    
    return keywords_by_cluster


def generate_cluster_titles(keywords_by_cluster: Dict[int, List[str]], title_keywords: int = 3) -> Dict[int, str]:
    """
    Generate descriptive titles for clusters based on extracted keywords.
    
    Args:
        keywords_by_cluster: Dictionary mapping cluster IDs to keyword lists
        title_keywords: Number of keywords to use in the title (default: 3)
        
    Returns:
        Dictionary mapping cluster IDs to descriptive titles
    """
    titles = {}
    
    for cluster_id, keywords in keywords_by_cluster.items():
        if not keywords:
            titles[cluster_id] = f"Empty Cluster {cluster_id}"
            continue
            
        # Create a title from the top keywords (limited to title_keywords)
        title_words = keywords[:title_keywords]
        title = " & ".join(title_words).title()
        
        # Add the title to the dictionary
        titles[cluster_id] = title
    
    return titles 