"""
Embeddings module for generating vector representations of text.

This module provides functionality to convert conversation text into
numerical vector representations using sentence transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


def get_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """
    Generate embeddings for a list of texts using sentence transformers.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the sentence transformer model to use
        
    Returns:
        Array of embeddings
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings) 