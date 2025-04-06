"""
Tests for the clustering module.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from clustering.embeddings import get_embeddings
from clustering.kmeans import perform_clustering, find_optimal_clusters
from clustering.analysis import extract_cluster_keywords, generate_cluster_titles


class MockSentenceTransformer:
    """Mock SentenceTransformer for testing."""
    
    def __init__(self, *args, **kwargs):
        pass
        
    def encode(self, texts, show_progress_bar=False):
        """Return dummy embeddings for testing."""
        return np.random.rand(len(texts), 10)


class TestClustering(unittest.TestCase):
    """Test cases for the clustering module."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_texts = [
            "This is a test about technology and programming",
            "Another test about coding and software",
            "This text is about cooking and recipes",
            "Food preparation and kitchen tips",
            "Programming languages and coding practices"
        ]
        
        self.sample_df = pd.DataFrame({
            "conversation_id": range(5),
            "text": self.sample_texts,
            "create_time": [1600000000 + i * 86400 for i in range(5)]
        })
        
        # Add cluster assignments
        self.sample_df["cluster"] = [0, 0, 1, 1, 0]
        
        # Create sample embeddings (10-dimensional)
        self.sample_embeddings = np.random.rand(5, 10)
        
    def test_perform_clustering(self):
        """Test perform_clustering function."""
        num_clusters = 2
        kmeans = perform_clustering(self.sample_embeddings, num_clusters, random_seed=42)
        
        self.assertIsInstance(kmeans, KMeans)
        self.assertEqual(kmeans.n_clusters, num_clusters)
        self.assertEqual(kmeans.cluster_centers_.shape, (num_clusters, 10))
        
    def test_extract_cluster_keywords(self):
        """Test extract_cluster_keywords function."""
        keywords_by_cluster = extract_cluster_keywords(self.sample_df, n_keywords=3)
        
        self.assertIsInstance(keywords_by_cluster, dict)
        self.assertIn(0, keywords_by_cluster)
        self.assertIn(1, keywords_by_cluster)
        
        # Check that we have keywords for each cluster
        self.assertTrue(len(keywords_by_cluster[0]) > 0)
        self.assertTrue(len(keywords_by_cluster[1]) > 0)
        
    def test_generate_cluster_titles(self):
        """Test generate_cluster_titles function."""
        keywords_by_cluster = {
            0: ["programming", "technology", "coding"],
            1: ["cooking", "food", "recipes"]
        }
        
        titles = generate_cluster_titles(keywords_by_cluster, title_keywords=2)
        
        self.assertIsInstance(titles, dict)
        self.assertIn(0, titles)
        self.assertIn(1, titles)
        
        # Check that titles are generated correctly
        self.assertEqual(titles[0], "Programming & Technology")
        self.assertEqual(titles[1], "Cooking & Food")


if __name__ == "__main__":
    unittest.main() 