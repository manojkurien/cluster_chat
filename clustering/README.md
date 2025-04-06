# Clustering Module

This module provides functionality for clustering conversation data based on text embeddings.

## Components

The module is organized into three main components:

### 1. Embeddings (`embeddings.py`)

Handles conversion of text to vector representations using sentence transformers.

- `get_embeddings()`: Generates embeddings for a list of texts using the specified sentence transformer model.

### 2. K-means Clustering (`kmeans.py`)

Implements K-means clustering functionality for text embeddings.

- `find_optimal_clusters()`: Determines the optimal number of clusters using silhouette scores.
- `perform_clustering()`: Performs K-means clustering on the provided embeddings.

### 3. Cluster Analysis (`analysis.py`)

Provides tools for analyzing and interpreting clusters.

- `extract_cluster_keywords()`: Extracts the most representative keywords for each cluster using TF-IDF.
- `generate_cluster_titles()`: Generates descriptive titles for clusters based on the extracted keywords.

## Usage

The clustering module is designed to be used as part of a larger conversation analysis pipeline. A typical workflow:

```python
from clustering import get_embeddings, find_optimal_clusters, perform_clustering
from clustering import extract_cluster_keywords, generate_cluster_titles

# Generate embeddings from text data
embeddings = get_embeddings(texts, model_name)

# Find optimal number of clusters
num_clusters, _ = find_optimal_clusters(embeddings, min_clusters, max_clusters, random_seed)

# Perform clustering
kmeans = perform_clustering(embeddings, num_clusters, random_seed)

# Extract keywords and generate titles
keywords_by_cluster = extract_cluster_keywords(df_with_clusters)
cluster_titles = generate_cluster_titles(keywords_by_cluster)
```

## Testing

The clustering module includes unit tests located in `tests/test_clustering.py`. These tests verify the functionality of each component:

### Running the Tests

```bash
# Run all tests in the project
python -m unittest discover

# Run only the clustering tests
python -m unittest tests/test_clustering.py
```

### Test Coverage

The tests cover:

1. **Embedding Generation**: Tests the conversion of text to vector representations
2. **K-means Clustering**: Tests the clustering algorithm functionality
3. **Keyword Extraction**: Tests the extraction of representative keywords from clusters
4. **Title Generation**: Tests the generation of descriptive titles from keywords

### Extending the Tests

When adding new functionality to the clustering module, consider adding corresponding tests to maintain code quality and reliability. 