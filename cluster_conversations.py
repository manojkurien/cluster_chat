"""
ChatGPT Conversation Analysis Tool

This script analyzes ChatGPT conversation history by:
1. Loading and processing conversation data from JSON exports
2. Converting conversations into embeddings using sentence transformers
3. Clustering conversations into topics using K-means
4. Generating visualizations of conversation patterns and trends
"""

# Standard library imports
import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable
from abc import ABC, abstractmethod

# Data processing imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Machine learning imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Visualization imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d
import seaborn as sns


# === CONFIGURATION ===
class Config:
    """Configuration settings for the analysis script."""
    
    # Input and output paths
    INPUT_DIR: str = "inputs"
    TIMESTAMP: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Output directory
    OUTPUT_DIR: str = f"outputs/run_{TIMESTAMP}"
    
    # Sample mode directories
    SAMPLE_INPUT_DIR: str = "sample_input"
    SAMPLE_OUTPUT_DIR: str = "sample_output"
    
    # Use sample mode
    USE_SAMPLE: bool = False
    
    # Output file paths
    OUTPUT_CSV: str = None  # Will be set in __init__
    OUTPUT_TRENDS_CSV: str = None  # Will be set in __init__
    
    # Output visualization paths
    OUTPUT_PLOT: str = None  # Will be set in __init__
    OUTPUT_HISTOGRAM: str = None  # Will be set in __init__
    OUTPUT_2D_PLOT: str = None  # Will be set in __init__
    OUTPUT_MONTHLY_TOTALS: str = None  # Will be set in __init__
    OUTPUT_SILHOUETTE: str = None  # Will be set in __init__
    
    # Analysis parameters
    MIN_CLUSTERS: int = 5   # Minimum number of clusters to consider
    MAX_CLUSTERS: int = 20  # Maximum number of clusters to consider
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    SMOOTHING_SIGMA: float = 3.0  # Gaussian smoothing parameter
    RANDOM_SEED: int = 42
    
    # Visualization settings
    FOOTER_TEXT: str = "Created by Manoj Kurien 2025"  # Footer text for plots
    
    # Format specification
    DATA_FORMAT: str = "chatgpt"  # Supported: "chatgpt", or add other formats
    
    def __init__(self, use_sample: bool = False):
        """
        Initialize configuration and create input/output directories if they don't exist.
        
        Args:
            use_sample: Whether to use sample input/output directories
        """
        self.USE_SAMPLE = use_sample
        
        # Set input and output directories based on the sample mode
        if use_sample:
            self.INPUT_DIR = self.SAMPLE_INPUT_DIR
            self.OUTPUT_DIR = self.SAMPLE_OUTPUT_DIR
            
        # Set output file paths based on chosen output directory
        self.OUTPUT_CSV = f"{self.OUTPUT_DIR}/clustered_conversations.csv"
        self.OUTPUT_TRENDS_CSV = f"{self.OUTPUT_DIR}/clustered_conversations_trends.csv"
        self.OUTPUT_PLOT = f"{self.OUTPUT_DIR}/conversation_trends.png"
        self.OUTPUT_HISTOGRAM = f"{self.OUTPUT_DIR}/conversation_histogram.png"
        self.OUTPUT_2D_PLOT = f"{self.OUTPUT_DIR}/conversation_clusters_2D.png"
        self.OUTPUT_MONTHLY_TOTALS = f"{self.OUTPUT_DIR}/conversation_monthly_totals.png"
        self.OUTPUT_SILHOUETTE = f"{self.OUTPUT_DIR}/silhouette_scores.png"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        print(f"Output files will be saved to: {self.OUTPUT_DIR}")
        
        # Create input directory if it doesn't exist
        os.makedirs(self.INPUT_DIR, exist_ok=True)
        
    def get_input_files(self) -> List[str]:
        """Get all files in the input directory."""
        # Get all files in the input directory
        all_files = [os.path.join(self.INPUT_DIR, f) for f in os.listdir(self.INPUT_DIR) 
                     if os.path.isfile(os.path.join(self.INPUT_DIR, f))]
        
        if not all_files:
            print(f"Warning: No files found in '{self.INPUT_DIR}' directory.")
            print(f"Please place your data files in the inputs directory.")
        else:
            file_names = [os.path.basename(f) for f in all_files]
            print(f"Found {len(all_files)} files: {', '.join(file_names)}")
        
        return all_files


# === DATA LOADING INTERFACE ===
class ConversationLoader(ABC):
    """
    Abstract base class for conversation data loaders.
    
    This class defines the interface for loading and processing conversation data
    from different sources. Implementations should handle specific file formats
    and convert them to a standardized DataFrame structure for analysis.
    """
    
    @abstractmethod
    def load_data(self, files: List[str]) -> List[Dict[str, Any]]:
        """
        Load conversation data from files.
        
        This method should handle opening and parsing the specified files,
        returning the raw conversation data in its original structure.
        
        Args:
            files: List of file paths to load data from
            
        Returns:
            List of dictionaries containing the raw conversation data
        """
        pass
    
    @abstractmethod
    def process_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process raw data into a structured DataFrame for analysis.
        
        This method should transform the raw conversation data into a standardized
        DataFrame structure required for clustering and visualization.
        
        Args:
            raw_data: List of dictionaries containing raw conversation data
            
        Returns:
            DataFrame with the following required columns:
                - conversation_id (str/int): A unique identifier for each conversation
                - text (str): The text content to analyze and cluster. This should 
                  typically combine all relevant user messages from the conversation.
                - create_time (int): Unix timestamp of conversation creation in seconds
                  since epoch. Used for trend analysis and time-based visualization.
                  
        Notes:
            - The 'text' column should contain all the text content that you want 
              to be considered for clustering. This is what will be embedded and 
              used to determine conversation topics.
            - It's important that 'create_time' is a Unix timestamp (seconds since epoch)
              as it will be converted to datetime using pd.to_datetime(..., unit='s')
            - Empty or irrelevant conversations should be filtered out before returning
            - Additional columns may be included but are not used by default
        """
        pass


class ChatGPTLoader(ConversationLoader):
    """Data loader for ChatGPT conversation exports."""
    
    def load_data(self, files: List[str]) -> List[Dict[str, Any]]:
        """
        Load conversation data from ChatGPT JSON files.
        
        Args:
            files: List of JSON file paths to load
            
        Returns:
            List of conversation dictionaries
        """
        conversations = []
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conversations += data
                    print(f"Successfully loaded {len(data)} conversations from {os.path.basename(path)}")
            except json.JSONDecodeError:
                print(f"Warning: Skipping {os.path.basename(path)} - not a valid JSON file")
            except Exception as e:
                print(f"Warning: Error loading {os.path.basename(path)}: {str(e)}")
        return conversations
    
    def _extract_user_messages(self, mapping: Dict[str, Any]) -> List[str]:
        """
        Extract all user messages from a ChatGPT conversation export.
        
        Args:
            mapping: The 'mapping' dictionary from a ChatGPT conversation export
            
        Returns:
            List of text messages sent by the user in the conversation
        """
        user_texts = []
        for node in mapping.values():
            msg = node.get("message")
            if not msg:
                continue
            if msg.get("author", {}).get("role") != "user":
                continue
            parts = msg.get("content", {}).get("parts", [])
            for part in parts:
                if isinstance(part, str):
                    text = part.strip()
                elif isinstance(part, dict) and "text" in part:
                    text = part["text"].strip()
                else:
                    continue
                if text:
                    user_texts.append(text)
        return user_texts
    
    def process_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process ChatGPT conversation export data into a structured DataFrame.
        
        Args:
            raw_data: List of conversation dictionaries from ChatGPT's data export files
            
        Returns:
            DataFrame with processed conversation data containing:
            - conversation_id: Unique identifier for each conversation
            - text: Combined title and user messages for analysis
            - create_time: Unix timestamp of conversation creation
        """
        processed_conversations = []
        for conversation in raw_data:
            mapping = conversation.get("mapping", {})
            user_messages = self._extract_user_messages(mapping)
            if not user_messages:
                continue
            
            # Get the title but only use it as part of the text field
            title = conversation.get("title", "").strip()
            text = f"{title} — {' '.join(user_messages)}"
            
            processed_conversations.append({
                "conversation_id": conversation.get("conversation_id"),
                "text": text,
                "create_time": conversation.get("create_time")
            })
        return pd.DataFrame(processed_conversations)


# Factory method to get appropriate loader
def get_loader(format_name: str) -> ConversationLoader:
    """
    Factory method to create the appropriate data loader.
    
    Args:
        format_name: String identifier for the data format
        
    Returns:
        Appropriate ConversationLoader instance
    
    Raises:
        ValueError: If the specified format is not supported
    """
    loaders = {
        "chatgpt": ChatGPTLoader,
        # Add more loaders here as formats are supported
    }
    
    if format_name not in loaders:
        supported = ", ".join(loaders.keys())
        raise ValueError(f"Unsupported data format: {format_name}. Supported formats: {supported}")
    
    return loaders[format_name]()


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that a DataFrame has the required structure for analysis.
    
    This function checks that the DataFrame contains all required columns
    with the correct data types and validates that the data is suitable
    for analysis.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message) where:
        - is_valid: Boolean indicating if the DataFrame is valid
        - error_message: String describing the validation error, empty if valid
    """
    # Check if DataFrame is empty
    if df.empty:
        return False, "DataFrame is empty. No conversations to analyze."
    
    # Check for required columns
    required_columns = ["conversation_id", "text", "create_time"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for non-empty text values
    if df["text"].isna().any() or (df["text"] == "").any():
        return False, "Some conversations have empty 'text' values."
    
    # Check that create_time contains valid timestamps
    try:
        # This will raise if create_time can't be converted to datetime
        pd.to_datetime(df["create_time"], unit="s")
    except (ValueError, TypeError):
        return False, "Column 'create_time' must contain valid Unix timestamps (seconds since epoch)."
    
    # Check for duplicate conversation IDs
    if df["conversation_id"].duplicated().any():
        return False, "Duplicate conversation IDs found. Each conversation must have a unique ID."
    
    return True, ""


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


def prepare_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to the DataFrame.
    
    Args:
        df: DataFrame with conversation data
        
    Returns:
        DataFrame with time features added
    """
    df_copy = df.copy()
    df_copy["created_at"] = pd.to_datetime(df_copy["create_time"], unit="s")
    df_copy["month"] = df_copy["created_at"].dt.to_period("M").astype(str)
    return df_copy


def create_trend_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pivot table of conversation counts by cluster and month.
    
    Args:
        df: DataFrame with cluster assignments and time features
        
    Returns:
        Pivot table of counts by cluster and month
    """
    return df.pivot_table(
        index="month",
        columns="cluster",
        values="conversation_id",
        aggfunc="count",
        fill_value=0
    )


# === CLUSTERING AND ANALYSIS ===
def find_optimal_clusters(embeddings: np.ndarray, min_clusters: int, max_clusters: int, random_seed: int) -> Tuple[int, List[float]]:
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


# === VISUALIZATION FUNCTIONS ===
def setup_plot(figsize: Tuple[int, int], title: str, xlabel: str, ylabel: str) -> Tuple[Figure, Axes]:
    """
    Set up a matplotlib figure and axes with common styling.
    
    Args:
        figsize: Figure size (width, height)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig, ax


def save_plot(fig: Figure, ax: Axes, output_path: str, config: Config, add_footer: bool = True) -> None:
    """
    Save a plot with optional footer.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        output_path: Path to save the figure
        config: Configuration object containing footer text
        add_footer: Whether to add footer with attribution
    """
    if add_footer:
        plt.figtext(0.99, 0.01, config.FOOTER_TEXT, 
                   horizontalalignment='right', fontsize=10)
    
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close(fig)  # Close the figure to free memory


def plot_silhouette_scores(cluster_range: range, scores: List[float], output_path: str, config: Config) -> None:
    """
    Plot silhouette scores for different numbers of clusters.
    
    Args:
        cluster_range: Range of cluster numbers
        scores: Corresponding silhouette scores
        output_path: Path to save the plot
        config: Configuration object containing settings
    """
    fig, ax = setup_plot(
        figsize=(12, 6),
        title='Silhouette Score vs Number of Clusters',
        xlabel='Number of Clusters (k)',
        ylabel='Silhouette Score'
    )
    
    ax.plot(cluster_range, scores, 'bo-', linewidth=2, markersize=8)
    
    # Add value labels on points
    for i, score in enumerate(scores):
        ax.annotate(
            f'{score:.3f}', 
            xy=(cluster_range[i], score),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center'
        )
    
    save_plot(fig, ax, output_path, config)


def plot_monthly_totals(trend_df: pd.DataFrame, output_path: str, smoothing_sigma: float, config: Config) -> None:
    """
    Plot the total number of conversations per month over time.
    
    Args:
        trend_df: Pivot table of conversation counts by cluster and month
        output_path: Path to save the plot
        smoothing_sigma: Gaussian smoothing parameter
        config: Configuration object containing settings
    """
    fig, ax = setup_plot(
        figsize=(16, 9),
        title="Conversations per Month",
        xlabel="Month",
        ylabel="Conversations Count"
    )
    
    monthly_totals = trend_df.sum(axis=1)
    smooth_totals = gaussian_filter1d(monthly_totals.values, sigma=smoothing_sigma)
    
    ax.plot(
        trend_df.index,
        smooth_totals,
        linewidth=4.0,
        color='#2ecc71'
    )
    
    ax.tick_params(axis='x', rotation=45)
    save_plot(fig, ax, output_path, config)


def plot_trend_percentages(trend_df: pd.DataFrame, cluster_titles: Dict[int, str], 
                          output_path: str, smoothing_sigma: float, config: Config) -> None:
    """
    Plot trend lines showing percentage of conversations per cluster over time.
    
    Args:
        trend_df: Pivot table of conversation counts by cluster and month
        cluster_titles: Dictionary mapping cluster IDs to titles
        output_path: Path to save the plot
        smoothing_sigma: Gaussian smoothing parameter
        config: Configuration object containing settings
    """
    fig, ax = setup_plot(
        figsize=(16, 9),
        title="Conversation Trends by Month",
        xlabel="Month",
        ylabel="% of Monthly Conversations"
    )
    
    palette = sns.color_palette("tab10", len(trend_df.columns))
    monthly_totals = trend_df.sum(axis=1)
    
    for i, cluster in enumerate(trend_df.columns):
        counts = trend_df[cluster].values
        percentages = (counts / monthly_totals) * 100
        smooth_percentages = gaussian_filter1d(percentages, sigma=smoothing_sigma)
        
        ax.plot(
            trend_df.index,
            smooth_percentages,
            label=cluster_titles.get(cluster, f"Cluster {cluster}"),
            linewidth=2.5,
            color=palette[i]
        )
    
    ax.tick_params(axis='x', rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Topics")
    save_plot(fig, ax, output_path, config)


def plot_topic_distribution(df: pd.DataFrame, cluster_titles: Dict[int, str], output_path: str, config: Config) -> None:
    """
    Plot histogram showing distribution of conversations across topics.
    
    Args:
        df: Processed conversation data with cluster assignments
        cluster_titles: Dictionary mapping cluster IDs to titles
        output_path: Path to save the plot
        config: Configuration object containing settings
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    cluster_counts = df["cluster"].value_counts().sort_values(ascending=False)
    num_clusters = len(cluster_counts)
    palette = sns.color_palette("tab10", num_clusters)
    
    titles = [cluster_titles.get(c, f"Cluster {c}") for c in cluster_counts.index]
    
    # Create bar plot with index-based colors
    bars = ax.bar(
        range(len(titles)), 
        cluster_counts.values, 
        color=[palette[i] for i in range(len(titles))]
    )

    ax.set_title("Conversation Topics", fontsize=18)
    ax.set_ylabel("Number of Conversations", fontsize=14)
    ax.set_xlabel("Topic", fontsize=14)
    ax.set_xticks(range(len(titles)))
    ax.set_xticklabels(titles, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 1,
            str(int(height)), 
            ha='center', 
            fontsize=12
        )

    save_plot(fig, ax, output_path, config)


def plot_cluster_visualization(df: pd.DataFrame, embeddings: np.ndarray, 
                              kmeans: KMeans, cluster_titles: Dict[int, str],
                              output_path: str, config: Config) -> None:
    """
    Create 2D visualization of conversation clusters using PCA.
    
    Args:
        df: Processed conversation data with cluster assignments
        embeddings: High-dimensional embeddings
        kmeans: Fitted clustering model
        cluster_titles: Dictionary mapping cluster IDs to titles
        output_path: Path to save the cluster plot
        config: Configuration object containing settings
    """
    # Fit PCA with 2 components for visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    num_clusters = len(df["cluster"].unique())
    palette = sns.color_palette("tab10", num_clusters)
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    # Plot points for each cluster
    for i, cluster in enumerate(sorted(df["cluster"].unique())):
        indices = df[df["cluster"] == cluster].index
        ax.scatter(
            reduced[indices, 0],
            reduced[indices, 1],
            label=cluster_titles.get(cluster, f"Cluster {cluster}"),
            alpha=0.7,
            s=30,
            color=palette[i]
        )

    # Plot cluster centers
    centers_2d = pca.transform(kmeans.cluster_centers_)
    ax.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c="black",
        s=100,
        marker="x",
        label="Cluster Centers"
    )

    ax.set_title("Conversation Clusters in 2D Space", fontsize=16)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    save_plot(fig, ax, output_path, config)


def export_cluster_info_for_external_titling(df: pd.DataFrame, keywords_by_cluster: Dict[int, List[str]], config: Config) -> None:
    """
    Export cluster information to a CSV file for external title generation.
    
    This function creates a CSV file containing cluster numbers, conversation counts,
    and the most representative keywords for each cluster. This can be used by
    external systems (like GPT) to generate more descriptive titles.
    
    Args:
        df: DataFrame with cluster assignments
        keywords_by_cluster: Dictionary mapping cluster IDs to keyword lists
        config: Configuration settings
    """
    # Create a list to store the cluster information
    cluster_info = []
    
    # For each cluster, gather the required information
    for cluster_id in sorted(df["cluster"].unique()):
        # Count the number of conversations in this cluster
        conversation_count = len(df[df["cluster"] == cluster_id])
        
        # Get the keywords for this cluster
        keywords = keywords_by_cluster.get(cluster_id, [])
        keywords_str = ", ".join(keywords)
        
        # Add to the list
        cluster_info.append({
            "cluster_id": cluster_id,
            "conversation_count": conversation_count,
            "keywords": keywords_str,
            # Add 5 example texts from this cluster (truncated to 100 chars each)
            "example_texts": "; ".join(df[df["cluster"] == cluster_id]["text"].str[:100].sample(min(5, conversation_count)).tolist())
        })
    
    # Convert to DataFrame and export to CSV
    cluster_info_df = pd.DataFrame(cluster_info)
    export_path = f"{config.OUTPUT_DIR}/cluster_info_for_titling.csv"
    cluster_info_df.to_csv(export_path, index=False)
    print(f"Exported cluster information for external titling to {export_path}")


def analyze_and_visualize_clusters(df: pd.DataFrame, embeddings: np.ndarray, kmeans: KMeans, 
                                  cluster_titles: Dict[int, str], keywords_by_cluster: Dict[int, List[str]], 
                                  config: Config) -> None:
    """
    Analyze clusters and generate visualizations.
    
    Args:
        df: Processed conversation data with cluster assignments
        embeddings: Text embeddings
        kmeans: Fitted clustering model
        cluster_titles: Dictionary mapping cluster IDs to titles
        keywords_by_cluster: Dictionary mapping cluster IDs to keyword lists
        config: Configuration settings
    """
    # Add time features
    df_with_time = prepare_time_features(df)
    
    # Export cluster information for external titling
    export_cluster_info_for_external_titling(df, keywords_by_cluster, config)
    
    # Create trend data
    trend_df = create_trend_df(df_with_time)
    trend_df.to_csv(config.OUTPUT_TRENDS_CSV)
    print(f"Trend data saved to {config.OUTPUT_TRENDS_CSV}")

    # Generate visualizations
    plot_monthly_totals(trend_df, config.OUTPUT_MONTHLY_TOTALS, config.SMOOTHING_SIGMA, config)
    plot_trend_percentages(trend_df, cluster_titles, config.OUTPUT_PLOT, config.SMOOTHING_SIGMA, config)
    plot_topic_distribution(df_with_time, cluster_titles, config.OUTPUT_HISTOGRAM, config)
    plot_cluster_visualization(df_with_time, embeddings, kmeans, cluster_titles, 
                              config.OUTPUT_2D_PLOT, config)


def main() -> None:
    """Main function to run the analysis pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Cluster conversation data and visualize trends")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample input/output directories for demo purposes"
    )
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config(use_sample=args.sample)
    print(f"Analysis started at {config.TIMESTAMP}")
    
    # Notify about sample mode
    if args.sample:
        print("SAMPLE MODE: Using sample_input directory for input and sample_output for results")
    
    # Get the appropriate loader for the specified format
    loader = get_loader(config.DATA_FORMAT)
    
    # Load and process data
    raw_data = loader.load_data(config.get_input_files())
    conversations_df = loader.process_data(raw_data)
    
    # Validate the DataFrame structure
    is_valid, error_message = validate_dataframe(conversations_df)
    if not is_valid:
        raise ValueError(f"Invalid DataFrame returned by {config.DATA_FORMAT} loader: {error_message}")
    
    # Continue with embeddings and clustering
    embeddings = get_embeddings(conversations_df["text"].tolist(), config.MODEL_NAME)

    # Find optimal number of clusters
    num_clusters, silhouette_scores = find_optimal_clusters(
        embeddings, 
        config.MIN_CLUSTERS,
        config.MAX_CLUSTERS, 
        config.RANDOM_SEED
    )
    
    # Plot silhouette scores
    plot_silhouette_scores(
        range(config.MIN_CLUSTERS, config.MAX_CLUSTERS + 1),
        silhouette_scores,
        config.OUTPUT_SILHOUETTE,
        config
    )
    
    # Perform clustering with optimal number of clusters
    print(f"\nProceeding with clustering using {num_clusters} clusters...")
    kmeans = perform_clustering(embeddings, num_clusters, config.RANDOM_SEED)
    conversations_df["cluster"] = kmeans.predict(embeddings)
    
    # Extract keywords and generate cluster titles
    keywords_by_cluster = extract_cluster_keywords(conversations_df, n_keywords=10)
    cluster_titles = generate_cluster_titles(keywords_by_cluster, title_keywords=3)
    
    # Print cluster information
    print("\nCluster titles:")
    for cluster_id, title in sorted(cluster_titles.items()):
        count = len(conversations_df[conversations_df["cluster"] == cluster_id])
        print(f"Cluster {cluster_id}: {title} ({count} conversations)")
        # Print all keywords for this cluster
        all_keywords = ", ".join(keywords_by_cluster[cluster_id])
        print(f"  Keywords: {all_keywords}")

    # Save clustered data
    conversations_df.to_csv(config.OUTPUT_CSV, index=False)
    print(f"Saved clustered results to {config.OUTPUT_CSV}")

    # Analyze and visualize clusters
    analyze_and_visualize_clusters(
        conversations_df, 
        embeddings, 
        kmeans, 
        cluster_titles,
        keywords_by_cluster,
        config
    )


if __name__ == "__main__":
    main()
