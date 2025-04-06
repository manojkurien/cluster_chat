"""
Configuration settings for the ChatGPT conversation analysis tool.

This module defines a Config class that manages various settings related to:
- Input/output paths
- Analysis parameters (clustering, embeddings)
- Visualization settings
"""

import os
from datetime import datetime


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
        
    def get_input_files(self) -> list[str]:
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