"""
Base interface for conversation data loaders.

This module defines the abstract base class that all conversation loaders must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict

import pandas as pd


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