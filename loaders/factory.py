"""
Loader factory and validation utilities.

This module provides a factory method to create appropriate loaders
based on data format and utilities to validate the processed data.
"""

from typing import Tuple

import pandas as pd

from loaders.base import ConversationLoader
from loaders.chatgpt import ChatGPTLoader


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