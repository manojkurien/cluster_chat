"""
Conversation loaders module.

This module contains the interface and implementations for loading
conversation data from various sources and formats.
"""

from loaders.base import ConversationLoader
from loaders.chatgpt import ChatGPTLoader
from loaders.factory import get_loader, validate_dataframe

__all__ = [
    "ConversationLoader",
    "ChatGPTLoader",
    "get_loader",
    "validate_dataframe"
] 