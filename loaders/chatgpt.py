"""
ChatGPT conversation loader implementation.

This module provides a loader for ChatGPT conversation export data.
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd

from loaders.base import ConversationLoader


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