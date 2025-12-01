"""
Dataset loader module for Twitter Financial News Sentiment dataset (2023).

This module loads the Twitter Financial News Sentiment dataset (Zeroshot, 2023),
a dataset from Twitter with clean labeling, ideal for baseline interpretability
and label quality analysis.

This module aligns with the CS5100 research proposal focusing on financial
social-media sentiment classification with label quality evaluation.

All labels are unified to: positive, neutral, negative.
"""

import pandas as pd
import os
import json
from typing import Optional


def load_twitter_financial(file_path: str) -> pd.DataFrame:
    """
    Load Twitter Financial News Sentiment dataset (Zeroshot, 2023).
    
    This dataset contains Twitter financial posts with 3-class sentiment labels:
    - 0 = neutral
    - 1 = positive
    - 2 = negative
    
    Labels are automatically mapped to unified format: positive, neutral, negative.
    
    Args:
        file_path: Path to the dataset file (CSV, TSV, or JSON)
        
    Returns:
        DataFrame with 'text' and 'label' columns (unified labels)
        
    Example:
        >>> df = load_twitter_financial('data/twitter_financial_train.csv')
        >>> print(df['label'].value_counts())
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Twitter Financial file not found: {file_path}")
    
    # Try different formats
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.json':
        try:
            df = pd.read_json(file_path)
        except:
            # Try JSONL format
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]
            df = pd.DataFrame(data)
    elif file_ext == '.tsv':
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)
    
    # Find text column
    text_col = None
    for col in ['text', 'Text', 'tweet', 'Tweet', 'content', 'Content', 'sentence', 'Sentence']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("Could not find text column in Twitter Financial dataset. Expected: 'text', 'Text', 'tweet', 'Tweet', 'content', 'Content'")
    
    # Find label column
    label_col = None
    for col in ['label', 'Label', 'sentiment', 'Sentiment', 'sentiment_label', 'class']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Could not find label column in Twitter Financial dataset. Expected: 'label', 'Label', 'sentiment', 'Sentiment'")
    
    # Map labels: numeric (0, 1, 2) or string formats to unified labels
    def map_label(value):
        """Map various label formats to unified 'positive', 'neutral', 'negative'."""
        if pd.isna(value):
            return 'neutral'
        
        # Try numeric mapping first (0=neutral, 1=positive, 2=negative)
        try:
            num_val = int(value)
            if num_val == 0:
                return 'neutral'
            elif num_val == 1:
                return 'positive'
            elif num_val == 2:
                return 'negative'
        except (ValueError, TypeError):
            pass
        
        # Try string mapping
        value_str = str(value).lower().strip()
        label_mapping = {
            'bullish': 'positive',
            'positive': 'positive',
            'pos': 'positive',
            '1': 'positive',
            'bearish': 'negative',
            'negative': 'negative',
            'neg': 'negative',
            '2': 'negative',
            'neutral': 'neutral',
            'neu': 'neutral',
            '0': 'neutral',
        }
        return label_mapping.get(value_str, 'neutral')
    
    # Create unified dataframe
    result_df = pd.DataFrame({
        'text': df[text_col].astype(str),
        'label': df[label_col].apply(map_label)
    })
    
    # Filter to only valid labels and remove empty texts
    valid_labels = ['positive', 'neutral', 'negative']
    result_df = result_df[result_df['label'].isin(valid_labels)]
    result_df = result_df[result_df['text'].str.len() > 0]
    
    # Reset index
    result_df = result_df.reset_index(drop=True)
    
    return result_df


def load_dataset(dataset_name: str, file_path: str, **kwargs) -> pd.DataFrame:
    """
    Unified dataset loader for Twitter Financial News Sentiment dataset.
    
    This function provides a consistent interface for loading the Twitter Financial
    News Sentiment dataset (Zeroshot, 2023).
    
    Args:
        dataset_name: Must be 'twitter_financial' (this project uses only one dataset)
        file_path: Path to dataset file
        **kwargs: Additional arguments (currently unused, reserved for future use)
        
    Returns:
        DataFrame with 'text' and 'label' columns (unified to: positive, neutral, negative)
        
    Example:
        >>> df = load_dataset('twitter_financial', 'data/twitter_financial_train.csv')
        
    Raises:
        ValueError: If dataset_name is not 'twitter_financial'
        FileNotFoundError: If the dataset file does not exist
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'twitter_financial':
        return load_twitter_financial(file_path)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"This project uses only 'twitter_financial' (Twitter Financial News Sentiment, Zeroshot, 2023)."
        )
