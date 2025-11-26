"""
Dataset loader module for financial sentiment datasets.

Supports:
- Financial PhraseBank
- SemEval-2017 Task 5 (microblogs)
- SEntFiN 1.0 (entity-sentiment dataset)
"""

import pandas as pd
import os
import ast
from typing import Optional
from collections import Counter


def load_phrasebank(file_path: str) -> pd.DataFrame:
    """
    Load Financial PhraseBank dataset.
    
    Expected format: CSV with columns for text and sentiment labels.
    Labels are unified to: positive, neutral, negative
    
    Args:
        file_path: Path to the PhraseBank CSV file
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PhraseBank file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Map common PhraseBank label formats to unified labels
    label_mapping = {
        'positive': 'positive',
        'neutral': 'neutral',
        'negative': 'negative',
        'Positive': 'positive',
        'Neutral': 'neutral',
        'Negative': 'negative',
        'pos': 'positive',
        'neu': 'neutral',
        'neg': 'negative',
    }
    
    # Try to find label column (common names)
    label_col = None
    for col in ['label', 'sentiment', 'Sentiment', 'Label', 'sentiment_label']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Could not find label column in PhraseBank dataset")
    
    # Try to find text column
    text_col = None
    for col in ['text', 'Text', 'sentence', 'Sentence', 'phrase', 'Phrase']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("Could not find text column in PhraseBank dataset")
    
    # Create unified dataframe
    result_df = pd.DataFrame({
        'text': df[text_col],
        'label': df[label_col].map(label_mapping).fillna(df[label_col])
    })
    
    # Filter to only valid labels
    valid_labels = ['positive', 'neutral', 'negative']
    result_df = result_df[result_df['label'].isin(valid_labels)]
    
    return result_df


def load_semeval(file_path: str) -> pd.DataFrame:
    """
    Load SemEval-2017 Task 5 dataset (microblogs).
    
    Expected format: CSV or TSV with text and sentiment scores/labels.
    Labels are unified to: positive, neutral, negative
    
    Args:
        file_path: Path to the SemEval dataset file
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SemEval file not found: {file_path}")
    
    # Try CSV first, then TSV
    try:
        df = pd.read_csv(file_path, sep=',')
    except:
        df = pd.read_csv(file_path, sep='\t')
    
    # SemEval typically has sentiment scores, convert to labels
    # Common column names: text, tweet, message, sentiment, score
    text_col = None
    for col in ['text', 'Text', 'tweet', 'Tweet', 'message', 'Message']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("Could not find text column in SemEval dataset")
    
    # Try to find sentiment score or label
    sentiment_col = None
    for col in ['sentiment', 'Sentiment', 'score', 'Score', 'label', 'Label']:
        if col in df.columns:
            sentiment_col = col
            break
    
    if sentiment_col is None:
        raise ValueError("Could not find sentiment column in SemEval dataset")
    
    # Convert scores to labels if needed
    def score_to_label(score):
        if isinstance(score, str):
            score_lower = score.lower()
            if 'positive' in score_lower or score_lower == 'pos':
                return 'positive'
            elif 'negative' in score_lower or score_lower == 'neg':
                return 'negative'
            else:
                return 'neutral'
        else:
            # Numeric score: > 0.1 = positive, < -0.1 = negative, else neutral
            try:
                score_val = float(score)
                if score_val > 0.1:
                    return 'positive'
                elif score_val < -0.1:
                    return 'negative'
                else:
                    return 'neutral'
            except:
                return 'neutral'
    
    result_df = pd.DataFrame({
        'text': df[text_col],
        'label': df[sentiment_col].apply(score_to_label)
    })
    
    # Filter to only valid labels
    valid_labels = ['positive', 'neutral', 'negative']
    result_df = result_df[result_df['label'].isin(valid_labels)]
    
    return result_df


def load_sentfin(file_path: str, aggregation_method: str = 'most_common') -> pd.DataFrame:
    """
    Load SEntFiN 1.0 dataset (entity-sentiment dataset).
    
    SEntFiN format has:
    - Title: the headline text
    - Decisions: dictionary string with entity-sentiment pairs like {'Entity': 'sentiment'}
    
    When a headline has multiple entities with different sentiments, we aggregate them.
    
    Args:
        file_path: Path to the SEntFiN CSV file
        aggregation_method: How to handle multiple sentiments per headline
            - 'most_common': Use the most frequent sentiment
            - 'first': Use the first entity's sentiment
            - 'expand': Create one row per entity (not implemented yet)
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SEntFiN file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Check for required columns
    if 'Title' not in df.columns:
        raise ValueError("SEntFiN dataset must have 'Title' column")
    if 'Decisions' not in df.columns:
        raise ValueError("SEntFiN dataset must have 'Decisions' column")
    
    def extract_sentiment(decisions_str):
        """
        Extract sentiment from Decisions dictionary string.
        Handles single or multiple entity-sentiment pairs.
        """
        if pd.isna(decisions_str):
            return 'neutral'
        
        try:
            # Parse the dictionary string
            decisions_dict = ast.literal_eval(str(decisions_str))
            
            if not isinstance(decisions_dict, dict):
                return 'neutral'
            
            # Extract all sentiments
            sentiments = list(decisions_dict.values())
            
            if len(sentiments) == 0:
                return 'neutral'
            
            # Aggregate based on method
            if aggregation_method == 'most_common':
                # Count sentiments
                sentiment_counts = Counter(sentiments)
                most_common = sentiment_counts.most_common(1)[0][0]
                return most_common
            elif aggregation_method == 'first':
                return sentiments[0]
            else:
                # Default to most common
                sentiment_counts = Counter(sentiments)
                return sentiment_counts.most_common(1)[0][0]
                
        except (ValueError, SyntaxError):
            # If parsing fails, try to extract sentiment directly
            decisions_str_lower = str(decisions_str).lower()
            if 'positive' in decisions_str_lower:
                return 'positive'
            elif 'negative' in decisions_str_lower:
                return 'negative'
            else:
                return 'neutral'
    
    # Extract labels
    labels = df['Decisions'].apply(extract_sentiment)
    
    # Map to standard format
    label_mapping = {
        'positive': 'positive',
        'neutral': 'neutral',
        'negative': 'negative',
        'pos': 'positive',
        'neu': 'neutral',
        'neg': 'negative',
    }
    
    labels = labels.map(label_mapping).fillna('neutral')
    
    # Create unified dataframe
    result_df = pd.DataFrame({
        'text': df['Title'],
        'label': labels
    })
    
    # Filter to only valid labels
    valid_labels = ['positive', 'neutral', 'negative']
    result_df = result_df[result_df['label'].isin(valid_labels)]
    
    return result_df


def load_dataset(dataset_name: str, file_path: str, **kwargs) -> pd.DataFrame:
    """
    Unified dataset loader.
    
    Args:
        dataset_name: Name of dataset ('phrasebank', 'semeval', or 'sentfin')
        file_path: Path to dataset file
        **kwargs: Additional arguments for specific loaders
            - For SEntFiN: aggregation_method ('most_common' or 'first')
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'phrasebank':
        return load_phrasebank(file_path)
    elif dataset_name == 'semeval':
        return load_semeval(file_path)
    elif dataset_name == 'sentfin':
        return load_sentfin(file_path, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'phrasebank', 'semeval', or 'sentfin'")

