"""
Dataset loader module for financial social-media sentiment datasets.

PRIMARY DATASETS (Post-2020, Social Media Financial Sentiment):
- Twitter Financial News Sentiment (Zeroshot, 2023): Clean Twitter data, ideal for baseline
- Financial Tweets Sentiment (TimKoornstra, 2023): Large-scale aggregated Twitter data
- TweetFinSent (JP Morgan, 2022): Expert-annotated high-quality Twitter data

LEGACY DATASETS (Deprecated - for reference only):
- Financial PhraseBank: Short financial sentences from news (LEGACY)
- SemEval-2017 Task 5: Microblogs (LEGACY)
- SEntFiN 1.0: Financial news headlines (LEGACY - not social media)

This module aligns with the CS5100 research proposal focusing on financial
social-media sentiment classification with label quality evaluation.

All datasets are unified to: positive, neutral, negative labels.
"""

import pandas as pd
import os
import ast
import json
from typing import Optional
from collections import Counter


def load_phrasebank(file_path: str) -> pd.DataFrame:
    """
    Load Financial PhraseBank dataset (LEGACY - Deprecated).
    
    NOTE: This is a legacy dataset. The project now uses modern social-media datasets.
    Use twitter_financial, financial_tweets_2023, or tweetfinsent instead.
    
    Financial PhraseBank contains short financial sentences from news articles,
    manually annotated by finance experts.
    
    Reference: Malo et al. (2014) "Good Debt or Bad Debt: Detecting Semantic 
    Orientations in Economic Texts"
    
    Expected format: CSV with columns for text and sentiment labels.
    Common formats:
    - Columns: 'sentence' or 'text', 'sentiment' or 'label'
    - Labels: 'positive', 'neutral', 'negative' (or 'pos', 'neu', 'neg')
    
    Args:
        file_path: Path to the PhraseBank CSV file
        
    Returns:
        DataFrame with 'text' and 'label' columns (unified labels)
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
    Load SemEval-2017 Task 5 dataset (LEGACY - Deprecated).
    
    NOTE: This is a legacy dataset. The project now uses modern social-media datasets.
    Use twitter_financial, financial_tweets_2023, or tweetfinsent instead.
    
    SemEval-2017 Task 5 contains financial microblogs with fine-grained sentiment annotations.
    
    Reference: Cortis et al. (2017) "SemEval-2017 Task 5: Fine-Grained Sentiment 
    Analysis on Financial Microblogs and News"
    
    Expected format: CSV or TSV with text and sentiment scores/labels.
    Common formats:
    - Columns: 'text', 'tweet', or 'message' for text
    - Sentiment: numeric scores or labels ('positive', 'neutral', 'negative')
    - Scores are converted: > 0.1 = positive, < -0.1 = negative, else neutral
    
    Args:
        file_path: Path to the SemEval dataset file (CSV or TSV)
        
    Returns:
        DataFrame with 'text' and 'label' columns (unified labels)
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
    Load SEntFiN 1.0 dataset (LEGACY - Deprecated).
    
    NOTE: This is a legacy dataset containing news headlines, not social media text.
    The project now uses modern social-media datasets exclusively.
    Use twitter_financial, financial_tweets_2023, or tweetfinsent instead.
    
    SEntFiN format has:
    - Title: the headline text
    - Decisions: dictionary string with entity-sentiment pairs like {'Entity': 'sentiment'}
    
    When a headline has multiple entities with different sentiments, we aggregate them.
    
    Reference: SEntFiN 1.0 (2023) - Entity-sentiment annotated financial news
    
    Args:
        file_path: Path to the SEntFiN CSV file
        aggregation_method: How to handle multiple sentiments per headline
            - 'most_common': Use the most frequent sentiment (default)
            - 'first': Use the first entity's sentiment
        
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


def load_twitter_financial(file_path: str) -> pd.DataFrame:
    """
    Load Twitter Financial News Sentiment dataset (Zeroshot, 2023).
    
    This is a modern post-2020 dataset from Twitter with clean labeling,
    ideal for baseline interpretability and label quality analysis.
    
    Expected format: CSV/JSON with text and sentiment labels.
    Labels: 'bullish' / 'neutral' / 'bearish' → maps to 'positive' / 'neutral' / 'negative'
    
    Args:
        file_path: Path to the dataset file (CSV, TSV, or JSON)
        
    Returns:
        DataFrame with 'text' and 'label' columns (unified labels)
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
        raise ValueError("Could not find text column in Twitter Financial dataset")
    
    # Find label column
    label_col = None
    for col in ['label', 'Label', 'sentiment', 'Sentiment', 'sentiment_label', 'class']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Could not find label column in Twitter Financial dataset")
    
    # Map labels: bullish/positive → positive, bearish/negative → negative, neutral → neutral
    # Note: This dataset uses numeric labels: 0=neutral, 1=positive, 2=negative
    label_mapping = {
        'bullish': 'positive',
        'positive': 'positive',
        'pos': 'positive',
        '1': 'positive',
        1: 'positive',
        'bearish': 'negative',
        'negative': 'negative',
        'neg': 'negative',
        '2': 'negative',
        2: 'negative',
        'neutral': 'neutral',
        'neu': 'neutral',
        '0': 'neutral',
        0: 'neutral',
    }
    
    # Handle numeric labels directly (for datasets with 0, 1, 2 format)
    def map_label(value):
        if pd.isna(value):
            return 'neutral'
        # Try numeric mapping first
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
        value_str = str(value).lower()
        return label_mapping.get(value_str, 'neutral')
    
    # Create unified dataframe
    result_df = pd.DataFrame({
        'text': df[text_col],
        'label': df[label_col].apply(map_label)
    })
    
    # Filter to only valid labels
    valid_labels = ['positive', 'neutral', 'negative']
    result_df = result_df[result_df['label'].isin(valid_labels)]
    
    return result_df


def load_financial_tweets_2023(file_path: str) -> pd.DataFrame:
    """
    Load Financial Tweets Sentiment dataset (TimKoornstra, 2023).
    
    This is a large-scale aggregated Twitter dataset from 2023, excellent for
    training robust models and analyzing noisy labels.
    
    Expected format: CSV/JSON with text and sentiment labels.
    Labels: 'bullish' / 'neutral' / 'bearish' → maps to 'positive' / 'neutral' / 'negative'
    
    Args:
        file_path: Path to the dataset file (CSV, TSV, or JSON)
        
    Returns:
        DataFrame with 'text' and 'label' columns (unified labels)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Financial Tweets 2023 file not found: {file_path}")
    
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
    for col in ['text', 'Text', 'tweet', 'Tweet', 'content', 'Content', 'message', 'Message']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("Could not find text column in Financial Tweets 2023 dataset")
    
    # Find label column
    label_col = None
    for col in ['label', 'Label', 'sentiment', 'Sentiment', 'sentiment_label', 'class', 'sentiment_class']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Could not find label column in Financial Tweets 2023 dataset")
    
    # Map labels: bullish/positive → positive, bearish/negative → negative, neutral → neutral
    label_mapping = {
        'bullish': 'positive',
        'positive': 'positive',
        'pos': 'positive',
        '1': 'positive',
        1: 'positive',
        'bearish': 'negative',
        'negative': 'negative',
        'neg': 'negative',
        '2': 'negative',
        2: 'negative',
        'neutral': 'neutral',
        'neu': 'neutral',
        '0': 'neutral',
        0: 'neutral',
    }
    
    # Create unified dataframe
    result_df = pd.DataFrame({
        'text': df[text_col],
        'label': df[label_col].astype(str).str.lower().map(label_mapping).fillna(df[label_col].astype(str).str.lower())
    })
    
    # Filter to only valid labels
    valid_labels = ['positive', 'neutral', 'negative']
    result_df = result_df[result_df['label'].isin(valid_labels)]
    
    return result_df


def load_tweetfinsent(file_path: str) -> pd.DataFrame:
    """
    Load TweetFinSent dataset (JP Morgan, 2022).
    
    This is an expert-annotated high-quality dataset from JP Morgan research team.
    Small but very high quality, ideal for label quality analysis.
    
    Expected format: CSV/JSON with text and sentiment labels.
    Labels: 'positive' / 'neutral' / 'negative' (stock price sentiment)
    
    Args:
        file_path: Path to the dataset file (CSV, TSV, or JSON)
        
    Returns:
        DataFrame with 'text' and 'label' columns (unified labels)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TweetFinSent file not found: {file_path}")
    
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
        raise ValueError("Could not find text column in TweetFinSent dataset")
    
    # Find label column
    label_col = None
    for col in ['label', 'Label', 'sentiment', 'Sentiment', 'sentiment_label', 'stock_sentiment']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Could not find label column in TweetFinSent dataset")
    
    # Map labels (TweetFinSent uses positive/neutral/negative directly)
    label_mapping = {
        'positive': 'positive',
        'pos': 'positive',
        '1': 'positive',
        1: 'positive',
        'negative': 'negative',
        'neg': 'negative',
        '-1': 'negative',
        -1: 'negative',
        'neutral': 'neutral',
        'neu': 'neutral',
        '0': 'neutral',
        0: 'neutral',
    }
    
    # Create unified dataframe
    result_df = pd.DataFrame({
        'text': df[text_col],
        'label': df[label_col].astype(str).str.lower().map(label_mapping).fillna(df[label_col].astype(str).str.lower())
    })
    
    # Filter to only valid labels
    valid_labels = ['positive', 'neutral', 'negative']
    result_df = result_df[result_df['label'].isin(valid_labels)]
    
    return result_df


def load_dataset(dataset_name: str, file_path: str, **kwargs) -> pd.DataFrame:
    """
    Unified dataset loader for financial social-media sentiment datasets.
    
    PRIMARY DATASETS (post-2020, recommended):
    - 'twitter_financial': Twitter Financial News Sentiment (Zeroshot, 2023)
    - 'financial_tweets_2023': Financial Tweets Sentiment (TimKoornstra, 2023)
    - 'tweetfinsent': TweetFinSent (JP Morgan, 2022)
    
    LEGACY DATASETS (deprecated - for reference only):
    - 'phrasebank': Financial PhraseBank (LEGACY)
    - 'semeval': SemEval-2017 Task 5 (LEGACY)
    - 'sentfin': SEntFiN 1.0 (LEGACY - news headlines, not social media)
    
    Args:
        dataset_name: Name of dataset
        file_path: Path to dataset file
        **kwargs: Additional arguments for specific loaders
            - For SEntFiN (legacy): aggregation_method ('most_common' or 'first')
        
    Returns:
        DataFrame with 'text' and 'label' columns (unified to: positive, neutral, negative)
        
    Example:
        >>> df = load_dataset('twitter_financial', 'data/twitter_financial_train.csv')
        >>> df = load_dataset('financial_tweets_2023', 'data/financial_tweets_2023.csv')
        >>> df = load_dataset('tweetfinsent', 'data/tweetfinsent.csv')
    """
    dataset_name = dataset_name.lower()
    
    # Primary datasets (post-2020)
    if dataset_name == 'twitter_financial':
        return load_twitter_financial(file_path)
    elif dataset_name == 'financial_tweets_2023':
        return load_financial_tweets_2023(file_path)
    elif dataset_name == 'tweetfinsent':
        return load_tweetfinsent(file_path)
    # Legacy datasets (deprecated)
    elif dataset_name == 'phrasebank':
        return load_phrasebank(file_path)
    elif dataset_name == 'semeval':
        return load_semeval(file_path)
    elif dataset_name == 'sentfin':
        return load_sentfin(file_path, **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Primary: 'twitter_financial', 'financial_tweets_2023', 'tweetfinsent'. "
            f"Legacy: 'phrasebank', 'semeval', 'sentfin'"
        )

