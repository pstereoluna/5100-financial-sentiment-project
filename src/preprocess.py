"""
Text preprocessing module for financial sentiment analysis.

This module provides functions to clean and preprocess financial social media text
by removing URLs, cashtags, hashtags, mentions, and normalizing text.
"""

import re
import string
from typing import Union, List
import pandas as pd

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Using simple tokenization.")


def clean_text(text: str) -> str:
    """
    Clean and preprocess text for financial sentiment analysis.
    
    Steps:
    1. Remove URLs
    2. Remove cashtags ($TSLA)
    3. Remove hashtags and @mentions
    4. Convert to lowercase
    5. Remove punctuation
    6. Tokenize and rejoin
    
    Args:
        text: Raw input text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove cashtags (e.g., $TSLA, $AAPL)
    text = re.sub(r'\$\w+', '', text)
    
    # Remove hashtags and @mentions
    text = re.sub(r'#\w+|@\w+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback if NLTK data not downloaded
            tokens = text.split()
    else:
        tokens = text.split()
    
    # Remove empty tokens and rejoin
    tokens = [t for t in tokens if t.strip()]
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


def preprocess_batch(texts: Union[List[str], pd.Series]) -> List[str]:
    """
    Preprocess a batch of texts.
    
    Args:
        texts: List or pandas Series of text strings
        
    Returns:
        List of cleaned text strings
    """
    return [clean_text(text) for text in texts]

