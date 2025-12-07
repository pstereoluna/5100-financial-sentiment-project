"""
Model definition for financial sentiment classification.

Uses TF-IDF vectorization (1-2 grams) with Logistic Regression classifier.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import List, Tuple


def build_model(max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)) -> Pipeline:
    """
    Build TF-IDF + Logistic Regression pipeline.
    
    Args:
        max_features: Maximum number of features for TF-IDF
        ngram_range: N-gram range for TF-IDF (default: (1, 2) for unigrams and bigrams)
        
    Returns:
        sklearn Pipeline with vectorizer and classifier
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced'  # Handle class imbalance
        # Note: multi_class='multinomial' is now default and deprecated parameter removed
    )
    
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])
    
    return pipeline


def get_top_features(pipeline: Pipeline, class_name: str, top_n: int = 20) -> List[Tuple[str, float]]:
    """
    Get top features (words/ngrams) for a given class.
    
    Args:
        pipeline: Trained sklearn Pipeline
        class_name: Class name ('positive', 'neutral', 'negative')
        top_n: Number of top features to return
        
    Returns:
        List of (feature, weight) tuples sorted by weight
    """
    if not hasattr(pipeline.named_steps['classifier'], 'coef_'):
        raise ValueError("Pipeline must be fitted before extracting features")
    
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['classifier']
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get class index
    classes = classifier.classes_
    try:
        class_idx = np.where(classes == class_name)[0][0]
    except IndexError:
        raise ValueError(f"Class '{class_name}' not found. Available classes: {classes}")
    
    # Get coefficients for this class
    coef = classifier.coef_[class_idx]
    
    # Sort by coefficient value (highest first)
    top_indices = np.argsort(coef)[::-1][:top_n]
    
    top_features = [(feature_names[i], coef[i]) for i in top_indices]
    
    return top_features


def get_all_top_features(pipeline: Pipeline, top_n: int = 20) -> dict:
    """
    Get top features for all classes.
    
    Args:
        pipeline: Trained sklearn Pipeline
        top_n: Number of top features per class
        
    Returns:
        Dictionary mapping class names to lists of (feature, weight) tuples
    """
    classes = pipeline.named_steps['classifier'].classes_
    result = {}
    
    for class_name in classes:
        result[class_name] = get_top_features(pipeline, class_name, top_n)
    
    return result

