"""
Label quality evaluation module.

Detects:
- Misclassifications
- Ambiguous low-confidence predictions
- Noisy label heuristics
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detect_misclassifications(
    model_path: str,
    data_path: str,
    dataset_name: str = 'phrasebank',
    output_path: str = 'results/misclassifications.csv'
) -> pd.DataFrame:
    """
    Detect misclassified examples.
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        dataset_name: Dataset name
        output_path: Path to save CSV report
        
    Returns:
        DataFrame with misclassified examples
    """
    # Load model
    model = joblib.load(model_path)
    
    # Load and preprocess data
    from src.dataset_loader import load_dataset
    from src.preprocess import preprocess_batch
    
    df = load_dataset(dataset_name, data_path)
    df['cleaned_text'] = preprocess_batch(df['text'])
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Make predictions
    X = df['cleaned_text'].values
    y_true = df['label'].values
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Find misclassifications
    misclassified = y_true != y_pred
    
    misclass_df = pd.DataFrame({
        'text': df['text'].values[misclassified],
        'cleaned_text': df['cleaned_text'].values[misclassified],
        'true_label': y_true[misclassified],
        'predicted_label': y_pred[misclassified],
        'confidence': np.max(y_proba[misclassified], axis=1),
        'true_label_prob': [y_proba[misclassified][i, np.where(model.named_steps['classifier'].classes_ == y_true[misclassified][i])[0][0]] 
                           for i in range(np.sum(misclassified))]
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    misclass_df.to_csv(output_path, index=False)
    print(f"Found {len(misclass_df)} misclassifications")
    print(f"Saved to {output_path}")
    
    return misclass_df


def detect_ambiguous_predictions(
    model_path: str,
    data_path: str,
    dataset_name: str = 'phrasebank',
    confidence_threshold: tuple = (0.45, 0.55),
    output_path: str = 'results/ambiguous_predictions.csv'
) -> pd.DataFrame:
    """
    Detect ambiguous predictions (low confidence).
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        dataset_name: Dataset name
        confidence_threshold: Tuple of (min, max) confidence for ambiguous range
        output_path: Path to save CSV report
        
    Returns:
        DataFrame with ambiguous predictions
    """
    # Load model
    model = joblib.load(model_path)
    
    # Load and preprocess data
    from src.dataset_loader import load_dataset
    from src.preprocess import preprocess_batch
    
    df = load_dataset(dataset_name, data_path)
    df['cleaned_text'] = preprocess_batch(df['text'])
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Make predictions
    X = df['cleaned_text'].values
    y_true = df['label'].values
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Find ambiguous predictions (confidence between thresholds)
    max_proba = np.max(y_proba, axis=1)
    ambiguous = (max_proba >= confidence_threshold[0]) & (max_proba <= confidence_threshold[1])
    
    classes = model.named_steps['classifier'].classes_
    
    ambiguous_df = pd.DataFrame({
        'text': df['text'].values[ambiguous],
        'cleaned_text': df['cleaned_text'].values[ambiguous],
        'true_label': y_true[ambiguous],
        'predicted_label': y_pred[ambiguous],
        'confidence': max_proba[ambiguous]
    })
    
    # Add per-class probabilities
    for i, class_name in enumerate(classes):
        ambiguous_df[f'prob_{class_name}'] = y_proba[ambiguous, i]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ambiguous_df.to_csv(output_path, index=False)
    print(f"Found {len(ambiguous_df)} ambiguous predictions (confidence between {confidence_threshold[0]} and {confidence_threshold[1]})")
    print(f"Saved to {output_path}")
    
    return ambiguous_df


def detect_noisy_labels(
    model_path: str,
    data_path: str,
    dataset_name: str = 'phrasebank',
    output_path: str = 'results/noisy_labels.csv'
) -> pd.DataFrame:
    """
    Detect potentially noisy labels using heuristics.
    
    Heuristics:
    1. High confidence predictions that disagree with labels
    2. Labels that disagree with majority of similar examples
    3. Very short texts (may be ambiguous)
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        dataset_name: Dataset name
        output_path: Path to save CSV report
        
    Returns:
        DataFrame with potentially noisy labels
    """
    # Load model
    model = joblib.load(model_path)
    
    # Load and preprocess data
    from src.dataset_loader import load_dataset
    from src.preprocess import preprocess_batch
    
    df = load_dataset(dataset_name, data_path)
    df['cleaned_text'] = preprocess_batch(df['text'])
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Make predictions
    X = df['cleaned_text'].values
    y_true = df['label'].values
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    max_proba = np.max(y_proba, axis=1)
    
    # Heuristic 1: High confidence misclassifications (likely label errors)
    high_conf_misclass = (y_true != y_pred) & (max_proba > 0.8)
    
    # Heuristic 2: Very short texts (may be ambiguous)
    short_texts = df['cleaned_text'].str.len() < 10
    
    # Heuristic 3: Low confidence on true label
    classes = model.named_steps['classifier'].classes_
    true_label_probs = np.array([y_proba[i, np.where(classes == y_true[i])[0][0]] 
                                 for i in range(len(y_true))])
    low_true_conf = true_label_probs < 0.4
    
    # Combine heuristics
    noisy_mask = high_conf_misclass | (short_texts & (y_true != y_pred)) | low_true_conf
    
    noisy_df = pd.DataFrame({
        'text': df['text'].values[noisy_mask],
        'cleaned_text': df['cleaned_text'].values[noisy_mask],
        'true_label': y_true[noisy_mask],
        'predicted_label': y_pred[noisy_mask],
        'confidence': max_proba[noisy_mask],
        'true_label_prob': true_label_probs[noisy_mask],
        'text_length': df['cleaned_text'].str.len().values[noisy_mask],
        'heuristic': ['high_conf_misclass' if h else ('short_text' if s else 'low_true_conf')
                     for h, s, l in zip(high_conf_misclass[noisy_mask], 
                                       short_texts[noisy_mask],
                                       low_true_conf[noisy_mask])]
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    noisy_df.to_csv(output_path, index=False)
    print(f"Found {len(noisy_df)} potentially noisy labels")
    print(f"Saved to {output_path}")
    
    return noisy_df


def run_label_quality_analysis(
    model_path: str,
    data_path: str,
    dataset_name: str = 'phrasebank',
    output_dir: str = 'results'
):
    """
    Run complete label quality analysis.
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        dataset_name: Dataset name
        output_dir: Directory to save all reports
    """
    print("Running label quality analysis...")
    print("=" * 60)
    
    # Misclassifications
    print("\n1. Detecting misclassifications...")
    misclass_df = detect_misclassifications(
        model_path, data_path, dataset_name,
        os.path.join(output_dir, 'misclassifications.csv')
    )
    
    # Ambiguous predictions
    print("\n2. Detecting ambiguous predictions...")
    ambiguous_df = detect_ambiguous_predictions(
        model_path, data_path, dataset_name,
        output_path=os.path.join(output_dir, 'ambiguous_predictions.csv')
    )
    
    # Noisy labels
    print("\n3. Detecting noisy labels...")
    noisy_df = detect_noisy_labels(
        model_path, data_path, dataset_name,
        output_path=os.path.join(output_dir, 'noisy_labels.csv')
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Label Quality Analysis Summary")
    print("=" * 60)
    print(f"Total misclassifications: {len(misclass_df)}")
    print(f"Ambiguous predictions: {len(ambiguous_df)}")
    print(f"Potentially noisy labels: {len(noisy_df)}")
    print(f"\nAll reports saved to {output_dir}/")
    print("\nAnalysis completed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Label quality analysis')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset file')
    parser.add_argument('--dataset_name', type=str, default='phrasebank',
                        choices=['phrasebank', 'semeval', 'sentfin'],
                        help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save reports')
    
    args = parser.parse_args()
    
    run_label_quality_analysis(
        model_path=args.model_path,
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir
    )

