"""
Label quality evaluation module for social media financial sentiment.

This module aligns with the CS5100 research proposal focusing on label quality
evaluation in noisy social media text. Social media financial posts (e.g., StockTwits)
are inherently noisier than news articles, making them ideal for studying:
- Annotation inconsistencies
- Borderline cases (positive vs neutral, negative vs neutral)
- Neutral ambiguous zone
- Dataset-inherent ambiguity

Detects:
- Misclassifications
- Ambiguous low-confidence predictions
- Noisy label heuristics
- Neutral zone ambiguity (social media specific)
- Borderline positive/negative vs neutral cases
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
    dataset_name: str = 'twitter_financial',
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
    dataset_name: str = 'twitter_financial',
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
    dataset_name: str = 'twitter_financial',
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


def analyze_neutral_ambiguous_zone(
    model_path: str,
    data_path: str,
    dataset_name: str = 'twitter_financial',
    output_path: str = 'results/neutral_ambiguous_zone.csv'
) -> pd.DataFrame:
    """
    Analyze the neutral ambiguous zone - cases where model struggles to distinguish
    between neutral and positive/negative.
    
    This is particularly relevant for social media text where many posts are
    borderline between neutral and sentiment-bearing.
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        dataset_name: Dataset name
        output_path: Path to save CSV report
        
    Returns:
        DataFrame with neutral ambiguous zone cases
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
    
    classes = model.named_steps['classifier'].classes_
    neutral_idx = np.where(classes == 'neutral')[0][0]
    
    # Get neutral probabilities
    neutral_probs = y_proba[:, neutral_idx]
    
    # Cases where neutral probability is high but label is positive/negative
    # OR where neutral probability is medium (0.3-0.5) regardless of label
    neutral_ambiguous = (
        ((neutral_probs > 0.3) & (neutral_probs < 0.5)) |  # Medium neutral confidence
        ((neutral_probs > 0.4) & (y_true != 'neutral'))    # High neutral prob but not neutral label
    )
    
    ambiguous_df = pd.DataFrame({
        'text': df['text'].values[neutral_ambiguous],
        'cleaned_text': df['cleaned_text'].values[neutral_ambiguous],
        'true_label': y_true[neutral_ambiguous],
        'predicted_label': y_pred[neutral_ambiguous],
        'neutral_prob': neutral_probs[neutral_ambiguous],
        'max_prob': np.max(y_proba[neutral_ambiguous], axis=1),
        'text_length': df['cleaned_text'].str.len().values[neutral_ambiguous]
    })
    
    # Add per-class probabilities
    for i, class_name in enumerate(classes):
        ambiguous_df[f'prob_{class_name}'] = y_proba[neutral_ambiguous, i]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ambiguous_df.to_csv(output_path, index=False)
    print(f"Found {len(ambiguous_df)} cases in neutral ambiguous zone")
    print(f"Saved to {output_path}")
    
    return ambiguous_df


def analyze_borderline_cases(
    model_path: str,
    data_path: str,
    dataset_name: str = 'twitter_financial',
    output_path: str = 'results/borderline_cases.csv'
) -> pd.DataFrame:
    """
    Analyze borderline cases: positive vs neutral and negative vs neutral.
    
    Social media text often has posts that are borderline between sentiment
    and neutral (e.g., "Stock is up" - positive or neutral?).
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        dataset_name: Dataset name
        output_path: Path to save CSV report
        
    Returns:
        DataFrame with borderline cases
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
    
    classes = model.named_steps['classifier'].classes_
    
    # Find positive and negative indices
    pos_idx = np.where(classes == 'positive')[0][0] if 'positive' in classes else None
    neg_idx = np.where(classes == 'negative')[0][0] if 'negative' in classes else None
    neu_idx = np.where(classes == 'neutral')[0][0] if 'neutral' in classes else None
    
    borderline_cases = []
    
    # Borderline positive vs neutral
    if pos_idx is not None and neu_idx is not None:
        pos_neu_diff = np.abs(y_proba[:, pos_idx] - y_proba[:, neu_idx])
        borderline_pos_neu = (pos_neu_diff < 0.15) & (
            ((y_true == 'positive') | (y_true == 'neutral')) |
            ((y_pred == 'positive') | (y_pred == 'neutral'))
        )
        borderline_cases.append(borderline_pos_neu)
    
    # Borderline negative vs neutral
    if neg_idx is not None and neu_idx is not None:
        neg_neu_diff = np.abs(y_proba[:, neg_idx] - y_proba[:, neu_idx])
        borderline_neg_neu = (neg_neu_diff < 0.15) & (
            ((y_true == 'negative') | (y_true == 'neutral')) |
            ((y_pred == 'negative') | (y_pred == 'neutral'))
        )
        borderline_cases.append(borderline_neg_neu)
    
    # Combine
    if borderline_cases:
        borderline_mask = np.any(borderline_cases, axis=0)
    else:
        borderline_mask = np.zeros(len(df), dtype=bool)
    
    borderline_df = pd.DataFrame({
        'text': df['text'].values[borderline_mask],
        'cleaned_text': df['cleaned_text'].values[borderline_mask],
        'true_label': y_true[borderline_mask],
        'predicted_label': y_pred[borderline_mask],
        'confidence': np.max(y_proba[borderline_mask], axis=1),
        'text_length': df['cleaned_text'].str.len().values[borderline_mask]
    })
    
    # Add per-class probabilities
    for i, class_name in enumerate(classes):
        borderline_df[f'prob_{class_name}'] = y_proba[borderline_mask, i]
    
    # Classify borderline type
    borderline_types = []
    for idx in range(len(borderline_df)):
        row = borderline_df.iloc[idx]
        if pos_idx is not None and neu_idx is not None:
            pos_neu_diff = abs(row[f'prob_{classes[pos_idx]}'] - row[f'prob_{classes[neu_idx]}'])
            if pos_neu_diff < 0.15:
                borderline_types.append('positive_vs_neutral')
                continue
        if neg_idx is not None and neu_idx is not None:
            neg_neu_diff = abs(row[f'prob_{classes[neg_idx]}'] - row[f'prob_{classes[neu_idx]}'])
            if neg_neu_diff < 0.15:
                borderline_types.append('negative_vs_neutral')
                continue
        borderline_types.append('other')
    
    borderline_df['borderline_type'] = borderline_types
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    borderline_df.to_csv(output_path, index=False)
    print(f"Found {len(borderline_df)} borderline cases")
    print(f"Saved to {output_path}")
    
    return borderline_df


def quantify_dataset_ambiguity(
    model_path: str,
    data_path: str,
    dataset_name: str = 'twitter_financial',
    output_path: str = 'results/dataset_ambiguity_metrics.csv'
) -> pd.DataFrame:
    """
    Quantify dataset-inherent ambiguity metrics.
    
    For social media datasets, we expect higher ambiguity due to:
    - Informal language
    - Missing context
    - Sarcasm/irony
    - Abbreviations
    
    This function computes various ambiguity metrics.
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        dataset_name: Dataset name
        output_path: Path to save CSV report
        
    Returns:
        DataFrame with ambiguity metrics
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
    
    # Compute ambiguity metrics
    max_proba = np.max(y_proba, axis=1)
    
    metrics = {
        'dataset': dataset_name,
        'total_samples': len(df),
        'avg_confidence': float(np.mean(max_proba)),
        'low_confidence_count': int(np.sum(max_proba < 0.6)),
        'low_confidence_pct': float(np.mean(max_proba < 0.6) * 100),
        'ambiguous_zone_count': int(np.sum((max_proba >= 0.45) & (max_proba <= 0.55))),
        'ambiguous_zone_pct': float(np.mean((max_proba >= 0.45) & (max_proba <= 0.55)) * 100),
        'high_confidence_count': int(np.sum(max_proba > 0.8)),
        'high_confidence_pct': float(np.mean(max_proba > 0.8) * 100),
        'misclassification_rate': float(np.mean(y_true != y_pred) * 100),
        'avg_text_length': float(df['cleaned_text'].str.len().mean()),
        'short_text_count': int((df['cleaned_text'].str.len() < 10).sum()),
        'short_text_pct': float((df['cleaned_text'].str.len() < 10).mean() * 100)
    }
    
    # Create DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    
    print("Dataset Ambiguity Metrics:")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print(f"\nSaved to {output_path}")
    
    return metrics_df


def run_label_quality_analysis(
    model_path: str,
    data_path: str,
    dataset_name: str = 'twitter_financial',
    output_dir: str = 'results',
    include_social_media_analysis: bool = True
):
    """
    Run complete label quality analysis with social media-specific metrics.
    
    This function aligns with the CS5100 research proposal by providing
    comprehensive label quality evaluation specifically designed for noisy
    social media financial text.
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        dataset_name: Dataset name
        output_dir: Directory to save all reports
        include_social_media_analysis: If True, include social media-specific
            ambiguity analysis (neutral zone, borderline cases, dataset metrics)
    """
    print("Running label quality analysis...")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    if include_social_media_analysis:
        print("Including social media-specific ambiguity analysis")
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
    
    # Social media-specific analysis
    if include_social_media_analysis:
        print("\n4. Analyzing neutral ambiguous zone...")
        neutral_ambiguous_df = analyze_neutral_ambiguous_zone(
            model_path, data_path, dataset_name,
            os.path.join(output_dir, 'neutral_ambiguous_zone.csv')
        )
        
        print("\n5. Analyzing borderline cases (positive/negative vs neutral)...")
        borderline_df = analyze_borderline_cases(
            model_path, data_path, dataset_name,
            os.path.join(output_dir, 'borderline_cases.csv')
        )
        
        print("\n6. Quantifying dataset-inherent ambiguity...")
        ambiguity_metrics = quantify_dataset_ambiguity(
            model_path, data_path, dataset_name,
            os.path.join(output_dir, 'dataset_ambiguity_metrics.csv')
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("Label Quality Analysis Summary")
    print("=" * 60)
    print(f"Total misclassifications: {len(misclass_df)}")
    print(f"Ambiguous predictions: {len(ambiguous_df)}")
    print(f"Potentially noisy labels: {len(noisy_df)}")
    
    if include_social_media_analysis:
        print(f"Neutral ambiguous zone cases: {len(neutral_ambiguous_df)}")
        print(f"Borderline cases: {len(borderline_df)}")
        print(f"\nDataset ambiguity metrics:")
        print(f"  Average confidence: {ambiguity_metrics['avg_confidence'].values[0]:.3f}")
        print(f"  Ambiguous zone: {ambiguity_metrics['ambiguous_zone_pct'].values[0]:.2f}%")
        print(f"  Low confidence: {ambiguity_metrics['low_confidence_pct'].values[0]:.2f}%")
    
    print(f"\nAll reports saved to {output_dir}/")
    print("\nAnalysis completed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Label quality analysis')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset file')
    parser.add_argument('--dataset_name', type=str, default='twitter_financial',
                        choices=['twitter_financial'],
                        help='Dataset name (must be twitter_financial)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save reports')
    
    args = parser.parse_args()
    
    run_label_quality_analysis(
        model_path=args.model_path,
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir
    )

