"""
Training script for financial sentiment classifier.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

from src.model import build_model
from src.preprocess import preprocess_batch


def train_model(
    data_path: str,
    valid_path: str,
    dataset_name: str = 'twitter_financial',
    max_features: int = 10000,
    model_save_path: str = 'results/model.joblib',
    results_dir: str = 'results'
):
    """
    Train the financial sentiment classifier.
    
    Args:
        data_path: Path to training dataset file
        valid_path: Path to validation dataset file (completely independent from training)
        dataset_name: Dataset name (must be 'twitter_financial')
        max_features: Maximum TF-IDF features
        model_save_path: Path to save trained model
        results_dir: Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load training dataset
    print(f"Loading training {dataset_name} dataset from {data_path}...")
    from src.dataset_loader import load_dataset
    train_df = load_dataset(dataset_name, data_path)
    print(f"Loaded {len(train_df)} training samples")
    print(f"Training label distribution:\n{train_df['label'].value_counts()}")
    
    # Preprocess training text
    print("Preprocessing training text...")
    train_df['cleaned_text'] = preprocess_batch(train_df['text'])
    
    # Remove empty texts
    train_df = train_df[train_df['cleaned_text'].str.len() > 0]
    print(f"After preprocessing: {len(train_df)} training samples")
    
    # Prepare training data
    X_train = train_df['cleaned_text'].values
    y_train = train_df['label'].values
    
    # Load validation dataset (completely independent)
    print(f"\nLoading validation {dataset_name} dataset from {valid_path}...")
    valid_df = load_dataset(dataset_name, valid_path)
    print(f"Loaded {len(valid_df)} validation samples")
    print(f"Validation label distribution:\n{valid_df['label'].value_counts()}")
    
    # Preprocess validation text
    print("Preprocessing validation text...")
    valid_df['cleaned_text'] = preprocess_batch(valid_df['text'])
    valid_df = valid_df[valid_df['cleaned_text'].str.len() > 0]
    print(f"After preprocessing: {len(valid_df)} validation samples")
    
    # Prepare validation data
    X_valid = valid_df['cleaned_text'].values
    y_valid = valid_df['label'].values
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_valid)} samples")
    
    # Build and train model
    print("Building model...")
    model = build_model(max_features=max_features)
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_pred = model.predict(X_valid)
    
    # Print classification report
    print("\nClassification Report (on validation set):")
    print(classification_report(y_valid, y_pred))
    
    # Save model
    print(f"\nSaving model to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    classes = model.named_steps['classifier'].classes_
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()
    
    # Save training summary
    summary = {
        'dataset': dataset_name,
        'train_samples': len(X_train),
        'validation_samples': len(X_valid),
        'max_features': max_features,
        'validation_accuracy': (y_pred == y_valid).mean()
    }
    
    summary_path = os.path.join(results_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Training Summary\n")
        f.write("=" * 50 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\nClassification Report (on validation set):\n")
        f.write(classification_report(y_valid, y_pred))
    
    print(f"Training summary saved to {summary_path}")
    print("\nTraining completed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train financial sentiment classifier')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training dataset file')
    parser.add_argument('--valid_path', type=str, required=True,
                        help='Path to validation dataset file (completely independent from training)')
    parser.add_argument('--dataset_name', type=str, default='twitter_financial',
                        choices=['twitter_financial'],
                        help='Dataset name (must be twitter_financial)')
    parser.add_argument('--max_features', type=int, default=10000,
                        help='Maximum TF-IDF features')
    parser.add_argument('--model_path', type=str, default='results/model.joblib',
                        help='Path to save model')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        valid_path=args.valid_path,
        dataset_name=args.dataset_name,
        max_features=args.max_features,
        model_save_path=args.model_path
    )

