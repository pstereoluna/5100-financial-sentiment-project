"""
Training script for financial sentiment classifier.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model
from src.preprocess import preprocess_batch


def train_model(
    data_path: str,
    dataset_name: str = 'phrasebank',
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10000,
    model_save_path: str = 'results/model.joblib',
    results_dir: str = 'results'
):
    """
    Train the financial sentiment classifier.
    
    Args:
        data_path: Path to dataset file
        dataset_name: Name of dataset ('phrasebank' or 'semeval')
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        max_features: Maximum TF-IDF features
        model_save_path: Path to save trained model
        results_dir: Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {dataset_name} dataset from {data_path}...")
    from src.dataset_loader import load_dataset
    df = load_dataset(dataset_name, data_path)
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Preprocess text
    print("Preprocessing text...")
    df['cleaned_text'] = preprocess_batch(df['text'])
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    print(f"After preprocessing: {len(df)} samples")
    
    # Split train/test
    X = df['cleaned_text'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Build and train model
    print("Building model...")
    model = build_model(max_features=max_features)
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    print(f"\nSaving model to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
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
        'test_samples': len(X_test),
        'max_features': max_features,
        'test_accuracy': (y_pred == y_test).mean()
    }
    
    summary_path = os.path.join(results_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Training Summary\n")
        f.write("=" * 50 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
    
    print(f"Training summary saved to {summary_path}")
    print("\nTraining completed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train financial sentiment classifier')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset file')
    parser.add_argument('--dataset_name', type=str, default='phrasebank',
                        choices=['phrasebank', 'semeval', 'sentfin'],
                        help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--max_features', type=int, default=10000,
                        help='Maximum TF-IDF features')
    parser.add_argument('--model_path', type=str, default='results/model.joblib',
                        help='Path to save model')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        test_size=args.test_size,
        max_features=args.max_features,
        model_save_path=args.model_path
    )

