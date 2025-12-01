"""
Evaluation script for financial sentiment classifier.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_all_top_features
from src.preprocess import preprocess_batch


def evaluate_model(
    model_path: str,
    data_path: str,
    dataset_name: str = 'twitter_financial',
    output_dir: str = 'results'
):
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model_path: Path to saved model file
        data_path: Path to test dataset file
        dataset_name: Name of dataset ('phrasebank' or 'semeval')
        output_dir: Directory to save evaluation results
    """
    # Load model
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print("Model loaded successfully")
    
    # Load dataset
    print(f"Loading {dataset_name} dataset from {data_path}...")
    from src.dataset_loader import load_dataset
    df = load_dataset(dataset_name, data_path)
    print(f"Loaded {len(df)} samples")
    
    # Preprocess text
    print("Preprocessing text...")
    df['cleaned_text'] = preprocess_batch(df['text'])
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Prepare data
    X = df['cleaned_text'].values
    y_true = df['label'].values
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = model.named_steps['classifier'].classes_
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=classes, columns=classes))
    
    # Top features
    print("\n" + "=" * 60)
    print("Top Features by Class:")
    print("=" * 60)
    top_features = get_all_top_features(model, top_n=20)
    
    for class_name, features in top_features.items():
        print(f"\n{class_name.upper()}:")
        for feature, weight in features[:10]:  # Show top 10
            print(f"  {feature:20s} {weight:8.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed predictions
    results_df = pd.DataFrame({
        'text': df['text'].values,
        'cleaned_text': df['cleaned_text'].values,
        'true_label': y_true,
        'predicted_label': y_pred,
        'confidence': np.max(y_proba, axis=1)
    })
    
    # Add per-class probabilities
    for i, class_name in enumerate(classes):
        results_df[f'prob_{class_name}'] = y_proba[:, i]
    
    results_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to {results_path}")
    
    # Save evaluation summary
    summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Evaluation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of samples: {len(df)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(pd.DataFrame(cm, index=classes, columns=classes)))
    
    print(f"Evaluation summary saved to {summary_path}")
    print("\nEvaluation completed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate financial sentiment classifier')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test dataset file')
    parser.add_argument('--dataset_name', type=str, default='twitter_financial',
                        choices=['twitter_financial', 'financial_tweets_2023', 'tweetfinsent', 'phrasebank', 'semeval', 'sentfin'],
                        help='Dataset name (primary: twitter_financial, financial_tweets_2023, tweetfinsent; legacy: phrasebank, semeval, sentfin)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir
    )

