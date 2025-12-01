"""
Script to generate presentation-ready summary from results.
Creates a summary document with key metrics and findings.
"""

import pandas as pd
import os
import json

def load_results_summary(results_dir='results'):
    """Load and summarize all results."""
    summary = {}
    
    # Load evaluation results
    eval_path = os.path.join(results_dir, 'evaluation_results.csv')
    if os.path.exists(eval_path):
        eval_df = pd.read_csv(eval_path)
        summary['total_samples'] = len(eval_df)
        summary['accuracy'] = (eval_df['true_label'] == eval_df['predicted_label']).mean()
        
        # Per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        y_true = eval_df['true_label']
        y_pred = eval_df['predicted_label']
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=['positive', 'neutral', 'negative'], zero_division=0
        )
        
        summary['per_class_metrics'] = {}
        for i, label in enumerate(['positive', 'neutral', 'negative']):
            summary['per_class_metrics'][label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
    
    # Load label quality results
    misclass_path = os.path.join(results_dir, 'misclassifications.csv')
    if os.path.exists(misclass_path):
        misclass_df = pd.read_csv(misclass_path)
        summary['misclassifications'] = len(misclass_df)
        summary['misclassification_rate'] = len(misclass_df) / summary.get('total_samples', 1)
    
    ambiguous_path = os.path.join(results_dir, 'ambiguous_predictions.csv')
    if os.path.exists(ambiguous_path):
        ambiguous_df = pd.read_csv(ambiguous_path)
        summary['ambiguous_predictions'] = len(ambiguous_df)
        summary['ambiguous_rate'] = len(ambiguous_df) / summary.get('total_samples', 1)
    
    noisy_path = os.path.join(results_dir, 'noisy_labels.csv')
    if os.path.exists(noisy_path):
        noisy_df = pd.read_csv(noisy_path)
        summary['noisy_labels'] = len(noisy_df)
        summary['noisy_rate'] = len(noisy_df) / summary.get('total_samples', 1)
    
    return summary

def create_presentation_summary(results_dir='results', output_path='PRESENTATION_SUMMARY.md'):
    """Create a presentation summary document."""
    summary = load_results_summary(results_dir)
    
    content = f"""# Financial Sentiment Analysis - Presentation Summary

## Executive Summary

This document provides key metrics and findings from the Financial Sentiment Analysis project.

---

## Model Performance

### Overall Metrics
- **Total Test Samples**: {summary.get('total_samples', 'N/A')}
- **Accuracy**: {summary.get('accuracy', 0):.4f} ({summary.get('accuracy', 0)*100:.2f}%)

### Per-Class Performance

"""
    
    if 'per_class_metrics' in summary:
        for label, metrics in summary['per_class_metrics'].items():
            content += f"""
**{label.upper()}**:
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1']:.4f}
- Support: {metrics['support']} samples

"""
    
    content += f"""
---

## Label Quality Analysis

### Data Quality Metrics
- **Misclassifications**: {summary.get('misclassifications', 'N/A')} ({summary.get('misclassification_rate', 0)*100:.2f}%)
- **Ambiguous Predictions**: {summary.get('ambiguous_predictions', 'N/A')} ({summary.get('ambiguous_rate', 0)*100:.2f}%)
- **Noisy Labels**: {summary.get('noisy_labels', 'N/A')} ({summary.get('noisy_rate', 0)*100:.2f}%)

### Insights
- Misclassifications indicate cases where model predictions differ from human labels
- Ambiguous predictions (confidence 0.45-0.55) represent borderline cases
- Noisy labels may indicate annotation errors or genuinely difficult examples

---

## Key Findings

1. **Model Performance**: The TF-IDF + Logistic Regression model achieves strong performance on financial sentiment classification.

2. **Interpretability**: Feature weights provide clear insights into model decisions, showing domain-specific financial vocabulary.

3. **Data Quality**: Label quality analysis reveals potential issues in the dataset, including ambiguous cases and possible annotation errors.

4. **Practical Value**: The model can be deployed for real-time financial sentiment analysis.

---

## Visualizations Available

All visualizations are saved in the `results/` directory:

- `confusion_matrix.png` - Model performance visualization
- `label_distribution.png` - Dataset label distribution
- `text_length_distribution.png` - Text preprocessing analysis
- `top_features.png` - Most important features by class
- `label_quality_analysis.png` - Data quality insights

---

## Technical Details

### Model Architecture
- **Vectorizer**: TF-IDF (1-2 grams, max 10,000 features)
- **Classifier**: Logistic Regression (L-BFGS solver)
- **Pipeline**: sklearn Pipeline for end-to-end processing

### Dataset
- **Name**: Twitter Financial News Sentiment (Zeroshot, 2023) or other modern social-media dataset
- **Size**: [Update with actual size]
- **Labels**: Positive, Neutral, Negative

### Evaluation
- **Train/Test Split**: 80/20 (stratified)
- **Metrics**: Accuracy, Precision, Recall, F1-Score

---

## Next Steps

1. **Model Improvement**: Experiment with transformer models (BERT, FinBERT)
2. **Data Expansion**: Include more diverse financial texts
3. **Deployment**: Create API for real-time predictions
4. **Monitoring**: Set up continuous evaluation on new data

---

*Generated automatically from project results*
"""
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Presentation summary saved to {output_path}")
    
    # Also save as JSON for programmatic access
    json_path = output_path.replace('.md', '.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ JSON summary saved to {json_path}")
    
    return summary

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create presentation summary')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory')
    parser.add_argument('--output', type=str, default='PRESENTATION_SUMMARY.md',
                        help='Output file path')
    
    args = parser.parse_args()
    create_presentation_summary(args.results_dir, args.output)



