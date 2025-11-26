# Financial Sentiment Analysis - Presentation Summary

## Executive Summary

This document provides key metrics and findings from the Financial Sentiment Analysis project.

---

## Model Performance

### Overall Metrics
- **Total Test Samples**: 2151
- **Accuracy**: 0.7792 (77.92%)

### Per-Class Performance


**POSITIVE**:
- Precision: 0.7597
- Recall: 0.8530
- F1-Score: 0.8037
- Support: 823 samples


**NEUTRAL**:
- Precision: 0.7703
- Recall: 0.7824
- F1-Score: 0.7763
- Support: 703 samples


**NEGATIVE**:
- Precision: 0.8265
- Recall: 0.6784
- F1-Score: 0.7452
- Support: 625 samples


---

## Label Quality Analysis

### Data Quality Metrics
- **Misclassifications**: 1323 (61.51%)
- **Ambiguous Predictions**: 1860 (86.47%)
- **Noisy Labels**: 1299 (60.39%)

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
- **Name**: SEntFiN 1.0
- **Size**: 10,753 samples
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
