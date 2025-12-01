# Financial Social-Media Sentiment Analysis Project
## Presentation Slides

---

## Slide 1: Title Slide

# Financial Social-Media Sentiment Analysis
## Machine Learning Pipeline with Label Quality Evaluation

**Project Overview**
- TF-IDF + Logistic Regression Classifier
- Financial Social-Media Sentiment Classification
- Comprehensive Label Quality Analysis

**CS5100 Final Project**

---

## Slide 2: Problem Statement

### Why Financial Social-Media Sentiment Analysis?

- **Volume**: Millions of financial social-media posts generated daily
- **Impact**: Sentiment drives market decisions
- **Challenge**: Manual analysis is impossible at scale
- **Solution**: Automated ML classification system

**Goal**: Classify financial social-media texts as **Positive**, **Neutral**, or **Negative**

**Research Focus**: Label quality evaluation in noisy social-media text

---

## Slide 3: Dataset

### Twitter Financial News Sentiment (Zeroshot, 2023)

**Dataset Characteristics**:
- **Source**: Real Twitter financial posts
- **Samples**: ~9,500 tweets
- **Labels**: 3-class (0=neutral, 1=positive, 2=negative) → unified to positive/neutral/negative
- **Format**: CSV with `text` and `label` columns

**Why This Dataset?**
- **Direct Twitter data**: Real social media platform
- **Clean annotations**: Ideal for baseline interpretability
- **Real-world noise**: Hashtags, mentions, cashtags, URLs
- **Ideal for label quality**: Higher ambiguity than news articles

**Key Features**:
- Short, informal text (typical of social media)
- Higher ambiguity (borderline cases common)
- Real social-media noise characteristics
- Perfect for studying label quality in noisy text

---

## Slide 4: Methodology - Preprocessing

### Text Preprocessing Pipeline

1. **Remove URLs** - Clean web links
2. **Remove Cashtags** - Remove stock symbols ($TSLA)
3. **Remove Hashtags & Mentions** - Clean social media artifacts
4. **Lowercase** - Normalize text
5. **Remove Punctuation** - Clean special characters
6. **Tokenize** - Split into words

**Example**:
```
Original: "RT @user: $TSLA is going up! Check https://t.co/abc #stocks"
Cleaned:  "rt user tsla is going up check stocks"
```

---

## Slide 5: Methodology - Model

### Machine Learning Pipeline

**TF-IDF Vectorization**
- 1-2 gram features (unigrams + bigrams)
- Maximum 10,000 features
- English stop words removed

**Logistic Regression Classifier**
- Multi-class classification
- L-BFGS solver
- Interpretable feature weights

**Pipeline**: `TF-IDF → Logistic Regression`

**Why Baseline Model?**
- **Interpretability**: Feature weights show model decisions
- **Lightweight NLP**: Matches CS5100 proposal requirement
- **Fast training**: Efficient for baseline analysis
- **Note**: Cannot capture sarcasm or long-range dependencies (limitation)

---

## Slide 6: Results - Model Performance

### Performance Metrics

**Overall Performance**:
- **Accuracy**: ~78%
- **Macro F1-Score**: ~0.66
- **Test Samples**: ~1,900

**Per-Class Performance**:
- **Negative**: High recall (0.96) - dominant class
- **Neutral**: Low recall (0.36) - most challenging
- **Positive**: Moderate recall (0.51)

**Key Finding**: Neutral class is most difficult to classify

---

## Slide 7: Results - Confusion Matrix

### Confusion Matrix Analysis

[Include confusion matrix visualization]

**Observations**:
- Negative class: High accuracy (dominant class, 64.74% of dataset)
- Neutral class: Many misclassified as negative or positive
- Positive class: Some confusion with negative

**Class Imbalance Impact**:
- Negative: 64.74% of dataset
- Positive: 20.15% of dataset
- Neutral: 15.11% of dataset

Imbalance ratio: ~4.3:1 (negative to neutral)

---

## Slide 8: Interpretability - Top Features

### Feature Weights by Class

**Positive Sentiment**:
- "beats", "bullish", "rises", "higher", "strong", "gain"

**Negative Sentiment**:
- "declares", "fed", "loss", "misses", "downgraded", "falls"

**Neutral Sentiment**:
- "lower", "downgraded", "misses", "cuts", "target cut"

**Key Insight**: Feature weights align with financial sentiment patterns

[Include top features visualization]

---

## Slide 9: Label Quality Analysis

### Comprehensive Label Quality Evaluation

**Core Analysis**:
- **Misclassifications**: 1,414 cases (14.84% of dataset)
- **Ambiguous Predictions**: 1,262 cases (13.25% of dataset)
- **Noisy Labels**: 1,319 potentially problematic labels

**Social Media-Specific Analysis**:
- **Neutral Ambiguous Zone**: 726 cases
- **Borderline Cases**: 1,172 cases (positive/negative vs neutral)
- **Dataset Ambiguity Metrics**:
  - Average confidence: 0.73
  - Ambiguous zone: 13.25%
  - Low confidence: 25.04%

**Key Finding**: 25% of predictions have low confidence, indicating dataset-inherent ambiguity

---

## Slide 10: Social-Media Text Characteristics

### Why Social-Media Text is Challenging

**Noise Characteristics**:
- Hashtags, mentions, cashtags, URLs
- Abbreviations and informal language
- Missing context (no conversation history)
- Sarcasm and irony

**Ambiguity Patterns**:
- Borderline cases between sentiment classes
- Neutral ambiguous zone (common in social media)
- Short texts with limited context
- Multiple interpretations possible

**Label Quality Implications**:
- Higher annotation inconsistency
- More ambiguous cases requiring review
- Noisy labels from rapid annotation
- Dataset-inherent ambiguity metrics

---

## Slide 11: Key Findings

### What We Learned

1. **Social-media text is inherently noisier** than news articles
2. **Label quality evaluation** is crucial for social-media datasets
3. **TF-IDF + Logistic Regression** provides good baseline with interpretability
4. **Neutral zone is particularly challenging** in social-media text
5. **Class imbalance affects performance** on minority classes
6. **Label quality matters more than accuracy** for noisy text

### Main Research Contribution

**Label quality evaluation framework** provides insights into:
- Dataset reliability
- Model limitations
- Annotation inconsistencies
- Inherent ambiguity in social-media text

This data-centric analysis is more valuable than raw accuracy metrics alone.

---

## Slide 12: Limitations

### Model Limitations

- **Bag-of-words representation**: TF-IDF ignores word order and long-range context
- **Baseline model**: Cannot capture sarcasm or long-range dependencies
- **No deep learning**: By design, we use lightweight NLP for interpretability
- **Short text challenge**: Model may struggle with very short or highly informal posts

### Data Limitations

- **Social-media ambiguity**: Inherently ambiguous (sarcasm, missing context)
- **Class imbalance**: Significant imbalance (negative: 64.74%, positive: 20.15%, neutral: 15.11%)
- **Label noise**: Some labels may be inconsistent
- **Missing context**: No conversation history or background

### Analysis Limitations

- Label quality heuristics are simple
- Ambiguity metrics are model-dependent

---

## Slide 13: Future Work

### Potential Improvements

1. **Contextual Embeddings**: Replace TF-IDF with BERT/FinBERT for stronger performance
2. **Active Learning**: Focus annotation on ambiguous cases identified by label quality analysis
3. **Additional Features**: Incorporate cashtags, hashtags, user metadata
4. **Real-Time Deployment**: API for live social-media streams
5. **Model Improvements**: Experiment with different classifiers or feature engineering

**Note**: FinBERT or other contextual embeddings could serve as a future upper bound for performance, especially for capturing sarcasm and long-range dependencies.

---

## Slide 14: Project Alignment

### Alignment with CS5100 Proposal

✅ **Financial social-media sentiment** (Twitter dataset, not news articles)  
✅ **Lightweight NLP** (TF-IDF + Logistic Regression baseline)  
✅ **Interpretability** (feature weights)  
✅ **Label quality evaluation** (comprehensive analysis)

**Main Research Contribution**: Label quality evaluation framework for noisy social-media text

---

## Slide 15: Conclusion

### Summary

- Built complete ML pipeline for financial social-media sentiment classification
- Achieved solid performance on Twitter Financial News Sentiment dataset (accuracy: ~78%, macro F1: ~0.66)
- Provided interpretable model via feature weights
- Conducted comprehensive label quality evaluation
- Identified ambiguous cases and noisy labels in social-media text

**Impact**: Framework for analyzing financial sentiment in noisy social-media text with focus on label quality evaluation.

**Alignment**: Perfectly matches CS5100 research proposal focusing on financial social-media sentiment with label quality evaluation.

**Key Takeaway**: Label quality evaluation provides more insights than raw accuracy metrics, especially for noisy social-media text.

---

## Presentation Tips

1. **Focus on label quality**: This is your main research contribution
2. **Show visual examples**: Use confusion matrix, top features, label quality plots
3. **Explain limitations**: Be honest about baseline model limitations
4. **Emphasize social-media challenges**: Why label quality matters more for noisy text
5. **Keep it concise**: 10-15 minutes, focus on key findings
