# Financial Social-Media Sentiment Analysis
## 8-Minute Presentation Deck (8-10 Slides)

---

## Slide 1: Title & Motivation

# Financial Social-Media Sentiment Analysis
## Label Quality Evaluation in Noisy Text

**Problem**
- Millions of financial posts daily → manual analysis impossible
- Sentiment drives market decisions
- **Challenge**: Noisy social-media text with ambiguous labels

**Goal**: Classify tweets as **Positive**, **Neutral**, or **Negative**  
**Research Focus**: Label quality evaluation in noisy social-media text

**CS5100 Final Project**

---

## Slide 2: Dataset

### Twitter Financial News Sentiment (Zeroshot, 2023)

- **~9,500 tweets** from real Twitter financial posts
- **3-class labels**: positive, neutral, negative
- **Characteristics**: Short, informal text with noise (hashtags, mentions, cashtags)
- **Why this dataset**: Real social-media noise + clean annotations → ideal for label quality research

---

## Slide 3: Preprocessing & Model

### Pipeline: Text Cleaning → TF-IDF → Logistic Regression

**Preprocessing**
- Remove URLs, cashtags, hashtags, mentions
- Lowercase, remove punctuation, tokenize

**Model**
- **TF-IDF**: 1-2 grams, max 10,000 features
- **Logistic Regression**: Multi-class, interpretable weights
- **Why baseline**: Lightweight, interpretable, matches CS5100 proposal

---

## Slide 4: Results Overview

### Performance Metrics

- **Accuracy**: ~80%
- **Macro F1**: ~0.74
- **Validation set**: 2,383 samples (independent hold-out)

**Per-Class Performance**
- **Neutral**: High recall - dominant class (~65%)
- **Positive**: Moderate recall (~20%)
- **Negative**: Lower recall - minority class (~15%)

**Key Finding**: Neutral class is majority, but minority classes (positive/negative) are more challenging

---

## Slide 5: Confusion Matrix Insights

[Include confusion matrix visualization]

**Observations**
- Neutral: High accuracy - dominant class (~65%)
- Negative: Lower accuracy - minority class (~15%), most challenging
- Positive: Moderate accuracy (~20%)

**Class Imbalance Impact**: 4.3:1 ratio (neutral to negative) affects minority class performance

---

## Slide 6: Interpretability - Top Features

### Feature Weights by Class

**Positive**: "bullish", "beats", "rises", "higher", "strong", "positive"

**Negative**: "downgraded", "lower", "misses", "falls", "cuts", "target cut"

**Neutral**: "declares", "reports", "conference", "presentation", "results", "fed"

**Insight**: Feature weights align with financial sentiment patterns

[Include top features visualization]

---

## Slide 7: Label Quality Analysis (Core Contribution)

### Comprehensive Label Quality Evaluation

**Key Metrics** (on validation set with 2,383 samples)
- **Misclassifications**: 469 instances
- **Ambiguous Predictions**: 494 instances
- **Noisy Labels**: 132 instances (using stricter threshold: true_label_prob < 0.2)
- **Neutral Ambiguous Zone**: 524 instances
- **Borderline Cases**: 497 instances

**Key Finding**: Approximately 21% of predictions exhibited low confidence, indicating dataset-inherent ambiguity in social-media text

**Research Contribution**: Label quality framework provides insights into dataset reliability and annotation inconsistencies. The stricter noisy label threshold (0.2) flags only high-confidence disagreements as potential label errors.

---

## Slide 8: Limitations & Future Work

### Limitations

**Model**: Bag-of-words ignores word order, cannot capture sarcasm  
**Data**: Class imbalance (4.3:1), social-media ambiguity, missing context  
**Analysis**: Label quality heuristics are simple, metrics are model-dependent

### Future Work

- **Contextual Embeddings**: BERT/FinBERT for stronger performance
- **Active Learning**: Focus annotation on ambiguous cases
- **Additional Features**: Cashtags, hashtags, user metadata

---

## Slide 9: Conclusion

### Summary

- Built ML pipeline for financial social-media sentiment classification
- Achieved solid baseline performance (accuracy: ~80%, macro F1: ~0.74) on independent validation set
- Provided interpretable model via feature weights
- **Main Contribution**: Label quality evaluation framework for noisy social-media text

**Key Takeaway**: Label quality evaluation provides more insights than raw accuracy metrics, especially for noisy social-media text.

**Alignment**: Matches CS5100 proposal focusing on financial social-media sentiment with label quality evaluation.

---

## Presentation Tips

1. **Focus on label quality**: This is your main research contribution
2. **Show visual examples**: Use confusion matrix, top features, label quality plots
3. **Explain limitations**: Be honest about baseline model limitations
4. **Keep it concise**: 8 minutes presentation + 2 minutes Q&A
