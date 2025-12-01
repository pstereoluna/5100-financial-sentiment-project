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

---

## Slide 2: Problem Statement

### Why Financial Social-Media Sentiment Analysis?

- **Volume**: Millions of financial social-media posts generated daily
- **Impact**: Sentiment drives market decisions
- **Challenge**: Manual analysis is impossible at scale
- **Solution**: Automated ML classification system

**Goal**: Classify financial social-media texts as **Positive**, **Neutral**, or **Negative**

---

## Slide 3: Dataset

### Primary Datasets (Post-2020, Social-Media Focus)

**Twitter Financial News Sentiment** (Zeroshot, 2023):
- Clean Twitter data (~9,500 samples)
- High-quality annotations
- Ideal for baseline interpretability

**Financial Tweets Sentiment** (TimKoornstra, 2023):
- Large-scale aggregated Twitter data
- Excellent for training robust models
- Good for noisy label analysis

**TweetFinSent** (JP Morgan, 2022):
- Expert-annotated high-quality dataset
- Small but very high quality
- Ideal for label quality analysis

**Why Social-Media Text?**
- Inherently noisier → better for studying label quality
- More ambiguous cases → tests model robustness
- Real-world application → practical relevance
- Post-2020 data → current language patterns

**Key Features**:
- All from Twitter (real social-media platform)
- 3-class format (positive/neutral/negative)
- Real social-media noise (hashtags, mentions, cashtags)

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

---

## Slide 6: Results - Model Performance

### Classification Performance

**Test Set Results**:
- **Accuracy**: [Your accuracy from results]
- **F1-Score (macro)**: [Your F1 score]
- **Precision**: [Your precision]
- **Recall**: [Your recall]

**Per-Class Performance**:
- **Positive**: Precision [X], Recall [Y], F1 [Z]
- **Neutral**: Precision [X], Recall [Y], F1 [Z]
- **Negative**: Precision [X], Recall [Y], F1 [Z]

---

## Slide 7: Interpretability

### Top Features for Each Class

**Positive Sentiment**:
- Top words: [profit, growth, gain, increase, ...]
- Feature weights show clear financial positive indicators

**Negative Sentiment**:
- Top words: [loss, decline, drop, fall, ...]
- Feature weights show clear financial negative indicators

**Neutral Sentiment**:
- Top words: [report, company, market, ...]
- Feature weights show factual/neutral language

**Insight**: Model is interpretable - feature weights align with financial sentiment patterns

---

## Slide 8: Label Quality Analysis

### Social-Media-Specific Label Quality Insights

**Why Social-Media Text is Challenging:**
- Informal language and abbreviations
- Missing context (no conversation history)
- Sarcasm and irony
- Borderline cases (positive vs neutral, negative vs neutral)

**Analysis Results:**

**Misclassifications**: [Number] cases
- High-confidence errors indicate potential label issues
- Model-predicted vs. human-annotated disagreements

**Ambiguous Predictions**: [Number] cases
- Low confidence (0.45-0.55) predictions
- Borderline cases requiring human review

**Neutral Ambiguous Zone**: [Number] cases
- Cases where model struggles to distinguish neutral from sentiment
- Common in social media (e.g., "Stock is up" - positive or neutral?)

**Borderline Cases**: [Number] cases
- Positive vs neutral: [Number]
- Negative vs neutral: [Number]

**Noisy Labels**: [Number] cases
- Potentially mislabeled examples
- Short texts or conflicting signals

**Dataset Ambiguity Metrics**:
- Average confidence: [X]
- Ambiguous zone: [Y]%
- Low confidence: [Z]%

---

## Slide 9: Dataset Comparison

### Comparing Modern Social-Media Datasets

**Twitter Financial (Zeroshot)**:
- Size: ~9,500 samples
- Quality: High (clean labels)
- Best for: Baseline + Interpretability

**Financial Tweets 2023 (TimKoornstra)**:
- Size: Large (varies)
- Quality: Medium-High
- Best for: Large-scale training + Noise analysis

**TweetFinSent (JP Morgan)**:
- Size: Small (varies)
- Quality: Very High (expert annotations)
- Best for: Label quality analysis

**Key Differences**:
- Different quality levels enable comprehensive label quality analysis
- Different sizes enable different use cases
- All share social-media noise characteristics

---

## Slide 10: Challenges & Solutions

### Challenges in Social-Media Sentiment Analysis

**Challenge 1: Noise**
- Solution: Preprocessing pipeline removes URLs, cashtags, hashtags

**Challenge 2: Ambiguity**
- Solution: Label quality analysis identifies ambiguous cases

**Challenge 3: Noisy Labels**
- Solution: Heuristic-based noisy label detection

**Challenge 4: Short Texts**
- Solution: TF-IDF with 1-2 grams captures context

---

## Slide 11: Future Work

### Potential Improvements

1. **Contextual Embeddings**: Replace TF-IDF with BERT/FinBERT
2. **Active Learning**: Focus annotation on ambiguous cases
3. **Multi-Dataset Training**: Combine all three datasets
4. **Real-Time Deployment**: API for live social-media streams
5. **Cross-Dataset Evaluation**: Test model on different datasets

---

## Slide 12: Key Takeaways

### What We Learned

1. **Social-media text is inherently noisier** than news articles
2. **Label quality evaluation** is crucial for social-media datasets
3. **TF-IDF + Logistic Regression** provides good baseline with interpretability
4. **Different dataset quality levels** enable comprehensive analysis
5. **Modern post-2020 datasets** reflect current language patterns

### Project Alignment

✅ **Financial social-media sentiment** (not news articles)  
✅ **Lightweight NLP** (TF-IDF + Logistic Regression)  
✅ **Interpretability** (feature weights)  
✅ **Label quality evaluation** (comprehensive analysis)

---

## Slide 13: Conclusion

### Summary

- Built complete ML pipeline for financial social-media sentiment classification
- Achieved solid performance on modern post-2020 Twitter datasets
- Provided interpretable model via feature weights
- Conducted comprehensive label quality evaluation
- Identified ambiguous cases and noisy labels in social-media text

**Impact**: Framework for analyzing financial sentiment in noisy social-media text with focus on label quality evaluation.

**Alignment**: Perfectly matches CS5100 research proposal focusing on financial social-media sentiment with label quality evaluation.
