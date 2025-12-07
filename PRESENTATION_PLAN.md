# 3-Person Presentation Plan
## 8-Minute Presentation + 2-Minute Q&A

---

## Division of Labor

### Person A: Opening + Problem + Dataset
**Slides 1-3** | **Time: ~2.5 minutes**

**Slides Covered:**
- Slide 1: Title & Motivation
- Slide 2: Dataset
- Slide 3: Preprocessing & Model

**Speaking Script Outline:**

**Slide 1 (Title & Motivation) - ~45 seconds**
- "Good morning/afternoon. Today we present our work on Financial Social-Media Sentiment Analysis with a focus on label quality evaluation."
- "The problem: millions of financial posts are generated daily on social media, making manual analysis impossible. Sentiment drives market decisions, but social-media text is inherently noisy with ambiguous labels."
- "Our goal is to classify tweets as positive, neutral, or negative, with a research focus on label quality evaluation in noisy text."

**Slide 2 (Dataset) - ~40 seconds**
- "We use the Twitter Financial News Sentiment dataset from 2023, containing approximately 9,500 tweets from real Twitter financial posts."
- "The dataset has 3-class labels: positive, neutral, and negative. The text is short and informal, with typical social-media noise like hashtags, mentions, and cashtags."
- "We chose this dataset because it combines real social-media noise with clean annotations, making it ideal for label quality research."

**Slide 3 (Preprocessing & Model) - ~45 seconds**
- "Our pipeline consists of three steps: text cleaning, TF-IDF vectorization, and Logistic Regression classification."
- "Preprocessing removes URLs, cashtags, hashtags, and mentions, then normalizes the text. We use TF-IDF with 1-2 grams and a maximum of 10,000 features."
- "We chose Logistic Regression as our baseline because it's lightweight, interpretable, and aligns with our CS5100 proposal requirements."
- **Transition**: "Now I'll hand it off to [Person B] to present our methodology and results."

---

### Person B: Method + Results + Interpretation
**Slides 4-7** | **Time: ~3 minutes**

**Slides Covered:**
- Slide 4: Results Overview
- Slide 5: Confusion Matrix Insights
- Slide 6: Interpretability - Top Features
- Slide 7: Label Quality Analysis (Core Contribution)

**Speaking Script Outline:**

**Slide 4 (Results Overview) - ~50 seconds**
- "Thank you, [Person A]. Let me present our results."
- "Our model achieved an accuracy of approximately 80% and a macro F1-score of 0.74 on an independent validation set of 2,383 samples."
- "Looking at per-class performance, the neutral class is the dominant class at approximately 65% of the dataset."
- "The positive class represents about 20% of the data, and the negative class is the minority at about 15%."
- "The key finding here is that minority classes, especially the negative class, are more challenging to classify due to class imbalance."

**Slide 5 (Confusion Matrix Insights) - ~40 seconds**
- "The confusion matrix reveals several important patterns."
- "The neutral class shows high accuracy, which is expected given its dominance in the dataset at approximately 65%."
- "The minority classes (positive and negative) are frequently misclassified as neutral, indicating the challenge of identifying sentiment in social-media text."
- "There's also some confusion between positive and negative classes."
- "The class imbalance, with a 4.3-to-1 ratio of neutral to negative, significantly affects performance on minority classes."

**Slide 6 (Interpretability - Top Features) - ~40 seconds**
- "One advantage of our baseline model is interpretability through feature weights."
- "For positive sentiment, top features include words like 'beats', 'bullish', 'rises', 'higher', 'strong', and 'gain'."
- "For negative sentiment, we see words like 'declares', 'fed', 'loss', 'misses', 'downgraded', and 'falls'."
- "Interestingly, neutral sentiment shares some features with negative, such as 'lower', 'downgraded', 'misses', and 'cuts'."
- "These feature weights align with expected financial sentiment patterns, demonstrating the model's interpretability."

**Slide 7 (Label Quality Analysis) - ~50 seconds**
- "Now, let me present our main research contribution: comprehensive label quality evaluation."
- "We performed label quality analysis on our independent validation set to identify misclassifications, ambiguous predictions, and potentially noisy labels."
- "The analysis reveals patterns in dataset ambiguity, including cases in the neutral ambiguous zone and borderline cases between classes."
- "A significant portion of predictions have low confidence, indicating dataset-inherent ambiguity in social-media text."
- "This label quality framework provides insights into dataset reliability and annotation inconsistencies, which is more valuable than raw accuracy metrics alone."
- **Transition**: "Now I'll hand it off to [Person C] to discuss limitations and conclusions."

---

### Person C: Label Quality + Limitations + Conclusion
**Slides 8-9** | **Time: ~2.5 minutes**

**Slides Covered:**
- Slide 8: Limitations & Future Work
- Slide 9: Conclusion

**Speaking Script Outline:**

**Slide 8 (Limitations & Future Work) - ~70 seconds**
- "Thank you, [Person B]. Let me discuss the limitations of our work and potential future directions."
- "Our model has several limitations. As a bag-of-words approach, TF-IDF ignores word order and cannot capture sarcasm or long-range dependencies. This is by design, as we chose a lightweight baseline for interpretability."
- "The data itself presents challenges: significant class imbalance at a 4.3-to-1 ratio, inherent social-media ambiguity, and missing context that could help disambiguate sentiment."
- "Our label quality analysis also has limitations: the heuristics are simple, and the metrics are model-dependent."
- "For future work, we could replace TF-IDF with contextual embeddings like BERT or FinBERT for stronger performance, especially for capturing sarcasm."
- "We could also implement active learning to focus annotation efforts on the ambiguous cases we've identified through label quality analysis."
- "Additionally, we could incorporate additional features like cashtags, hashtags, and user metadata to improve performance."

**Slide 9 (Conclusion) - ~60 seconds**
- "In conclusion, we built a complete ML pipeline for financial social-media sentiment classification."
- "We achieved solid baseline performance with 80% accuracy and 0.74 macro F1-score on an independent validation set, while providing an interpretable model through feature weights."
- "Our main research contribution is the label quality evaluation framework for noisy social-media text."
- "The key takeaway is that label quality evaluation provides more insights than raw accuracy metrics alone, especially for noisy social-media text like Twitter posts."
- "This work aligns perfectly with our CS5100 proposal, focusing on financial social-media sentiment with comprehensive label quality evaluation."
- "Thank you for your attention. We're now open to questions."

---

## Time Allocation Summary

| Person | Slides | Time | Content Focus |
|--------|--------|------|---------------|
| **Person A** | 1-3 | ~2.5 min | Opening, problem, dataset, pipeline overview |
| **Person B** | 4-7 | ~3.0 min | Results, confusion matrix, interpretability, label quality |
| **Person C** | 8-9 | ~2.5 min | Limitations, future work, conclusion |
| **Total** | 9 slides | ~8 min | + 2 min Q&A = 10 minutes total |

---

## Transition Phrases

- **Person A → Person B**: "Now I'll hand it off to [Person B] to present our methodology and results."
- **Person B → Person C**: "Now I'll hand it off to [Person C] to discuss limitations and conclusions."

---

## Rehearsal Checklist

- [ ] Each person practices their section independently
- [ ] Practice transitions between speakers
- [ ] Time each section (Person A: 2.5 min, Person B: 3 min, Person C: 2.5 min)
- [ ] Ensure visual aids (confusion matrix, top features) are ready
- [ ] Prepare answers for common Q&A questions:
  - Why not use deep learning?
  - How does this compare to other sentiment analysis methods?
  - What makes label quality evaluation important?
  - How would you improve the model?

---

## Notes

- **Balance**: Workload is evenly distributed (2.5-3 minutes per person)
- **No Overlap**: Each person covers distinct content
- **Smooth Transitions**: Clear handoff phrases between speakers
- **Focus**: Person B emphasizes the main research contribution (label quality)
- **Conclusion**: Person C wraps up with limitations and future work

