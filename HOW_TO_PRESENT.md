# How to Create a Presentation

This guide shows you multiple ways to create a presentation from your project results.

## Option 1: Export Jupyter Notebook as Slides (Recommended)

### Step 1: Configure Notebook for Slides

1. Open your notebook: `notebooks/02_train_baseline.ipynb` or `notebooks/03_label_quality.ipynb` or `notebooks/04_final_report.ipynb`
2. In Jupyter, go to **View → Cell Toolbar → Slideshow**
3. Set each cell's slide type:
   - **Slide**: Main content slides
   - **Sub-slide**: Supporting points
   - **Fragment**: Appear on click
   - **Skip**: Hidden from presentation
   - **Notes**: Speaker notes

### Step 2: Recommended Slide Structure

```
Cell 1: Title Slide (Slide)
Cell 2: Problem Statement (Slide)
Cell 3: Dataset Overview (Slide)
Cell 4: Methodology (Slide)
Cell 5: Results - Performance (Slide)
Cell 6: Results - Visualizations (Slide)
Cell 7: Label Quality Analysis (Slide)
Cell 8: Key Findings (Slide)
Cell 9: Limitations (Slide)
Cell 10: Future Work (Slide)
Cell 11: Conclusion (Slide)
```

### Step 3: Export to Slides

**Method A: Using Jupyter**
```bash
# Install reveal.js exporter
pip install jupyter-reveal

# Export notebook to reveal.js slides
jupyter nbconvert notebooks/04_final_report.ipynb --to slides
```

**Method B: Using nbconvert**
```bash
# Export to HTML slides
jupyter nbconvert notebooks/04_final_report.ipynb \
    --to slides \
    --reveal-prefix reveal.js \
    --output presentation.html
```

**Method C: Export to PDF**
```bash
# Export to PDF (via HTML)
jupyter nbconvert notebooks/04_final_report.ipynb \
    --to pdf \
    --output presentation.pdf
```

---

## Option 2: Use the Presentation Markdown

1. **Open**: `PRESENTATION.md`
2. **Convert to slides** using:
   - **Marp** (VS Code extension)
   - **reveal.js**
   - **Pandoc** to PowerPoint
   - **Google Slides** (copy content)

### Using Marp (VS Code)

1. Install Marp extension in VS Code
2. Open `PRESENTATION.md`
3. Click "Marp: Export Slide Deck"
4. Choose format (PDF, HTML, PPTX)

### Using Pandoc

```bash
# Install pandoc
brew install pandoc  # Mac
# or download from https://pandoc.org

# Convert to PowerPoint
pandoc PRESENTATION.md -o presentation.pptx

# Convert to PDF (requires LaTeX)
pandoc PRESENTATION.md -o presentation.pdf
```

---

## Option 3: Create PowerPoint from Results

### Manual Creation

Use the results from `results/` directory:
- Confusion matrix: `results/confusion_matrix.png`
- Top features: `results/top_features.png`
- Label quality analysis: `results/label_quality_analysis.png`
- CSV reports: `results/*.csv`

Create slides manually in PowerPoint/Google Slides using these visualizations.

---

## Presentation Structure

### Recommended Slides (10-15 minutes)

1. **Title Slide** (30 sec)
   - Project title
   - Your name
   - CS5100 Final Project

2. **Problem Statement** (1 min)
   - Why financial sentiment analysis?
   - Why social-media text?
   - Research goals

3. **Dataset** (1 min)
   - Twitter Financial News Sentiment (Zeroshot, 2023)
   - Dataset characteristics
   - Label distribution (show class imbalance)

4. **Methodology** (2 min)
   - Preprocessing pipeline
   - TF-IDF + Logistic Regression
   - Why baseline model (interpretability)

5. **Results - Performance** (2 min)
   - Accuracy, macro F1-score
   - Confusion matrix
   - Per-class performance
   - Highlight neutral class difficulty

6. **Results - Interpretability** (1 min)
   - Top features for each class
   - Feature weight visualization

7. **Label Quality Analysis** (3 min)
   - Misclassifications
   - Ambiguous predictions
   - Noisy labels
   - Neutral ambiguous zone
   - Borderline cases
   - Dataset ambiguity metrics

8. **Key Findings** (2 min)
   - Social-media text is noisier
   - Neutral zone is challenging
   - Class imbalance affects performance
   - Label quality matters more than accuracy

9. **Limitations** (1 min)
   - Baseline model (cannot capture sarcasm/long-range dependencies)
   - Class imbalance
   - Social-media ambiguity

10. **Future Work** (1 min)
    - FinBERT for stronger performance
    - Active learning for ambiguous cases
    - Additional features

11. **Conclusion** (30 sec)
    - Summary
    - Alignment with proposal
    - Main contribution: label quality evaluation

---

## Tips for Effective Presentation

1. **Focus on label quality**: This is your main research contribution
2. **Show visual examples**: Use confusion matrix, top features, label quality plots
3. **Explain limitations**: Be honest about baseline model limitations
4. **Emphasize social-media challenges**: Why label quality matters more for noisy text
5. **Keep it concise**: 10-15 minutes, focus on key findings

---

## Key Points to Emphasize

- **Why social-media text**: Inherently noisier, better for label quality research
- **Why baseline model**: Interpretability + lightweight NLP requirement
- **Label quality contribution**: More valuable than raw accuracy metrics
- **Neutral class difficulty**: Key finding from label quality analysis
- **Class imbalance**: Affects model performance on minority classes

---

## Visual Aids

Make sure to include:
- Confusion matrix (shows neutral class difficulty)
- Top features visualization (interpretability)
- Label quality summary plots
- Example error cases (high-confidence misclassifications)
- Ambiguous case examples

---

## Practice

1. Time yourself (aim for 10-15 minutes)
2. Practice explaining label quality findings
3. Be ready to discuss limitations and future work
4. Prepare answers for questions about:
   - Why not use deep learning?
   - How to improve performance?
   - What makes this research contribution valuable?
