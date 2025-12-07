# Financial Social-Media Sentiment Classification with Label Quality Evaluation

A machine learning pipeline for **financial social-media sentiment classification** with comprehensive label quality evaluation, aligned with CS5100 research proposal.

## Project Overview

This project implements a complete ML pipeline for classifying financial sentiment from **Twitter social-media text** using:
- **TF-IDF vectorization** (1-2 grams)
- **Logistic Regression** classifier
- **Interpretability** via feature weights
- **Label quality evaluation** with misclassification detection, ambiguous label analysis, and noisy label detection

### Research Focus

This project aligns with the CS5100 research proposal focusing on:
1. **Financial social-media sentiment** (Twitter dataset, not news articles)
2. **Lightweight NLP** (TF-IDF + Logistic Regression baseline)
3. **Interpretability** (feature weights for model decisions)
4. **Label quality evaluation** (identifying ambiguous and noisy labels in social-media text)

**Why social-media text?** Social-media financial posts (Twitter) are inherently noisier than news articles, making them ideal for studying label quality issues, annotation inconsistencies, and model robustness.

## Dataset

**Twitter Financial News Sentiment** (Zeroshot, 2023):
- **Source**: Real Twitter financial posts
- **Samples**: ~9,500 tweets
- **Labels**: 3-class (0=Bearish/negative, 1=Bullish/positive, 2=Neutral) → unified to positive/neutral/negative
- **Format**: CSV with `text` and `label` columns
- **Characteristics**: 
  - Short, informal text (typical of social media)
  - Real-world noise (hashtags, mentions, cashtags)
  - Higher ambiguity than news articles
  - Ideal for label quality evaluation research

### Dataset Setup

1. Download the dataset from [Hugging Face](https://huggingface.co/datasets/Zeroshot/twitter-financial-news-sentiment) or place your CSV file in `data/twitter_financial_train.csv`

**Note**: The `data/` directory may contain other files (e.g., `SEntFiN.csv`). This project uses only the Twitter Financial News Sentiment dataset. Other files in `data/` are ignored by git (see `.gitignore`).

2. Expected CSV format:
   ```csv
   text,label
   "Stock prices are rising!",1
   "Market is stable",0
   "Prices falling",2
   ```

## Installation

### Local Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd financial-sentiment-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (if using NLTK tokenization):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Google Colab

See [COLAB_SETUP.md](COLAB_SETUP.md) for detailed Google Colab setup instructions.

## Quick Start

### 1. Train Model

```bash
python src/train.py \
    --data_path data/twitter_financial_train.csv \
    --valid_path data/twitter_financial_valid.csv \
    --dataset_name twitter_financial \
    --max_features 10000
```

This will:
- Load and preprocess the training dataset (uses ALL training data, no split)
- Train TF-IDF + Logistic Regression model
- Evaluate on the independent validation set
- Save model to `results/model.joblib`
- Generate confusion matrix plot

**Note**: The validation set is completely independent from training data, ensuring unbiased evaluation.

### 2. Evaluate Model

```bash
python src/evaluate.py \
    --model_path results/model.joblib \
    --data_path data/twitter_financial_valid.csv \
    --dataset_name twitter_financial
```

This will:
- Load trained model
- Evaluate on the specified dataset (typically validation set)
- Print classification report and confusion matrix
- Show top features by class
- Save detailed results to `results/evaluation_results.csv`

### 3. Label Quality Analysis

```bash
python src/label_quality.py \
    --model_path results/model.joblib \
    --data_path data/twitter_financial_valid.csv \
    --dataset_name twitter_financial \
    --output_dir results
```

**Note**: Can be run on either training or validation data to analyze label quality.

This will generate:
- `misclassifications.csv`: Cases where model disagrees with labels
- `ambiguous_predictions.csv`: Low-confidence predictions (0.45-0.55)
- `noisy_labels.csv`: Potentially mislabeled examples
- `neutral_ambiguous_zone.csv`: Cases where neutral vs sentiment is ambiguous
- `borderline_cases.csv`: Borderline positive/negative vs neutral
- `dataset_ambiguity_metrics.csv`: Overall ambiguity metrics

## Directory Structure

```
financial-sentiment-project/
├── data/                    # Dataset files (not included in repo)
│   └── twitter_financial_train.csv
├── src/                     # Source code
│   ├── preprocess.py        # Text preprocessing (URLs, hashtags, mentions)
│   ├── dataset_loader.py    # Twitter dataset loader
│   ├── model.py            # TF-IDF + Logistic Regression model
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── label_quality.py    # Label quality analysis
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_train_baseline.ipynb    # Model training
│   ├── 03_label_quality.ipynb    # Label quality analysis
│   └── 04_final_report.ipynb      # Complete project report
├── tests/                  # Unit tests
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_label_quality.py
├── results/                # Output directory (models, plots, reports)
├── README.md
├── HOW_TO_RUN.md           # Detailed usage guide
├── HOW_TO_PRESENT.md       # Presentation guide
├── PRESENTATION.md         # Presentation content
├── FINAL_REPORT.md         # Complete project report
├── COLAB_SETUP.md          # Google Colab setup
├── requirements.txt
└── .gitignore
```

## Methodology

### Preprocessing

Text preprocessing optimized for social media:
- Remove URLs, cashtags, hashtags, @mentions
- Lowercase conversion
- Remove punctuation
- Tokenization

### Model

- **Feature Extraction**: TF-IDF with 1-2 gram features (max 10,000 features)
- **Classifier**: Multinomial Logistic Regression
- **Interpretability**: Feature weights show which words contribute to each sentiment class

**Note**: This is a **baseline model** using lightweight NLP. For stronger performance, consider contextual embeddings (e.g., FinBERT) as a future upper bound.

### Label Quality Evaluation

Comprehensive analysis designed for noisy social-media text:

1. **Misclassifications**: Model predictions that disagree with labels
2. **Ambiguous Predictions**: Low-confidence predictions (0.45-0.55 probability)
3. **Noisy Labels**: High-confidence disagreements or suspicious patterns
4. **Neutral Ambiguous Zone**: Cases where model struggles with neutral vs sentiment
5. **Borderline Cases**: Positive/negative vs neutral boundary cases
6. **Dataset Ambiguity Metrics**: Overall ambiguity quantification

## Usage Examples

### Using Jupyter Notebooks

1. **EDA**: `notebooks/01_eda.ipynb` - Explore dataset characteristics
2. **Training**: `notebooks/02_train_baseline.ipynb` - Train and evaluate model
3. **Label Quality**: `notebooks/03_label_quality.ipynb` - Comprehensive label quality analysis
4. **Final Report**: `notebooks/04_final_report.ipynb` - Complete project report

### Using Command-Line Scripts

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for detailed usage examples.

## Results

After training and evaluation, results are saved to `results/`:
- `model.joblib`: Trained model
- `confusion_matrix.png`: Confusion matrix visualization
- `evaluation_results.csv`: Detailed predictions
- Label quality CSV reports (see above)

## Key Features

- **Social-media optimized**: Preprocessing handles Twitter-specific noise
- **Interpretable**: Feature weights show model decisions
- **Label quality focus**: Comprehensive analysis of dataset quality
- **Academic alignment**: Matches CS5100 research proposal requirements
- **Clean codebase**: Modular, well-documented, tested

## Limitations

- **Bag-of-words representation**: TF-IDF ignores word order and long-range context
- **Social-media ambiguity**: Social-media text is inherently ambiguous (sarcasm, missing context, abbreviations), making some cases difficult even for humans
- **Short text challenge**: The model may struggle with very short or highly informal social-media posts
- **Baseline model**: TF-IDF + Logistic Regression is a lightweight baseline; **cannot capture sarcasm or long-range dependencies**
- **No deep learning**: By design, we use lightweight NLP for interpretability, but this limits model capacity

**Future improvements**: Consider FinBERT or other contextual embeddings for stronger performance, especially for capturing sarcasm and long-range dependencies.

## Future Work

- Replace TF-IDF with contextual embeddings (e.g., BERT or FinBERT)
- Incorporate additional features (cashtags, hashtags, user metadata)
- Active learning for ambiguous/noisy cases
- Real-time API deployment

## Citation

If you use this project, please cite:

```
Financial Social-Media Sentiment Classification with Label Quality Evaluation
CS5100 Final Project
```

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues or pull requests for improvements.
