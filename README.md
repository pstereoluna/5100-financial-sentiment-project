# Financial Social-Media Sentiment Classification Project

A machine learning pipeline for **financial social-media sentiment classification** with label quality evaluation.

## Project Overview

This project implements a complete ML pipeline for classifying financial sentiment from **social-media text** (Twitter) using:
- **TF-IDF vectorization** (1-2 grams)
- **Logistic Regression** classifier
- **Interpretability** via feature weights
- **Label quality evaluation** with misclassification detection, ambiguous label analysis, and noisy label detection

### Research Focus

This project aligns with the CS5100 research proposal focusing on:
1. **Financial social-media sentiment** (post-2020 Twitter datasets, not news articles)
2. **Lightweight NLP** (TF-IDF + Logistic Regression baseline)
3. **Interpretability** (feature weights for model decisions)
4. **Label quality evaluation** (identifying ambiguous and noisy labels in social-media text)

**Why social-media text?** Social-media financial posts (Twitter) are inherently noisier than news articles, making them ideal for studying label quality issues, annotation inconsistencies, and model robustness.

## Features

- **Text preprocessing** optimized for social media (URL removal, cashtag removal, hashtag/mention removal)
- **Primary datasets**: Three modern post-2020 Twitter financial sentiment datasets
- **Model training and evaluation pipeline** with TF-IDF + Logistic Regression
- **Label quality analysis** specifically designed for noisy social-media text:
  - Misclassification patterns
  - Ambiguous predictions (low-confidence cases)
  - Noisy label detection (high-confidence disagreements)
  - Neutral zone analysis (borderline positive/negative vs neutral)
  - Borderline case detection
  - Dataset-inherent ambiguity quantification
- Comprehensive test suite
- Jupyter notebooks for EDA, baseline training, and label quality analysis

## Directory Structure

```
financial-sentiment-project/
├── data/                    # Dataset files (not included in repo)
├── src/                     # Source code
│   ├── preprocess.py        # Text preprocessing
│   ├── dataset_loader.py    # Dataset loaders
│   ├── model.py            # Model definition
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── label_quality.py    # Label quality analysis
│   └── download_modern_datasets.py  # Dataset download helper
├── notebooks/              # Jupyter notebooks
│   ├── 01_modern_dataset_eda.ipynb       # EDA for modern datasets
│   ├── 02_train_baseline_modern.ipynb    # Baseline training
│   └── 03_label_quality_modern.ipynb    # Label quality analysis
├── tests/                  # Unit tests
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_label_quality.py
├── results/                # Output directory (models, plots, reports)
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

### Local Installation

1. Clone or download this repository

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

**项目完全支持Google Colab！** 查看 [COLAB_SETUP.md](COLAB_SETUP.md) 获取详细指南。

## Dataset Setup

### Primary Datasets (Post-2020, Social-Media)

This project uses **three modern post-2020 Twitter financial sentiment datasets**:

1. **Twitter Financial News Sentiment** (Zeroshot, 2023)
   - Clean labeling, ideal for baseline interpretability
   - 3-class: bullish / neutral / bearish → positive / neutral / negative

2. **Financial Tweets Sentiment** (TimKoornstra, 2023)
   - Large-scale aggregated dataset
   - Excellent for large-scale training + noisy label analysis
   - 3-class: bullish / neutral / bearish → positive / neutral / negative

3. **TweetFinSent** (JP Morgan, 2022)
   - Expert annotated (positive/neutral/negative as "stock price sentiment")
   - Small but very high-quality
   - Ideal for label quality analysis

**Why these datasets?**
- All from Twitter (real social-media platform)
- Post-2020 data (modern language patterns)
- 3-class format (positive/neutral/negative)
- Real social-media noise (hashtags, mentions, cashtags, abbreviations)
- Different quality levels enable comprehensive label quality analysis

### Quick Start: Download Datasets

**Automated download (recommended):**
```bash
# Install datasets library if needed
pip install datasets requests

# Download all modern datasets
python src/download_modern_datasets.py --all

# Or download individually
python src/download_modern_datasets.py --dataset twitter_financial
python src/download_modern_datasets.py --dataset financial_tweets_2023
python src/download_modern_datasets.py --dataset tweetfinsent
```

**Manual download:**
See `data/DATASET_RECOMMENDATIONS.md` for detailed download links and instructions.

**Note**: If you already have `twitter_financial_train.csv` in the `data/` directory, you can use it directly.

## Usage

### Training

Train a model using the command-line script:

**Primary datasets** (recommended):
```bash
# Train on Twitter Financial News Sentiment
python src/train.py \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial \
    --test_size 0.2 \
    --max_features 10000

# Train on Financial Tweets 2023
python src/train.py \
    --data_path data/financial_tweets_2023.csv \
    --dataset_name financial_tweets_2023 \
    --test_size 0.2

# Train on TweetFinSent
python src/train.py \
    --data_path data/tweetfinsent.csv \
    --dataset_name tweetfinsent \
    --test_size 0.2
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/02_train_baseline_modern.ipynb
```

### Evaluation

Evaluate a trained model:

```bash
python src/evaluate.py \
    --model_path results/model.joblib \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial \
    --output_dir results
```

### Label Quality Analysis

Run label quality analysis to detect misclassifications, ambiguous predictions, and noisy labels:

```bash
python src/label_quality.py \
    --model_path results/model.joblib \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial \
    --output_dir results
```

This will generate multiple CSV reports:
- `misclassifications.csv`: Examples where predictions don't match true labels
- `ambiguous_predictions.csv`: Low-confidence predictions (confidence between 0.45-0.55)
- `noisy_labels.csv`: Potentially noisy labels based on heuristics
- `neutral_ambiguous_zone.csv`: Cases where model struggles to distinguish neutral from sentiment
- `borderline_cases.csv`: Borderline positive/negative vs neutral cases
- `dataset_ambiguity_metrics.csv`: Dataset-inherent ambiguity metrics

### Dataset Comparison

Run the dataset comparison notebook to analyze differences between the three modern datasets:
```bash
jupyter notebook notebooks/03_dataset_comparison.ipynb
```

### Running Tests

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_preprocess.py
python -m pytest tests/test_model.py
python -m pytest tests/test_label_quality.py
```

Or using unittest:

```bash
python -m unittest discover tests/
```

## Module Documentation

### preprocess.py

Text preprocessing functions:
- `clean_text(text: str) -> str`: Clean a single text string
- `preprocess_batch(texts)`: Preprocess a batch of texts

Removes URLs, cashtags ($TSLA), hashtags, @mentions, converts to lowercase, removes punctuation, and tokenizes.

### dataset_loader.py

**Primary dataset loaders**:
- `load_twitter_financial(file_path)`: Load Twitter Financial News Sentiment (Zeroshot, 2023)
- `load_financial_tweets_2023(file_path)`: Load Financial Tweets Sentiment (TimKoornstra, 2023)
- `load_tweetfinsent(file_path)`: Load TweetFinSent (JP Morgan, 2022)
- `load_dataset(dataset_name, file_path)`: Unified loader

All loaders return a pandas DataFrame with 'text' and 'label' columns, with labels unified to: **positive, neutral, negative**.

**Legacy loaders** (deprecated):
- `load_phrasebank()`, `load_semeval()`, `load_sentfin()` - marked as legacy

### model.py

Model building functions:
- `build_model(max_features, ngram_range)`: Build TF-IDF + Logistic Regression pipeline
- `get_top_features(pipeline, class_name, top_n)`: Get top features for a class
- `get_all_top_features(pipeline, top_n)`: Get top features for all classes

### train.py

Training script that:
- Loads and preprocesses data
- Splits into train/test sets
- Trains the model
- Saves model and confusion matrix
- Generates training summary

### evaluate.py

Evaluation script that:
- Loads a trained model
- Evaluates on test data
- Prints classification report and confusion matrix
- Shows top features for each class
- Saves detailed results to CSV

### label_quality.py

Label quality analysis functions:
- `detect_misclassifications()`: Find misclassified examples
- `detect_ambiguous_predictions()`: Find low-confidence predictions
- `detect_noisy_labels()`: Find potentially noisy labels using heuristics
- `analyze_neutral_ambiguous_zone()`: Analyze neutral ambiguous zone (social-media specific)
- `analyze_borderline_cases()`: Analyze borderline positive/negative vs neutral cases
- `quantify_dataset_ambiguity()`: Quantify dataset-inherent ambiguity metrics
- `run_label_quality_analysis()`: Run complete analysis with social-media-specific metrics

## Results

All outputs are saved to the `results/` directory:
- `model.joblib`: Trained model
- `confusion_matrix.png`: Confusion matrix visualization
- `training_summary.txt`: Training metrics
- `evaluation_results.csv`: Detailed predictions
- `evaluation_summary.txt`: Evaluation metrics
- `misclassifications.csv`: Misclassified examples
- `ambiguous_predictions.csv`: Ambiguous predictions
- `noisy_labels.csv`: Potentially noisy labels
- `neutral_ambiguous_zone.csv`: Neutral ambiguous zone cases
- `borderline_cases.csv`: Borderline cases
- `dataset_ambiguity_metrics.csv`: Dataset ambiguity metrics

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- joblib
- jupyter
- datasets (for downloading from HuggingFace)
- requests (for downloading from GitHub)

See `requirements.txt` for specific versions.

## Why Modern Social-Media Datasets?

**Why we use post-2020 Twitter datasets:**
- **Social-media source**: Direct Twitter data (not news headlines)
- **Recent data**: Post-2020 reflects current social-media language patterns
- **3-class format**: All support positive/neutral/negative classification
- **Noise characteristics**: Real social-media noise (hashtags, mentions, cashtags, abbreviations)
- **Label quality research**: Different quality levels enable comprehensive label quality analysis
- **Interpretability**: Clean labels support TF-IDF + Logistic Regression baseline analysis

**How they enable better label quality analysis:**
- **Ambiguity analysis**: Different quality levels reveal different ambiguity patterns
- **Noisy label detection**: Large-scale and high-quality datasets provide comprehensive coverage
- **Low-confidence cases**: Social-media noise creates natural low-confidence scenarios

## Alignment with CS5100 Proposal

This project perfectly matches the CS5100 research proposal by:
1. Focusing on **financial social-media sentiment** (not news articles)
2. Using **lightweight NLP** (TF-IDF + Logistic Regression baseline)
3. Providing **interpretability** via feature weights
4. Conducting **comprehensive label quality evaluation** in noisy social-media text

See `PROPOSAL_CONSISTENCY_CHECK.md` for detailed alignment analysis.

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues or pull requests for improvements.
