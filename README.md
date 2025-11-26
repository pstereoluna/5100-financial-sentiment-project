# Financial Sentiment Project

A machine learning pipeline for financial social media sentiment classification with label quality evaluation.

## Project Overview

This project implements a complete ML pipeline for classifying financial sentiment from social media text using:
- **TF-IDF vectorization** (1-2 grams)
- **Logistic Regression** classifier
- **Interpretability** via feature weights
- **Label quality evaluation** with misclassification detection and ambiguous label analysis

## Features

- Text preprocessing (URL removal, cashtag removal, hashtag/mention removal)
- Support for multiple datasets (Financial PhraseBank, SemEval-2017 Task 5, SEntFiN 1.0)
- Model training and evaluation pipeline
- Label quality analysis (misclassifications, ambiguous predictions, noisy labels)
- Comprehensive test suite
- Jupyter notebooks for EDA and baseline training

## Directory Structure

```
financial-sentiment-project/
├── data/                    # Dataset files (not included)
├── src/                     # Source code
│   ├── preprocess.py        # Text preprocessing
│   ├── dataset_loader.py    # Dataset loaders
│   ├── model.py            # Model definition
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── label_quality.py    # Label quality analysis
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   └── 02_train_baseline.ipynb  # Baseline training
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

**快速开始**:
1. 上传项目到Colab或从GitHub克隆
2. 运行 `notebooks/COLAB_快速开始.ipynb`
3. 按照notebook中的步骤操作

**优势**:
- 无需本地安装
- 免费GPU支持（如果需要）
- 易于分享和协作

## Dataset Setup

### Quick Start: Download from Hugging Face

The easiest way to get started is using the helper script:

```bash
# Install datasets library if not already installed
pip install datasets

# Download Financial PhraseBank
python src/download_datasets.py --dataset phrasebank

# Or download Financial-Sentiment-Analysis dataset
python src/download_datasets.py --dataset financial_sentiment
```

### Manual Dataset Setup

#### Financial PhraseBank

1. Download the Financial PhraseBank dataset from:
   - [Hugging Face](https://huggingface.co/datasets/financial_phrasebank)
   - [ResearchGate](https://www.researchgate.net/publication/251231364_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts)
   - Kaggle (search for "Financial PhraseBank")
2. Place the CSV file in the `data/` directory
3. Ensure the CSV has columns for text and sentiment labels

#### SemEval-2017 Task 5

1. Download the SemEval-2017 Task 5 dataset (microblogs) from:
   - [Official SemEval-2017 Task 5](https://alt.qcri.org/semeval2017/task5/)
2. Place the file in the `data/` directory
3. The loader supports both CSV and TSV formats

#### SEntFiN 1.0

1. Download the SEntFiN 1.0 dataset from:
   - [arXiv:2305.12257](https://arxiv.org/abs/2305.12257) (check paper for download link)
   - The dataset should have columns: `Title`, `Decisions`
2. Place the CSV file in the `data/` directory
3. The loader automatically handles entity-sentiment dictionaries and aggregates multiple sentiments per headline

**Note:** If you already have `SEntFiN.csv` in the `data/` directory, it's ready to use! The loader will automatically extract sentiments from the `Decisions` column.

#### Formatting Custom Datasets

If you have your own dataset, format it using:

```bash
python src/download_datasets.py --dataset custom \
    --input your_dataset.csv \
    --text_col sentence \
    --label_col sentiment \
    --output data/formatted_dataset.csv
```

**Note:** See `data/DATASET_RECOMMENDATIONS.md` for detailed dataset recommendations and sources.

## Usage

### Training

Train a model using the command-line script:

```bash
python src/train.py \
    --data_path data/financial_phrasebank.csv \
    --dataset_name phrasebank \
    --test_size 0.2 \
    --max_features 10000 \
    --model_path results/model.joblib
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/02_train_baseline.ipynb
```

### Evaluation

Evaluate a trained model:

```bash
python src/evaluate.py \
    --model_path results/model.joblib \
    --data_path data/financial_phrasebank.csv \
    --dataset_name phrasebank \
    --output_dir results
```

### Label Quality Analysis

Run label quality analysis to detect misclassifications, ambiguous predictions, and noisy labels:

```bash
python src/label_quality.py \
    --model_path results/model.joblib \
    --data_path data/financial_phrasebank.csv \
    --dataset_name phrasebank \
    --output_dir results
```

This will generate three CSV reports:
- `misclassifications.csv`: Examples where predictions don't match true labels
- `ambiguous_predictions.csv`: Low-confidence predictions (confidence between 0.45-0.55)
- `noisy_labels.csv`: Potentially noisy labels based on heuristics

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

Dataset loading functions:
- `load_phrasebank(file_path)`: Load Financial PhraseBank dataset
- `load_semeval(file_path)`: Load SemEval-2017 Task 5 dataset
- `load_sentfin(file_path, aggregation_method)`: Load SEntFiN 1.0 dataset
- `load_dataset(dataset_name, file_path)`: Unified loader

All loaders return a pandas DataFrame with 'text' and 'label' columns, with labels unified to: positive, neutral, negative.

**SEntFiN Notes:** The SEntFiN dataset contains entity-sentiment pairs. When a headline has multiple entities with different sentiments, the loader aggregates them using the most common sentiment by default. You can change this with the `aggregation_method` parameter ('most_common' or 'first').

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
- `run_label_quality_analysis()`: Run complete analysis

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

See `requirements.txt` for specific versions.

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues or pull requests for improvements.

