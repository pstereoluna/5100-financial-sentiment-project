# How to Run the Financial Social-Media Sentiment Project

This guide explains how to run the code with the Twitter Financial News Sentiment dataset (2023).

## Quick Start

### Step 1: Download Dataset

**Option 1: From Hugging Face**
```bash
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('Zeroshot/twitter-financial-news-sentiment'); ds['train'].to_pandas().to_csv('data/twitter_financial_train.csv', index=False)"
```

**Option 2: Manual Download**
- Download from [Hugging Face](https://huggingface.co/datasets/Zeroshot/twitter-financial-news-sentiment)
- Place CSV file in `data/twitter_financial_train.csv`
- Expected format: CSV with `text` and `label` columns (labels: 0=Bearish/negative, 1=Bullish/positive, 2=Neutral)

### Step 2: Run Everything in Jupyter Notebook (Recommended)

The easiest way is to use the notebooks:

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_train_baseline.ipynb
jupyter notebook notebooks/03_label_quality.ipynb
jupyter notebook notebooks/04_final_report.ipynb
```

**Update the configuration at the top of each notebook:**
```python
DATA_PATH = 'data/twitter_financial_train.csv'  # Your dataset path
DATASET_NAME = 'twitter_financial'
```

### Step 3: Run from Command Line

#### Train the Model

```bash
python src/train.py \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial \
    --test_size 0.2 \
    --max_features 10000 \
    --model_path results/model.joblib
```

**Output:**
- Model saved to: `results/model.joblib`
- Confusion matrix: `results/confusion_matrix.png`
- Training summary: `results/training_summary.txt`

#### Evaluate the Model

```bash
python src/evaluate.py \
    --model_path results/model.joblib \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial \
    --output_dir results
```

**Output:**
- Evaluation results: `results/evaluation_results.csv`
- Evaluation summary: `results/evaluation_summary.txt`
- Top features printed to console

#### Label Quality Analysis

```bash
python src/label_quality.py \
    --model_path results/model.joblib \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial \
    --output_dir results
```

**Output:**
- `misclassifications.csv`: Cases where model disagrees with labels
- `ambiguous_predictions.csv`: Low-confidence predictions (0.45-0.55)
- `noisy_labels.csv`: Potentially mislabeled examples
- `neutral_ambiguous_zone.csv`: Cases where neutral vs sentiment is ambiguous
- `borderline_cases.csv`: Borderline positive/negative vs neutral
- `dataset_ambiguity_metrics.csv`: Overall ambiguity metrics

## Where Results Are Saved

All results are saved to the `results/` directory:

### Training Results
- `model.joblib`: Trained model (save this!)
- `confusion_matrix.png`: Confusion matrix visualization
- `training_summary.txt`: Training metrics and classification report

### Evaluation Results
- `evaluation_results.csv`: Detailed predictions with probabilities
- `evaluation_summary.txt`: Summary metrics

### Label Quality Results
- `misclassifications.csv`: Misclassified examples
- `ambiguous_predictions.csv`: Ambiguous predictions
- `noisy_labels.csv`: Potentially noisy labels
- `neutral_ambiguous_zone.csv`: Neutral ambiguous cases
- `borderline_cases.csv`: Borderline cases
- `dataset_ambiguity_metrics.csv`: Overall ambiguity metrics
- `label_quality_analysis.png`: Comprehensive visualization (from notebook)

## Notebook Workflow

### 1. Exploratory Data Analysis (`01_eda.ipynb`)

- Load and explore dataset
- Analyze label distribution
- Text length analysis
- Sample examples by label
- Social-media noise indicators
- Class imbalance analysis
- Dataset quality notes (ambiguous tweets, etc.)

### 2. Model Training (`02_train_baseline.ipynb`)

- Load and preprocess data
- Split train/test
- Build TF-IDF + Logistic Regression model
- Train model
- Evaluate on test set
- Visualize confusion matrix
- Analyze top features (interpretability)
- Report macro-F1 score

### 3. Label Quality Analysis (`03_label_quality.ipynb`)

- Load trained model
- Run comprehensive label quality analysis
- Visualize results
- Generate summary insights
- Focus on ambiguous neutral zone, noisy labels, borderline cases

### 4. Final Report (`04_final_report.ipynb`)

- Complete end-to-end report
- Dataset overview and EDA
- Model training and evaluation
- Interpretability analysis
- Comprehensive label quality evaluation
- Limitations and future work

## Command-Line Arguments

### `train.py`
- `--data_path`: Path to dataset CSV file (required)
- `--dataset_name`: Must be `twitter_financial` (default)
- `--test_size`: Test set proportion (default: 0.2)
- `--max_features`: Maximum TF-IDF features (default: 10000)
- `--model_path`: Path to save model (default: `results/model.joblib`)

### `evaluate.py`
- `--model_path`: Path to saved model (required)
- `--data_path`: Path to dataset CSV file (required)
- `--dataset_name`: Must be `twitter_financial` (default)
- `--output_dir`: Directory to save results (default: `results`)

### `label_quality.py`
- `--model_path`: Path to saved model (required)
- `--data_path`: Path to dataset CSV file (required)
- `--dataset_name`: Must be `twitter_financial` (default)
- `--output_dir`: Directory to save reports (default: `results`)

## Troubleshooting

### FileNotFoundError: Dataset file not found
- Make sure `data/twitter_financial_train.csv` exists
- Check the file path in your command/notebook

### ModuleNotFoundError
- Install dependencies: `pip install -r requirements.txt`

### Model file not found
- Train the model first using `train.py` or `02_train_baseline.ipynb`

### Empty results
- Make sure the dataset has valid labels (0, 1, 2 or positive, neutral, negative)
- Check that preprocessing didn't remove all text

## Next Steps

1. Review label quality reports to identify problematic cases
2. Analyze top features to understand model decisions
3. Use findings to improve preprocessing or model
4. See `FINAL_REPORT.md` for complete project report
