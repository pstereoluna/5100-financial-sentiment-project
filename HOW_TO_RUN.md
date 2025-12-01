# How to Run the Financial Social-Media Sentiment Project

This guide explains how to run the code with modern post-2020 social-media datasets and where to find results.

## Quick Start

### Step 1: Download Datasets

**Automated download (recommended):**
```bash
cd financial-sentiment-project
pip install datasets requests
python src/download_modern_datasets.py --all
```

**Manual download:**
- See `data/DATASET_RECOMMENDATIONS.md` for download links
- Place files in `data/` directory:
  - `twitter_financial_train.csv`
  - `financial_tweets_2023.csv`
  - `tweetfinsent.csv`

### Step 2: Run Everything in Jupyter Notebook (Recommended)

The easiest way is to use the modern notebooks:

```bash
jupyter notebook notebooks/01_modern_dataset_eda.ipynb
jupyter notebook notebooks/02_train_baseline_modern.ipynb
jupyter notebook notebooks/03_label_quality_modern.ipynb
```

**Update the configuration at the top:**
```python
DATA_PATH = 'data/twitter_financial_train.csv'  # Your dataset path
DATASET_NAME = 'twitter_financial'  # 'twitter_financial', 'financial_tweets_2023', or 'tweetfinsent'
```

### Option 2: Run from Command Line

#### Step 1: Train the Model

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

#### Step 2: Evaluate the Model

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

#### Step 3: Label Quality Analysis

```bash
python src/label_quality.py \
    --model_path results/model.joblib \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial \
    --output_dir results
```

**Output:**
- Misclassifications: `results/misclassifications.csv`
- Ambiguous predictions: `results/ambiguous_predictions.csv`
- Noisy labels: `results/noisy_labels.csv`
- Neutral ambiguous zone: `results/neutral_ambiguous_zone.csv`
- Borderline cases: `results/borderline_cases.csv`
- Dataset ambiguity metrics: `results/dataset_ambiguity_metrics.csv`

## Where Are Results Saved?

All results are saved to the **`results/`** directory:

```
results/
├── model.joblib                    # Trained model
├── confusion_matrix.png            # Confusion matrix visualization
├── training_summary.txt            # Training metrics
├── evaluation_results.csv          # Detailed predictions with probabilities
├── evaluation_summary.txt          # Evaluation metrics
├── top_features.png               # Top features visualization
├── label_distribution.png         # Label distribution
├── text_length_distribution.png   # Text length analysis
├── label_quality_analysis.png     # Label quality visualization
├── misclassifications.csv         # Examples where model was wrong
├── ambiguous_predictions.csv      # Low-confidence predictions
├── noisy_labels.csv               # Potentially problematic labels
├── neutral_ambiguous_zone.csv     # Neutral ambiguous zone cases
├── borderline_cases.csv           # Borderline positive/negative vs neutral
└── dataset_ambiguity_metrics.csv # Dataset-inherent ambiguity metrics
```

## Creating a Report in Jupyter Notebook

### Option A: Use the Modern Notebooks

1. **EDA Notebook** (`01_modern_dataset_eda.ipynb`): Explore modern social-media datasets
2. **Training Notebook** (`02_train_baseline_modern.ipynb`): Train and evaluate on modern datasets
3. **Label Quality Notebook** (`03_label_quality_modern.ipynb`): Comprehensive label quality analysis

**To use them:**
1. Open the notebook
2. Update the configuration at the top
3. Run all cells (Cell → Run All)
4. Export as PDF/HTML if needed (File → Download as)

### Option B: Create a Custom Report Notebook

You can create your own report by loading saved results:

```python
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load saved model
model = joblib.load('results/model.joblib')

# Load evaluation results
eval_results = pd.read_csv('results/evaluation_results.csv')

# Load label quality reports
misclass = pd.read_csv('results/misclassifications.csv')
ambiguous = pd.read_csv('results/ambiguous_predictions.csv')
noisy = pd.read_csv('results/noisy_labels.csv')
neutral_ambiguous = pd.read_csv('results/neutral_ambiguous_zone.csv')
borderline = pd.read_csv('results/borderline_cases.csv')
ambiguity_metrics = pd.read_csv('results/dataset_ambiguity_metrics.csv')

# Create your visualizations and analysis
# ...
```

## Example: Complete Workflow

```bash
# 1. Navigate to project directory
cd financial-sentiment-project

# 2. Download datasets (if not already done)
python src/download_modern_datasets.py --dataset twitter_financial

# 3. Train model
python src/train.py \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial

# 4. Evaluate
python src/evaluate.py \
    --model_path results/model.joblib \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial

# 5. Label quality analysis
python src/label_quality.py \
    --model_path results/model.joblib \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial

# 6. Open notebook to view results
jupyter notebook notebooks/03_label_quality_modern.ipynb
```

## Working with Different Datasets

### Twitter Financial News Sentiment
```bash
python src/train.py \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial
```

### Financial Tweets 2023
```bash
python src/train.py \
    --data_path data/financial_tweets_2023.csv \
    --dataset_name financial_tweets_2023
```

### TweetFinSent
```bash
python src/train.py \
    --data_path data/tweetfinsent.csv \
    --dataset_name tweetfinsent
```

## Viewing Results

### View CSV Files
```bash
# View evaluation results
head -20 results/evaluation_results.csv

# View misclassifications
head -20 results/misclassifications.csv

# View label quality metrics
cat results/dataset_ambiguity_metrics.csv
```

### View Images
```bash
# On Mac
open results/confusion_matrix.png

# On Linux
xdg-open results/confusion_matrix.png

# Or just open in any image viewer
```

### View Text Summaries
```bash
cat results/training_summary.txt
cat results/evaluation_summary.txt
```

## Tips

1. **For quick testing**: Use the Jupyter notebooks
2. **For production**: Use command-line scripts
3. **For reports**: Export the notebook as PDF/HTML
4. **For sharing**: Share the `results/` directory or export notebook
5. **For dataset comparison**: Use `notebooks/03_dataset_comparison.ipynb`

## Troubleshooting

**If results directory doesn't exist:**
```bash
mkdir -p results
```

**If you get import errors:**
```bash
pip install -r requirements.txt
```

**If model file not found:**
Make sure you've run the training script first!

**If dataset file not found:**
- Check that the file exists in `data/` directory
- Verify the file name matches (e.g., `twitter_financial_train.csv`)
- Run the download script: `python src/download_modern_datasets.py --dataset twitter_financial`
