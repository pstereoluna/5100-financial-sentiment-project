# How to Run the Financial Sentiment Project

This guide explains how to run the code and where to find results.

## Quick Start

### Option 1: Run Everything in Jupyter Notebook (Recommended)

The easiest way is to use the complete pipeline notebook:

```bash
cd financial-sentiment-project
jupyter notebook notebooks/03_full_pipeline_report.ipynb
```

This notebook will:
- Load and preprocess your data
- Train the model
- Evaluate performance
- Run label quality analysis
- Create visualizations
- Save all results

**Just update the configuration at the top:**
```python
DATA_PATH = 'data/SEntFiN.csv'  # Your dataset path
DATASET_NAME = 'sentfin'  # 'phrasebank', 'semeval', or 'sentfin'
```

### Option 2: Run from Command Line

#### Step 1: Train the Model

```bash
python src/train.py \
    --data_path data/SEntFiN.csv \
    --dataset_name sentfin \
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
    --data_path data/SEntFiN.csv \
    --dataset_name sentfin \
    --output_dir results
```

**Output:**
- Evaluation results: `results/evaluation_results.csv`
- Evaluation summary: `results/evaluation_summary.txt`

#### Step 3: Label Quality Analysis

```bash
python src/label_quality.py \
    --model_path results/model.joblib \
    --data_path data/SEntFiN.csv \
    --dataset_name sentfin \
    --output_dir results
```

**Output:**
- Misclassifications: `results/misclassifications.csv`
- Ambiguous predictions: `results/ambiguous_predictions.csv`
- Noisy labels: `results/noisy_labels.csv`

## Where Are Results Saved?

All results are saved to the **`results/`** directory:

```
results/
├── model.joblib                    # Trained model
├── confusion_matrix.png            # Confusion matrix visualization
├── training_summary.txt            # Training metrics
├── evaluation_results.csv          # Detailed predictions with probabilities
├── evaluation_summary.txt          # Evaluation metrics
├── top_features.png               # Top features visualization (if using notebook)
├── label_distribution.png         # Label distribution (if using notebook)
├── text_length_distribution.png   # Text length analysis (if using notebook)
├── label_quality_analysis.png     # Label quality visualization (if using notebook)
├── misclassifications.csv         # Examples where model was wrong
├── ambiguous_predictions.csv      # Low-confidence predictions
└── noisy_labels.csv               # Potentially problematic labels
```

## Creating a Report in Jupyter Notebook

### Option A: Use the Complete Pipeline Notebook

The `03_full_pipeline_report.ipynb` notebook is a complete report that:
- Runs the entire pipeline
- Creates all visualizations
- Generates a comprehensive report
- Saves all results

**To use it:**
1. Open: `notebooks/03_full_pipeline_report.ipynb`
2. Update the configuration at the top
3. Run all cells (Cell → Run All)
4. Export as PDF/HTML if needed (File → Download as)

### Option B: Use Individual Notebooks

1. **EDA Notebook** (`01_eda.ipynb`): Explore your data
2. **Training Notebook** (`02_train_baseline.ipynb`): Train and evaluate
3. **Create your own report notebook** combining results

### Option C: Create a Custom Report Notebook

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

# Create your visualizations and analysis
# ...
```

## Example: Complete Workflow

```bash
# 1. Navigate to project directory
cd financial-sentiment-project

# 2. Train model
python src/train.py --data_path data/SEntFiN.csv --dataset_name sentfin

# 3. Evaluate
python src/evaluate.py --model_path results/model.joblib --data_path data/SEntFiN.csv --dataset_name sentfin

# 4. Label quality analysis
python src/label_quality.py --model_path results/model.joblib --data_path data/SEntFiN.csv --dataset_name sentfin

# 5. Open notebook to view results
jupyter notebook notebooks/03_full_pipeline_report.ipynb
```

## Viewing Results

### View CSV Files
```bash
# View evaluation results
head -20 results/evaluation_results.csv

# View misclassifications
head -20 results/misclassifications.csv
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

1. **For quick testing**: Use the Jupyter notebook (`03_full_pipeline_report.ipynb`)
2. **For production**: Use command-line scripts
3. **For reports**: Export the notebook as PDF/HTML
4. **For sharing**: Share the `results/` directory or export notebook

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

