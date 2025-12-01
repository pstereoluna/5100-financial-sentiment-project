# Google Colab Setup Guide

## ✅ Project can be used in Google Colab!

This guide will help you quickly run the project in Colab.

---

## Method 1: Upload Project to Colab

### Step 1: Upload Project

**Option A: From GitHub (if uploaded)**
```python
# In Colab's first cell
!git clone https://github.com/your-username/financial-sentiment-project.git
%cd financial-sentiment-project
```

**Option B: Manual Upload**
1. In Colab: Files → Upload → Select project folder
2. Or use Google Drive

### Step 2: Install Dependencies

```python
# In Colab cell
!pip install -r requirements.txt

# Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Step 3: Upload Dataset

```python
# Method 1: From local upload
from google.colab import files
uploaded = files.upload()  # Select your dataset file
# Move to data directory
!mkdir -p data
!mv twitter_financial_train.csv data/

# Method 2: From Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Copy file
!cp /content/drive/MyDrive/twitter_financial_train.csv data/
```

### Step 4: Run Notebooks

Use the notebooks directly:
- `notebooks/01_eda.ipynb`
- `notebooks/02_train_baseline.ipynb`
- `notebooks/03_label_quality.ipynb`
- `notebooks/04_final_report.ipynb`

---

## Colab-Specific Path Adjustments

### Path Setup

```python
# Colab path setup
import os
import sys

# Colab default is /content
if 'google.colab' in str(get_ipython()):
    PROJECT_ROOT = '/content/financial-sentiment-project'
    os.chdir(PROJECT_ROOT)
else:
    PROJECT_ROOT = os.getcwd()

# Add src to path
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
```

### Data File Path

```python
# Colab data path
DATA_PATH = '/content/financial-sentiment-project/data/twitter_financial_train.csv'
# Or if using Drive
DATA_PATH = '/content/drive/MyDrive/twitter_financial_train.csv'

DATASET_NAME = 'twitter_financial'  # Must be 'twitter_financial'
```

### Results Directory

```python
# Save results to Drive (persistent)
RESULTS_DIR = '/content/drive/MyDrive/results'
# Or local (temporary)
RESULTS_DIR = '/content/financial-sentiment-project/results'
```

---

## Quick Start in Colab

### Complete Workflow

```python
# Cell 1: Setup
!git clone https://github.com/your-username/financial-sentiment-project.git
%cd financial-sentiment-project
!pip install -r requirements.txt

# Cell 2: Upload dataset
from google.colab import files
uploaded = files.upload()
!mkdir -p data
!mv twitter_financial_train.csv data/

# Cell 3: Run training
!python src/train.py \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial

# Cell 4: Run evaluation
!python src/evaluate.py \
    --model_path results/model.joblib \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial

# Cell 5: Label quality analysis
!python src/label_quality.py \
    --model_path results/model.joblib \
    --data_path data/twitter_financial_train.csv \
    --dataset_name twitter_financial
```

---

## Using Notebooks in Colab

1. Upload the entire project folder to Colab
2. Open `notebooks/01_eda.ipynb` in Colab
3. Update paths in the first cell:
   ```python
   DATA_PATH = '/content/financial-sentiment-project/data/twitter_financial_train.csv'
   DATASET_NAME = 'twitter_financial'
   ```
4. Run all cells

---

## Saving Results in Colab

### Save to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp -r results /content/drive/MyDrive/
```

### Download Results

```python
from google.colab import files

# Download specific files
files.download('results/model.joblib')
files.download('results/confusion_matrix.png')
files.download('results/misclassifications.csv')
```

---

## Troubleshooting

**Import errors**: Make sure you've added `src/` to `sys.path` in the notebook

**File not found**: Check that paths are absolute (use `/content/...`)

**Out of memory**: Use smaller `max_features` or reduce dataset size

**Slow execution**: Colab free tier has limitations; consider using Colab Pro

---

## Tips

1. **Save frequently**: Colab sessions can timeout
2. **Use Drive**: Save important results to Google Drive
3. **Check GPU**: Colab provides free GPU (not needed for this project, but available)
4. **Download results**: Download CSV reports and visualizations before session ends
