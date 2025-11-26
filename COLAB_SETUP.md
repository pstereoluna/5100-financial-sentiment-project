# Google Colab 使用指南

## ✅ 项目可以在Google Colab中使用！

只需要做一些小的调整。本指南将帮助你快速在Colab中运行项目。

---

## 方法1: 使用Colab适配的Notebook（推荐）

### 步骤1: 上传项目到Colab

**选项A: 从GitHub克隆（如果已上传）**
```python
# 在Colab的第一个cell中运行
!git clone https://github.com/your-username/financial-sentiment-project.git
%cd financial-sentiment-project
```

**选项B: 手动上传**
1. 在Colab中：文件 → 上传 → 选择项目文件夹
2. 或使用Google Drive

### 步骤2: 安装依赖

```python
# 在Colab cell中运行
!pip install -r requirements.txt

# 下载NLTK数据
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 步骤3: 上传数据集

```python
# 方法1: 从本地上传
from google.colab import files
uploaded = files.upload()  # 选择你的数据集文件

# 方法2: 从Google Drive挂载
from google.colab import drive
drive.mount('/content/drive')
# 然后复制文件
!cp /content/drive/MyDrive/your_dataset.csv data/
```

### 步骤4: 运行Notebook

直接使用 `notebooks/03_full_pipeline_report.ipynb`，但需要修改路径设置。

---

## 方法2: 创建Colab专用Notebook

我已经为你创建了一个Colab适配版本（见下方）。

---

## Colab适配的关键修改

### 1. 路径处理

Colab的工作目录是 `/content/`，需要调整：

```python
# Colab中的路径设置
import os
import sys

# Colab默认在/content目录
if 'google.colab' in str(get_ipython()):
    # 在Colab中
    PROJECT_ROOT = '/content/financial-sentiment-project'
    os.chdir(PROJECT_ROOT)
else:
    # 本地环境
    PROJECT_ROOT = os.getcwd()

# 添加src到路径
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
```

### 2. 数据文件路径

```python
# Colab中，数据可能在/content/或/content/drive/MyDrive/
DATA_PATH = '/content/financial-sentiment-project/data/SEntFiN.csv'
# 或
DATA_PATH = '/content/drive/MyDrive/SEntFiN.csv'
```

### 3. 结果保存

```python
# Colab中，结果可以保存到Google Drive
RESULTS_DIR = '/content/drive/MyDrive/results'  # 保存到Drive
# 或
RESULTS_DIR = '/content/financial-sentiment-project/results'  # 保存到Colab（临时）
```

---

## 快速开始（Colab）

### 完整设置代码（复制到Colab第一个cell）

```python
# ============================================
# Google Colab 设置 - 复制到第一个cell运行
# ============================================

import os
import sys

# 检测是否在Colab中
try:
    import google.colab
    IN_COLAB = True
    print("✓ 检测到Google Colab环境")
except:
    IN_COLAB = False
    print("✓ 本地环境")

# 设置项目路径
if IN_COLAB:
    # 克隆项目（如果还没克隆）
    if not os.path.exists('/content/financial-sentiment-project'):
        print("正在克隆项目...")
        !git clone https://github.com/your-username/financial-sentiment-project.git
        # 或者手动上传后，设置路径
        # PROJECT_ROOT = '/content/你的项目文件夹名'
    
    PROJECT_ROOT = '/content/financial-sentiment-project'
    os.chdir(PROJECT_ROOT)
    print(f"✓ 项目根目录: {PROJECT_ROOT}")
else:
    # 本地环境
    PROJECT_ROOT = os.getcwd()
    if os.path.basename(PROJECT_ROOT) == 'notebooks':
        PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
        os.chdir(PROJECT_ROOT)

# 安装依赖
print("\n正在安装依赖...")
!pip install -q pandas numpy scikit-learn nltk matplotlib seaborn joblib jupyter datasets

# 下载NLTK数据
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("✓ NLTK数据已下载")
except:
    print("⚠ NLTK数据下载失败，但可以继续")

# 添加src到路径
src_path = os.path.join(PROJECT_ROOT, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"✓ 已添加src到路径: {src_path}")

# 导入项目模块
try:
    from dataset_loader import load_dataset
    from preprocess import preprocess_batch
    from model import build_model, get_all_top_features
    print("✓ 项目模块导入成功")
except Exception as e:
    print(f"⚠ 导入错误: {e}")
    print("请确保项目文件已正确上传")

print("\n" + "="*60)
print("设置完成！现在可以使用项目了")
print("="*60)
```

---

## 在Colab中使用项目的步骤

### 步骤1: 创建新的Colab Notebook

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 创建新notebook

### 步骤2: 运行设置代码

将上面的"完整设置代码"复制到第一个cell并运行。

### 步骤3: 上传数据集

```python
# 方法1: 从本地上传
from google.colab import files
uploaded = files.upload()

# 查看上传的文件
for filename in uploaded.keys():
    print(f"已上传: {filename}")
    # 移动到data目录
    !mkdir -p data
    !mv {filename} data/
```

### 步骤4: 运行训练

```python
# 配置
DATA_PATH = 'data/SEntFiN.csv'  # 或你上传的文件名
DATASET_NAME = 'sentfin'
RESULTS_DIR = 'results'

# 运行训练（使用命令行或直接调用函数）
!python src/train.py \
    --data_path {DATA_PATH} \
    --dataset_name {DATASET_NAME} \
    --test_size 0.2
```

### 步骤5: 查看结果

```python
# 查看结果文件
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image

# 显示混淆矩阵
Image('results/confusion_matrix.png')

# 查看评估结果
results = pd.read_csv('results/evaluation_results.csv')
results.head()
```

---

## Colab特有功能

### 1. 保存结果到Google Drive

```python
# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 保存结果到Drive
RESULTS_DIR = '/content/drive/MyDrive/financial-sentiment-results'
!mkdir -p {RESULTS_DIR}

# 训练时指定路径
!python src/train.py \
    --data_path data/SEntFiN.csv \
    --dataset_name sentfin \
    --model_path {RESULTS_DIR}/model.joblib
```

### 2. 使用Colab的GPU（如果需要）

```python
# 检查GPU
!nvidia-smi

# 注意：当前项目使用CPU即可，不需要GPU
# 但如果未来升级到深度学习模型，可以使用GPU
```

### 3. 下载结果文件

```python
# 下载结果到本地
from google.colab import files

# 下载模型
files.download('results/model.joblib')

# 下载可视化
files.download('results/confusion_matrix.png')

# 下载CSV结果
files.download('results/evaluation_results.csv')
```

---

## 常见问题

### Q1: 找不到模块？

**解决方案**:
```python
# 确保路径正确
import sys
sys.path.insert(0, '/content/financial-sentiment-project/src')

# 或使用绝对导入
from src.model import build_model
```

### Q2: 文件路径错误？

**解决方案**:
```python
# 检查当前目录
import os
print("当前目录:", os.getcwd())
print("文件列表:", os.listdir('.'))

# 使用绝对路径
DATA_PATH = '/content/financial-sentiment-project/data/SEntFiN.csv'
```

### Q3: Colab会话断开后文件丢失？

**解决方案**:
- 使用Google Drive保存重要文件
- 或定期下载结果文件

### Q4: 内存不足？

**解决方案**:
```python
# 减少特征数量
MAX_FEATURES = 5000  # 从10000减少到5000

# 使用更小的数据集
# 只使用部分数据训练
```

---

## Colab vs 本地环境对比

| 特性 | Colab | 本地 |
|------|-------|------|
| 安装依赖 | `!pip install` | `pip install` |
| 文件路径 | `/content/` | 项目根目录 |
| 数据上传 | 需要上传 | 本地文件 |
| 结果保存 | 临时/Drive | 本地文件 |
| GPU支持 | 免费GPU | 需要本地GPU |
| 会话持久性 | 12小时 | 永久 |

---

## 推荐的Colab工作流程

1. **第一次使用**:
   - 克隆/上传项目
   - 运行设置代码
   - 上传数据集

2. **日常使用**:
   - 打开Colab notebook
   - 运行设置代码（如果会话重启）
   - 运行训练/评估

3. **保存结果**:
   - 重要结果保存到Google Drive
   - 或下载到本地

---

## 快速测试

在Colab中运行这个测试cell，确认一切正常：

```python
# 测试代码
import sys
import os

# 检查路径
print("Python路径:", sys.path[:3])
print("当前目录:", os.getcwd())

# 检查模块
try:
    from dataset_loader import load_dataset
    print("✓ dataset_loader 导入成功")
except Exception as e:
    print(f"✗ dataset_loader 导入失败: {e}")

try:
    from model import build_model
    print("✓ model 导入成功")
except Exception as e:
    print(f"✗ model 导入失败: {e}")

# 检查数据目录
if os.path.exists('data'):
    print("✓ data目录存在")
    print("  文件:", os.listdir('data'))
else:
    print("⚠ data目录不存在，需要创建并上传数据")
```

---

## 总结

✅ **项目完全可以在Colab中使用！**

只需要：
1. 上传项目文件到Colab
2. 运行设置代码安装依赖
3. 上传数据集
4. 调整路径（如果需要）

**优势**:
- 无需本地安装
- 免费GPU（如果需要）
- 易于分享和协作

**注意事项**:
- Colab会话会断开，重要文件要保存到Drive
- 文件路径可能需要调整
- 大文件上传可能需要时间

---

**需要帮助？** 查看主README.md获取更多信息。

