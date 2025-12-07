# 技术概述：金融社交媒体情感分类

本文档提供了金融社交媒体情感分类项目的全面技术说明，包括问题定义、数据处理、模型架构、训练方法、评估指标和标签质量分析。

---

## 1. 问题定义

### 1.1 任务

**单数据集金融社交媒体情感分类**，使用轻量级、可解释的机器学习基线模型。

- **输入**：金融社交媒体文本（Twitter 推文）
- **输出**：3 类情感标签：`positive`（积极）、`neutral`（中性）、`negative`（消极）
- **数据集**：Twitter Financial News Sentiment (Zeroshot, 2023)，来自 `data/twitter_financial_train.csv`

### 1.2 研究重点

本项目强调**标签质量评估**，而非追求最先进的准确率。核心研究贡献包括：

- 识别噪声社交媒体文本中的模糊案例
- 检测可能被错误标注的样本
- 量化数据集固有的模糊性
- 分析边界案例（积极/消极 vs 中性）
- 通过标签质量分析理解模型局限性

**为什么选择社交媒体文本？** 社交媒体金融帖子本质上比新闻文章更嘈杂，使其成为研究的理想选择：
- 标注不一致性
- 情感类别之间的边界案例
- 中性模糊区域（社交媒体中常见）
- 数据集固有的模糊性

### 1.3 与 CS5100 提案的对齐

- **单一社交媒体数据集**（非多个数据集或新闻文章）
- **轻量级、可解释的基线**（TF-IDF + 逻辑回归）
- **强调数据/标签质量**，而非追求 SOTA 模型性能
- **全面的标签质量评估**作为主要研究贡献

---

## 2. 数据和预处理

### 2.1 数据集

**来源**：Twitter Financial News Sentiment (Zeroshot, 2023)
- **文件**：`data/twitter_financial_train.csv`
- **格式**：CSV，包含 `text` 和 `label` 列
- **标签**：数字格式（0=Bearish/消极，1=Bullish/积极，2=Neutral/中性）→ 统一为 `positive`、`neutral`、`negative`
- **规模**：约 9,500 个样本
- **特征**：
  - 短文本、非正式语言（典型的社交媒体特征）
  - 真实世界的噪声（标签、提及、股票代码、URL）
  - 类别不平衡：中性（~65%）、积极（~20%）、消极（~15%）

**加载**：数据集通过 `src/dataset_loader.py` 中的 `load_twitter_financial()` 函数加载，该函数：
- 处理 CSV/TSV/JSON 格式
- 自动检测文本和标签列
- 将数字标签（0, 1, 2）映射到统一的字符串标签
- 过滤无效标签和空文本

### 2.2 文本预处理

预处理在 `src/preprocess.py` 中实现，针对社交媒体文本进行了优化：

**步骤**（按顺序）：
1. **URL 移除**：移除所有 URL（`http://...`、`www....`）
2. **股票代码移除**：移除股票符号（如 `$TSLA`、`$AAPL`）
3. **标签/提及移除**：移除 `#hashtags` 和 `@mentions`
4. **小写转换**：将所有文本转换为小写
5. **标点移除**：移除标点符号
6. **分词**：将文本分割为词元（如果可用则使用 NLTK，否则使用简单分割）
7. **重新连接**：将词元重新连接为清理后的文本字符串

**实现细节**：
- 使用正则表达式进行模式匹配（URL、股票代码、标签、提及）
- 优雅处理缺失的 NLTK（回退到简单分词）
- 保留词序（词袋表示）
- 假设英文文本和 UTF-8 编码

**示例**：
```
原始文本: "RT @user: $TSLA is going up! Check https://t.co/abc #stocks"
清理后:   "rt user tsla is going up check stocks"
```

**为什么需要这种预处理？**
- 社交媒体文本包含对情感无贡献的噪声（标签、提及、股票代码）
- URL 被移除，因为它们不包含情感信息
- 小写和标点移除为词袋表示规范化文本
- 停用词移除减少噪声（尽管 TF-IDF 已经降低常见词的权重）

---

## 3. 模型架构

### 3.1 特征提取：TF-IDF

**词频-逆文档频率（TF-IDF）**向量化，使用 1-2 元组特征。

**参数**（来自 `src/model.py`）：
- **N-元组范围**：`(1, 2)` → 一元组（单词）和二元组（词对）
- **最大特征数**：`10,000`（按 TF-IDF 分数排序的最重要特征）
- **停用词**：移除英文停用词
- **最小文档频率**：`min_df=2`（特征必须出现在至少 2 个文档中）
- **最大文档频率**：`max_df=0.95`（特征必须出现在最多 95% 的文档中）
- **小写**：`True`（已在预处理中处理）

**为什么使用 TF-IDF？**
- **轻量级**：计算快速，无需 GPU
- **可解释**：特征权重显示哪些词/二元组重要
- **基线**：文本分类的标准基线
- **词袋**：捕获词的存在和重要性，但忽略词序和长距离上下文

**局限性**：
- 无法捕获词序或长距离依赖
- 无法捕获讽刺或上下文相关的含义
- 忽略词之间的语义关系

### 3.2 分类器：多项逻辑回归

**逻辑回归**用于多类分类。

**参数**：
- **求解器**：`lbfgs`（适用于中小型数据集）
- **最大迭代次数**：`1000`
- **随机种子**：`42`（用于可重现性）
- **多类**：`multinomial`（默认，处理 3+ 类）

**为什么使用逻辑回归？**
- **可解释**：特征权重显示哪些词对每个类别有贡献
- **快速**：高效的训练和推理
- **基线**：文本分类的标准基线
- **轻量级**：无深度学习依赖
- **线性模型**：简单、可解释的决策边界

**局限性**：
- **线性决策边界**：无法捕获非线性模式
- **有限容量**：无法捕获复杂模式（讽刺、上下文）
- **基线模型**：为可解释性设计，非 SOTA 性能

### 3.3 流水线

模型实现为 **scikit-learn 流水线**：
```
Pipeline([
    ('tfidf', TfidfVectorizer(...)),
    ('classifier', LogisticRegression(...))
])
```

这确保了：
- 训练和推理期间一致的预处理
- 使用 `joblib` 轻松保存/加载模型
- 用于预测和特征提取的清晰接口

---

## 4. 训练和评估逻辑

### 4.1 训练过程

**数据划分**：
- **训练集**：`twitter_financial_train.csv`（~9,500 样本）- 仅用于训练
- **验证集**：`twitter_financial_valid.csv`（~2,400 样本）- 完全独立的测试集

使用独立的验证集（而不是从训练数据中随机划分）确保了无偏的评估和更准确的性能估计。

**训练步骤**（来自 `src/train.py`）：
1. 使用 `load_dataset('twitter_financial', train_path)` 加载训练数据集
2. 使用 `preprocess_batch()` 预处理训练文本
3. 过滤空文本
4. 使用 `load_dataset('twitter_financial', valid_path)` 加载独立验证集
5. 预处理验证集文本
6. 使用 `build_model(max_features=10000, ngram_range=(1, 2))` 构建模型
7. 训练模型：`model.fit(X_train, y_train)`（仅使用训练集）
8. 在独立验证集上评估：`model.predict(X_valid)`
9. 保存模型：`joblib.dump(model, 'results/model.joblib')`

**模型保存**：
- 保存到 `results/model.joblib`（或在报告 notebook 中为 `results/model_report.joblib`）
- 使用 `joblib` 进行高效序列化
- 在流水线中包括向量化器和分类器

### 4.2 评估指标

**计算的指标**（来自 `src/evaluate.py`）：
- **准确率**：整体分类准确率
- **宏 F1 分数**：所有类别的平均 F1 分数（处理类别不平衡）
- **每类指标**：每个类别的精确率、召回率、F1 分数
- **混淆矩阵**：显示预测与真实标签的分布

**为什么使用宏 F1？**
- **类别不平衡**：数据集存在显著不平衡（中性：~65%，积极：~20%，消极：~15%）
- **平衡评估**：宏 F1 对所有类别给予相等权重，不偏向多数类
- **优于准确率**：在类别不平衡的情况下，准确率可能具有误导性

**输出文件**：
- `results/evaluation_results.csv`：每个样本的详细预测和概率
- `results/evaluation_summary.txt`：摘要指标（准确率、宏 F1、每类指标）
- `results/confusion_matrix.png`：混淆矩阵的可视化

### 4.3 混淆矩阵

混淆矩阵显示：
- **行**：真实标签
- **列**：预测标签
- **值**：样本数量

**关键观察**：
- **中性类**：高召回率，由于类别不平衡（多数类）
- **积极类**：中等召回率（~20%）
- **消极类**：较低召回率（~15%）- 少数类更具挑战性

**可视化**：使用 `seaborn.heatmap()` 保存到 `results/confusion_matrix.png`

---

## 5. 标签质量分析逻辑

标签质量分析是本项目的**主要研究贡献**。它识别模糊案例、可能被错误标注的样本，以及噪声社交媒体文本中数据集固有的模糊性。

所有标签质量函数在 `src/label_quality.py` 中实现，并使用：
- **模型预测**：`model.predict(X)` → 预测标签
- **模型概率**：`model.predict_proba(X)` → 类别上的概率分布

### 5.1 误分类

**函数**：`detect_misclassifications()`

**逻辑**：
1. 对所有样本进行预测
2. 比较预测与真实标签：`y_true != y_pred`
3. 提取误分类样本，包括：
   - 原始文本
   - 真实标签
   - 预测标签
   - 置信度（最大概率）
   - 真实标签概率

**输出**：`results/misclassifications.csv`（或 `results/report_misclassifications.csv`）

**用例**：识别模型与标签不一致的案例。高置信度误分类可能表示标签错误。

### 5.2 模糊预测

**函数**：`detect_ambiguous_predictions()`

**逻辑**：
1. 进行预测并获取概率
2. 找到最大概率在 `confidence_threshold` 之间（默认：0.45-0.55）的样本
3. 这些是模型不确定的低置信度预测

**输出**：`results/ambiguous_predictions.csv`（或 `results/report_ambiguous_predictions.csv`）

**用例**：识别模型（可能还有人类）会不确定的真正模糊案例。由于缺少上下文、讽刺或边界情感，这在社交媒体文本中很常见。

### 5.3 噪声标签

**函数**：`detect_noisy_labels()`

**逻辑**：
1. 找到高置信度误分类（置信度 > 0.8）
2. 应用启发式方法：
   - 高置信度不一致（模型非常确信但错误）
   - 重复模式（相同文本但不同标签）
   - 短文本模糊性（非常短的文本更可能模糊）

**输出**：`results/noisy_labels.csv`（或 `results/report_noisy_labels.csv`）

**用例**：识别应手动审查的可能被错误标注的样本。高置信度不一致表明标签错误而非模型错误。

### 5.4 中性模糊区域

**函数**：`analyze_neutral_ambiguous_zone()`

**逻辑**：
1. 获取所有样本的概率分布
2. 找到中性概率高但模型预测积极/消极的样本（或反之）
3. 这些是模型难以区分中性与情感的案例

**输出**：`results/neutral_ambiguous_zone.csv`（或 `results/report_neutral_ambiguous_zone.csv`）

**用例**：社交媒体特定分析。中性在社交媒体文本中本质上模糊（许多边界案例）。这识别了中性与情感不明确的案例。

### 5.5 边界案例

**函数**：`analyze_borderline_cases()`

**逻辑**：
1. 找到模型预测积极/消极但中性概率也很高的样本
2. 分类边界类型：
   - `positive_vs_neutral`：模型预测积极但中性概率高
   - `negative_vs_neutral`：模型预测消极但中性概率高

**输出**：`results/borderline_cases.csv`（或 `results/report_borderline_cases.csv`）

**用例**：识别积极/消极与中性之间的边界案例。在情感微妙或依赖上下文的社交媒体文本中很常见。

### 5.6 数据集模糊性指标

**函数**：`quantify_dataset_ambiguity()`

**逻辑**：
1. 计算整体指标：
   - 所有预测的平均置信度
   - 低置信度百分比（置信度 < 0.6）
   - 模糊区域百分比（置信度在 0.45-0.55 之间）
   - 高置信度百分比（置信度 > 0.8）
   - 误分类率
   - 平均文本长度
   - 短文本百分比（< 10 个字符）

**输出**：`results/dataset_ambiguity_metrics.csv`（或 `results/report_dataset_ambiguity_metrics.csv`）

**用例**：量化数据集中的整体模糊性。社交媒体文本本质上比新闻文章更嘈杂，这些指标捕获了这一点。

### 5.7 标签质量可视化

**文件**：`results/label_quality_analysis.png`

**内容**：
- 误分类置信度分布
- 按标签的模糊预测
- 按启发式的噪声标签
- 中性模糊区域分布
- 按类型的边界案例
- 摘要统计

**生成者**：Notebooks（如 `notebooks/03_label_quality.ipynb` 或 `notebooks/04_final_report.ipynb`）

---

## 6. 局限性和未来工作

### 6.1 模型局限性

**词袋表示**：
- TF-IDF 忽略词序和长距离上下文
- 无法捕获像 "not good" 这样的短语（会被分别处理为 "not" 和 "good"）
- 无法捕获讽刺或上下文相关的含义

**线性模型容量**：
- 逻辑回归具有线性决策边界
- 无法捕获非线性模式
- 对复杂情感模式的容量有限

**基线模型**：
- 为可解释性设计，非 SOTA 性能
- 无法捕获讽刺或长距离依赖
- 可能难以处理非常短或高度非正式的帖子

### 6.2 数据局限性

**类别不平衡**：
- 显著不平衡（中性：~65%，积极：~20%，消极：~15%）
- 模型偏向多数类（中性）
- 在少数类（积极、消极）上表现不佳

**社交媒体模糊性**：
- 本质上模糊（讽刺、缺少上下文、缩写）
- 某些案例即使对人类也真正困难
- 快速标注导致的标签噪声

**缺少上下文**：
- 无对话历史或背景信息
- 上下文有限的短文本
- 多种解释可能

### 6.3 分析局限性

**标签质量启发式**：
- 简单的启发式可能无法捕获所有类型的标签错误
- 高置信度误分类可能是模型错误，而非标签错误
- 模糊性指标依赖于模型

**模型依赖的指标**：
- 模糊性指标取决于使用的特定模型
- 不同模型可能识别不同的模糊案例
- 指标可能随模型改进而变化

### 6.4 未来工作

**模型改进**：
1. **上下文嵌入**：用 BERT 或 FinBERT 替换 TF-IDF，以获得更丰富的语义和更好地处理讽刺/长距离依赖
2. **额外特征**：将股票代码、标签、用户元数据作为特征
3. **模型改进**：尝试不同的分类器或特征工程
4. **处理类别不平衡**：使用类别权重、SMOTE 或其他技术

**标签质量改进**：
1. **主动学习**：使用标签质量发现来专注于模糊/噪声案例的标注工作
2. **手动审查**：审查高置信度误分类以识别真正的标签错误
3. **错误分析**：深入分析标签质量分析识别的特定错误案例
4. **交叉验证**：在不同训练/测试划分上验证标签质量发现

**部署**：
1. **实时 API**：将模型部署为 API 并连接到实时社交媒体流
2. **模型监控**：随时间跟踪模型性能和标签质量指标
3. **A/B 测试**：比较不同模型或预处理策略

**注意**：FinBERT 或其他上下文嵌入可以作为性能的未来上界，特别是对于捕获基线模型无法处理的讽刺和长距离依赖。

---

## 7. 与 CS5100 提案的对齐

本项目与 CS5100 研究提案对齐，通过：

1. **单一社交媒体数据集**：仅使用 Twitter Financial News Sentiment (Zeroshot, 2023)，非多个数据集或新闻文章
2. **轻量级、可解释的基线**：TF-IDF + 逻辑回归提供可解释的特征权重
3. **强调数据/标签质量**：标签质量评估是主要研究贡献，而非追求 SOTA 模型性能
4. **全面的标签质量分析**：识别模糊案例、噪声标签、边界案例和数据集固有的模糊性

**主要研究贡献**：标签质量评估框架提供了对数据集可靠性和模型局限性的见解，特别是对于噪声社交媒体文本。这种以数据为中心的分析比原始准确率指标更有价值。

**关键发现**：
- 社交媒体文本比新闻文章具有更高的模糊性
- 中性类别表现最好（召回率：0.85），因为它是多数类（~65%）
- 类别不平衡影响模型在少数类上的性能，负面类别（15%）最具挑战性
- 对于噪声文本，标签质量比准确率更重要
- 使用独立验证集（而非从训练数据中随机划分）确保了无偏的评估和更准确的性能估计

---

## 8. 文件结构和输出

### 8.1 源代码

- `src/dataset_loader.py`：数据集加载（仅 Twitter Financial）
- `src/preprocess.py`：文本预处理
- `src/model.py`：模型定义（TF-IDF + 逻辑回归）
- `src/train.py`：训练脚本
- `src/evaluate.py`：评估脚本
- `src/label_quality.py`：标签质量分析函数

### 8.2 Notebooks

- `notebooks/01_eda.ipynb`：探索性数据分析
- `notebooks/02_train_baseline.ipynb`：模型训练和评估
- `notebooks/03_label_quality.ipynb`：标签质量分析
- `notebooks/04_final_report.ipynb`：完整项目报告

### 8.3 输出文件

**模型**：
- `results/model.joblib`：训练好的模型（或 `results/model_report.joblib`）

**评估**：
- `results/evaluation_results.csv`：带有概率的详细预测
- `results/evaluation_summary.txt`：摘要指标
- `results/confusion_matrix.png`：混淆矩阵可视化

**标签质量**：
- `results/misclassifications.csv`：误分类样本
- `results/ambiguous_predictions.csv`：低置信度预测
- `results/noisy_labels.csv`：可能的噪声标签
- `results/neutral_ambiguous_zone.csv`：中性模糊区域案例
- `results/borderline_cases.csv`：边界案例
- `results/dataset_ambiguity_metrics.csv`：数据集模糊性指标
- `results/label_quality_analysis.png`：全面可视化

---

## 9. 可重现性

要重现结果：

1. **下载数据集**：将 `twitter_financial_train.csv` 和 `twitter_financial_valid.csv` 放在 `data/` 目录中
2. **安装依赖**：`pip install -r requirements.txt`
3. **训练模型**：`python src/train.py --data_path data/twitter_financial_train.csv --valid_path data/twitter_financial_valid.csv --dataset_name twitter_financial`
4. **评估**：`python src/evaluate.py --model_path results/model.joblib --data_path data/twitter_financial_valid.csv --dataset_name twitter_financial`
5. **标签质量**：`python src/label_quality.py --model_path results/model.joblib --data_path data/twitter_financial_train.csv --dataset_name twitter_financial`

或使用 Jupyter notebooks 进行交互式分析。

---

## 总结

本项目实现了使用 TF-IDF + 逻辑回归的**轻量级、可解释基线**，用于金融社交媒体情感分类。主要研究贡献是**全面的标签质量评估**，识别噪声社交媒体文本中的模糊案例、噪声标签和数据集固有的模糊性。

**关键技术要点**：
- 单一数据集：Twitter Financial News Sentiment (Zeroshot, 2023)
- 预处理：针对社交媒体噪声优化（URL、股票代码、标签、提及）
- 模型：TF-IDF（1-2 元组）+ 多项逻辑回归
- 评估：准确率、宏 F1、每类指标、混淆矩阵
- 标签质量：误分类、模糊预测、噪声标签、中性模糊区域、边界案例、数据集模糊性指标
- 局限性：基线模型无法捕获讽刺/长距离依赖；类别不平衡影响性能
- 未来工作：FinBERT/transformers 以获得更强的性能；主动学习用于模糊案例

这种方法与 CS5100 提案对齐，通过专注于**数据/标签质量评估**而非追求 SOTA 模型性能，使其成为理解噪声社交媒体文本分类的宝贵贡献。
