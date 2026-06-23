# RAVDESS Emotion Acoustic Baseline

本项目用于完成“基于 RAVDESS 情感语音库的跨情绪声学特征差异与分类基线研究”。

核心流程：

1. 解析 RAVDESS 文件名，筛选 `neutral / happy / sad / angry` 四类情绪；
2. 使用 `librosa` 提取传统声学特征；
3. 可选使用 `parselmouth` 调用 Praat 提取 Jitter、Shimmer、HNR、Formant；
4. 对声学特征做单因素 ANOVA 与 FDR 校正；
5. 使用 Logistic Regression、SVM、Random Forest 构建说话人独立分类基线；
6. 输出分类结果、显著性特征表、混淆矩阵和特征箱线图。

## 1. 项目结构

```text
ravdess_emotion_baseline/
│
├── data/
│   ├── raw/
│   │   └── RAVDESS/
│   └── processed/
│       ├── ravdess_metadata.csv
│       └── ravdess_acoustic_features.csv
│
├── results/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── 01_parse_metadata.py
│   ├── 02_extract_features.py
│   ├── 03_statistical_analysis.py
│   ├── 04_train_baseline_models.py
│   └── 05_visualize_results.py
│
├── requirements.txt
└── README.md
```

## 2. 安装依赖

建议使用 Python 3.9 或以上版本。

```bash
pip install -r requirements.txt
```

如果需要提取 Jitter、Shimmer、HNR、Formant，请安装：

```bash
pip install praat-parselmouth
```

如果暂时不想安装 Praat/Parselmouth，也可以只用 librosa 特征完成第一版实验。

## 3. 准备数据

将 RAVDESS 的 Audio-only Speech `.wav` 文件解压到：

```text
data/raw/RAVDESS/
```

可以保留原始的 Actor_01、Actor_02 等文件夹结构，脚本会递归搜索 `.wav` 文件。
（由于 RAVDESS 原始音频文件较大，且数据集应由使用者从官方渠道获取，本仓库不包含原始 wav 文件。复现实验时，请将 RAVDESS Audio-only Speech 数据解压至 data/raw/RAVDESS/ 目录下。）

## 4. 运行流程

### 4.1 解析元数据

```bash
python src/01_parse_metadata.py \
  --raw_dir data/raw/RAVDESS \
  --out_csv data/processed/ravdess_metadata.csv
```

### 4.2 提取声学特征

基础版本，只使用 librosa：

```bash
python src/02_extract_features.py \
  --metadata_csv data/processed/ravdess_metadata.csv \
  --out_csv data/processed/ravdess_acoustic_features.csv
```

加入 Praat/Parselmouth 特征：

```bash
python src/02_extract_features.py \
  --metadata_csv data/processed/ravdess_metadata.csv \
  --out_csv data/processed/ravdess_acoustic_features.csv \
  --use_praat
```

### 4.3 统计分析

```bash
python src/03_statistical_analysis.py \
  --features_csv data/processed/ravdess_acoustic_features.csv \
  --out_dir results/tables
```

### 4.4 训练分类基线

四分类：neutral / happy / sad / angry。

```bash
python src/04_train_baseline_models.py \
  --features_csv data/processed/ravdess_acoustic_features.csv \
  --out_dir results/tables \
  --task four_class \
  --cv groupkfold \
  --n_splits 6
```

二分类：neutral vs negative，其中 negative = sad + angry。

```bash
python src/04_train_baseline_models.py \
  --features_csv data/processed/ravdess_acoustic_features.csv \
  --out_dir results/tables \
  --task neutral_vs_negative \
  --cv groupkfold \
  --n_splits 6
```

### 4.5 可视化结果

```bash
python src/05_visualize_results.py \
  --features_csv data/processed/ravdess_acoustic_features.csv \
  --anova_csv results/tables/anova_results.csv \
  --classification_csv results/tables/classification_results.csv \
  --predictions_csv results/tables/overall_predictions.csv \
  --fig_dir results/figures
```

## 5. 主要输出文件

```text
data/processed/ravdess_metadata.csv
data/processed/ravdess_acoustic_features.csv
results/tables/anova_results.csv
results/tables/tukey_results.csv
results/tables/fold_metrics.csv
results/tables/classification_results.csv
results/tables/overall_predictions.csv
results/figures/*.png
```

## 6. 注意事项

1. RAVDESS 的 neutral 样本数量少于 happy/sad/angry，因为 neutral 没有 strong intensity 条件。
2. 分类实验必须使用说话人独立划分，不能随机划分音频样本。
3. 标准化、缺失值填补必须放在交叉验证内部，本项目已通过 sklearn Pipeline 实现。
4. Jitter、Shimmer、HNR、Formant 可能因为个别音频条件提取失败，失败值会记录为 NaN，后续由中位数填补。
5. 第一版建议先不加 `--use_praat`，等 librosa 流程跑通后再补充音质特征。
