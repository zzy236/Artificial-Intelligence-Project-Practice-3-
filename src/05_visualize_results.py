"""
05_visualize_results.py

根据前面脚本输出的表格生成可视化结果：

1. FDR 显著特征的箱线图；
2. 不同模型的 Macro F1 对比图；
3. 各模型的混淆矩阵。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_feature_boxplots(
    features_df: pd.DataFrame,
    anova_df: pd.DataFrame,
    fig_dir: Path,
    top_n: int = 10,
    group_col: str = "emotion",
) -> None:
    """保存前若干个显著特征的箱线图。"""
    if "significant_fdr_0.05" in anova_df.columns:
        selected = anova_df.loc[anova_df["significant_fdr_0.05"] == True, "feature"].head(top_n).tolist()
    else:
        selected = anova_df["feature"].head(top_n).tolist()

    if not selected:
        selected = anova_df["feature"].head(top_n).tolist()

    groups = sorted(features_df[group_col].dropna().unique().tolist())

    for feature in selected:
        if feature not in features_df.columns:
            continue

        data = [features_df.loc[features_df[group_col] == group, feature].dropna().values for group in groups]
        if sum(len(x) for x in data) == 0:
            continue

        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=groups, showmeans=True)
        plt.xlabel(group_col)
        plt.ylabel(feature)
        plt.title(f"{feature} by {group_col}")
        plt.tight_layout()

        out_path = fig_dir / f"boxplot_{feature}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


def save_model_comparison(
    classification_df: pd.DataFrame,
    fig_dir: Path,
    metric: str = "macro_f1",
) -> None:
    """保存模型指标对比柱状图。"""
    if metric not in classification_df.columns:
        return

    df = classification_df.sort_values(metric, ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df[metric])
    plt.ylabel(metric)
    plt.xlabel("model")
    plt.title(f"Model comparison: {metric}")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_path = fig_dir / f"model_comparison_{metric}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_confusion_matrices(
    predictions_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """根据样本级预测表保存混淆矩阵。"""
    required = {"model", "y_true", "y_pred"}
    if not required.issubset(predictions_df.columns):
        return

    for model_name, sub in predictions_df.groupby("model"):
        labels = sorted(sub["y_true"].dropna().unique().tolist())
        cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, values_format="d")
        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        safe_model_name = str(model_name).replace("/", "_").replace(" ", "_")
        out_path = fig_dir / f"confusion_matrix_{safe_model_name}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


def save_confusion_from_summary_if_needed(
    classification_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """备用：如果没有 predictions_df，则尝试从 classification_results 中解析混淆矩阵。"""
    if "confusion_matrix" not in classification_df.columns:
        return

    for _, row in classification_df.iterrows():
        model_name = row["model"]
        labels = str(row.get("labels", "")).split(",")

        try:
            cm = json.loads(row["confusion_matrix"])
        except Exception:
            try:
                cm = ast.literal_eval(row["confusion_matrix"])
            except Exception:
                continue

        disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, values_format="d")
        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        safe_model_name = str(model_name).replace("/", "_").replace(" ", "_")
        out_path = fig_dir / f"confusion_matrix_{safe_model_name}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RAVDESS acoustic baseline results.")
    parser.add_argument("--features_csv", type=str, default="data/processed/ravdess_acoustic_features.csv")
    parser.add_argument("--anova_csv", type=str, default="results/tables/anova_results.csv")
    parser.add_argument("--classification_csv", type=str, default="results/tables/classification_results.csv")
    parser.add_argument("--predictions_csv", type=str, default="results/tables/overall_predictions.csv")
    parser.add_argument("--fig_dir", type=str, default="results/figures")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    anova_csv = Path(args.anova_csv)
    classification_csv = Path(args.classification_csv)
    predictions_csv = Path(args.predictions_csv)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if features_csv.exists() and anova_csv.exists():
        features_df = pd.read_csv(features_csv)
        anova_df = pd.read_csv(anova_csv)
        save_feature_boxplots(features_df, anova_df, fig_dir, top_n=args.top_n)
        print("[完成] 特征箱线图已保存。")
    else:
        print("[跳过] 未找到 features_csv 或 anova_csv，跳过箱线图。")

    if classification_csv.exists():
        classification_df = pd.read_csv(classification_csv)
        save_model_comparison(classification_df, fig_dir, metric="macro_f1")
        save_model_comparison(classification_df, fig_dir, metric="accuracy")
        print("[完成] 模型对比图已保存。")

        if predictions_csv.exists():
            predictions_df = pd.read_csv(predictions_csv)
            save_confusion_matrices(predictions_df, fig_dir)
            print("[完成] 混淆矩阵已保存。")
        else:
            save_confusion_from_summary_if_needed(classification_df, fig_dir)
            print("[完成] 混淆矩阵已保存。")
    else:
        print("[跳过] 未找到 classification_csv，跳过模型图与混淆矩阵。")

    print(f"[完成] 图像目录：{fig_dir}")


if __name__ == "__main__":
    main()
