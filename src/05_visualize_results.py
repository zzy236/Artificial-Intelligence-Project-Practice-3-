"""
05_visualize_results.py

根据前面脚本输出的表格生成可视化结果：

1. FDR 显著特征的箱线图；
2. 不同模型的 Macro F1 对比图；
3. 各模型的混淆矩阵。
4. 特征类型 × 分类器的 Accuracy / Macro-F1 / AUC 对比表及分组柱状图。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import ast
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_LABELS = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "auc": "AUC"}
FEATURE_ORDER = ["traditional", "wav2vec2", "hubert", "egemaps"]
MODEL_ORDER = ["LogisticRegression", "SVM_RBF", "RandomForest"]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', "_", str(value))


def build_comparison_table(classification_df: pd.DataFrame) -> pd.DataFrame:
    """长表一行对应一个特征类型/分类器，指标使用汇总的 OOF 预测结果。"""
    df = classification_df.copy()
    if "feature_type" not in df:
        df["feature_type"] = "traditional"  # 兼容原有单特征结果表。
    required = {"feature_type", "model", *METRIC_LABELS}
    if required - set(df.columns):
        raise ValueError(f"分类结果缺少列：{sorted(required - set(df.columns))}")
    if df[["feature_type", "model"]].isna().any().any():
        raise ValueError("特征类型或分类器名称缺失。")
    if df.duplicated(["feature_type", "model"]).any():
        raise ValueError("存在重复的特征类型/分类器组合，请为不同任务分别保存结果。")
    columns = ["feature_type", "model"]
    if "n_features" in df:
        columns.append("n_features")
    columns += list(METRIC_LABELS)
    table = df[columns].copy()
    feature_rank = {name: i for i, name in enumerate(FEATURE_ORDER)}
    model_rank = {name: i for i, name in enumerate(MODEL_ORDER)}
    table["_feature_order"] = table["feature_type"].map(feature_rank).fillna(len(feature_rank))
    table["_model_order"] = table["model"].map(model_rank).fillna(len(model_rank))
    table = table.sort_values(["_feature_order", "feature_type", "_model_order", "model"])
    return table.drop(columns=["_feature_order", "_model_order"]).rename(
        columns={"model": "classifier", **METRIC_LABELS}
    ).reset_index(drop=True)


def save_comparison_tables(classification_df: pd.DataFrame, table_dir: Path,
                           fig_dir: Path) -> pd.DataFrame:
    table = build_comparison_table(classification_df)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_dir / "feature_classifier_comparison.csv", index=False, encoding="utf-8-sig")
    # 宽表：行是特征类型，列是 分类器 × 指标，便于论文/报告横向比较。
    wide = table.pivot(index="feature_type", columns="classifier", values=list(METRIC_LABELS.values()))
    features = table["feature_type"].drop_duplicates().tolist()
    classifiers = table["classifier"].drop_duplicates().tolist()
    wide = wide.reindex(index=features, columns=pd.MultiIndex.from_product([
        list(METRIC_LABELS.values()), classifiers,
    ]))
    wide.columns = [f"{classifier}_{metric}" for metric, classifier in wide.columns]
    wide.to_csv(table_dir / "feature_classifier_comparison_wide.csv", encoding="utf-8-sig")

    display = table.copy()
    for metric in METRIC_LABELS.values():
        display[metric] = display[metric].map(lambda value: f"{value:.4f}" if pd.notna(value) else "N/A")
    fig, ax = plt.subplots(figsize=(12, max(3.2, 0.38 * len(display) + 1.6)))
    ax.axis("off")
    artist = ax.table(cellText=display.values.tolist(), colLabels=display.columns.tolist(),
                      cellLoc="center", loc="center")
    artist.auto_set_font_size(False)
    artist.set_fontsize(10)
    artist.auto_set_column_width(col=list(range(len(display.columns))))
    artist.scale(1, 1.5)
    for (row, _), cell in artist.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e7eff8")
            cell.set_text_props(weight="bold")
    ax.set_title("Feature type x classifier: pooled OOF metrics", pad=18)
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_classifier_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[对比表] Accuracy/Macro-F1/AUC（汇总 OOF 测试预测；不是折均值）")
    print(table.to_string(index=False))
    return table


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

    df = classification_df.copy()
    if "feature_type" not in df:
        df["feature_type"] = "traditional"
    features = [name for name in FEATURE_ORDER if name in set(df["feature_type"])]
    features += sorted(set(df["feature_type"]) - set(features))
    models = [name for name in MODEL_ORDER if name in set(df["model"])]
    models += sorted(set(df["model"]) - set(models))
    matrix = df.pivot(index="model", columns="feature_type", values=metric).reindex(index=models, columns=features)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.8 / len(features)
    for i, feature in enumerate(features):
        bars = ax.bar(x + (i - (len(features) - 1) / 2) * width,
                      matrix[feature].to_numpy(), width=width, label=feature)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xlabel("Classifier")
    ax.set_title(f"Feature type x classifier: {METRIC_LABELS.get(metric, metric)}")
    ax.legend(title="Feature type", loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / f"model_comparison_{metric}.png", dpi=300)
    plt.close(fig)


def save_confusion_matrices(
    predictions_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """根据样本级预测表保存混淆矩阵。"""
    required = {"model", "y_true", "y_pred"}
    if not required.issubset(predictions_df.columns):
        return

    group_columns = ["feature_type", "model"] if "feature_type" in predictions_df else ["model"]
    for keys, sub in predictions_df.groupby(group_columns):
        names = keys if isinstance(keys, tuple) else (keys,)
        model_name = " / ".join(map(str, names))
        labels = sorted(sub["y_true"].dropna().unique().tolist())
        cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, values_format="d")
        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        safe_model_name = safe_name("_".join(map(str, names)))
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
        if "feature_type" in classification_df:
            model_name = f"{row['feature_type']}_{model_name}"
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

        safe_model_name = safe_name(model_name)
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
    parser.add_argument("--skip_boxplots", action="store_true", help="只绘制分类结果，跳过声学特征箱线图。")
    parser.add_argument("--table_dir", help="主分类结果的对比表输出目录，默认与 classification_csv 同目录。")
    parser.add_argument("--comparison_csv", default="results/tables/feature_comparison/classification_results.csv")
    parser.add_argument("--comparison_predictions_csv", help="默认使用 comparison_csv 同目录的 overall_predictions.csv。")
    parser.add_argument("--comparison_fig_dir", default="results/figures/feature_comparison")
    args = parser.parse_args()

    features_csv = project_path(args.features_csv)
    anova_csv = project_path(args.anova_csv)
    classification_csv = project_path(args.classification_csv)
    predictions_csv = project_path(args.predictions_csv)
    fig_dir = project_path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_boxplots:
        print("[跳过] 已指定 --skip_boxplots。")
    elif features_csv.exists() and anova_csv.exists():
        features_df = pd.read_csv(features_csv)
        anova_df = pd.read_csv(anova_csv)
        save_feature_boxplots(features_df, anova_df, fig_dir, top_n=args.top_n)
        print("[完成] 特征箱线图已保存。")
    else:
        print("[跳过] 未找到 features_csv 或 anova_csv，跳过箱线图。")

    if classification_csv.exists():
        classification_df = pd.read_csv(classification_csv)
        save_comparison_tables(classification_df,
                               project_path(args.table_dir) if args.table_dir else classification_csv.parent, fig_dir)
        for metric in METRIC_LABELS:
            save_model_comparison(classification_df, fig_dir, metric=metric)
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

    comparison_csv = project_path(args.comparison_csv)
    if comparison_csv.exists() and comparison_csv.resolve() != classification_csv.resolve():
        comparison_df = pd.read_csv(comparison_csv)
        comparison_fig_dir = project_path(args.comparison_fig_dir)
        save_comparison_tables(comparison_df, comparison_csv.parent, comparison_fig_dir)
        for metric in METRIC_LABELS:
            save_model_comparison(comparison_df, comparison_fig_dir, metric)
        comparison_predictions = (project_path(args.comparison_predictions_csv)
                                  if args.comparison_predictions_csv else comparison_csv.parent / "overall_predictions.csv")
        if comparison_predictions.exists():
            save_confusion_matrices(pd.read_csv(comparison_predictions), comparison_fig_dir)
        else:
            save_confusion_from_summary_if_needed(comparison_df, comparison_fig_dir)
        print(f"[完成] 特征类型 × 分类器的对比表和图像：{comparison_fig_dir}")

    print(f"[完成] 图像目录：{fig_dir}")


if __name__ == "__main__":
    main()
