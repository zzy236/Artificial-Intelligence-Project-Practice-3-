"""
03_statistical_analysis.py

对 RAVDESS 四类情绪声学特征进行统计差异分析。

默认流程：
1. 自动识别数值型声学特征列；
2. 对每个特征进行单因素 ANOVA；
3. 进行 Benjamini-Hochberg FDR 多重检验校正；
4. 对显著特征进行 Tukey HSD 事后检验；
5. 输出 anova_results.csv 与 tukey_results.csv。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd


META_COLUMNS = {
    "sample_id",
    "file_name",
    "file_path",
    "modality_id",
    "modality",
    "vocal_channel_id",
    "vocal_channel",
    "emotion_id",
    "emotion",
    "intensity_id",
    "intensity",
    "statement_id",
    "statement",
    "repetition_id",
    "repetition",
    "actor",
    "gender",
    "load_error",
    "n_samples",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """识别数值型声学特征列。"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in META_COLUMNS]


def compute_eta_squared(df: pd.DataFrame, feature: str, group_col: str) -> float:
    """计算 one-way ANOVA 的 eta-squared 效应量。"""
    sub = df[[feature, group_col]].dropna()
    if sub.empty:
        return np.nan

    grand_mean = sub[feature].mean()
    ss_total = ((sub[feature] - grand_mean) ** 2).sum()

    if ss_total == 0:
        return np.nan

    ss_between = 0.0
    for _, group in sub.groupby(group_col):
        ss_between += len(group) * (group[feature].mean() - grand_mean) ** 2

    return float(ss_between / ss_total)


def run_anova(
    df: pd.DataFrame,
    group_col: str = "emotion",
    feature_cols: list[str] | None = None,
    min_group_size: int = 2,
) -> pd.DataFrame:
    """对每个声学特征执行单因素 ANOVA。"""
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    emotions = sorted(df[group_col].dropna().unique().tolist())
    rows = []

    for feature in feature_cols:
        groups = []
        group_sizes = {}
        group_means = {}
        group_stds = {}

        for emo in emotions:
            values = df.loc[df[group_col] == emo, feature].dropna().values
            group_sizes[emo] = len(values)
            group_means[emo] = float(np.mean(values)) if len(values) > 0 else np.nan
            group_stds[emo] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            if len(values) >= min_group_size:
                groups.append(values)

        if len(groups) < 2:
            f_stat, p_value = np.nan, np.nan
        else:
            try:
                f_stat, p_value = f_oneway(*groups)
            except Exception:
                f_stat, p_value = np.nan, np.nan

        row = {
            "feature": feature,
            "F": float(f_stat) if np.isfinite(f_stat) else np.nan,
            "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            "eta_squared": compute_eta_squared(df, feature, group_col),
        }

        for emo in emotions:
            row[f"{emo}_n"] = group_sizes[emo]
            row[f"{emo}_mean"] = group_means[emo]
            row[f"{emo}_std"] = group_stds[emo]

        rows.append(row)

    result = pd.DataFrame(rows)

    valid_p = result["p_value"].notna()
    result["p_fdr_bh"] = np.nan
    result["significant_fdr_0.05"] = False

    if valid_p.any():
        reject, p_adj, _, _ = multipletests(result.loc[valid_p, "p_value"].values, alpha=0.05, method="fdr_bh")
        result.loc[valid_p, "p_fdr_bh"] = p_adj
        result.loc[valid_p, "significant_fdr_0.05"] = reject

    result = result.sort_values(["p_fdr_bh", "p_value"], na_position="last").reset_index(drop=True)
    return result


def run_tukey_for_significant_features(
    df: pd.DataFrame,
    anova_df: pd.DataFrame,
    group_col: str = "emotion",
    max_features: int = 30,
) -> pd.DataFrame:
    """对 FDR 显著的前若干个特征做 Tukey HSD 事后检验。"""
    sig_features = anova_df.loc[anova_df["significant_fdr_0.05"], "feature"].head(max_features).tolist()
    rows = []

    for feature in sig_features:
        sub = df[[feature, group_col]].dropna()
        if sub[group_col].nunique() < 2:
            continue

        try:
            tukey = pairwise_tukeyhsd(endog=sub[feature].values, groups=sub[group_col].values, alpha=0.05)
            table = pd.DataFrame(
                tukey.summary().data[1:],
                columns=tukey.summary().data[0],
            )
            table.insert(0, "feature", feature)
            rows.append(table)
        except Exception as exc:
            rows.append(
                pd.DataFrame(
                    [{
                        "feature": feature,
                        "group1": np.nan,
                        "group2": np.nan,
                        "meandiff": np.nan,
                        "p-adj": np.nan,
                        "lower": np.nan,
                        "upper": np.nan,
                        "reject": np.nan,
                        "error": str(exc),
                    }]
                )
            )

    if not rows:
        return pd.DataFrame(columns=["feature", "group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"])

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ANOVA and Tukey HSD on acoustic features.")
    parser.add_argument("--features_csv", type=str, default="data/processed/ravdess_acoustic_features.csv")
    parser.add_argument("--out_dir", type=str, default="results/tables")
    parser.add_argument("--group_col", type=str, default="emotion")
    parser.add_argument("--max_tukey_features", type=int, default=30)
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv 不存在：{features_csv}")

    df = pd.read_csv(features_csv)
    feature_cols = get_feature_columns(df)

    if not feature_cols:
        raise RuntimeError("没有识别到数值型声学特征列。")

    anova_df = run_anova(df, group_col=args.group_col, feature_cols=feature_cols)
    tukey_df = run_tukey_for_significant_features(
        df,
        anova_df=anova_df,
        group_col=args.group_col,
        max_features=args.max_tukey_features,
    )

    anova_path = out_dir / "anova_results.csv"
    tukey_path = out_dir / "tukey_results.csv"

    anova_df.to_csv(anova_path, index=False, encoding="utf-8-sig")
    tukey_df.to_csv(tukey_path, index=False, encoding="utf-8-sig")

    print(f"[完成] ANOVA 结果已保存：{anova_path}")
    print(f"[完成] Tukey HSD 结果已保存：{tukey_path}")
    print("[信息] FDR 显著特征数量：", int(anova_df["significant_fdr_0.05"].sum()))
    print("[信息] 前 10 个特征：")
    print(anova_df[["feature", "F", "p_value", "p_fdr_bh", "eta_squared", "significant_fdr_0.05"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
