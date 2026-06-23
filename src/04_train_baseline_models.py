"""
04_train_baseline_models.py

构建 RAVDESS 情绪分类传统机器学习基线。

支持任务：
1. four_class: neutral / happy / sad / angry 四分类
2. neutral_vs_negative: neutral vs negative，其中 negative = sad + angry，happy 会被排除

支持交叉验证：
1. groupkfold: 按 actor 分组的 K 折交叉验证
2. logo: Leave-One-Group-Out，即留一说话人交叉验证

模型：
1. Logistic Regression
2. SVM
3. Random Forest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


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
    """识别用于建模的数值特征列。"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in META_COLUMNS]


def prepare_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """根据任务设置标签。"""
    df = df.copy()

    if task == "four_class":
        keep = ["neutral", "happy", "sad", "angry"]
        df = df[df["emotion"].isin(keep)].copy()
        df["target"] = df["emotion"]

    elif task == "neutral_vs_negative":
        keep = ["neutral", "sad", "angry"]
        df = df[df["emotion"].isin(keep)].copy()
        df["target"] = np.where(df["emotion"] == "neutral", "neutral", "negative")

    else:
        raise ValueError(f"未知任务：{task}")

    df = df.reset_index(drop=True)
    return df


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """构建三个基线模型。"""
    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "SVM_RBF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }

    return models


def get_cv(groups: np.ndarray, cv_name: str, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成交叉验证索引。"""
    dummy_x = np.zeros(len(groups))
    dummy_y = np.zeros(len(groups))

    if cv_name == "groupkfold":
        n_groups = len(np.unique(groups))
        if n_splits > n_groups:
            raise ValueError(f"n_splits={n_splits} 大于说话人数 {n_groups}")
        cv = GroupKFold(n_splits=n_splits)
        return list(cv.split(dummy_x, dummy_y, groups))

    if cv_name == "logo":
        cv = LeaveOneGroupOut()
        return list(cv.split(dummy_x, dummy_y, groups))

    raise ValueError(f"未知 cv：{cv_name}")


def safe_auc_score(
    y_true_encoded: np.ndarray,
    proba: np.ndarray,
    n_classes: int,
) -> float:
    """计算 AUC。多分类用 macro OVR，二分类用正类概率。"""
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true_encoded, proba[:, 1]))
        return float(roc_auc_score(
            y_true_encoded,
            proba,
            multi_class="ovr",
            average="macro",
            labels=list(range(n_classes)),
        ))
    except Exception:
        return np.nan


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_encoded: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
) -> Dict[str, Any]:
    """计算整体评价指标。"""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "auc": safe_auc_score(y_true_encoded, proba, n_classes=len(labels)),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    cv_name: str,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """训练并评估所有模型。"""
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df["target"].astype(str).values
    groups = df["actor"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    labels = label_encoder.classes_.tolist()
    n_classes = len(labels)

    cv_splits = get_cv(groups=groups, cv_name=cv_name, n_splits=n_splits)
    models = build_models(random_state=random_state)

    summary_rows = []
    fold_rows = []
    prediction_rows = []

    for model_name, model in models.items():
        print(f"[模型] {model_name}")

        oof_pred = np.empty(len(df), dtype=object)
        oof_proba = np.full((len(df), n_classes), np.nan, dtype=float)

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, start=1):
            estimator = clone(model)

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            y_test_encoded = y_encoded[test_idx]

            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)

            # 预测概率并按照全局标签顺序对齐
            if hasattr(estimator, "predict_proba"):
                fold_proba_raw = estimator.predict_proba(X_test)
                fold_classes = estimator.classes_.tolist()
                fold_proba = np.zeros((len(test_idx), n_classes), dtype=float)
                for j, cls in enumerate(fold_classes):
                    global_j = labels.index(cls)
                    fold_proba[:, global_j] = fold_proba_raw[:, j]
            else:
                fold_proba = np.full((len(test_idx), n_classes), np.nan)

            oof_pred[test_idx] = pred
            oof_proba[test_idx] = fold_proba

            fold_metrics = evaluate_predictions(
                y_true=y_test,
                y_pred=pred,
                y_true_encoded=y_test_encoded,
                proba=fold_proba,
                labels=labels,
            )
            fold_metrics.update({
                "model": model_name,
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_actors": ",".join(map(str, sorted(set(groups[test_idx])))),
            })
            fold_rows.append(fold_metrics)

        summary = evaluate_predictions(
            y_true=y,
            y_pred=oof_pred,
            y_true_encoded=y_encoded,
            proba=oof_proba,
            labels=labels,
        )
        summary.update({
            "model": model_name,
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "n_classes": n_classes,
            "labels": ",".join(labels),
            "cv": cv_name,
        })

        cm = confusion_matrix(y, oof_pred, labels=labels)
        summary["confusion_matrix"] = json_dumps_matrix(cm.tolist())

        summary_rows.append(summary)

        for i in range(len(df)):
            row = {
                "sample_id": df.loc[i, "sample_id"] if "sample_id" in df.columns else i,
                "file_name": df.loc[i, "file_name"] if "file_name" in df.columns else "",
                "actor": df.loc[i, "actor"],
                "emotion": df.loc[i, "emotion"],
                "target": y[i],
                "model": model_name,
                "y_true": y[i],
                "y_pred": oof_pred[i],
            }
            for j, label in enumerate(labels):
                row[f"proba_{label}"] = oof_proba[i, j]
            prediction_rows.append(row)

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
    )


def json_dumps_matrix(matrix: list[list[int]]) -> str:
    """将混淆矩阵转换为字符串，避免额外依赖。"""
    import json
    return json.dumps(matrix, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train speaker-independent emotion classification baselines.")
    parser.add_argument("--features_csv", type=str, default="data/processed/ravdess_acoustic_features.csv")
    parser.add_argument("--out_dir", type=str, default="results/tables")
    parser.add_argument("--task", type=str, default="four_class", choices=["four_class", "neutral_vs_negative"])
    parser.add_argument("--cv", type=str, default="groupkfold", choices=["groupkfold", "logo"])
    parser.add_argument("--n_splits", type=int, default=6)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv 不存在：{features_csv}")

    df = pd.read_csv(features_csv)
    df = prepare_task(df, task=args.task)
    feature_cols = get_feature_columns(df)

    if not feature_cols:
        raise RuntimeError("没有识别到可用于训练的数值型声学特征列。")

    print(f"[信息] 任务：{args.task}")
    print(f"[信息] 样本数：{len(df)}")
    print(f"[信息] 特征数：{len(feature_cols)}")
    print("[信息] 类别分布：")
    print(df["target"].value_counts().to_string())
    print(f"[信息] 说话人数：{df['actor'].nunique()}")

    summary_df, fold_df, pred_df = train_and_evaluate(
        df=df,
        feature_cols=feature_cols,
        cv_name=args.cv,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    summary_path = out_dir / "classification_results.csv"
    fold_path = out_dir / "fold_metrics.csv"
    pred_path = out_dir / "overall_predictions.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"[完成] 分类汇总结果已保存：{summary_path}")
    print(f"[完成] 每折结果已保存：{fold_path}")
    print(f"[完成] 样本级预测结果已保存：{pred_path}")
    print("[信息] 分类结果摘要：")
    print(summary_df[["model", "accuracy", "balanced_accuracy", "macro_f1", "auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
