"""
04_train_baseline_models.py

构建 RAVDESS 情绪分类基线，支持传统特征与冻结 SSL embedding 的公平对比。

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIGN_COLUMNS = ["sample_id", "file_name", "actor", "emotion"]
FEATURE_TYPES = ["traditional", "wav2vec2", "hubert", "egemaps"]


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
    "target",
    "feature_type",
    "fold",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """识别用于建模的数值特征列。"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in META_COLUMNS]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def align_feature_table(reference: pd.DataFrame, candidate: pd.DataFrame,
                        feature_type: str) -> pd.DataFrame:
    """按 sample_id 对齐，并拒绝样本缺失、重复或标签/说话人冲突。"""
    for name, table in [("reference", reference), (feature_type, candidate)]:
        missing = set(ALIGN_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f"{name} 缺少样本对齐列：{sorted(missing)}")
        if table[ALIGN_COLUMNS].isna().any().any() or table["sample_id"].duplicated().any():
            raise ValueError(f"{name} 样本标识缺失或 sample_id 重复。")
    if set(reference["sample_id"]) != set(candidate["sample_id"]):
        raise ValueError(f"{feature_type} 与传统特征表的样本集合不同，不能公平比较。")
    aligned = candidate.set_index("sample_id", drop=False).loc[reference["sample_id"]].reset_index(drop=True)
    if not reference[ALIGN_COLUMNS].reset_index(drop=True).equals(aligned[ALIGN_COLUMNS]):
        raise ValueError(f"{feature_type} 的文件名、说话人或情绪标签与传统特征不一致。")
    return aligned


def select_feature_columns(df: pd.DataFrame, feature_type: str) -> list[str]:
    columns = get_feature_columns(df)
    if feature_type in {"wav2vec2", "hubert"}:
        expected = [f"{feature_type}_emb_{i:04d}" for i in range(1, 769)]
        if any(col not in columns for col in expected):
            raise ValueError(f"{feature_type} 需要完整的 768 维数值 embedding。")
        columns = expected
    elif feature_type == "egemaps" and len(columns) != 88:
        raise ValueError(f"eGeMAPS Functionals 应有 88 维特征，实际为 {len(columns)}。")
    if not columns:
        raise ValueError(f"{feature_type} 没有可用于训练的数值特征。")
    return columns


def validate_cv_splits(df: pd.DataFrame, splits) -> np.ndarray:
    """确保每条样本只测试一次、训练/测试说话人不交叉、训练包含全部类别。"""
    counts = np.zeros(len(df), dtype=int)
    assignments = np.zeros(len(df), dtype=int)
    all_labels = set(df["target"])
    for fold, (train, test) in enumerate(splits, start=1):
        if not len(train) or not len(test):
            raise ValueError("交叉验证中存在空训练集或测试集。")
        if np.intersect1d(train, test).size:
            raise ValueError("训练样本与测试样本发生交叉。")
        if set(df.iloc[train]["actor"]) & set(df.iloc[test]["actor"]):
            raise ValueError("训练说话人与测试说话人发生交叉。")
        if set(df.iloc[train]["target"]) != all_labels:
            raise ValueError(f"第 {fold} 折训练集缺少类别。")
        counts[test] += 1
        assignments[test] = fold
    if not np.all(counts == 1):
        raise ValueError("每条样本必须恰好作为测试样本一次。")
    return assignments


def restore_groupkfold_protocol(df: pd.DataFrame, fold_metrics_csv: Path,
                                n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """复用原实验的测试说话人列表，避免库版本导致等大小分组的排序变化。"""
    stored = pd.read_csv(fold_metrics_csv)
    if not {"fold", "test_actors"}.issubset(stored.columns):
        raise ValueError("折协议文件必须包含 fold、test_actors 列（fold_metrics.csv）。")
    if stored[["fold", "test_actors"]].isna().any().any():
        raise ValueError("折协议中有缺失的折号或测试说话人。")
    if set(stored["fold"]) != set(range(1, n_splits + 1)):
        raise ValueError(f"折协议必须包含连续的 {n_splits} 折。")
    actors = df["actor"].astype(str).to_numpy()
    all_actors = set(actors)
    splits = []
    for fold in range(1, n_splits + 1):
        actor_lists = {
            frozenset(part.strip() for part in str(value).split(","))
            for value in stored.loc[stored["fold"] == fold, "test_actors"]
        }
        if len(actor_lists) != 1:
            raise ValueError(f"折协议中第 {fold} 折的测试说话人列表不一致。")
        test_actors = set(next(iter(actor_lists)))
        if not test_actors or not test_actors.issubset(all_actors):
            raise ValueError(f"折协议中第 {fold} 折包含当前数据不存在的说话人。")
        mask = np.isin(actors, list(test_actors))
        splits.append((np.flatnonzero(~mask), np.flatnonzero(mask)))
    validate_cv_splits(df, splits)
    return splits


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
    feature_type: str = "traditional",
    cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """训练并评估所有模型。"""
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df["target"].astype(str).values
    groups = df["actor"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    labels = label_encoder.classes_.tolist()
    n_classes = len(labels)

    if cv_splits is None:
        cv_splits = get_cv(groups=groups, cv_name=cv_name, n_splits=n_splits)
    assignments = validate_cv_splits(df, cv_splits)
    models = build_models(random_state=random_state)

    summary_rows = []
    fold_rows = []
    prediction_rows = []

    for model_name, model in models.items():
        print(f"[模型] {feature_type} × {model_name}", flush=True)

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
                "feature_type": feature_type,
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
            "feature_type": feature_type,
            "n_splits": len(cv_splits),
            "random_state": random_state,
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
                "feature_type": feature_type,
                "fold": int(assignments[i]),
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
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--compare_features", action="store_true", help="对比 traditional/wav2vec2/hubert，复用同一组 CV 划分。")
    parser.add_argument("--feature_sets", nargs="+", choices=FEATURE_TYPES, help="显式选择需要训练的特征类型。")
    parser.add_argument("--wav2vec2_csv", default="data/processed/ravdess_wav2vec2_base_embeddings.csv")
    parser.add_argument("--hubert_csv", default="data/processed/ravdess_hubert_base_embeddings.csv")
    parser.add_argument("--egemaps_csv", help="可选的 88 维 eGeMAPS Functionals CSV，需带样本对齐列。")
    parser.add_argument("--fold_protocol_csv", help="复用已有 fold_metrics.csv 中的说话人划分；比较模式六折默认复用原结果。")
    parser.add_argument("--task", type=str, default="four_class", choices=["four_class", "neutral_vs_negative"])
    parser.add_argument("--cv", type=str, default="groupkfold", choices=["groupkfold", "logo"])
    parser.add_argument("--n_splits", type=int, default=6)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    selected = list(dict.fromkeys(args.feature_sets or (
        ["traditional", "wav2vec2", "hubert"] if args.compare_features else ["traditional"]
    )))
    comparison = args.compare_features or args.feature_sets is not None
    out_dir = project_path(args.out_dir or (
        "results/tables/feature_comparison" if comparison else "results/tables"
    ))
    paths = {"traditional": args.features_csv, "wav2vec2": args.wav2vec2_csv,
             "hubert": args.hubert_csv, "egemaps": args.egemaps_csv}
    reference = pd.read_csv(project_path(args.features_csv))
    # 在训练任何模型之前校验所有输入；统一对齐之后再按 task 筛选。
    tables = {}
    for feature_type in selected:
        if not paths[feature_type]:
            parser.error(f"选择 {feature_type} 必须提供 --{feature_type}_csv。")
        candidate = (reference.copy() if feature_type == "traditional"
                     else pd.read_csv(project_path(paths[feature_type])))
        candidate = align_feature_table(reference, candidate, feature_type)
        tables[feature_type] = prepare_task(candidate, task=args.task)
        select_feature_columns(tables[feature_type], feature_type)

    canonical = prepare_task(reference, task=args.task)
    if canonical.empty or canonical["target"].nunique() < 2:
        raise ValueError("当前任务至少需要两个类别及非空样本。")
    # 只生成一次划分，全部特征类型及分类器共享同一组训练/测试索引。
    protocol_path = project_path(args.fold_protocol_csv) if args.fold_protocol_csv else None
    original_protocol = project_path("results/tables/fold_metrics.csv")
    if protocol_path is None and comparison and args.cv == "groupkfold" and args.n_splits == 6 and original_protocol.exists():
        protocol_path = original_protocol
    if protocol_path is not None:
        if args.cv != "groupkfold":
            parser.error("--fold_protocol_csv 只适用于 groupkfold。")
        cv_splits = restore_groupkfold_protocol(canonical, protocol_path, args.n_splits)
        print(f"[折协议] 复用已有 GroupKFold 的说话人划分：{protocol_path}", flush=True)
    else:
        cv_splits = get_cv(canonical["actor"].to_numpy(), args.cv, args.n_splits)
    assignments = validate_cv_splits(canonical, cv_splits)
    print(f"[协议] task={args.task}, cv={args.cv}, folds={len(cv_splits)}, "
          f"samples={len(canonical)}, actors={canonical['actor'].nunique()}", flush=True)
    outputs = []
    for feature_type, df in tables.items():
        feature_cols = select_feature_columns(df, feature_type)
        print(f"[特征] {feature_type}: {len(feature_cols)} 维", flush=True)
        outputs.append(train_and_evaluate(
            df, feature_cols, args.cv, args.n_splits, args.random_state,
            feature_type=feature_type, cv_splits=cv_splits,
        ))
    summary_df, fold_df, pred_df = (
        pd.concat([output[i] for output in outputs], ignore_index=True) for i in range(3)
    )
    for table in [summary_df, fold_df, pred_df]:
        table["task"] = args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    split_df = canonical[ALIGN_COLUMNS + ["target"]].copy()
    split_df["fold"] = assignments
    split_df.to_csv(out_dir / "cv_assignments.csv", index=False, encoding="utf-8-sig")

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
    print(summary_df[["feature_type", "model", "accuracy", "balanced_accuracy", "macro_f1", "auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
