"""
02_extract_features.py

从 RAVDESS 音频中提取传统声学特征。

默认提取 librosa 特征：
1. 时长特征
2. F0 基频统计量
3. RMS 能量统计量
4. 过零率
5. 频谱质心、频谱带宽、频谱滚降、频谱平坦度
6. MFCC 及其一阶/二阶差分统计量

可选提取 Praat/Parselmouth 特征：
1. Jitter
2. Shimmer
3. HNR
4. Formant F1/F2/F3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


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
}


def safe_stat(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """对一组数值计算常用统计量。空数组或全 NaN 时返回 NaN。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_median": np.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_median": float(np.median(values)),
    }


def load_audio(
    file_path: str | Path,
    sr: int = 16000,
    trim: bool = True,
    top_db: int = 30,
) -> tuple[np.ndarray, int]:
    """读取音频，统一采样率、单声道，并可选去除首尾静音。"""
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    if trim:
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        if len(y_trimmed) > 0:
            y = y_trimmed

    return y.astype(np.float32), sr


def extract_librosa_features(
    file_path: str | Path,
    sr: int = 16000,
    trim: bool = True,
    top_db: int = 30,
    n_mfcc: int = 13,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Dict[str, Any]:
    """使用 librosa 提取传统声学特征。"""
    features: Dict[str, Any] = {}

    try:
        y, sr = load_audio(file_path, sr=sr, trim=trim, top_db=top_db)
    except Exception as exc:
        warnings.warn(f"读取音频失败：{file_path}，原因：{exc}")
        features["load_error"] = 1
        return features

    features["load_error"] = 0
    features["duration"] = float(librosa.get_duration(y=y, sr=sr))
    features["n_samples"] = int(len(y))

    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)), mode="constant")

    # F0 / pitch
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        features.update(safe_stat(f0, "f0"))
        features["voiced_ratio"] = float(np.mean(voiced_flag)) if voiced_flag is not None else np.nan
        features["voiced_prob_mean"] = float(np.nanmean(voiced_prob)) if voiced_prob is not None else np.nan
    except Exception as exc:
        warnings.warn(f"F0 提取失败：{file_path}，原因：{exc}")
        for key, value in safe_stat(np.array([]), "f0").items():
            features[key] = value
        features["voiced_ratio"] = np.nan
        features["voiced_prob_mean"] = np.nan

    # RMS energy
    try:
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        features.update(safe_stat(rms, "rms"))
    except Exception:
        features.update(safe_stat(np.array([]), "rms"))

    # Zero crossing rate
    try:
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        features.update(safe_stat(zcr, "zcr"))
    except Exception:
        features.update(safe_stat(np.array([]), "zcr"))

    # Spectral features
    spectral_extractors = {
        "spectral_centroid": lambda: librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0],
        "spectral_bandwidth": lambda: librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0],
        "spectral_rolloff": lambda: librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0],
        "spectral_flatness": lambda: librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0],
    }

    for name, extractor in spectral_extractors.items():
        try:
            values = extractor()
            features.update(safe_stat(values, name))
        except Exception:
            features.update(safe_stat(np.array([]), name))

    # MFCC + delta + delta2
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        for i in range(n_mfcc):
            idx = i + 1
            features[f"mfcc_{idx}_mean"] = float(np.mean(mfcc[i]))
            features[f"mfcc_{idx}_std"] = float(np.std(mfcc[i], ddof=1)) if mfcc.shape[1] > 1 else 0.0

            features[f"delta_mfcc_{idx}_mean"] = float(np.mean(delta[i]))
            features[f"delta_mfcc_{idx}_std"] = float(np.std(delta[i], ddof=1)) if delta.shape[1] > 1 else 0.0

            features[f"delta2_mfcc_{idx}_mean"] = float(np.mean(delta2[i]))
            features[f"delta2_mfcc_{idx}_std"] = float(np.std(delta2[i], ddof=1)) if delta2.shape[1] > 1 else 0.0
    except Exception as exc:
        warnings.warn(f"MFCC 提取失败：{file_path}，原因：{exc}")
        for i in range(n_mfcc):
            idx = i + 1
            for prefix in ["mfcc", "delta_mfcc", "delta2_mfcc"]:
                features[f"{prefix}_{idx}_mean"] = np.nan
                features[f"{prefix}_{idx}_std"] = np.nan

    # 简化语速代理变量：有声帧数 / 总时长
    if features.get("duration", 0) and np.isfinite(features.get("duration", np.nan)):
        features["voiced_frames_per_second"] = features.get("voiced_ratio", np.nan) / features["duration"]
    else:
        features["voiced_frames_per_second"] = np.nan

    return features


def extract_praat_features(
    file_path: str | Path,
    pitch_floor: float = 75,
    pitch_ceiling: float = 600,
) -> Dict[str, Any]:
    """使用 Parselmouth 调用 Praat 提取 Jitter、Shimmer、HNR、Formant。"""
    features: Dict[str, Any] = {
        "jitter_local": np.nan,
        "shimmer_local": np.nan,
        "hnr_mean": np.nan,
        "formant_f1_mean": np.nan,
        "formant_f2_mean": np.nan,
        "formant_f3_mean": np.nan,
    }

    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        warnings.warn("未安装 praat-parselmouth，跳过 Praat 特征。")
        return features

    try:
        snd = parselmouth.Sound(str(file_path))

        # Jitter / Shimmer
        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
            features["jitter_local"] = float(
                call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            )
            features["shimmer_local"] = float(
                call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            )
        except Exception as exc:
            warnings.warn(f"Jitter/Shimmer 提取失败：{file_path}，原因：{exc}")

        # HNR
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 1.0)
            features["hnr_mean"] = float(call(harmonicity, "Get mean", 0, 0))
        except Exception as exc:
            warnings.warn(f"HNR 提取失败：{file_path}，原因：{exc}")

        # Formants
        try:
            # 男女性混合时 max_formant 取 5500 Hz 作为通用基线设置。
            formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            for formant_number in [1, 2, 3]:
                try:
                    value = call(formant, "Get mean", formant_number, 0, 0, "Hertz")
                    features[f"formant_f{formant_number}_mean"] = float(value)
                except Exception:
                    features[f"formant_f{formant_number}_mean"] = np.nan
        except Exception as exc:
            warnings.warn(f"Formant 提取失败：{file_path}，原因：{exc}")

    except Exception as exc:
        warnings.warn(f"Praat 特征整体提取失败：{file_path}，原因：{exc}")

    return features


def extract_features_for_dataframe(
    metadata_df: pd.DataFrame,
    sr: int = 16000,
    trim: bool = True,
    top_db: int = 30,
    n_mfcc: int = 13,
    use_praat: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """为 metadata_df 中的每个音频提取特征。"""
    if limit is not None:
        metadata_df = metadata_df.head(limit).copy()

    rows = []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting features"):
        file_path = row["file_path"]

        feature_row = row.to_dict()
        librosa_features = extract_librosa_features(
            file_path=file_path,
            sr=sr,
            trim=trim,
            top_db=top_db,
            n_mfcc=n_mfcc,
        )
        feature_row.update(librosa_features)

        if use_praat:
            praat_features = extract_praat_features(file_path=file_path)
            feature_row.update(praat_features)

        rows.append(feature_row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract acoustic features from RAVDESS wav files.")
    parser.add_argument("--metadata_csv", type=str, default="data/processed/ravdess_metadata.csv")
    parser.add_argument("--out_csv", type=str, default="data/processed/ravdess_acoustic_features.csv")
    parser.add_argument("--sr", type=int, default=16000, help="目标采样率。")
    parser.add_argument("--n_mfcc", type=int, default=13)
    parser.add_argument("--top_db", type=int, default=30)
    parser.add_argument("--no_trim", action="store_true", help="不去除首尾静音。")
    parser.add_argument("--use_praat", action="store_true", help="是否额外提取 Praat/Parselmouth 特征。")
    parser.add_argument("--limit", type=int, default=None, help="调试用，只处理前 N 条样本。")
    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata_csv 不存在：{metadata_csv}")

    metadata_df = pd.read_csv(metadata_csv)
    feature_df = extract_features_for_dataframe(
        metadata_df=metadata_df,
        sr=args.sr,
        trim=not args.no_trim,
        top_db=args.top_db,
        n_mfcc=args.n_mfcc,
        use_praat=args.use_praat,
        limit=args.limit,
    )

    feature_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[完成] 特征表已保存：{out_csv}")
    print(f"[信息] 样本数：{len(feature_df)}")
    print(f"[信息] 特征列数：{len([c for c in feature_df.columns if c not in META_COLUMNS])}")


if __name__ == "__main__":
    main()
