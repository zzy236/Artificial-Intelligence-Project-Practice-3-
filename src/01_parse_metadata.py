"""
01_parse_metadata.py

解析 RAVDESS 文件名，生成实验所需的元数据表。

RAVDESS 文件名格式：
Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav

例如：
03-01-05-01-02-02-12.wav

字段含义：
03 = audio-only
01 = speech
05 = angry
01 = normal intensity
02 = statement 2
02 = repetition 2
12 = actor 12
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


MODALITY_MAP = {
    "01": "full_av",
    "02": "video_only",
    "03": "audio_only",
}

VOCAL_CHANNEL_MAP = {
    "01": "speech",
    "02": "song",
}

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}

STATEMENT_MAP = {
    "01": "kids_are_talking_by_the_door",
    "02": "dogs_are_sitting_by_the_door",
}

REPETITION_MAP = {
    "01": "first",
    "02": "second",
}


def parse_ravdess_filename(file_path: Path) -> dict:
    """解析单个 RAVDESS wav 文件名。"""
    stem = file_path.stem
    parts = stem.split("-")

    if len(parts) != 7:
        raise ValueError(f"文件名字段数量不是 7 个，无法解析：{file_path.name}")

    modality_id, vocal_channel_id, emotion_id, intensity_id, statement_id, repetition_id, actor_id = parts

    if emotion_id not in EMOTION_MAP:
        raise ValueError(f"未知 emotion_id={emotion_id}: {file_path.name}")

    actor = int(actor_id)
    gender = "male" if actor % 2 == 1 else "female"

    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "modality_id": modality_id,
        "modality": MODALITY_MAP.get(modality_id, "unknown"),
        "vocal_channel_id": vocal_channel_id,
        "vocal_channel": VOCAL_CHANNEL_MAP.get(vocal_channel_id, "unknown"),
        "emotion_id": emotion_id,
        "emotion": EMOTION_MAP.get(emotion_id, "unknown"),
        "intensity_id": intensity_id,
        "intensity": INTENSITY_MAP.get(intensity_id, "unknown"),
        "statement_id": statement_id,
        "statement": STATEMENT_MAP.get(statement_id, "unknown"),
        "repetition_id": repetition_id,
        "repetition": REPETITION_MAP.get(repetition_id, "unknown"),
        "actor": actor,
        "gender": gender,
    }


def build_metadata(
    raw_dir: Path,
    emotions: set[str],
    modality: str = "audio_only",
    vocal_channel: str = "speech",
) -> pd.DataFrame:
    """递归搜索 wav 文件并生成元数据 DataFrame。"""
    wav_files = sorted(raw_dir.rglob("*.wav"))

    rows = []
    bad_files = []

    for wav_path in wav_files:
        try:
            row = parse_ravdess_filename(wav_path)
        except Exception as exc:
            bad_files.append((str(wav_path), str(exc)))
            continue

        if row["modality"] != modality:
            continue
        if row["vocal_channel"] != vocal_channel:
            continue
        if row["emotion"] not in emotions:
            continue

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "没有找到符合条件的 RAVDESS wav 文件。请检查 --raw_dir 是否正确，"
            "以及文件名是否保持 RAVDESS 原始命名格式。"
        )

    df = df.sort_values(["actor", "emotion_id", "intensity_id", "statement_id", "repetition_id"]).reset_index(drop=True)
    df.insert(0, "sample_id", range(len(df)))

    if bad_files:
        print(f"[警告] 有 {len(bad_files)} 个文件无法解析，已跳过。")
        for path, reason in bad_files[:5]:
            print(f"  - {path}: {reason}")
        if len(bad_files) > 5:
            print("  ...")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse RAVDESS metadata.")
    parser.add_argument("--raw_dir", type=str, default="data/raw/RAVDESS", help="RAVDESS wav 文件根目录。")
    parser.add_argument("--out_csv", type=str, default="data/processed/ravdess_metadata.csv", help="输出元数据 CSV。")
    parser.add_argument(
        "--emotions",
        type=str,
        nargs="+",
        default=["neutral", "happy", "sad", "angry"],
        help="需要保留的情绪类别。",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir 不存在：{raw_dir}")

    df = build_metadata(raw_dir=raw_dir, emotions=set(args.emotions))
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[完成] 元数据已保存：{out_csv}")
    print(f"[信息] 样本数：{len(df)}")
    print("[信息] 情绪分布：")
    print(df["emotion"].value_counts().to_string())
    print("[信息] 说话人数：", df["actor"].nunique())


if __name__ == "__main__":
    main()
