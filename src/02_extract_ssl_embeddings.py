"""为已有 672 条 speech 样本提取冻结的自监督语音表示。

默认模型：facebook/wav2vec2-base、facebook/hubert-base-ls960。
每条音频单独推理，对 last_hidden_state 的时间轴取均值，得到 768 维向量。
单条推理不补零，因此不存在把 padding 帧计入 mean pooling 的问题。
仅提取特征；不使用标签训练、不做跨样本标准化，也不生成交叉验证划分。

运行（从任意工作目录均可）：
    python src/02_extract_ssl_embeddings.py --validate_only
    python src/02_extract_ssl_embeddings.py
    python src/02_extract_ssl_embeddings.py --limit 2 --out_dir results/ssl_smoke

依赖：pip install -r requirements.txt
首次运行会下载模型；--local_files_only 可使用已缓存或本地模型。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = {
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
}
OUTPUT_NAMES = {
    "wav2vec2": "ravdess_wav2vec2_base_embeddings.csv",
    "hubert": "ravdess_hubert_base_embeddings.csv",
}
META_COLUMNS = [
    "sample_id", "file_name", "file_path", "modality_id", "modality",
    "vocal_channel_id", "vocal_channel", "emotion_id", "emotion",
    "intensity_id", "intensity", "statement_id", "statement",
    "repetition_id", "repetition", "actor", "gender",
]
ALIGN_COLUMNS = ["sample_id", "file_name", "actor", "emotion"]
SAMPLE_RATE = 16000
EMBEDDING_DIM = 768


def project_path(value: str | Path) -> Path:
    """相对路径统一以项目根目录为基准，不依赖启动目录。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_metadata(metadata_csv: Path, reference_csv: Path,
                      expected_samples: int = 672) -> tuple[pd.DataFrame, list[Path]]:
    """校验完整样本集及顺序；不重新扫描或筛选原始音频。"""
    metadata = pd.read_csv(metadata_csv, dtype={col: str for col in META_COLUMNS})
    missing = set(META_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValueError(f"元数据缺少列：{sorted(missing)}")
    metadata = metadata[META_COLUMNS].copy()
    if len(metadata) != expected_samples:
        raise ValueError(f"应有 {expected_samples} 条样本，实际为 {len(metadata)} 条。")
    if metadata[META_COLUMNS].isna().any().any():
        raise ValueError("元数据存在缺失字段。")
    if metadata["sample_id"].duplicated().any() or metadata["file_name"].duplicated().any():
        raise ValueError("sample_id 或 file_name 重复。")
    if not metadata["vocal_channel"].eq("speech").all():
        raise ValueError("存在非 speech 样本。")
    if not metadata["modality"].eq("audio_only").all():
        raise ValueError("存在非 audio_only 样本。")
    if set(metadata["emotion"]) != {"neutral", "happy", "sad", "angry"}:
        raise ValueError("元数据应恰好包含 neutral/happy/sad/angry 四类。")

    reference = pd.read_csv(reference_csv, usecols=ALIGN_COLUMNS, dtype=str)
    if not metadata[ALIGN_COLUMNS].reset_index(drop=True).equals(reference[ALIGN_COLUMNS]):
        raise ValueError("元数据与原有传统特征表的样本标识、标签或顺序不一致。")

    audio_paths = [project_path(value) for value in metadata["file_path"]]
    missing_paths = [str(path) for path in audio_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"有 {len(missing_paths)} 条音频不存在，例如：{missing_paths[:3]}")
    return metadata, audio_paths


def load_dependencies():
    """延迟导入，让 --help 和 --validate_only 无需模型推理依赖。"""
    try:
        import librosa
        import torch
        import transformers
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "缺少 SSL 推理依赖，请执行：python -m pip install -r requirements.txt"
        ) from exc
    return librosa, torch, transformers, tqdm


def load_audio(path: Path, librosa, trim: bool, top_db: float) -> np.ndarray:
    """与传统特征脚本相同：16 kHz、单声道、可选首尾静音裁剪。"""
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"音频为空或含非有限值：{path}")
    if trim:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        if trimmed.size > 0:
            audio = trimmed
    return np.asarray(audio, dtype=np.float32)


def mean_pool_hidden_state(hidden_state) -> np.ndarray:
    """输入 [1, T, 768]，只对实际时间帧求平均，输出 float32 [768]。"""
    if hidden_state.ndim != 3 or hidden_state.shape[0] != 1:
        raise ValueError(f"单条语音的 hidden_state 形状异常：{hidden_state.shape}")
    if hidden_state.shape[1] < 1 or hidden_state.shape[2] != EMBEDDING_DIM:
        raise ValueError(f"预期非空的 768 维表示，实际为：{hidden_state.shape}")
    embedding = hidden_state.mean(dim=1)[0].detach().float().cpu().numpy()
    if not np.isfinite(embedding).all():
        raise ValueError("模型产生非有限 embedding，终止提取。")
    return embedding.astype(np.float32, copy=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_model_embeddings(metadata: pd.DataFrame, audio_paths: list[Path],
                             model_name: str, args, dependencies) -> tuple[pd.DataFrame, dict]:
    librosa, torch, transformers, tqdm = dependencies
    model_id = getattr(args, f"{model_name}_model")
    revision = getattr(args, f"{model_name}_revision")
    load_kwargs = {
        "cache_dir": str(project_path(args.cache_dir)),
        "local_files_only": args.local_files_only,
        "revision": revision,
    }
    extractor = transformers.AutoFeatureExtractor.from_pretrained(model_id, **load_kwargs)
    model_class = (transformers.Wav2Vec2Model if model_name == "wav2vec2"
                   else transformers.HubertModel)
    model = model_class.from_pretrained(model_id, **load_kwargs)
    if model.config.hidden_size != EMBEDDING_DIM:
        raise ValueError(f"要求 Base 模型的 768 维表示，实际为 {model.config.hidden_size} 维。")
    if extractor.sampling_rate != SAMPLE_RATE:
        raise ValueError(f"模型采样率必须为 {SAMPLE_RATE} Hz。")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无可用 CUDA 设备。")
    torch.manual_seed(42)
    model.to(device).eval()
    model.requires_grad_(False)

    # 卷积前端的最小输入长度（这两个 Base 模型均为 400 个采样点）。
    min_samples, stride = 1, 1
    for kernel, layer_stride in zip(model.config.conv_kernel, model.config.conv_stride):
        min_samples += (kernel - 1) * stride
        stride *= layer_stride

    print(f"[模型] {model_name}: {model_id}，device={device}，逐条推理，无 padding")
    embeddings = []
    with torch.inference_mode():
        for index, path in enumerate(tqdm(audio_paths, desc=f"Extracting {model_name}")):
            try:
                audio = load_audio(path, librosa, trim=not args.no_trim, top_db=args.top_db)
                if len(audio) < min_samples:
                    raise ValueError(f"音频不足 {min_samples} 个采样点。")
                inputs = extractor(
                    audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                    padding=False, return_attention_mask=False,
                )
                # 两个 Base 模型使用 group norm；单条无 padding 输入无需 attention_mask。
                output = model(input_values=inputs["input_values"].to(device), return_dict=True)
                embeddings.append(mean_pool_hidden_state(output.last_hidden_state))
            except Exception as exc:
                raise RuntimeError(
                    f"提取失败：model={model_name}, sample_id={metadata.iloc[index]['sample_id']}, "
                    f"file={path}。未跳过样本，未写出本模型的不完整 CSV。"
                ) from exc

    matrix = np.stack(embeddings)
    columns = [f"{model_name}_emb_{index:04d}" for index in range(1, EMBEDDING_DIM + 1)]
    result = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(matrix, columns=columns)], axis=1)
    manifest = {
        "model": model_name,
        "model_id_or_path": model_id,
        "requested_revision": revision,
        "resolved_commit": getattr(model.config, "_commit_hash", None),
        "n_samples": len(metadata),
        "embedding_dim": EMBEDDING_DIM,
        "representation": "last_hidden_state",
        "pooling": "mean over time; single utterance; no padding",
        "sampling_rate": SAMPLE_RATE,
        "mono": True,
        "trim_silence": not args.no_trim,
        "trim_top_db": args.top_db,
        "waveform_normalization": bool(extractor.do_normalize),
        "dataset_feature_standardization": False,
        "frozen_model": True,
        "device": device,
        "metadata_csv": str(project_path(args.metadata_csv)),
        "metadata_sha256": file_sha256(project_path(args.metadata_csv)),
        "reference_features_csv": str(project_path(args.reference_features_csv)),
        "reference_features_sha256": file_sha256(project_path(args.reference_features_csv)),
        "sample_ids_in_order": metadata["sample_id"].tolist(),
        "debug_limit": args.limit,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "librosa": librosa.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata_csv", default="data/processed/ravdess_metadata.csv")
    parser.add_argument("--reference_features_csv", default="data/processed/ravdess_acoustic_features.csv")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_IDS), default=list(MODEL_IDS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cache_dir", default=".cache/huggingface")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_trim", action="store_true", help="默认与传统特征一致，裁剪首尾静音。")
    parser.add_argument("--top_db", type=float, default=30)
    parser.add_argument("--expected_samples", type=int, default=672)
    parser.add_argument("--limit", type=int, help="仅调试：校验完整样本集后提取前 N 条。")
    parser.add_argument("--validate_only", action="store_true", help="只检查样本、顺序和音频路径，不加载模型。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 SSL 输出文件。")
    for name, model_id in MODEL_IDS.items():
        parser.add_argument(f"--{name}_model", default=model_id, help="Hugging Face 模型 ID 或本地模型目录。")
        parser.add_argument(f"--{name}_revision", default="main", help="可指定 commit hash 固定模型版本。")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.expected_samples < 1 or (args.limit is not None and args.limit < 1) or args.top_db < 0:
        parser.error("expected_samples/limit 必须为正数，top_db 不能为负数。")
    metadata, audio_paths = validate_metadata(
        project_path(args.metadata_csv), project_path(args.reference_features_csv), args.expected_samples,
    )
    print(f"[校验通过] {len(metadata)} 条 speech，{metadata['actor'].nunique()} 位说话人；"
          "与传统特征表的样本标识、标签、顺序完全一致，音频文件均存在。")
    print(metadata["emotion"].value_counts().to_string())
    if args.validate_only:
        return
    if args.limit is not None:
        metadata = metadata.head(args.limit).copy()
        audio_paths = audio_paths[:args.limit]
        print(f"[调试] 仅提取前 {len(metadata)} 条，不能作为完整实验数据。")
    out_dir = project_path(args.out_dir)
    selected_models = list(dict.fromkeys(args.models))
    for name in selected_models:
        output = out_dir / OUTPUT_NAMES[name]
        for path in [output, output.with_suffix(".json")]:
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"输出已存在：{path}；换用 --out_dir 或显式传入 --overwrite。")
    dependencies = load_dependencies()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in selected_models:
        result, manifest = extract_model_embeddings(metadata, audio_paths, name, args, dependencies)
        output = out_dir / OUTPUT_NAMES[name]
        temporary = output.with_suffix(".csv.tmp")
        result.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.9g")
        temporary.replace(output)
        manifest_path = output.with_suffix(".json")
        manifest["output_csv_sha256"] = file_sha256(output)
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_manifest.replace(manifest_path)
        print(f"[完成] {output}：{len(result)} 条 × {EMBEDDING_DIM} 维（另含元数据列）")
        print(f"[记录] {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"[原因] {exc.__cause__}", file=sys.stderr)
        sys.exit(1)
