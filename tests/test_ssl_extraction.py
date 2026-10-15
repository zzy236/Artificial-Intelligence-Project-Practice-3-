"""校验样本对齐约束；如安装了 PyTorch，同时校验实际张量的 mean pooling。

运行：python -m unittest discover -s tests -v
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "02_extract_ssl_embeddings.py"
SPEC = importlib.util.spec_from_file_location("ssl_extraction", SCRIPT)
ssl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ssl)

try:
    import torch
except ImportError:
    torch = None


class SampleAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rows = []
        for index, emotion in enumerate(["neutral", "happy", "sad", "angry"]):
            audio = self.root / f"sample_{index}.wav"
            audio.touch()  # 数据校验只检查文件存在；不调用模型或音频读取。
            row = {col: "01" for col in ssl.META_COLUMNS}
            row.update(sample_id=str(index), file_name=audio.name, file_path=str(audio),
                       actor="1", emotion=emotion, vocal_channel="speech", modality="audio_only")
            rows.append(row)
        self.metadata = pd.DataFrame(rows)
        self.meta_path = self.root / "metadata.csv"
        self.ref_path = self.root / "reference.csv"
        self.metadata.to_csv(self.meta_path, index=False)
        self.metadata.to_csv(self.ref_path, index=False)

    def validate(self):
        return ssl.validate_metadata(self.meta_path, self.ref_path, expected_samples=4)

    def test_preserves_sample_order_and_identifiers(self):
        actual, paths = self.validate()
        pd.testing.assert_frame_equal(actual, self.metadata)
        self.assertEqual(paths, [Path(value) for value in self.metadata.file_path])

    def test_rejects_reordered_reference(self):
        self.metadata.iloc[::-1].to_csv(self.ref_path, index=False)
        with self.assertRaisesRegex(ValueError, "顺序不一致"):
            self.validate()

    def test_rejects_reference_label_change(self):
        changed = self.metadata.copy()
        changed.loc[0, "emotion"] = "sad"
        changed.to_csv(self.ref_path, index=False)
        with self.assertRaisesRegex(ValueError, "不一致"):
            self.validate()

    def test_rejects_duplicate_sample_id(self):
        self.metadata.loc[1, "sample_id"] = "0"
        self.metadata.to_csv(self.meta_path, index=False)
        with self.assertRaisesRegex(ValueError, "重复"):
            self.validate()

    def test_rejects_song(self):
        self.metadata.loc[0, "vocal_channel"] = "song"
        self.metadata.to_csv(self.meta_path, index=False)
        with self.assertRaisesRegex(ValueError, "非 speech"):
            self.validate()

    def test_rejects_missing_audio(self):
        Path(self.metadata.loc[0, "file_path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "音频不存在"):
            self.validate()

    def test_rejects_wrong_sample_count(self):
        with self.assertRaisesRegex(ValueError, "672"):
            ssl.validate_metadata(self.meta_path, self.ref_path)


@unittest.skipIf(torch is None, "未安装 PyTorch，不能验证实际模型张量池化")
class MeanPoolingTests(unittest.TestCase):
    def test_averages_time_and_preserves_embedding_dimension(self):
        values = torch.stack([torch.ones(768), torch.full((768,), 3.0)]).unsqueeze(0)
        actual = ssl.mean_pool_hidden_state(values)
        np.testing.assert_array_equal(actual, np.full(768, 2.0, dtype=np.float32))
        self.assertEqual(actual.dtype, np.float32)

    def test_rejects_batch_or_wrong_dimension(self):
        for shape in [(2, 3, 768), (1, 3, 512), (1, 0, 768)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                ssl.mean_pool_hidden_state(torch.empty(shape))

    def test_rejects_nonfinite_embedding(self):
        with self.assertRaisesRegex(ValueError, "非有限"):
            ssl.mean_pool_hidden_state(torch.full((1, 2, 768), float("nan")))


if __name__ == "__main__":
    unittest.main()
