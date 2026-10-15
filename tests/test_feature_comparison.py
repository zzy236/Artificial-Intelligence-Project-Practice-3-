"""比较实验的关键约束：样本对齐、说话人隔离、概率列及图表组合独立性。"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load_script("baseline_train", ROOT / "src/04_train_baseline_models.py")
visualize = load_script("baseline_visualize", ROOT / "src/05_visualize_results.py")


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "sample_id": [0, 1, 2, 3], "file_name": ["a.wav", "b.wav", "c.wav", "d.wav"],
            "actor": [1, 1, 2, 2], "emotion": ["sad", "neutral", "sad", "neutral"],
            "value": [1.0, 2.0, 3.0, 4.0],
        })

    def test_reordered_embeddings_are_aligned_by_identity(self):
        other = self.df.iloc[[2, 0, 3, 1]].copy()
        other["value"] += 10
        actual = train.align_feature_table(self.df, other, "hubert")
        self.assertEqual(actual.sample_id.tolist(), [0, 1, 2, 3])
        self.assertEqual(actual.value.tolist(), [11, 12, 13, 14])

    def test_rejects_dropped_duplicate_or_relabeled_samples(self):
        relabeled = self.df.copy()
        relabeled.loc[0, "emotion"] = "angry"
        for other in [self.df.iloc[:-1], pd.concat([self.df, self.df.iloc[:1]]), relabeled]:
            with self.subTest(rows=len(other)), self.assertRaises(ValueError):
                train.align_feature_table(self.df, other, "hubert")

    def test_rejects_speaker_leakage_and_repeated_test_samples(self):
        df = train.prepare_task(self.df, "four_class")
        leaking = [(np.array([0, 2]), np.array([1, 3])), (np.array([1, 3]), np.array([0, 2]))]
        repeated = [(np.array([0, 1]), np.array([2, 3]))] * 2
        for splits in [leaking, repeated]:
            with self.subTest(splits=splits), self.assertRaises(ValueError):
                train.validate_cv_splits(df, splits)

    def test_neutral_negative_task_and_feature_columns_exclude_labels(self):
        df = train.prepare_task(self.df, "neutral_vs_negative")
        self.assertEqual(set(df.target), {"neutral", "negative"})
        df["target"] = [0, 1, 0, 1]
        self.assertEqual(train.get_feature_columns(df), ["value"])


class EvaluationTests(unittest.TestCase):
    def test_restores_saved_speaker_protocol_exactly(self):
        df = pd.DataFrame({"actor": [1, 1, 2, 2, 3, 3, 4, 4],
                           "target": ["sad", "neutral"] * 4})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold_metrics.csv"
            pd.DataFrame({"fold": [1, 2, 1, 2], "test_actors": ["1,3", "2,4", "1,3", "2,4"]}).to_csv(path, index=False)
            splits = train.restore_groupkfold_protocol(df, path, 2)
            np.testing.assert_array_equal(splits[0][1], [0, 1, 4, 5])
            np.testing.assert_array_equal(splits[1][1], [2, 3, 6, 7])
            pd.DataFrame({"fold": [1, 2, 1], "test_actors": ["1,3", "2,4", "1,4"]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                train.restore_groupkfold_protocol(df, path, 2)

    def test_multiclass_auc_is_macro_ovr(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        proba = np.array([[.8, .1, .1], [.2, .6, .2], [.2, .1, .7],
                          [.4, .5, .1], [.1, .7, .2], [.2, .6, .2]])
        self.assertAlmostEqual(train.safe_auc_score(y, proba, 3),
                               roc_auc_score(y, proba, multi_class="ovr", average="macro"))

    def test_shared_splits_reach_predictions_for_both_feature_types(self):
        rows = []
        for actor in range(1, 7):
            for emotion, value in [("neutral", -1.0), ("sad", 1.0)]:
                rows.append({"sample_id": len(rows), "file_name": f"{actor}_{emotion}.wav",
                             "actor": actor, "emotion": emotion, "value": value + actor * .01})
        df = train.prepare_task(pd.DataFrame(rows), "four_class")
        splits = train.get_cv(df.actor.to_numpy(), "groupkfold", 3)
        lr = {"LogisticRegression": train.build_models()["LogisticRegression"]}
        predictions = []
        with patch.object(train, "build_models", return_value=lr):
            for feature in ["traditional", "hubert"]:
                summary, folds, pred = train.train_and_evaluate(
                    df, ["value"], "groupkfold", 3, 42, feature, splits)
                self.assertEqual(summary.loc[0, "feature_type"], feature)
                self.assertEqual(summary.loc[0, "accuracy"], 1.0)
                self.assertEqual(len(folds), 3)
                self.assertEqual(len(pred), len(df))
                self.assertEqual(set(pred.proba_neutral + pred.proba_sad), {1.0})
                predictions.append(pred)
        pd.testing.assert_frame_equal(predictions[0][["sample_id", "fold"]],
                                      predictions[1][["sample_id", "fold"]])


class VisualizationTests(unittest.TestCase):
    def summary(self):
        return pd.DataFrame([
            {"feature_type": feature, "model": model, "accuracy": .6 + i * .1,
             "macro_f1": .5 + i * .1, "auc": .7 + i * .1, "n_features": 124 if i == 0 else 768,
             "labels": "neutral,sad", "confusion_matrix": json.dumps([[3, 1], [1, 3]])}
            for i, feature in enumerate(["traditional", "hubert"])
            for model in ["LogisticRegression", "SVM_RBF"]
        ])

    def test_tables_preserve_each_feature_classifier_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = visualize.save_comparison_tables(self.summary(), root, root)
            self.assertEqual(len(table), 4)
            wide = pd.read_csv(root / "feature_classifier_comparison_wide.csv")
            self.assertEqual(wide.shape, (2, 7))
            self.assertEqual(set(table.classifier), {"LogisticRegression", "SVM_RBF"})
            self.assertTrue((root / "feature_classifier_comparison.png").is_file())
            for metric in visualize.METRIC_LABELS:
                visualize.save_model_comparison(self.summary(), root, metric)
                self.assertTrue((root / f"model_comparison_{metric}.png").is_file())

    def test_confusion_matrix_files_do_not_overwrite_other_feature_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            visualize.save_confusion_from_summary_if_needed(self.summary(), root)
            self.assertEqual(len(list(root.glob("confusion_matrix_*.png"))), 4)

    def test_legacy_summary_and_duplicate_combinations(self):
        legacy = self.summary().query("feature_type == 'traditional'").drop(columns="feature_type")
        self.assertEqual(set(visualize.build_comparison_table(legacy).feature_type), {"traditional"})
        with self.assertRaises(ValueError):
            visualize.build_comparison_table(pd.concat([self.summary(), self.summary().iloc[:1]]))


if __name__ == "__main__":
    unittest.main()
