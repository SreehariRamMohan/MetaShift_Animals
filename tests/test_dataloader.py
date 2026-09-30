"""Small loader/metric tests; full split integrity is checked by verify.py."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from metashift_animals import MetaShiftAnimals, evaluate_predictions
from manifest import CLASS_NAMES


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.relative_path = "train/bear/bear(tree)/123.jpg"
        image_path = self.root / self.relative_path
        image_path.parent.mkdir(parents=True)
        Image.new("L", (7, 5), color=100).save(image_path)
        self.rows = [
            {
                "row_id": "0", "split": "train", "image_id": "123",
                "class_id": "0", "class_name": "bear", "context": "tree",
                "relative_path": self.relative_path, "image_sha256": "unused-fixture",
            },
            {
                "row_id": "1", "split": "train", "image_id": "123",
                "class_id": "0", "class_name": "bear", "context": "water",
                "relative_path": self.relative_path, "image_sha256": "unused-fixture",
            },
        ]
        manifest_patch = patch("metashift_animals.load_manifest", return_value=self.rows)
        self.mock_manifest = manifest_patch.start()
        self.addCleanup(manifest_patch.stop)

    def test_rgb_and_metadata_preserve_duplicate_rows(self):
        dataset = MetaShiftAnimals(self.root, return_metadata=True)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.classes, list(CLASS_NAMES))
        self.assertEqual(dataset.class_to_idx["zebra"], 7)
        self.assertEqual(dataset.targets, [0, 0])
        self.assertEqual(dataset.groups, [0, 0])
        first_image, first_target, first_metadata = dataset[0]
        second_image, second_target, second_metadata = dataset[1]
        self.assertEqual(first_image.mode, "RGB")
        self.assertEqual(first_image.size, (7, 5))
        self.assertEqual(first_target, second_target)
        self.assertEqual(first_metadata["image_id"], second_metadata["image_id"])
        self.assertNotEqual(first_metadata["row_id"], second_metadata["row_id"])
        self.assertEqual(dataset.contexts, ["tree", "water"])
        self.assertEqual(second_metadata["context"], "water")
        self.assertEqual(second_image.tobytes(), first_image.tobytes())

    def test_optional_transforms_and_manifest_directory(self):
        dataset = MetaShiftAnimals(
            self.root,
            transform=lambda image: image.size,
            target_transform=lambda target: target + 10,
            manifest_dir=self.root / "manifests",
        )
        self.assertEqual(dataset[0], ((7, 5), 10))
        self.mock_manifest.assert_called_once_with("train", manifest_dir=self.root / "manifests")

    def test_flat_upstream_images(self):
        Image.new("RGB", (3, 2)).save(self.root / "123.jpg")
        dataset = MetaShiftAnimals(images_root=self.root)
        self.assertEqual(dataset[0][0].size, (3, 2))
        self.assertEqual(len(dataset), 2)

    def test_rejects_ambiguous_root_and_nonexistent_validation_split(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MetaShiftAnimals()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MetaShiftAnimals(self.root, images_root=self.root)
        with self.assertRaisesRegex(ValueError, "no validation split"):
            MetaShiftAnimals(self.root, split="val")


class EvaluationTests(unittest.TestCase):
    def test_accuracy_and_worst_class_in_percent(self):
        targets = [index for index in range(8) for _ in range(2)]
        predictions = targets.copy()
        predictions[0] = predictions[1] = 1
        predictions[2] = 0
        result = evaluate_predictions(targets, predictions)
        self.assertEqual(result["accuracy"], 81.25)
        self.assertEqual(result["worst_class_accuracy"], 0.0)
        self.assertEqual(result["per_class_accuracy"]["bear"], 0.0)
        self.assertEqual(result["per_class_accuracy"]["bird"], 50.0)
        self.assertEqual(result["per_class_accuracy"]["zebra"], 100.0)

    def test_accuracy_weights_rows_not_class_averages(self):
        targets = [0] * 9 + list(range(1, 8))
        predictions = [1] * 9 + list(range(1, 8))
        result = evaluate_predictions(targets, predictions)
        self.assertEqual(result["accuracy"], 43.75)
        self.assertEqual(result["worst_class_accuracy"], 0.0)

    def test_cpu_tensors(self):
        labels = torch.arange(8)
        result = evaluate_predictions(labels, labels)
        self.assertEqual(result["accuracy"], 100.0)
        self.assertEqual(result["worst_class_accuracy"], 100.0)

    def test_missing_classes_fail_unless_explicitly_partial(self):
        with self.assertRaisesRegex(ValueError, "missing target classes"):
            evaluate_predictions([0, 0, 1], [0, 1, 1])
        result = evaluate_predictions([0, 0, 1], [0, 1, 1], require_all_classes=False)
        self.assertAlmostEqual(result["accuracy"], 100.0 * 2 / 3)
        self.assertEqual(result["worst_class_accuracy"], 50.0)
        self.assertEqual(set(result["per_class_accuracy"]), {"bear", "bird"})

    def test_invalid_prediction_inputs(self):
        cases = [
            ([], []), ([0], []), ([0], [8]), ([0], [0.0]),
            ([0], [[0]]), ([True], [0]), (0, 0),
        ]
        for targets, predictions in cases:
            with self.subTest(targets=targets, predictions=predictions):
                with self.assertRaises(ValueError):
                    evaluate_predictions(targets, predictions, require_all_classes=False)


if __name__ == "__main__":
    unittest.main()
