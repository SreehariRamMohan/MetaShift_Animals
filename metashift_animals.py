"""PyTorch loader and row-weighted evaluation for the frozen MetaShift Animals split.

Each manifest row is one benchmark sample. Repeated image IDs are intentional:
images may have multiple context assignments, including across animal classes.
"""

from numbers import Integral
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from manifest import CLASS_NAMES, load_manifest


class MetaShiftAnimals(Dataset):
    """Load the eight-class MetaShift-derived benchmark without changing its split.

    Args:
        root: Directory produced by ``prepare.py``, containing train/ and test/.
        split: Exactly ``"train"`` or ``"test"``. There is no validation split.
        transform: Optional callable applied to the RGB PIL image.
        target_transform: Optional callable applied to the integer class label.
        return_metadata: Return ``(image, target, metadata)`` instead of a pair.
        manifest_dir: Optional directory containing the frozen split manifests.
        images_root: Alternative to root: original flat directory of upstream
            JPEGs named ``<image_id>.jpg``. No prepared copy is needed in this mode.

    ``classes`` defines the label order; ``targets`` and ``groups`` both contain
    animal-class IDs. The benchmark reports worst-class accuracy, not accuracy
    of the worst context group. Context annotations are available in ``contexts``
    and, optionally, each sample's metadata.
    """

    def __init__(
        self,
        root=None,
        split="train",
        transform=None,
        target_transform=None,
        return_metadata=False,
        manifest_dir=None,
        *,
        images_root=None,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'; no validation split is defined")
        if (root is None) == (images_root is None):
            raise ValueError("provide exactly one of root or images_root")
        self.root = Path(root).expanduser() if root is not None else None
        self.images_root = Path(images_root).expanduser() if images_root is not None else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = return_metadata
        self.classes = list(CLASS_NAMES)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self._rows = (
            load_manifest(split)
            if manifest_dir is None
            else load_manifest(split, manifest_dir=Path(manifest_dir))
        )
        self.targets = [int(row["class_id"]) for row in self._rows]
        self.contexts = [row["context"] for row in self._rows]
        self.groups = self.targets.copy()
        self.samples = [
            (
                str(self.images_root / f"{row['image_id']}.jpg")
                if self.images_root is not None
                else str(self.root / row["relative_path"]),
                target,
            )
            for row, target in zip(self._rows, self.targets)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_metadata:
            row = self._rows[index]
            metadata = {
                "row_id": int(row["row_id"]),
                "image_id": row["image_id"],
                "context": row["context"],
                "class_name": row["class_name"],
            }
            return image, target, metadata
        return image, target


def _labels(values, name):
    """Accept sequences, NumPy arrays, and tensors without importing NumPy."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence of class IDs") from exc
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must contain integer class IDs, not scores or probabilities")
        if not 0 <= value < len(CLASS_NAMES):
            raise ValueError(f"{name} contains a class ID outside [0, {len(CLASS_NAMES) - 1}]")
        result.append(int(value))
    return result


def evaluate_predictions(targets, predictions, *, require_all_classes=True):
    """Return row-weighted accuracy and worst-class accuracy, in percent (0–100).

    Inputs must be one-dimensional integer class IDs in manifest order, with one
    prediction for every evaluated row, including repeated image IDs. NumPy arrays,
    Python sequences, and CPU tensors are accepted; logits are not.

    Full benchmark evaluation requires all eight target classes. For an explicitly
    partial diagnostic, ``require_all_classes=False`` computes the minimum only
    over classes present and omits absent classes from ``per_class_accuracy``.
    The caller is responsible for supplying the complete frozen test split when
    reporting benchmark results; this function does not identify a split by size.
    """
    targets = _labels(targets, "targets")
    predictions = _labels(predictions, "predictions")
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    if not targets:
        raise ValueError("cannot evaluate an empty set of predictions")
    counts = [0] * len(CLASS_NAMES)
    correct = [0] * len(CLASS_NAMES)
    for target, prediction in zip(targets, predictions):
        counts[target] += 1
        correct[target] += int(target == prediction)
    missing = [name for name, count in zip(CLASS_NAMES, counts) if count == 0]
    if missing and require_all_classes:
        raise ValueError("missing target classes: " + ", ".join(missing))
    per_class_accuracy = {
        name: 100.0 * hits / count
        for name, hits, count in zip(CLASS_NAMES, correct, counts)
        if count
    }
    return {
        "accuracy": 100.0 * sum(correct) / len(targets),
        "worst_class_accuracy": min(per_class_accuracy.values()),
        "per_class_accuracy": per_class_accuracy,
    }
