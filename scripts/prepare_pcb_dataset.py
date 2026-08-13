"""Download PCB Defects and build leak-safe cropped-classification splits.

The Kaggle source is a Pascal VOC object-detection dataset.  The agent's ML
contract is image classification, so each annotated bounding box is
materialized as a small image while every crop from one source image remains
in the same split.
"""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


DATASET_HANDLE = "akhatova/pcb-defects"
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


@dataclass(frozen=True)
class Annotation:
    label: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class SourceImage:
    image: Path
    annotations: tuple[Annotation, ...]

    @property
    def label(self) -> str:
        labels = {annotation.label for annotation in self.annotations}
        if len(labels) != 1:
            raise ValueError(f"Expected one defect class in {self.image}, got {labels}")
        return next(iter(labels))


def download_dataset() -> Path:
    """Download the required Kaggle dataset and return its version directory."""
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_HANDLE)).resolve()


def _normalise_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def inspect_dataset(dataset_path: Path) -> list[SourceImage]:
    """Parse and validate all original image/XML pairs."""
    pcb_root = dataset_path / "PCB_DATASET"
    image_root = pcb_root / "images"
    annotation_root = pcb_root / "Annotations"
    if not image_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(
            f"Expected PCB_DATASET/images and PCB_DATASET/Annotations under {dataset_path}"
        )

    sources: list[SourceImage] = []
    image_classes = {path.name for path in image_root.iterdir() if path.is_dir()}
    annotation_classes = {
        path.name for path in annotation_root.iterdir() if path.is_dir()
    }
    if image_classes != annotation_classes:
        raise ValueError(
            "Image and annotation class directories differ: "
            f"images={sorted(image_classes)}, annotations={sorted(annotation_classes)}"
        )

    for class_name in sorted(image_classes):
        expected_label = _normalise_label(class_name)
        images = {path.stem: path for path in (image_root / class_name).glob("*.jpg")}
        xml_files = {
            path.stem: path for path in (annotation_root / class_name).glob("*.xml")
        }
        if images.keys() != xml_files.keys():
            raise ValueError(
                f"Image/XML mismatch for {class_name}: "
                f"missing_xml={sorted(images.keys() - xml_files.keys())}, "
                f"missing_image={sorted(xml_files.keys() - images.keys())}"
            )

        for stem in sorted(images):
            root = ET.parse(xml_files[stem]).getroot()
            annotations: list[Annotation] = []
            for obj in root.findall("object"):
                label = _normalise_label(obj.findtext("name") or "")
                if label != expected_label:
                    raise ValueError(
                        f"Unexpected label {label!r} in {xml_files[stem]}; "
                        f"expected {expected_label!r}"
                    )
                bbox = obj.find("bndbox")
                if bbox is None:
                    raise ValueError(f"Object without bndbox in {xml_files[stem]}")
                box = tuple(int(bbox.findtext(name) or "0") for name in ("xmin", "ymin", "xmax", "ymax"))
                if box[0] >= box[2] or box[1] >= box[3]:
                    raise ValueError(f"Invalid box {box} in {xml_files[stem]}")
                annotations.append(Annotation(label=label, box=box))
            if not annotations:
                raise ValueError(f"No objects found in {xml_files[stem]}")
            sources.append(SourceImage(images[stem].resolve(), tuple(annotations)))
    return sources


def stratified_group_split(
    sources: Iterable[SourceImage], seed: int = 42
) -> dict[str, list[SourceImage]]:
    """Stratify crop counts while keeping source-image groups intact."""
    by_class: dict[str, list[SourceImage]] = {}
    for source in sources:
        by_class.setdefault(source.label, []).append(source)

    result = {split: [] for split in SPLIT_FRACTIONS}
    for label, class_sources in sorted(by_class.items()):
        shuffled = sorted(class_sources, key=lambda source: source.image.as_posix())
        random.Random(f"{seed}:{label}").shuffle(shuffled)
        # Large groups first improves the fit; the prior shuffle randomises ties.
        shuffled.sort(key=lambda source: len(source.annotations), reverse=True)
        total = sum(len(source.annotations) for source in shuffled)
        targets = {
            split: total * fraction for split, fraction in SPLIT_FRACTIONS.items()
        }
        assigned = Counter()
        for source in shuffled:
            split = max(
                SPLIT_FRACTIONS,
                key=lambda name: (targets[name] - assigned[name]) / targets[name],
            )
            result[split].append(source)
            assigned[split] += len(source.annotations)
    return result


def _padded_crop_box(
    box: tuple[int, int, int, int], width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    # Pascal VOC coordinates are one-based; Pillow uses a zero-based half-open box.
    left, top = xmin - 1, ymin - 1
    right, bottom = xmax, ymax
    margin = max(4, round(max(right - left, bottom - top) * padding))
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(width, right + margin),
        min(height, bottom + margin),
    )


def _manifest_path(path: Path, working_directory: Path) -> str:
    try:
        return path.resolve().relative_to(working_directory.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_manifest(
    dataset_path: Path,
    output_path: Path = Path("data/pcb_splits.json"),
    crop_directory: Path = Path("data/pcb_crops"),
    seed: int = 42,
    padding: float = 0.25,
) -> dict:
    """Create cropped images and return/write the ML track's manifest schema."""
    sources = inspect_dataset(dataset_path)
    grouped_splits = stratified_group_split(sources, seed=seed)
    crop_directory.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    working_directory = Path.cwd()
    manifest_splits: dict[str, list[dict[str, str]]] = {
        split: [] for split in SPLIT_FRACTIONS
    }
    split_class_distribution: dict[str, dict[str, int]] = {}

    for split, split_sources in grouped_splits.items():
        distribution = Counter()
        for source in split_sources:
            label = source.label
            destination_dir = crop_directory / label
            destination_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(source.image) as image:
                image = image.convert("RGB")
                for index, annotation in enumerate(source.annotations, start=1):
                    destination = destination_dir / f"{source.image.stem}__obj_{index:02d}.jpg"
                    crop_box = _padded_crop_box(
                        annotation.box, image.width, image.height, padding
                    )
                    image.crop(crop_box).save(destination, format="JPEG", quality=95)
                    manifest_splits[split].append(
                        {
                            "image": _manifest_path(destination, working_directory),
                            "label": label,
                        }
                    )
                    distribution[label] += 1
        manifest_splits[split].sort(key=lambda item: (item["label"], item["image"]))
        split_class_distribution[split] = dict(sorted(distribution.items()))

    class_distribution = Counter(source.label for source in sources for _ in source.annotations)
    class_counts = list(class_distribution.values())
    summary = {
        "dataset": DATASET_HANDLE,
        "representation": "cropped_defect_classification",
        "annotation_format": "Pascal VOC XML bounding boxes",
        "total_source_images": len(sources),
        "total_samples": sum(class_counts),
        "classes": sorted(class_distribution),
        "class_distribution": dict(sorted(class_distribution.items())),
        "class_imbalance_ratio": round(max(class_counts) / min(class_counts), 4),
        "train_samples": len(manifest_splits["train"]),
        "validation_samples": len(manifest_splits["validation"]),
        "test_samples": len(manifest_splits["test"]),
        "split_class_distribution": split_class_distribution,
        "split_seed": seed,
        "split_fractions": SPLIT_FRACTIONS,
        "grouping": "all crops from a source image stay in one split",
        "rotated_derivatives_excluded": True,
    }
    manifest = {
        "classes": summary["classes"],
        "splits": manifest_splits,
        "dataset_summary": summary,
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/pcb_splits.json"))
    parser.add_argument("--crop-directory", type=Path, default=Path("data/pcb_crops"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", type=float, default=0.25)
    args = parser.parse_args()
    dataset_path = args.dataset_path.resolve() if args.dataset_path else download_dataset()
    print(f"Dataset path: {dataset_path}")
    manifest = build_manifest(
        dataset_path,
        output_path=args.output,
        crop_directory=args.crop_directory,
        seed=args.seed,
        padding=args.padding,
    )
    print(json.dumps(manifest["dataset_summary"], indent=2))
    print(f"Manifest: {args.output.resolve()}")


if __name__ == "__main__":
    main()
