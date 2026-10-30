"""Download and check original datasets. No splitting, label changes or training."""

import argparse
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR100, SVHN


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def require(condition, message):
    """Stop immediately when a check fails."""
    if not condition:
        raise RuntimeError(message)


def check_dataset(name, dataset, expected_count, classes, expected_shape):
    images = dataset.data

    if isinstance(dataset, CIFAR100):
        labels = np.asarray(dataset.targets)
    else:
        labels = np.asarray(dataset.labels)

    require(
        len(dataset) == expected_count,
        f"{name}: expected {expected_count} images, got {len(dataset)}",
    )
    require(
        images.shape == expected_shape,
        f"{name}: unexpected image shape {images.shape}",
    )
    require(
        images.dtype == np.uint8,
        f"{name}: unexpected image type {images.dtype}",
    )
    require(
        labels.shape == (expected_count,),
        f"{name}: unexpected label shape {labels.shape}",
    )
    require(
        np.issubdtype(labels.dtype, np.integer),
        f"{name}: labels must be integers",
    )
    require(
        np.array_equal(np.unique(labels), np.arange(classes)),
        f"{name}: labels must cover 0 through {classes - 1}",
    )

    counts = np.bincount(labels, minlength=classes)

    if isinstance(dataset, CIFAR100):
        require(
            np.all(counts == expected_count // classes),
            f"{name}: unexpected CIFAR-100 class counts",
        )

    print(
        f"[OK] {name}: "
        f"images={len(dataset)}, "
        f"shape={images.shape}, "
        f"classes={classes}, "
        f"label_range={labels.min()}..{labels.max()}, "
        f"per_class_min={counts.min()}, "
        f"per_class_max={counts.max()}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check existing files without downloading.",
    )
    args = parser.parse_args()

    download = not args.check_only
    print(f"Data directory: {DATA_ROOT}", flush=True)

    # torchvision checks the original files using its built-in checksums.
    for train, split, count in (
        (True, "train", 50000),
        (False, "test", 10000),
    ):
        print(f"\nPreparing CIFAR-100 {split}...", flush=True)
        dataset = CIFAR100(
            root=str(DATA_ROOT / "cifar100"),
            train=train,
            download=download,
        )
        check_dataset(
            name=f"CIFAR100/{split}",
            dataset=dataset,
            expected_count=count,
            classes=100,
            expected_shape=(count, 32, 32, 3),
        )
        del dataset

    # Deliberately exclude the SVHN "extra" split.
    for split, count in (
        ("train", 73257),
        ("test", 26032),
    ):
        print(f"\nPreparing SVHN {split}...", flush=True)
        dataset = SVHN(
            root=str(DATA_ROOT / "svhn"),
            split=split,
            download=download,
        )
        check_dataset(
            name=f"SVHN/{split}",
            dataset=dataset,
            expected_count=count,
            classes=10,
            expected_shape=(count, 3, 32, 32),
        )
        del dataset

    print("\nDATASET_CHECKS_COMPLETED", flush=True)
    print(
        "Original datasets only. "
        "No split manifests, noisy labels or training results were created.",
        flush=True,
    )


if __name__ == "__main__":
    main()