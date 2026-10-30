"""Create fixed clean splits for CIFAR-100 and Oxford-IIIT Pet."""

import hashlib
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR100


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MANIFEST_ROOT = ROOT / "manifests"

SEED = 11


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_labels(labels, expected_count, num_classes, name):
    labels = np.asarray(labels, dtype=np.int64)

    require(
        labels.shape == (expected_count,),
        f"{name}: unexpected label count/shape: {labels.shape}",
    )
    require(
        np.array_equal(np.unique(labels), np.arange(num_classes)),
        f"{name}: expected all labels from 0 to {num_classes - 1}",
    )
    return labels


def make_partition(source_split, indices, source_labels, source_ids):
    indices = np.asarray(indices, dtype=np.int64)

    return {
        "source_split": source_split,
        "indices": indices.tolist(),
        "sample_ids": [source_ids[int(i)] for i in indices],
        "labels_clean": source_labels[indices].tolist(),
    }


def check_partitions(partitions, num_classes):
    seen = set()

    for name, part in partitions.items():
        ids = part["sample_ids"]
        indices = part["indices"]
        labels = part["labels_clean"]

        require(
            len(ids) == len(indices) == len(labels),
            f"{name}: IDs, indices and labels have different lengths",
        )
        require(
            len(set(ids)) == len(ids),
            f"{name}: duplicate sample IDs",
        )
        require(
            len(set(indices)) == len(indices),
            f"{name}: duplicate source indices",
        )
        require(
            all(0 <= label < num_classes for label in labels),
            f"{name}: invalid class label",
        )

        overlap = seen.intersection(ids)
        require(
            not overlap,
            f"{name}: sample overlap detected: {sorted(overlap)[:5]}",
        )
        seen.update(ids)


def prepare_cifar100():
    print("\nPreparing CIFAR-100 fixed split...", flush=True)

    source_train = CIFAR100(
        root=str(DATA_ROOT / "cifar100"),
        train=True,
        download=False,
    )
    source_test = CIFAR100(
        root=str(DATA_ROOT / "cifar100"),
        train=False,
        download=False,
    )

    train_labels = check_labels(
        source_train.targets, 50000, 100, "CIFAR100 official train"
    )
    test_labels = check_labels(
        source_test.targets, 10000, 100, "CIFAR100 official test"
    )

    require(
        np.all(np.bincount(train_labels, minlength=100) == 500),
        "CIFAR100 official train must contain 500 images per class",
    )
    require(
        np.all(np.bincount(test_labels, minlength=100) == 100),
        "CIFAR100 official test must contain 100 images per class",
    )

    # A dedicated generator avoids changing global/model random states.
    rng = np.random.default_rng(SEED)

    selected_train = []
    selected_val = []
    selected_unused = []

    for class_id in range(100):
        class_indices = np.flatnonzero(train_labels == class_id)
        shuffled = rng.permutation(class_indices)

        selected_train.extend(shuffled[:170])
        selected_val.extend(shuffled[170:200])
        selected_unused.extend(shuffled[200:])

    # Stable order in the manifest; training-time shuffling is separate.
    selected_train = np.sort(
        np.asarray(selected_train, dtype=np.int64)
    )
    selected_val = np.sort(
        np.asarray(selected_val, dtype=np.int64)
    )
    selected_unused = np.sort(
        np.asarray(selected_unused, dtype=np.int64)
    )

    reconstructed = np.concatenate(
        [selected_train, selected_val, selected_unused]
    )
    require(
        np.array_equal(np.sort(reconstructed), np.arange(50000)),
        "CIFAR100 partitions do not cover official train exactly once",
    )

    # Include the official split in each ID:
    # train/0 and test/0 are different original samples.
    train_ids = [f"train/{i}" for i in range(50000)]
    test_ids = [f"test/{i}" for i in range(10000)]

    partitions = {
        "train": make_partition(
            "train", selected_train, train_labels, train_ids
        ),
        "validation": make_partition(
            "train", selected_val, train_labels, train_ids
        ),
        "unused": make_partition(
            "train", selected_unused, train_labels, train_ids
        ),
        "test": make_partition(
            "test", np.arange(10000), test_labels, test_ids
        ),
    }

    expected_per_class = {
        "train": 170,
        "validation": 30,
        "unused": 300,
        "test": 100,
    }

    for name, per_class in expected_per_class.items():
        counts = np.bincount(
            partitions[name]["labels_clean"], minlength=100
        )
        require(
            np.all(counts == per_class),
            f"CIFAR100 {name}: incorrect per-class counts",
        )

    check_partitions(partitions, num_classes=100)

    return {
        "schema_version": 1,
        "dataset": "cifar100",
        "seed": SEED,
        "num_classes": 100,
        "split_method": (
            "Per-class permutation using numpy.default_rng(11); "
            "170 train, 30 validation, 300 unused per class."
        ),
        "partitions": partitions,
    }


def read_pet_annotation(split):
    path = (
        DATA_ROOT
        / "oxford-iiit-pet"
        / "annotations"
        / f"{split}.txt"
    )

    ids = []
    labels = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.split()
        require(
            len(fields) == 4,
            f"{path.name}:{line_number}: expected four fields",
        )

        image_id = fields[0]
        require(
            "/" not in image_id and "\\" not in image_id,
            f"{path.name}:{line_number}: invalid image ID",
        )

        # Official breed labels are 1..37.
        # Convert them to the model's required 0..36.
        category = int(fields[1]) - 1

        ids.append(f"images/{image_id}.jpg")
        labels.append(category)

    require(
        len(ids) == len(set(ids)),
        f"Pet {split}: duplicate image IDs in annotation file",
    )

    return ids, np.asarray(labels, dtype=np.int64)


def prepare_pet():
    print("\nPreparing Oxford-IIIT Pet official splits...", flush=True)

    train_ids, train_labels = read_pet_annotation("trainval")
    test_ids, test_labels = read_pet_annotation("test")

    train_labels = check_labels(
        train_labels, 3680, 37, "Pet official trainval"
    )
    test_labels = check_labels(
        test_labels, 3669, 37, "Pet official test"
    )

    partitions = {
        "train": make_partition(
            "trainval",
            np.arange(len(train_ids)),
            train_labels,
            train_ids,
        ),
        "test": make_partition(
            "test",
            np.arange(len(test_ids)),
            test_labels,
            test_ids,
        ),
    }

    check_partitions(partitions, num_classes=37)

    return {
        "schema_version": 1,
        "dataset": "oxford_pet",
        "seed": SEED,
        "num_classes": 37,
        "split_method": (
            "Use all official trainval for training and all official "
            "test for final evaluation. No independent validation split."
        ),
        "partitions": partitions,
    }


def save_without_overwriting(path, content):
    if path.exists():
        require(
            path.read_bytes() == content,
            f"Existing file differs from current protocol: {path}\n"
            "Stopped without overwriting. Inspect the difference first.",
        )
        return "REUSED"

    with path.open("xb") as stream:
        stream.write(content)

    return "CREATED"


def save_manifest(manifest):
    filename = f"{manifest['dataset']}_seed{SEED}_splits.json"
    path = MANIFEST_ROOT / filename

    # No timestamps: unchanged data/protocol produce identical bytes.
    content = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    digest = hashlib.sha256(content).hexdigest()
    status = save_without_overwriting(path, content)

    checksum_path = path.with_name(path.name + ".sha256")
    checksum_content = f"{digest}  {filename}\n".encode("utf-8")
    save_without_overwriting(checksum_path, checksum_content)

    print(f"\n[{status}] {path}")
    print(f"SHA256: {digest}")

    for name, part in manifest["partitions"].items():
        counts = np.bincount(
            part["labels_clean"],
            minlength=manifest["num_classes"],
        )
        print(
            f"  {name}: count={len(part['indices'])}, "
            f"class_count_min={counts.min()}, "
            f"class_count_max={counts.max()}"
        )

    print("  sample_identity_disjointness=PASS")
    print("  clean_label_range=PASS")


def main():
    # Validate both datasets before writing either manifest.
    manifests = [
        prepare_cifar100(),
        prepare_pet(),
    ]

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    for manifest in manifests:
        save_manifest(manifest)

    print("\nFIXED_SPLITS_CHECKS_COMPLETED")
    print("Seed: 11")
    print("Labels are clean; no label noise has been generated.")
    print("Pet has no independent validation split.")
    print("No model training or test-set accuracy evaluation was performed.")


if __name__ == "__main__":
    main()