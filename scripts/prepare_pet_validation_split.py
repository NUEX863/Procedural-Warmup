"""Create a fixed stratified Pet train/validation split using seed 11."""

import copy
import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests" / "oxford_pet_seed11_splits.json"
OUTPUT = ROOT / "manifests" / "pet90_10_seed11_splits.json"

SEED = 11
NUM_CLASSES = 37
VALIDATION_COUNT = 368


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def read_source():
    content = SOURCE.read_bytes()
    digest = sha256(content)
    checksum_path = SOURCE.with_name(SOURCE.name + ".sha256")

    recorded = checksum_path.read_text(encoding="utf-8").split()
    require(
        recorded == [digest, SOURCE.name],
        "Original manifest checksum mismatch",
    )

    manifest = json.loads(content)
    require(
        manifest["dataset"] == "oxford_pet"
        and manifest["num_classes"] == NUM_CLASSES
        and manifest["seed"] == SEED,
        "Unexpected original manifest",
    )
    require(
        set(manifest["partitions"]) == {"train", "test"},
        "Expected the original trainval/test manifest",
    )

    return manifest, digest


def check_partition(partition, expected_count, expected_source):
    require(
        partition["source_split"] == expected_source,
        "Unexpected official source split",
    )

    indices = partition["indices"]
    sample_ids = partition["sample_ids"]
    labels = partition["labels_clean"]

    require(
        len(indices) == len(sample_ids) == len(labels) == expected_count,
        "Partition length mismatch",
    )
    require(len(set(indices)) == expected_count, "Duplicate indices")
    require(len(set(sample_ids)) == expected_count, "Duplicate images")
    require(
        all(type(label) is int and 0 <= label < NUM_CLASSES for label in labels),
        "Invalid clean labels",
    )
    require(set(labels) == set(range(NUM_CLASSES)), "Missing classes")


def select_partition(source, positions):
    positions = sorted(positions)
    labels = [source["labels_clean"][i] for i in positions]

    return {
        "source_split": "trainval",
        "count": len(positions),
        "indices": [source["indices"][i] for i in positions],
        "sample_ids": [source["sample_ids"][i] for i in positions],
        "labels_clean": labels,
        "class_counts": [
            labels.count(class_id) for class_id in range(NUM_CLASSES)
        ],
    }


def build_manifest(original, source_digest):
    trainval = original["partitions"]["train"]
    test = original["partitions"]["test"]

    check_partition(trainval, 3680, "trainval")
    check_partition(test, 3669, "test")

    require(
        set(trainval["indices"]) == set(range(3680)),
        "Original train partition must contain all official trainval indices",
    )
    require(
        set(test["indices"]) == set(range(3669)),
        "Original test partition must contain all official test indices",
    )
    require(
        set(trainval["sample_ids"]).isdisjoint(test["sample_ids"]),
        "Official trainval/test image identities overlap",
    )

    labels = trainval["labels_clean"]
    members = {
        class_id: [
            index for index, label in enumerate(labels)
            if label == class_id
        ]
        for class_id in range(NUM_CLASSES)
    }
    counts = {class_id: len(rows) for class_id, rows in members.items()}

    # Start with floor(10% of each class), then distribute the remainder.
    # Larger fractional remainders receive one additional validation image.
    # Equal remainders are resolved by ascending class ID.
    validation_counts = {
        class_id: count // 10 for class_id, count in counts.items()
    }
    remaining = VALIDATION_COUNT - sum(validation_counts.values())

    allocation_order = sorted(
        range(NUM_CLASSES),
        key=lambda class_id: (-(counts[class_id] % 10), class_id),
    )
    require(
        0 <= remaining <= NUM_CLASSES,
        "Invalid validation allocation",
    )
    for class_id in allocation_order[:remaining]:
        validation_counts[class_id] += 1

    rng = random.Random(SEED)
    train_positions = []
    validation_positions = []

    for class_id in range(NUM_CLASSES):
        positions = list(members[class_id])
        rng.shuffle(positions)
        count = validation_counts[class_id]

        require(
            0 < count < len(positions),
            "Every class must occur in train and validation",
        )
        validation_positions.extend(positions[:count])
        train_positions.extend(positions[count:])

    train = select_partition(trainval, train_positions)
    validation = select_partition(trainval, validation_positions)

    check_partition(train, 3312, "trainval")
    check_partition(validation, 368, "trainval")

    train_ids = set(train["sample_ids"])
    validation_ids = set(validation["sample_ids"])
    test_ids = set(test["sample_ids"])

    require(train_ids.isdisjoint(validation_ids), "Train/validation overlap")
    require(train_ids.isdisjoint(test_ids), "Train/test overlap")
    require(validation_ids.isdisjoint(test_ids), "Validation/test overlap")
    require(
        train_ids | validation_ids == set(trainval["sample_ids"]),
        "Train/validation do not reconstruct official trainval",
    )
    require(
        set(train["indices"]).isdisjoint(validation["indices"]),
        "Train/validation source indices overlap",
    )

    return {
        "schema_version": 2,
        "protocol": "pet90_10_v1",
        "dataset": "oxford_pet",
        "num_classes": NUM_CLASSES,
        "seed": SEED,
        "source_manifest": SOURCE.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": source_digest,
        "split_method": {
            "training_count": 3312,
            "validation_count": VALIDATION_COUNT,
            "stratification": "Official class labels",
            "allocation": (
                "Largest remainder allocation for exactly 368 validation "
                "images; fractional ties resolved by ascending class ID"
            ),
            "sampling": (
                "Python random.Random(11); shuffle each class in ascending "
                "class order; take validation prefix; sort saved positions"
            ),
        },
        "partitions": {
            "train": train,
            "validation": validation,
            "test": copy.deepcopy(test),
        },
    }


def main():
    original, source_digest = read_source()
    manifest = build_manifest(original, source_digest)

    # Repeating the algorithm must give exactly the same result.
    require(
        manifest == build_manifest(original, source_digest),
        "Split generation is not reproducible",
    )
    require(
        manifest["partitions"]["test"] == original["partitions"]["test"],
        "Official test partition changed",
    )

    content = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")

    digest = sha256(content)
    checksum_path = OUTPUT.with_name(OUTPUT.name + ".sha256")
    checksum_content = f"{digest}  {OUTPUT.name}\n".encode("utf-8")

    outputs = [
        (OUTPUT, content),
        (checksum_path, checksum_content),
    ]

    # Check both paths before writing either file.
    for path, expected in outputs:
        if path.exists():
            require(
                path.read_bytes() == expected,
                f"Existing file differs; refusing to overwrite: {path}",
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for path, expected in outputs:
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(expected)
            print(f"[CREATED] {path}")
        else:
            print(f"[UNCHANGED] {path}")

    require(
        sha256(SOURCE.read_bytes()) == source_digest,
        "Original manifest changed",
    )

    for split in ("train", "validation", "test"):
        labels = manifest["partitions"][split]["labels_clean"]
        counts = [labels.count(c) for c in range(NUM_CLASSES)]
        print(
            f"{split}: count={len(labels)}, classes={NUM_CLASSES}, "
            f"per_class_min={min(counts)}, per_class_max={max(counts)}"
        )

    print(f"SHA256: {digest}")
    print("split_reproducibility=PASS")
    print("sample_identity_disjointness=PASS")
    print("clean_labels_preserved=PASS")
    print("official_test_partition_unchanged=True")
    print("original_manifest_unchanged=True")
    print("test_images_opened=False")
    print("model_training_started=False")
    print("PET_VALIDATION_SPLIT_COMPLETED")
    print("status=PASS")


if __name__ == "__main__":
    main()