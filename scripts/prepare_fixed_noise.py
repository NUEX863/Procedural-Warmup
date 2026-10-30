"""Generate fixed, paired, nested label-noise manifests."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "manifests"

SEED = 11
NOISE_PERCENTAGES = (0, 10, 20, 30)

# SHA256 values confirmed in the previous split-generation step.
DATASETS = {
    "cifar100": {
        "dataset_stream": 0,
        "num_classes": 100,
        "train_count": 17000,
        "split_sha256": (
            "01e9ef20f9343330ff5d95c96af2a29b9"
            "c1a2e7f22e6d8b8a090d738f4398f48"
        ),
    },
    "oxford_pet": {
        "dataset_stream": 1,
        "num_classes": 37,
        "train_count": 3680,
        "split_sha256": (
            "aaaf23ab73d7aabb1cb392298580c165"
            "fb2f440af104234111daa1a6af6021a3"
        ),
    },
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def encode_json(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def load_fixed_split(dataset_name, spec):
    path = (
        MANIFEST_ROOT
        / f"{dataset_name}_seed{SEED}_splits.json"
    )
    content = path.read_bytes()
    actual_hash = sha256_bytes(content)

    require(
        actual_hash == spec["split_sha256"],
        f"{dataset_name}: split manifest differs from the approved SHA256",
    )

    checksum_path = path.with_name(path.name + ".sha256")
    checksum_parts = checksum_path.read_text(
        encoding="utf-8"
    ).split()

    require(
        checksum_parts == [actual_hash, path.name],
        f"{dataset_name}: split checksum file does not match",
    )

    manifest = json.loads(content)

    require(
        manifest["schema_version"] == 1,
        f"{dataset_name}: unsupported split schema",
    )
    require(
        manifest["dataset"] == dataset_name
        and manifest["seed"] == SEED
        and manifest["num_classes"] == spec["num_classes"],
        f"{dataset_name}: split metadata mismatch",
    )

    train = manifest["partitions"]["train"]
    labels = np.asarray(train["labels_clean"], dtype=np.int64)
    count = spec["train_count"]

    require(
        len(train["indices"])
        == len(train["sample_ids"])
        == len(labels)
        == count,
        f"{dataset_name}: incorrect training count",
    )
    require(
        len(set(train["sample_ids"])) == count,
        f"{dataset_name}: duplicate training sample IDs",
    )
    require(
        np.array_equal(
            np.unique(labels),
            np.arange(spec["num_classes"]),
        ),
        f"{dataset_name}: incorrect clean label range",
    )

    train_ids = set(train["sample_ids"])
    for name, partition in manifest["partitions"].items():
        if name != "train":
            require(
                not train_ids.intersection(partition["sample_ids"]),
                f"{dataset_name}: train overlaps with {name}",
            )

    return path, content, train, labels


def allocate_class_counts(class_sizes, percentage, tie_rank):
    """Largest-remainder allocation with a fixed tie-breaking order."""
    total = int(class_sizes.sum())

    require(
        total * percentage % 100 == 0,
        "Requested global corruption count is not an integer",
    )
    target = total * percentage // 100

    # Integer arithmetic avoids floating-point rounding ambiguity.
    numerators = class_sizes * percentage
    quotas = numerators // 100
    remainders = numerators % 100

    remaining = target - int(quotas.sum())

    order = sorted(
        range(len(class_sizes)),
        key=lambda c: (-int(remainders[c]), int(tie_rank[c])),
    )

    require(
        0 <= remaining <= len(class_sizes),
        "Invalid remainder allocation",
    )

    for class_id in order[:remaining]:
        quotas[class_id] += 1

    require(
        int(quotas.sum()) == target,
        "Per-class quotas do not sum to the global target",
    )
    require(
        np.all((quotas >= 0) & (quotas <= class_sizes)),
        "Per-class corruption count is outside its valid range",
    )
    require(
        np.all(np.abs(quotas * 100 - numerators) < 100),
        "A class differs from its exact target by at least one image",
    )

    return quotas, target


def prepare_dataset(dataset_name, spec):
    path, original_bytes, train, clean = load_fixed_split(
        dataset_name, spec
    )

    count = len(clean)
    num_classes = spec["num_classes"]

    # One experiment seed. These are deterministic substreams derived
    # from seed=11, not additional experimental seed repetitions.
    seed_sequence = np.random.SeedSequence(
        SEED,
        spawn_key=(spec["dataset_stream"],),
    )
    selection_seed, replacement_seed, tie_seed = seed_sequence.spawn(3)

    selection_rng = np.random.default_rng(selection_seed)
    replacement_rng = np.random.default_rng(replacement_seed)
    tie_rng = np.random.default_rng(tie_seed)

    class_sizes = np.bincount(clean, minlength=num_classes)

    # Generate each class's selection order once for all noise levels.
    class_orders = [
        selection_rng.permutation(np.flatnonzero(clean == class_id))
        for class_id in range(num_classes)
    ]

    # Generate one fixed wrong label per training image.
    # Mapping around the true class gives a uniform choice among K-1
    # incorrect classes, with no chance of drawing the true label.
    wrong_labels = replacement_rng.integers(
        0,
        num_classes - 1,
        size=count,
        dtype=np.int64,
    )
    wrong_labels += wrong_labels >= clean

    require(
        np.all(wrong_labels != clean),
        f"{dataset_name}: a replacement equals its true label",
    )

    tie_order = tie_rng.permutation(num_classes)
    tie_rank = np.empty(num_classes, dtype=np.int64)
    tie_rank[tie_order] = np.arange(num_classes)

    previous_mask = np.zeros(count, dtype=bool)
    previous_observed = clean.copy()
    previous_quotas = np.zeros(num_classes, dtype=np.int64)

    prepared = []

    for percentage in NOISE_PERCENTAGES:
        quotas, target = allocate_class_counts(
            class_sizes, percentage, tie_rank
        )

        # Explicitly verify nesting rather than assuming that rounding
        # necessarily preserves it for every possible dataset.
        require(
            np.all(quotas >= previous_quotas),
            f"{dataset_name}: class quotas are not nested",
        )

        selected = np.concatenate([
            class_orders[class_id][:int(quotas[class_id])]
            for class_id in range(num_classes)
        ])

        mask = np.zeros(count, dtype=bool)
        mask[selected] = True

        observed = clean.copy()
        observed[mask] = wrong_labels[mask]

        actual_mask = observed != clean
        actual_per_class = np.bincount(
            clean[actual_mask],
            minlength=num_classes,
        )

        require(
            int(actual_mask.sum()) == target,
            f"{dataset_name}/{percentage}: incorrect corruption count",
        )
        require(
            np.array_equal(actual_mask, mask),
            f"{dataset_name}/{percentage}: mask/label mismatch",
        )
        require(
            np.array_equal(actual_per_class, quotas),
            f"{dataset_name}/{percentage}: class quota mismatch",
        )
        require(
            np.all((observed >= 0) & (observed < num_classes)),
            f"{dataset_name}/{percentage}: invalid observed label",
        )
        require(
            np.array_equal(observed[~mask], clean[~mask]),
            f"{dataset_name}/{percentage}: clean samples were changed",
        )
        require(
            np.all(~previous_mask | mask),
            f"{dataset_name}/{percentage}: corruption sets are not nested",
        )
        require(
            np.array_equal(
                observed[previous_mask],
                previous_observed[previous_mask],
            ),
            f"{dataset_name}/{percentage}: a previous wrong label changed",
        )

        if dataset_name == "cifar100":
            require(
                np.all(quotas == 170 * percentage // 100),
                "CIFAR100 per-class corruption count is incorrect",
            )

        manifest = {
            "schema_version": 1,
            "dataset": dataset_name,
            "seed": SEED,
            "num_classes": num_classes,
            "noise_percent": percentage,
            "corrupted_count": target,
            "shared_by_initializations": ["random", "procedural"],
            "split_manifest": path.relative_to(ROOT).as_posix(),
            "split_manifest_sha256": spec["split_sha256"],
            "noise_rule": (
                "Fixed symmetric wrong-class labels; nested class-wise "
                "selection; largest-remainder integer quotas."
            ),
            "rng": {
                "base_seed": SEED,
                "dataset_stream": spec["dataset_stream"],
                "child_stream_roles": [
                    "selection",
                    "replacement",
                    "quota_tie_breaking",
                ],
            },
            "class_train_counts": class_sizes.tolist(),
            "class_corruption_counts": quotas.tolist(),
            "training": {
                "source_split": train["source_split"],
                "indices": train["indices"],
                "sample_ids": train["sample_ids"],
                "labels_clean": clean.tolist(),
                "labels_observed": observed.tolist(),
                "is_corrupted": mask.tolist(),
            },
        }

        filename = (
            f"{dataset_name}_seed{SEED}_noise{percentage:02d}.json"
        )
        output_path = MANIFEST_ROOT / filename
        output_bytes = encode_json(manifest)
        digest = sha256_bytes(output_bytes)

        prepared.append({
            "path": output_path,
            "content": output_bytes,
            "sha256": digest,
            "percentage": percentage,
            "count": target,
            "class_min": int(quotas.min()),
            "class_max": int(quotas.max()),
        })

        previous_mask = mask
        previous_observed = observed
        previous_quotas = quotas

    return prepared, path, original_bytes


def main():
    prepared = []
    original_splits = []

    for dataset_name, spec in DATASETS.items():
        print(f"Preparing {dataset_name} noise manifests...", flush=True)

        items, split_path, split_bytes = prepare_dataset(
            dataset_name, spec
        )
        prepared.extend(items)
        original_splits.append((split_path, split_bytes))

    # Validate every existing output before writing any new file.
    artifacts = []

    for item in prepared:
        path = item["path"]
        checksum_path = path.with_name(path.name + ".sha256")
        checksum_content = (
            f"{item['sha256']}  {path.name}\n"
        ).encode("utf-8")

        artifacts.append((path, item["content"]))
        artifacts.append((checksum_path, checksum_content))

    for path, content in artifacts:
        if path.exists():
            require(
                path.read_bytes() == content,
                f"Existing output differs: {path}\n"
                "Stopped without overwriting.",
            )

    for path, content in artifacts:
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(content)

        require(
            path.read_bytes() == content,
            f"Saved file verification failed: {path}",
        )

    # The original split files include clean validation/test labels.
    # Verify that their complete contents remain unchanged.
    for path, original_bytes in original_splits:
        require(
            path.read_bytes() == original_bytes,
            f"Original clean split was modified: {path}",
        )

    for item in prepared:
        print(f"\n[OK] {item['path'].name}")
        print(
            f"  noise={item['percentage']}%, "
            f"corrupted={item['count']}, "
            f"per_class_min={item['class_min']}, "
            f"per_class_max={item['class_max']}"
        )
        print(f"  SHA256={item['sha256']}")

    print("\nexact_corruption_counts=PASS")
    print("wrong_labels_differ_from_clean_labels=PASS")
    print("nested_corruption_sets=PASS")
    print("shared_wrong_labels_across_noise_levels=PASS")
    print("original_clean_split_files_unchanged=PASS")
    print("pairing=one_shared_manifest_for_Random_and_Procedural")
    print("NOISE_MANIFEST_CHECKS_COMPLETED")
    print("No training was performed.")


if __name__ == "__main__":
    main()