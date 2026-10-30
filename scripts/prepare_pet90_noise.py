"""Prepare fixed, paired label-noise manifests for the Pet 90/10 protocol."""

import copy
import hashlib
import json
import os
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "manifests"
SPLIT_PATH = MANIFEST_ROOT / "pet90_10_seed11_splits.json"

EXPECTED_SPLIT_SHA256 = (
    "2bf0bc11ba4452bc3396081962135d4498d01c458e692e7134f6e44693740b3e"
)

SEED = 11
NUM_CLASSES = 37
NOISE_LEVELS = (0, 10, 20, 30)
EXPECTED_COUNTS = {0: 0, 10: 331, 20: 662, 30: 994}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def derived_seed(purpose):
    content = f"{SEED}:{purpose}".encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(content).digest()[:8], "little"
    ) % (2**63)


def encode_json(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def read_split():
    content = SPLIT_PATH.read_bytes()
    digest = sha256(content)

    require(
        digest == EXPECTED_SPLIT_SHA256,
        "Fixed split differs from the previously approved split",
    )

    checksum_path = SPLIT_PATH.with_name(SPLIT_PATH.name + ".sha256")
    checksum_content = checksum_path.read_bytes()
    require(
        checksum_content.decode("utf-8").split()
        == [digest, SPLIT_PATH.name],
        "Split checksum sidecar mismatch",
    )

    manifest = json.loads(content)
    require(
        manifest["schema_version"] == 2
        and manifest["protocol"] == "pet90_10_v1"
        and manifest["dataset"] == "oxford_pet"
        and manifest["seed"] == SEED
        and manifest["num_classes"] == NUM_CLASSES,
        "Unexpected split metadata",
    )

    partitions = manifest["partitions"]
    require(
        set(partitions) == {"train", "validation", "test"},
        "Unexpected partitions",
    )

    identities = {}
    for split, expected_count in (
        ("train", 3312),
        ("validation", 368),
        ("test", 3669),
    ):
        partition = partitions[split]
        ids = partition["sample_ids"]
        indices = partition["indices"]
        labels = partition["labels_clean"]

        require(
            len(ids) == len(indices) == len(labels) == expected_count,
            f"Incorrect partition size: {split}",
        )
        require(
            len(set(ids)) == len(set(indices)) == expected_count,
            f"Duplicate samples: {split}",
        )
        require(
            all(type(label) is int and 0 <= label < NUM_CLASSES
                for label in labels),
            f"Invalid labels: {split}",
        )
        require(
            set(labels) == set(range(NUM_CLASSES)),
            f"Missing classes: {split}",
        )

        expected_source = "test" if split == "test" else "trainval"
        require(
            partition["source_split"] == expected_source,
            f"Unexpected source split: {split}",
        )
        identities[split] = set(ids)

    require(
        identities["train"].isdisjoint(identities["validation"])
        and identities["train"].isdisjoint(identities["test"])
        and identities["validation"].isdisjoint(identities["test"]),
        "Partition image identities overlap",
    )

    return manifest, content, checksum_content


def class_quotas(class_counts, noise_percent, tie_rank):
    total = sum(class_counts)
    # Round to the nearest whole image; exact halves round upward.
    target = (total * noise_percent + 50) // 100

    require(
        target == EXPECTED_COUNTS[noise_percent],
        "Unexpected global corruption count",
    )

    quotas = [
        count * noise_percent // 100 for count in class_counts
    ]
    remaining = target - sum(quotas)

    # Largest remainders first; use a fixed seeded order to resolve ties.
    priority = sorted(
        range(NUM_CLASSES),
        key=lambda class_id: (
            -(class_counts[class_id] * noise_percent % 100),
            tie_rank[class_id],
        ),
    )
    require(
        0 <= remaining <= NUM_CLASSES,
        "Invalid class-quota allocation",
    )

    for class_id in priority[:remaining]:
        quotas[class_id] += 1

    require(sum(quotas) == target, "Class quotas do not sum to target")
    require(
        all(0 <= quota <= count
            for quota, count in zip(quotas, class_counts)),
        "Class quota outside valid range",
    )
    require(
        all(
            abs(100 * quota - count * noise_percent) < 100
            for quota, count in zip(quotas, class_counts)
        ),
        "Per-class allocation differs from its target by at least one image",
    )
    return quotas


def build_all(split_manifest):
    source = split_manifest["partitions"]["train"]
    clean = list(source["labels_clean"])
    sample_ids = list(source["sample_ids"])
    total = len(clean)

    members = [
        [i for i, label in enumerate(clean) if label == class_id]
        for class_id in range(NUM_CLASSES)
    ]
    class_counts = [len(rows) for rows in members]

    # Each class has one permanent ordering of candidate corrupted images.
    selection_orders = []
    for class_id, rows in enumerate(members):
        order = list(rows)
        random.Random(
            derived_seed(f"noise-selection:class:{class_id}")
        ).shuffle(order)
        selection_orders.append(order)

    tie_order = list(range(NUM_CLASSES))
    random.Random(derived_seed("noise-quota-ties")).shuffle(tie_order)
    tie_rank = {
        class_id: rank for rank, class_id in enumerate(tie_order)
    }

    # Each image receives one permanent candidate wrong label.
    # Draw uniformly from the other 36 classes, excluding its true class.
    wrong_labels = []
    for sample_id, true_label in zip(sample_ids, clean):
        candidate = random.Random(
            derived_seed(f"wrong-label:{sample_id}")
        ).randrange(NUM_CLASSES - 1)

        wrong_label = candidate + int(candidate >= true_label)
        require(wrong_label != true_label, "Wrong label equals clean label")
        wrong_labels.append(wrong_label)

    outputs = {}
    previous_changed = set()
    previous_observed = list(clean)
    previous_quotas = [0] * NUM_CLASSES

    for noise_percent in NOISE_LEVELS:
        quotas = class_quotas(class_counts, noise_percent, tie_rank)
        require(
            all(current >= previous
                for current, previous in zip(quotas, previous_quotas)),
            "Per-class quotas are not nested",
        )

        changed_positions = {
            position
            for class_id in range(NUM_CLASSES)
            for position in selection_orders[class_id][:quotas[class_id]]
        }

        observed = list(clean)
        for position in changed_positions:
            observed[position] = wrong_labels[position]

        flags = [
            observed[i] != clean[i] for i in range(total)
        ]
        actual_changed = {
            i for i, changed in enumerate(flags) if changed
        }

        require(
            actual_changed == changed_positions,
            "Corruption flags do not match selected images",
        )
        require(
            len(actual_changed) == EXPECTED_COUNTS[noise_percent],
            "Incorrect corruption count",
        )
        require(
            all(0 <= label < NUM_CLASSES for label in observed),
            "Observed label outside valid range",
        )
        require(
            all(observed[i] != clean[i] for i in actual_changed),
            "A corrupted image retained its clean label",
        )
        require(
            all(observed[i] == clean[i]
                for i in range(total) if i not in actual_changed),
            "An unselected image changed label",
        )
        require(
            previous_changed <= actual_changed,
            "Corruption sets are not nested",
        )
        require(
            all(observed[i] == previous_observed[i]
                for i in previous_changed),
            "Wrong labels changed across noise levels",
        )

        actual_per_class = [
            sum(flags[i] for i in members[class_id])
            for class_id in range(NUM_CLASSES)
        ]
        require(actual_per_class == quotas, "Per-class counts differ from quotas")

        training = copy.deepcopy(source)
        training["labels_observed"] = observed
        training["is_corrupted"] = flags

        for field in (
            "source_split", "indices", "sample_ids", "labels_clean"
        ):
            require(
                training[field] == source[field],
                f"Original clean training data changed: {field}",
            )

        outputs[noise_percent] = {
            "schema_version": 2,
            "protocol": "pet90_noise_v1",
            "dataset": "oxford_pet",
            "num_classes": NUM_CLASSES,
            "seed": SEED,
            "noise_percent": noise_percent,
            "actual_noise_percent": 100 * len(actual_changed) / total,
            "training_count": total,
            "corrupted_count": len(actual_changed),
            "corrupted_count_per_clean_class": actual_per_class,
            "shared_by_initializations": ["random", "procedural"],
            "split_manifest": SPLIT_PATH.relative_to(ROOT).as_posix(),
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "noise_method": {
                "type": "Symmetric wrong-class label noise",
                "global_count": "Nearest integer; exact halves rounded upward",
                "class_allocation": (
                    "Largest remainder allocation by clean class; "
                    "fixed seed-derived tie order"
                ),
                "image_selection": (
                    "One fixed seed-derived permutation per clean class; "
                    "select a prefix at each noise level"
                ),
                "wrong_label": (
                    "One fixed seed-derived label per image, sampled "
                    "uniformly from the other 36 classes"
                ),
                "nested_corruption_sets": True,
                "shared_wrong_labels_across_noise_levels": True,
            },
            "validation_and_test": (
                "Read clean labels only from the fixed split manifest; "
                "never apply this training-noise mapping to them"
            ),
            "training": training,
        }

        previous_changed = actual_changed
        previous_observed = observed
        previous_quotas = quotas

    return outputs


def main():
    split_manifest, split_content, checksum_content = read_split()
    outputs = build_all(split_manifest)

    # Repeat from the same seed and inputs; require byte-for-byte identity.
    repeated = build_all(split_manifest)
    require(
        all(
            encode_json(outputs[noise]) == encode_json(repeated[noise])
            for noise in NOISE_LEVELS
        ),
        "Noise generation is not reproducible",
    )

    files_to_write = []
    summaries = []

    for noise_percent in NOISE_LEVELS:
        manifest = outputs[noise_percent]
        path = MANIFEST_ROOT / f"pet90_seed11_noise{noise_percent:02d}.json"
        content = encode_json(manifest)
        digest = sha256(content)

        checksum_path = path.with_name(path.name + ".sha256")
        checksum = f"{digest}  {path.name}\n".encode("utf-8")

        files_to_write.extend([
            (path, content),
            (checksum_path, checksum),
        ])
        summaries.append((path, digest, manifest))

    # Check every existing destination before creating any new files.
    # Never overwrite a different manifest.
    for path, content in files_to_write:
        if path.exists():
            require(
                path.read_bytes() == content,
                f"Existing file differs; refusing to overwrite: {path}",
            )

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    for path, content in files_to_write:
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        require(path.read_bytes() == content, f"Saved-file mismatch: {path}")

    require(
        SPLIT_PATH.read_bytes() == split_content,
        "Fixed split file changed",
    )
    require(
        SPLIT_PATH.with_name(SPLIT_PATH.name + ".sha256").read_bytes()
        == checksum_content,
        "Fixed split checksum file changed",
    )

    for path, digest, manifest in summaries:
        counts = manifest["corrupted_count_per_clean_class"]
        print(f"\n[OK] {path.name}")
        print(
            f"  nominal_noise={manifest['noise_percent']}%, "
            f"actual_noise={manifest['actual_noise_percent']:.6f}%"
        )
        print(
            f"  corrupted={manifest['corrupted_count']}, "
            f"per_class_min={min(counts)}, per_class_max={max(counts)}"
        )
        print(f"  SHA256={digest}")

    print("\nnoise_reproducibility=PASS")
    print("exact_corruption_counts=PASS")
    print("class_stratified_corruption=PASS")
    print("wrong_labels_differ_from_clean_labels=PASS")
    print("nested_corruption_sets=PASS")
    print("shared_wrong_labels_across_noise_levels=PASS")
    print("training_image_ids_and_clean_labels_unchanged=PASS")
    print("fixed_split_files_unchanged=PASS")
    print("pairing=one_shared_manifest_for_Random_and_Procedural")
    print("validation_and_test_noise_generated=False")
    print("images_opened=False")
    print("formal_training_started=False")
    print("PET90_NOISE_MANIFEST_CHECKS_COMPLETED")
    print("status=PASS")


if __name__ == "__main__":
    main()