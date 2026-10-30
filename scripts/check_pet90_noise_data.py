"""Check actual Pet90 batches without model creation or training."""

import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.pet90_data import (
    Pet90Dataset,
    accumulation_group_sizes,
    file_sha256,
    load_config,
    make_loader,
    require,
)
from src.pet90_noise_data import (
    EXPECTED_COUNTS,
    NOISE_HASHES,
    Pet90NoiseDataset,
    noise_path,
)


def add_json(digest, value):
    digest.update(
        (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")
    )


def scan(loader, dataset, epoch):
    """Validate and fingerprint every actual batch, including image tensors."""
    image_digest = hashlib.sha256()
    batch_digest = hashlib.sha256()

    if dataset.split == "train":
        expected_order = [position for _, position in loader.sampler]
    else:
        expected_order = list(range(len(dataset)))

    seen = []
    changed = {}
    offset = 0

    for batch in loader:
        positions = batch["position"].tolist()
        size = len(positions)

        require(
            positions == expected_order[offset:offset + size],
            "Actual batch order differs from the sampler",
        )
        require(
            tuple(batch["image"].shape) == (size, 3, 224, 224)
            and batch["image"].dtype == torch.float32,
            "Unexpected image tensor shape or dtype",
        )
        require(
            bool(torch.isfinite(batch["image"]).all()),
            "Image tensors contain non-finite values",
        )

        ids = [dataset.sample_ids[p] for p in positions]
        indices = [dataset.indices[p] for p in positions]
        labels = [dataset.labels[p] for p in positions]

        if isinstance(dataset, Pet90NoiseDataset):
            clean = [dataset.clean_labels[p] for p in positions]
            flags = [dataset.is_corrupted[p] for p in positions]
        else:
            clean = list(labels)
            flags = [False] * size

        expected_epoch = epoch if dataset.split == "train" else -1

        require(list(batch["sample_id"]) == ids, "Image IDs differ")
        require(
            batch["source_index"].tolist() == indices,
            "Source indices differ",
        )
        require(batch["label"].tolist() == labels, "Observed labels differ")
        require(
            batch["clean_label"].tolist() == clean,
            "Clean labels differ",
        )
        require(
            batch["is_corrupted"].tolist() == flags,
            "Corruption flags differ",
        )
        require(
            batch["epoch"].tolist() == [expected_epoch] * size,
            "Worker received the wrong epoch",
        )
        require(
            flags == [a != b for a, b in zip(labels, clean)],
            "Actual batch corruption does not match its labels",
        )

        image_bytes = batch["image"].contiguous().numpy().tobytes()
        add_json(image_digest, [positions, ids, indices])
        image_digest.update(image_bytes)

        add_json(
            batch_digest,
            [positions, ids, indices, labels, clean, flags, expected_epoch],
        )
        batch_digest.update(image_bytes)

        for sample_id, label, flag in zip(ids, labels, flags):
            if flag:
                changed[sample_id] = label

        seen.extend(positions)
        offset += size

    require(
        offset == len(dataset)
        and sorted(seen) == list(range(len(dataset))),
        "Missing or repeated samples",
    )
    return {
        "count": offset,
        "image_sha256": image_digest.hexdigest(),
        "batch_sha256": batch_digest.hexdigest(),
        "changed": changed,
    }


def check_next_epoch(loader, dataset):
    """Check that already-running persistent workers receive epoch 1."""
    loader.sampler.set_epoch(1)
    iterator = iter(loader)
    batch = next(iterator)

    expected_keys = list(loader.sampler)[:len(batch["position"])]
    require(
        batch["position"].tolist() == [p for _, p in expected_keys],
        "Epoch-1 sampling order differs",
    )

    for row, key in enumerate(expected_keys):
        expected = dataset[key]
        require(
            torch.equal(batch["image"][row], expected["image"]),
            "Persistent worker used the wrong augmentation epoch",
        )
        for field in (
            "label", "clean_label", "is_corrupted",
            "source_index", "position", "epoch",
        ):
            require(
                batch[field][row].item() == expected[field],
                f"Epoch-1 field differs: {field}",
            )
        require(
            batch["sample_id"][row] == expected["sample_id"],
            "Epoch-1 image identity differs",
        )

    del iterator


def main():
    cfg = load_config()
    torch.set_num_threads(cfg["cpu_threads"])

    protected = [
        ROOT / "config" / "pet90_baseline.json",
        ROOT / cfg["split_manifest"],
        ROOT / "src" / "pet90_data.py",
        ROOT / "src" / "model_factory.py",
        ROOT / "src" / "training_core.py",
        ROOT / "scripts" / "run_pet90_training.py",
    ]
    split_path = ROOT / cfg["split_manifest"]
    protected.append(split_path.with_name(split_path.name + ".sha256"))

    for noise in NOISE_HASHES:
        path = noise_path(noise)
        protected.extend([path, path.with_name(path.name + ".sha256")])

    before = {path: file_sha256(path) for path in protected}

    # Fail before image access if anyone requests a test dataset.
    try:
        Pet90NoiseDataset("test", cfg)
    except ValueError:
        pass
    else:
        raise RuntimeError("Test access was not rejected")

    try:
        Pet90NoiseDataset("validation", cfg, noise_percent=10)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Validation accepted a noise setting")

    reference_images = None
    zero_noise_result = None
    previous_changed = {}

    for noise in NOISE_HASHES:
        print(f"\nChecking noise={noise}% ...", flush=True)
        pair_results = []

        # Independent reconstructions of the input streams for R and P.
        # Different global RNG states and worker counts must not change them.
        for initialization, workers, global_seed in (
            ("random", cfg["num_workers"], 12345),
            ("procedural", 0, 98765),
        ):
            torch.manual_seed(global_seed)
            dataset = Pet90NoiseDataset("train", cfg, noise)
            loader = make_loader(
                dataset, cfg, epoch=0, num_workers=workers
            )

            result = scan(loader, dataset, epoch=0)
            require(
                len(result["changed"]) == EXPECTED_COUNTS[noise],
                "Actual corrupted sample count differs",
            )

            if initialization == "random":
                check_next_epoch(loader, dataset)

            pair_results.append(result)
            print(
                f"  {initialization}: samples={result['count']}, "
                f"corrupted={len(result['changed'])}, workers={workers}",
                flush=True,
            )

            del loader, dataset
            gc.collect()

        require(
            pair_results[0] == pair_results[1],
            "Random/Procedural actual input streams differ",
        )

        result = pair_results[0]
        if reference_images is None:
            reference_images = result["image_sha256"]
            zero_noise_result = result

        require(
            result["image_sha256"] == reference_images,
            "Noise level changed image order or augmentation",
        )

        changed = result["changed"]
        require(
            set(previous_changed).issubset(changed),
            "Actual corrupted image sets are not nested",
        )
        require(
            all(changed[sample_id] == label
                for sample_id, label in previous_changed.items()),
            "Actual wrong labels differ across noise levels",
        )
        previous_changed = changed

        print(
            f"[PASS] noise={noise}%: images, labels, order and "
            "augmentation match between Random and Procedural",
            flush=True,
        )

    print("\nChecking zero-noise compatibility ...", flush=True)
    clean_train = Pet90Dataset("train", cfg)
    clean_loader = make_loader(clean_train, cfg, epoch=0, num_workers=0)
    clean_result = scan(clean_loader, clean_train, epoch=0)
    require(
        clean_result == zero_noise_result,
        "Zero-noise adapter differs from the verified clean pipeline",
    )
    del clean_loader, clean_train
    print("[PASS] Zero-noise pipeline unchanged", flush=True)

    print("\nChecking clean validation ...", flush=True)
    validation = Pet90NoiseDataset("validation", cfg)
    loader = make_loader(validation, cfg, num_workers=0)
    first = scan(loader, validation, epoch=-1)
    torch.manual_seed(54321)
    second = scan(loader, validation, epoch=-1)
    require(first == second, "Validation is not deterministic")
    require(
        first["count"] == 368 and not first["changed"],
        "Validation labels are not all clean",
    )
    del loader, validation
    print("[PASS] All 368 validation samples remain clean", flush=True)

    groups = accumulation_group_sizes(3312, cfg)
    require(
        groups == [64] * 51 + [48],
        "Unexpected accumulation groups",
    )

    require(
        before == {path: file_sha256(path) for path in protected},
        "A protected source, config or manifest changed",
    )

    print("\nactual_batch_pairing=PASS")
    print("cross_noise_image_order_and_augmentation=PASS")
    print("nested_actual_corruption_sets=PASS")
    print("shared_actual_wrong_labels=PASS")
    print("persistent_worker_epoch_update=PASS")
    print("zero_noise_matches_clean_pipeline=PASS")
    print("clean_validation=PASS")
    print("accumulation_groups=51*64+48")
    print("protected_files_unchanged=PASS")
    print("test_images_opened=False")
    print("model_created=False")
    print("formal_training_started=False")
    print("PET90_NOISE_DATA_CHECKS_COMPLETED")
    print("status=PASS")


if __name__ == "__main__":
    main()