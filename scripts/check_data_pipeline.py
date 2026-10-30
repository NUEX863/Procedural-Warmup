"""Check manifest loading and paired batches. No training or timing."""

import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import (
    build_dataset,
    make_loader,
    require,
)


def compare_pair(dataset_name, percentage):
    # These represent the data loaders for the two initialization runs.
    random_data = build_dataset(
        dataset_name, "train", noise_percent=percentage
    )
    procedural_data = build_dataset(
        dataset_name, "train", noise_percent=percentage
    )

    require(
        random_data.manifest_hashes == procedural_data.manifest_hashes,
        "R/P manifest hashes differ",
    )
    require(
        random_data.sample_ids == procedural_data.sample_ids,
        "R/P training image identities differ",
    )
    require(
        np.array_equal(
            random_data.observed_labels,
            procedural_data.observed_labels,
        ),
        "R/P observed labels differ",
    )

    # Check two complete sampling orders without reading all the images.
    order_loader_r = make_loader(random_data)
    order_loader_p = make_loader(procedural_data)

    for epoch in range(2):
        order_r = list(order_loader_r.sampler)
        order_p = list(order_loader_p.sampler)

        require(
            order_r == order_p,
            f"R/P sample order differs in check epoch {epoch + 1}",
        )
        require(
            sorted(order_r) == list(range(len(random_data))),
            "An epoch does not contain every training sample exactly once",
        )

    # Fresh loaders: the following batches start at the first epoch.
    loader_r = make_loader(random_data)
    loader_p = make_loader(procedural_data)

    iterator_r = iter(loader_r)
    iterator_p = iter(loader_p)

    for batch_number in range(2):
        batch_r = next(iterator_r)
        batch_p = next(iterator_p)

        require(
            batch_r["sample_id"] == batch_p["sample_id"],
            "R/P actual batch image identities differ",
        )
        for key in (
            "image",
            "label",
            "clean_label",
            "is_corrupted",
            "source_index",
        ):
            require(
                torch.equal(batch_r[key], batch_p[key]),
                f"R/P actual batch field differs: {key}",
            )

        images = batch_r["image"]
        require(
            tuple(images.shape) == (16, 3, 224, 224),
            f"Unexpected batch image shape: {tuple(images.shape)}",
        )
        require(
            images.dtype == torch.float32
            and torch.isfinite(images).all().item(),
            "Images have incorrect dtype or non-finite values",
        )
        require(
            torch.equal(
                batch_r["label"] != batch_r["clean_label"],
                batch_r["is_corrupted"],
            ),
            "Actual batch corruption flags are incorrect",
        )

    result = {
        "dataset": dataset_name,
        "noise_percent": percentage,
        "train_count": len(random_data),
        "corrupted_count": int(random_data.is_corrupted.sum()),
        "manifest_hashes": random_data.manifest_hashes,
        "all_sample_ids_and_labels_equal_R_P": True,
        "two_complete_epoch_orders_equal_R_P": True,
        "first_two_real_batches_equal_R_P": True,
        "image_shape_dtype_and_finiteness": "PASS",
    }

    print(
        f"[PASS] {dataset_name} noise={percentage}%: "
        "manifests, labels, sampling orders and actual batches match",
        flush=True,
    )
    return result


def expect_denied(**kwargs):
    try:
        build_dataset(**kwargs)
    except ValueError:
        return
    raise RuntimeError(f"Access should have been rejected: {kwargs}")


def main():
    torch.set_num_threads(4)
    results = []

    for dataset_name in ("cifar100", "oxford_pet"):
        for percentage in (0, 10, 20, 30):
            results.append(compare_pair(dataset_name, percentage))

    validation = build_dataset("cifar100", "validation")
    require(len(validation) == 3000, "Incorrect validation count")
    require(
        not validation.is_corrupted.any(),
        "CIFAR validation contains altered labels",
    )

    val_loader = make_loader(validation)
    require(
        list(val_loader.sampler) == list(range(3000)),
        "Validation order is not fixed",
    )

    val_batch = next(iter(val_loader))
    require(
        torch.equal(val_batch["label"], val_batch["clean_label"]),
        "Actual validation batch labels are not clean",
    )
    require(
        tuple(val_batch["image"].shape) == (16, 3, 224, 224)
        and torch.isfinite(val_batch["image"]).all().item(),
        "Invalid validation image batch",
    )

    # These calls must fail before opening either official test dataset.
    expect_denied(dataset_name="oxford_pet", split="validation")
    expect_denied(dataset_name="oxford_pet", split="test")
    expect_denied(dataset_name="cifar100", split="test")

    print(json.dumps({
        "status": "PASS",
        "checks": results,
        "cifar100_validation_count": 3000,
        "cifar100_validation_labels_clean": True,
        "pet_validation_request_rejected": True,
        "test_access_requires_explicit_permission": True,
        "test_images_loaded": False,
        "scope": (
            "All selected source IDs and labels checked; "
            "two complete sampling orders and first two real training "
            "batches per noise setting checked. No training or timing."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()