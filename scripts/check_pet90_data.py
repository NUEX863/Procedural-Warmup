"""Check Pet 90/10 data pairing and initialization without model training."""

import argparse
import gc
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.pet90_data import (
    Pet90Dataset,
    EpochKeySampler,
    accumulation_group_sizes,
    file_sha256,
    load_config,
    make_loader,
    require,
)
from src.model_factory import (
    FRESH_MODEL_KEYS,
    build_model,
    official_utils,
    read_checkpoint,
    validate_transfer,
)


def check_initialization():
    random_model = build_model("random", num_classes=37, seed=11)
    procedural_model = build_model("procedural", num_classes=37, seed=11)

    # Independently construct the author's model using the same seed.
    utils = official_utils()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(11)
        reference = utils.build_model(SimpleNamespace(
            model="vit_tiny",
            nb_classes=37,
            drop_path=0.0,
        ))

    random_state = random_model.state_dict()
    procedural_state = procedural_model.state_dict()

    require(
        set(random_state) == set(reference.state_dict())
        and len(random_state) == 152,
        "Unexpected state dictionary",
    )
    require(
        all(
            torch.equal(value, reference.state_dict()[name])
            for name, value in random_state.items()
        ),
        "Random initialization differs from the author",
    )

    warmup = read_checkpoint()
    transferred = validate_transfer(procedural_model, warmup)
    require(len(transferred) == 146, "Unexpected transfer count")

    require(
        all(
            torch.equal(procedural_state[name], value)
            for name, value in transferred.items()
        ),
        "A transferred tensor differs from the published checkpoint",
    )
    require(
        all(
            torch.equal(random_state[name], procedural_state[name])
            for name in FRESH_MODEL_KEYS
        ),
        "Fresh parameters differ between initializations",
    )

    # Use the author's full checkpoint-loading operation as a reference.
    utils.load_state_dict(reference, warmup, prefix="")
    require(
        all(
            torch.equal(value, reference.state_dict()[name])
            for name, value in procedural_state.items()
        ),
        "Procedural initialization differs from the author",
    )

    for model in (random_model, procedural_model):
        require(
            all(parameter.requires_grad for parameter in model.parameters()),
            "Unexpected frozen parameters",
        )
        require(
            all(torch.isfinite(value).all().item()
                for value in model.state_dict().values()),
            "Non-finite model state",
        )

    print("[PASS] Random: all 152 state tensors equal to author")
    print("[PASS] Procedural: all 152 state tensors equal to author")
    print("[PASS] 146 transferred tensors; 6 shared fresh tensors")
    print("[PASS] All model parameters remain trainable")

    del reference, random_model, procedural_model, warmup
    gc.collect()


def digest_loader(loader, dataset, expected_epoch):
    digest = hashlib.sha256()
    seen = set()
    batch_sizes = []

    for batch in loader:
        count = len(batch["label"])
        images = batch["image"]

        require(
            images.shape == (count, 3, 224, 224)
            and images.dtype == torch.float32
            and torch.isfinite(images).all().item(),
            "Invalid image batch",
        )
        require(
            torch.equal(batch["label"], batch["clean_label"])
            and not batch["is_corrupted"].any().item(),
            "This baseline must use clean labels",
        )
        require(
            batch["epoch"].eq(expected_epoch).all().item(),
            "Worker used an incorrect epoch",
        )

        positions = batch["position"].tolist()
        labels = batch["label"].tolist()
        source_indices = batch["source_index"].tolist()

        for position, sample_id, label, source_index in zip(
            positions, batch["sample_id"], labels, source_indices
        ):
            require(position not in seen, "Duplicate sample")
            seen.add(position)

            require(
                sample_id == dataset.sample_ids[position]
                and label == dataset.labels[position]
                and source_index == dataset.indices[position],
                "Batch identity/label mismatch",
            )

            digest.update((sample_id + "\n").encode("utf-8"))
            digest.update(int(label).to_bytes(1, "little"))

        digest.update(images.contiguous().numpy().tobytes())
        batch_sizes.append(count)

    require(seen == set(range(len(dataset))), "Missing samples")
    return digest.hexdigest(), batch_sizes


def main():
    cfg = load_config()
    torch.set_num_threads(cfg["cpu_threads"])
    torch.random.default_generator.manual_seed(cfg["seed"])

    protected_paths = [
        ROOT / "src/model_factory.py",
        ROOT / "src/data_pipeline.py",
        ROOT / "src/training_core.py",
        ROOT / "scripts/run_training.py",
        ROOT / "config/training.json",
        ROOT / "config/pet90_baseline.json",
        ROOT / "manifests/oxford_pet_seed11_splits.json",
        ROOT / cfg["split_manifest"],
    ]
    before = {str(path): file_sha256(path) for path in protected_paths}

    # This rejection happens before any test dataset can be constructed.
    try:
        Pet90Dataset("test", cfg)
    except ValueError:
        print("[PASS] Test dataset access is blocked")
    else:
        raise RuntimeError("Test access was not blocked")

    train_r = Pet90Dataset("train", cfg)
    train_p = Pet90Dataset("train", cfg)
    validation = Pet90Dataset("validation", cfg)

    require(len(train_r) == len(train_p) == 3312, "Wrong training size")
    require(len(validation) == 368, "Wrong validation size")
    print("[PASS] Fixed split: train=3312, validation=368")

    check_initialization()

    # Advancing unrelated RNG must not alter data order or augmentation.
    torch.rand(73)
    loader = make_loader(train_r, cfg, epoch=0)
    random_digest, batch_sizes = digest_loader(loader, train_r, 0)

    require(
        batch_sizes == [16] * 207,
        "Expected 207 micro batches of 16 images",
    )

    # Reuse the same persistent workers after changing epoch.
    loader.sampler.set_epoch(1)
    next_batch = next(iter(loader))

    expected_keys = list(
        EpochKeySampler(train_r, cfg["seed"], epoch=1)
    )[:16]

    require(
        next_batch["position"].tolist()
        == [position for _, position in expected_keys],
        "Epoch-1 sample order mismatch",
    )
    require(
        next_batch["epoch"].eq(1).all().item(),
        "Persistent workers retained the old epoch",
    )

    for index, key in enumerate(expected_keys):
        reference = train_p[key]
        require(
            torch.equal(next_batch["image"][index], reference["image"])
            and next_batch["sample_id"][index] == reference["sample_id"],
            "Persistent-worker augmentation differs from direct loading",
        )

    print("[PASS] Persistent workers receive the new epoch correctly")
    del next_batch, loader
    gc.collect()

    torch.rand(137)

    # Rebuild the loader in the main process, as an independent replay.
    loader = make_loader(train_p, cfg, epoch=0, num_workers=0)
    procedural_digest, replay_sizes = digest_loader(loader, train_p, 0)
    del loader

    require(
        random_digest == procedural_digest
        and batch_sizes == replay_sizes,
        "Random/Procedural actual image batches differ",
    )
    print("[PASS] Random/Procedural orders, labels and augmented tensors match")
    print("[PASS] Epoch replay is independent of worker count and unrelated RNG")
    print(f"Epoch-1 data SHA256: {random_digest}")

    order0 = [
        position for _, position
        in EpochKeySampler(train_r, cfg["seed"], epoch=0)
    ]
    order1 = [
        position for _, position
        in EpochKeySampler(train_r, cfg["seed"], epoch=1)
    ]
    require(order0 != order1, "Training order did not change across epochs")

    probes = (0, 100, 500, 1000, 2000, 3000)
    require(
        any(
            not torch.equal(
                train_r[(0, position)]["image"],
                train_r[(1, position)]["image"],
            )
            for position in probes
        ),
        "Training augmentation did not change across epochs",
    )
    print("[PASS] Training order and augmentation vary across epochs")

    rng_before = torch.random.get_rng_state().clone()
    train_r[(2, 0)]
    require(
        torch.equal(rng_before, torch.random.get_rng_state()),
        "Augmentation changed the caller's CPU RNG state",
    )
    print("[PASS] Augmentation preserves the caller's CPU RNG state")

    loader = make_loader(validation, cfg)
    val_digest1, val_sizes = digest_loader(loader, validation, -1)
    del loader

    torch.rand(91)
    loader = make_loader(validation, cfg, epoch=9)
    val_digest2, _ = digest_loader(loader, validation, -1)
    del loader

    require(val_digest1 == val_digest2, "Validation preprocessing is random")
    require(
        val_sizes == [32] * 11 + [16],
        "Unexpected validation batches",
    )
    print("[PASS] Validation is deterministic and labels remain clean")

    groups = accumulation_group_sizes(len(train_r), cfg)
    require(
        groups == [64] * 51 + [48]
        and sum(groups) == 3312,
        "Incorrect effective-batch group plan",
    )
    print("[PASS] Accumulation plan: 51 groups of 64, final group of 48")

    for path, digest in before.items():
        require(file_sha256(path) == digest, f"File changed: {path}")
    print("[PASS] Existing code, configuration and manifests unchanged")

    print("\nPET90_DATA_CHECKS_COMPLETED")
    print("status=PASS")
    print("train_samples=3312")
    print("validation_samples=368")
    print("micro_batch_size=16")
    print("accumulation_steps=4")
    print("effective_batch_size=64")
    print("planned_updates_per_epoch=52")
    print("last_effective_batch_size=48")
    print("test_images_opened=False")
    print("model_parameters_updated=False")
    print("formal_training_started=False")
    print("No speed benchmark was performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    main()