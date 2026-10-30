"""Read fixed experiment manifests without generating new splits or noise."""

import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MANIFEST_ROOT = ROOT / "manifests"

SEED = 11
NUM_CLASSES = {"cifar100": 100, "oxford_pet": 37}
NOISE_PERCENTAGES = (0, 10, 20, 30)

# Deterministic preprocessing shared by both initializations.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_verified_manifest(path):
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    checksum_path = path.with_name(path.name + ".sha256")
    recorded = checksum_path.read_text(encoding="utf-8").split()

    require(
        recorded == [digest, path.name],
        f"Manifest checksum mismatch: {path}",
    )

    manifest = json.loads(content)
    require(
        manifest.get("schema_version") == 1,
        f"Unsupported manifest schema: {path}",
    )
    return manifest, digest


@lru_cache(maxsize=4)
def load_official_source(dataset_name, source_split):
    """Cache original datasets; wrappers never modify their labels."""
    if dataset_name == "cifar100":
        require(source_split in {"train", "test"}, "Invalid CIFAR split")

        base = datasets.CIFAR100(
            root=str(DATA_ROOT / "cifar100"),
            train=source_split == "train",
            download=False,
        )
        labels = np.asarray(base.targets, dtype=np.int64)
        sample_ids = [
            f"{source_split}/{i}" for i in range(len(base))
        ]

    elif dataset_name == "oxford_pet":
        require(
            source_split in {"trainval", "test"},
            "Invalid Pet split",
        )

        base = datasets.OxfordIIITPet(
            root=str(DATA_ROOT),
            split=source_split,
            target_types="category",
            download=False,
        )

        # These attributes match the project's pinned torchvision version.
        labels = np.asarray(base._labels, dtype=np.int64)
        pet_root = DATA_ROOT / "oxford-iiit-pet"
        sample_ids = [
            Path(path).relative_to(pet_root).as_posix()
            for path in base._images
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return base, labels, sample_ids


class ManifestDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        split,
        partition,
        observed_labels,
        manifest_hashes,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.num_classes = NUM_CLASSES[dataset_name]
        self.manifest_hashes = manifest_hashes

        self.indices = np.asarray(
            partition["indices"], dtype=np.int64
        )
        self.sample_ids = list(partition["sample_ids"])
        self.clean_labels = np.asarray(
            partition["labels_clean"], dtype=np.int64
        )
        self.observed_labels = np.asarray(
            observed_labels, dtype=np.int64
        )
        self.is_corrupted = (
            self.observed_labels != self.clean_labels
        )

        base, source_labels, source_ids = load_official_source(
            dataset_name, partition["source_split"]
        )
        self.base = base

        count = len(self.indices)
        require(
            len(self.sample_ids)
            == len(self.clean_labels)
            == len(self.observed_labels)
            == count,
            "Manifest field lengths differ",
        )
        require(
            len(set(self.sample_ids)) == count
            and len(np.unique(self.indices)) == count,
            "Duplicate manifest samples",
        )
        require(
            np.all(
                (self.indices >= 0)
                & (self.indices < len(source_labels))
            ),
            "Source index outside official dataset",
        )
        require(
            np.array_equal(
                source_labels[self.indices], self.clean_labels
            ),
            "Manifest clean labels differ from official labels",
        )
        require(
            [
                source_ids[int(index)]
                for index in self.indices
            ] == self.sample_ids,
            "Manifest image IDs do not match official source indices",
        )
        require(
            np.all(
                (self.observed_labels >= 0)
                & (self.observed_labels < self.num_classes)
            ),
            "Observed labels outside valid class range",
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        source_index = int(self.indices[position])
        image, official_label = self.base[source_index]

        require(
            int(official_label) == int(self.clean_labels[position]),
            "Official source label changed",
        )

        image = IMAGE_TRANSFORM(image.convert("RGB"))

        return {
            "image": image,
            # Training must use this observed label.
            "label": int(self.observed_labels[position]),
            # The fields below are for auditing and logging.
            "clean_label": int(self.clean_labels[position]),
            "is_corrupted": bool(self.is_corrupted[position]),
            "sample_id": self.sample_ids[position],
            "source_index": source_index,
        }


def build_dataset(
    dataset_name,
    split,
    noise_percent=None,
    allow_test=False,
):
    if dataset_name not in NUM_CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported experimental split: {split}")

    if dataset_name == "oxford_pet" and split == "validation":
        raise ValueError("Pet has no independent validation split")

    if split == "test" and not allow_test:
        raise ValueError("Test access requires explicit final-evaluation permission")

    if split == "train":
        if noise_percent not in NOISE_PERCENTAGES:
            raise ValueError("Training requires noise_percent=0, 10, 20 or 30")
    elif noise_percent is not None:
        raise ValueError("Validation/test must not receive a noise setting")

    split_path = (
        MANIFEST_ROOT / f"{dataset_name}_seed{SEED}_splits.json"
    )
    split_manifest, split_hash = read_verified_manifest(split_path)

    require(
        split_manifest["dataset"] == dataset_name
        and split_manifest["seed"] == SEED
        and split_manifest["num_classes"] == NUM_CLASSES[dataset_name],
        "Split manifest metadata mismatch",
    )

    partition = split_manifest["partitions"][split]
    hashes = {"split_sha256": split_hash, "noise_sha256": None}

    if split == "train":
        noise_path = MANIFEST_ROOT / (
            f"{dataset_name}_seed{SEED}_noise{noise_percent:02d}.json"
        )
        noise, noise_hash = read_verified_manifest(noise_path)

        require(
            noise["dataset"] == dataset_name
            and noise["seed"] == SEED
            and noise["noise_percent"] == noise_percent
            and noise["num_classes"] == NUM_CLASSES[dataset_name],
            "Noise manifest metadata mismatch",
        )
        require(
            noise["split_manifest_sha256"] == split_hash
            and noise["split_manifest"]
            == split_path.relative_to(ROOT).as_posix(),
            "Noise manifest refers to a different split",
        )
        require(
            noise["shared_by_initializations"]
            == ["random", "procedural"],
            "Noise manifest must be shared by both initializations",
        )

        training = noise["training"]
        for key in (
            "source_split",
            "indices",
            "sample_ids",
            "labels_clean",
        ):
            require(
                training[key] == partition[key],
                f"Noise manifest changed the clean partition: {key}",
            )

        observed = np.asarray(
            training["labels_observed"], dtype=np.int64
        )
        clean = np.asarray(partition["labels_clean"], dtype=np.int64)
        require(observed.shape == clean.shape, "Label shape mismatch")

        changed = observed != clean
        require(
            np.array_equal(
                changed,
                np.asarray(training["is_corrupted"], dtype=bool),
            ),
            "Corruption flags do not match actual label changes",
        )
        require(
            len(clean) * noise_percent % 100 == 0
            and int(changed.sum())
            == noise["corrupted_count"]
            == len(clean) * noise_percent // 100,
            "Incorrect corruption count",
        )
        hashes["noise_sha256"] = noise_hash
    else:
        observed = partition["labels_clean"]

    return ManifestDataset(
        dataset_name,
        split,
        partition,
        observed,
        hashes,
    )


def seed_worker(worker_id):
    # PyTorch assigns each worker a deterministic seed from the loader.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset,
    batch_size=16,
    num_workers=0,
    pin_memory=False,
):
    # Separate generators prevent worker setup from changing sample order.
    order_generator = torch.Generator().manual_seed(SEED)
    worker_generator = torch.Generator().manual_seed(SEED)

    if dataset.split == "train":
        sampler = RandomSampler(
            dataset,
            replacement=False,
            generator=order_generator,
        )
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=worker_generator,
    )