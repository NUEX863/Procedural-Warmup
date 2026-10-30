"""Pet 90/10 training and validation data, with reproducible augmentation."""

import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "pet90_baseline.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_seed(seed, purpose):
    content = f"{seed}:{purpose}".encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(content).digest()[:8], "little"
    ) % (2**63)


def load_config():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))

    require(cfg["protocol"] == "pet90_baseline_v1", "Wrong protocol")
    require(cfg["seed"] == 11, "Expected seed 11")
    require(cfg["num_classes"] == 37, "Expected 37 classes")
    require(cfg["noise_percent"] == 0, "This stage supports clean training only")
    require(
        cfg["batch_size"] == 16 and cfg["accumulation_steps"] == 4,
        "Expected micro batch 16 and accumulation 4",
    )
    require(
        cfg["epochs"] == 150
        and cfg["warmup_epochs"] == 5
        and cfg["lr"] == 0.0003,
        "Unexpected baseline schedule",
    )
    require(
        cfg["drop_path"] == 0
        and cfg["label_smoothing"] == 0
        and cfg["mixup_alpha"] == 0
        and cfg["cutmix_alpha"] == 0
        and cfg["randaugment"] is False,
        "Unexpected baseline regularization",
    )
    require(
        cfg["image_size"] == 224
        and cfg["evaluation_resize"] == 256,
        "Unexpected image size",
    )
    require(cfg["num_workers"] >= 0, "Invalid worker count")
    require(cfg["eval_batch_size"] > 0, "Invalid evaluation batch size")
    return cfg


def read_manifest(cfg):
    path = ROOT / cfg["split_manifest"]
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    require(digest == cfg["split_sha256"], "Split fingerprint mismatch")
    require(
        path.with_name(path.name + ".sha256")
        .read_text(encoding="utf-8")
        .split() == [digest, path.name],
        "Split checksum sidecar mismatch",
    )

    manifest = json.loads(content)
    require(
        manifest["schema_version"] == 2
        and manifest["protocol"] == "pet90_10_v1"
        and manifest["dataset"] == "oxford_pet"
        and manifest["seed"] == 11
        and manifest["num_classes"] == 37,
        "Unexpected split metadata",
    )

    partitions = manifest["partitions"]
    require(
        set(partitions) == {"train", "validation", "test"},
        "Unexpected partitions",
    )

    identities = {}
    for split, count in (
        ("train", 3312),
        ("validation", 368),
        ("test", 3669),
    ):
        partition = partitions[split]
        ids = partition["sample_ids"]
        labels = partition["labels_clean"]
        indices = partition["indices"]

        require(
            len(ids) == len(labels) == len(indices) == count,
            f"Incorrect partition size: {split}",
        )
        require(
            len(set(ids)) == len(set(indices)) == count,
            f"Duplicate samples: {split}",
        )
        require(
            all(type(label) is int and 0 <= label < 37 for label in labels),
            f"Invalid labels: {split}",
        )
        require(set(labels) == set(range(37)), f"Missing class: {split}")

        expected_source = "test" if split == "test" else "trainval"
        require(
            partition["source_split"] == expected_source,
            f"Incorrect source split: {split}",
        )
        identities[split] = set(ids)

    require(
        identities["train"].isdisjoint(identities["validation"])
        and identities["train"].isdisjoint(identities["test"])
        and identities["validation"].isdisjoint(identities["test"]),
        "Image identities overlap",
    )
    require(
        set(partitions["train"]["indices"]).isdisjoint(
            partitions["validation"]["indices"]
        ),
        "Train/validation source indices overlap",
    )
    require(
        set(partitions["train"]["indices"])
        | set(partitions["validation"]["indices"])
        == set(range(3680)),
        "Train/validation do not cover official trainval",
    )

    # Only manifest metadata is read here, not test images.
    return manifest


class Pet90Dataset(Dataset):
    def __init__(self, split, cfg):
        # Reject test access before constructing any official dataset.
        if split not in {"train", "validation"}:
            raise ValueError("This loader permits train/validation only")

        self.split = split
        self.seed = cfg["seed"]
        self.num_classes = 37
        self.manifest_hash = cfg["split_sha256"]

        manifest = read_manifest(cfg)
        partition = manifest["partitions"][split]
        self.indices = list(partition["indices"])
        self.sample_ids = list(partition["sample_ids"])
        self.labels = list(partition["labels_clean"])

        # Both new partitions originate from official trainval.
        base = datasets.OxfordIIITPet(
            root=str(ROOT / "data"),
            split="trainval",
            target_types="category",
            download=False,
        )
        self.classes = list(base.classes)
        self.paths = []

        pet_root = ROOT / "data" / "oxford-iiit-pet"
        for index, sample_id, label in zip(
            self.indices, self.sample_ids, self.labels
        ):
            path = Path(base._images[index])
            require(
                path.relative_to(pet_root).as_posix() == sample_id,
                "Image identity differs from official source",
            )
            require(
                int(base._labels[index]) == label,
                "Clean label differs from official source",
            )
            self.paths.append(path)

        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    cfg["image_size"],
                    scale=tuple(cfg["crop_scale"]),
                    ratio=tuple(cfg["crop_ratio"]),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(
                    p=cfg["horizontal_flip_probability"]
                ),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    cfg["evaluation_resize"],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.CenterCrop(cfg["image_size"]),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if self.split == "train":
            require(
                isinstance(key, tuple) and len(key) == 2,
                "Training requires an explicit (epoch, position) key",
            )
            epoch, position = key
            require(epoch >= 0, "Invalid epoch")
        else:
            require(isinstance(key, int), "Invalid validation index")
            epoch, position = -1, key

        with Image.open(self.paths[position]) as opened:
            image = opened.convert("RGB")

        if self.split == "train":
            seed = derived_seed(
                self.seed,
                f"augmentation:{epoch}:{self.sample_ids[position]}",
            )
            # Preserve the caller's CPU RNG and do not touch CUDA RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                image = self.transform(image)
        else:
            image = self.transform(image)

        label = self.labels[position]
        return {
            "image": image,
            "label": label,
            "clean_label": label,
            "is_corrupted": False,
            "sample_id": self.sample_ids[position],
            "source_index": self.indices[position],
            "position": position,
            "epoch": epoch,
        }


class EpochKeySampler(Sampler):
    """Send the epoch inside each key, including to persistent workers."""

    def __init__(self, dataset, seed, epoch=0):
        self.count = len(dataset)
        self.seed = seed
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        require(type(epoch) is int and epoch >= 0, "Invalid epoch")
        self.epoch = epoch

    def __iter__(self):
        epoch = self.epoch
        generator = torch.Generator().manual_seed(
            derived_seed(self.seed, f"training-order:{epoch}")
        )
        order = torch.randperm(
            self.count, generator=generator
        ).tolist()
        return iter((epoch, position) for position in order)

    def __len__(self):
        return self.count


def make_loader(dataset, cfg, epoch=0, num_workers=None):
    training = dataset.split == "train"
    workers = (
        cfg["num_workers"] if training else 0
    ) if num_workers is None else num_workers

    options = {
        "dataset": dataset,
        "batch_size": (
            cfg["batch_size"] if training else cfg["eval_batch_size"]
        ),
        "sampler": (
            EpochKeySampler(dataset, cfg["seed"], epoch)
            if training else None
        ),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "generator": torch.Generator().manual_seed(
            derived_seed(cfg["seed"], f"loader-workers:{dataset.split}")
        ),
    }
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)

    return DataLoader(**options)


def accumulation_group_sizes(sample_count, cfg):
    effective = cfg["batch_size"] * cfg["accumulation_steps"]
    return [
        min(effective, sample_count - start)
        for start in range(0, sample_count, effective)
    ]