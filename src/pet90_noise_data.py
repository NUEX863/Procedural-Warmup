"""Fixed label-noise adapter for the verified Pet 90/10 data pipeline."""

import json

from torch.utils.data import Dataset

from src.pet90_data import (
    ROOT,
    Pet90Dataset,
    file_sha256,
    read_manifest,
    require,
)


NOISE_HASHES = {
    0: "ad5d6a80b5a07d6a86b162e59ace95ed15dcb26cbeef2099a393908aae235dbb",
    10: "907d971329b79798bae19b9022c8650b19ceacf09b4eba90ff6a78e4c34610eb",
    20: "7b0f17bd3911f056eabbd103e8decf9d54ccb6bd92c5a38b4b52a888b60a417d",
    30: "8b5f7381778a60bbaefa3e2a97c3a8b0331655610844403c2040f5ff8d58f25a",
}

EXPECTED_COUNTS = {0: 0, 10: 331, 20: 662, 30: 994}


def noise_path(noise_percent):
    require(
        type(noise_percent) is int and noise_percent in NOISE_HASHES,
        "Noise must be one of 0, 10, 20, 30",
    )
    return ROOT / "manifests" / (
        f"pet90_seed11_noise{noise_percent:02d}.json"
    )


def read_noise_manifest(cfg, noise_percent):
    path = noise_path(noise_percent)
    digest = file_sha256(path)
    require(
        digest == NOISE_HASHES[noise_percent],
        f"Noise manifest fingerprint mismatch: {path.name}",
    )
    require(
        path.with_name(path.name + ".sha256")
        .read_text(encoding="utf-8")
        .split() == [digest, path.name],
        "Noise checksum sidecar mismatch",
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(
        manifest["schema_version"] == 2
        and manifest["protocol"] == "pet90_noise_v1"
        and manifest["dataset"] == "oxford_pet"
        and manifest["seed"] == cfg["seed"] == 11
        and manifest["num_classes"] == cfg["num_classes"] == 37
        and manifest["noise_percent"] == noise_percent,
        "Unexpected noise metadata",
    )
    require(
        manifest["split_manifest"] == cfg["split_manifest"]
        and manifest["split_manifest_sha256"] == cfg["split_sha256"],
        "Noise manifest belongs to a different split",
    )
    require(
        manifest["shared_by_initializations"]
        == ["random", "procedural"],
        "Noise manifest must be shared by both initializations",
    )

    source = read_manifest(cfg)["partitions"]["train"]
    training = manifest["training"]

    # Every original field must match the fixed clean training partition.
    for field, value in source.items():
        require(
            training[field] == value,
            f"Clean training field changed: {field}",
        )

    clean = training["labels_clean"]
    observed = training["labels_observed"]
    flags = training["is_corrupted"]

    require(
        len(clean) == len(observed) == len(flags) == 3312,
        "Incorrect training label count",
    )
    require(
        all(type(label) is int and 0 <= label < 37
            for label in observed),
        "Invalid observed label",
    )
    require(
        all(type(flag) is bool for flag in flags),
        "Corruption flags must be Boolean",
    )
    require(
        flags == [a != b for a, b in zip(clean, observed)],
        "Corruption flags disagree with actual label changes",
    )

    count = sum(flags)
    require(
        count == manifest["corrupted_count"]
        == EXPECTED_COUNTS[noise_percent],
        "Incorrect corruption count",
    )
    require(
        manifest["training_count"] == 3312
        and abs(
            manifest["actual_noise_percent"] - 100 * count / 3312
        ) < 1e-10,
        "Incorrect actual noise rate",
    )

    per_class = [
        sum(flag for label, flag in zip(clean, flags) if label == c)
        for c in range(37)
    ]
    require(
        per_class == manifest["corrupted_count_per_clean_class"],
        "Incorrect corruption counts by clean class",
    )
    return manifest


class Pet90NoiseDataset(Dataset):
    """Use observed training labels without changing the clean base dataset."""

    def __init__(self, split, cfg, noise_percent=None):
        if split not in {"train", "validation"}:
            raise ValueError("Only train and validation are permitted")

        if split == "train":
            require(
                type(noise_percent) is int
                and noise_percent in NOISE_HASHES,
                "Training requires an explicit noise percentage",
            )
            manifest = read_noise_manifest(cfg, noise_percent)
        else:
            require(
                noise_percent is None,
                "Validation must not receive a noise setting",
            )
            manifest = None

        # Composition keeps the verified clean dataset unchanged.
        self.base = Pet90Dataset(split, cfg)
        self.split = split
        self.seed = self.base.seed
        self.num_classes = self.base.num_classes
        self.manifest_hash = self.base.manifest_hash
        self.indices = list(self.base.indices)
        self.sample_ids = list(self.base.sample_ids)
        self.classes = list(self.base.classes)
        self.clean_labels = list(self.base.labels)
        self.noise_percent = noise_percent

        if manifest is None:
            self.noise_manifest_hash = None
            self.actual_noise_percent = 0.0
            self.observed_labels = list(self.clean_labels)
            self.is_corrupted = [False] * len(self.base)
        else:
            training = manifest["training"]
            self.noise_manifest_hash = NOISE_HASHES[noise_percent]
            self.actual_noise_percent = manifest["actual_noise_percent"]
            self.observed_labels = list(training["labels_observed"])
            self.is_corrupted = list(training["is_corrupted"])

            require(
                training["indices"] == self.indices
                and training["sample_ids"] == self.sample_ids
                and training["labels_clean"] == self.clean_labels,
                "Noise mapping is not aligned with the image dataset",
            )

        # 'labels' always means the targets supplied to the training/eval loop.
        self.labels = list(self.observed_labels)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        item = dict(self.base[key])
        position = item["position"]

        require(
            item["clean_label"] == self.clean_labels[position],
            "Base clean label changed",
        )

        item["label"] = self.observed_labels[position]
        item["clean_label"] = self.clean_labels[position]
        item["is_corrupted"] = self.is_corrupted[position]
        return item