
import gc
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_factory import build_model


DATA_ROOT = ROOT / "data"

MODEL_SEED = 11
SPLIT_SEED = 11
NOISE_SEED = 20261026
LOADER_SEED = 1101

NOISE_RATES = (0.0, 0.2, 0.4)
EPOCHS = 20

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.05
USE_AMP = True

DATASETS = {
    "cifar100": {
        "classes": 100,
        "train_size": 10000,
        "val_size": 5000,
        "official_train_size": 50000,
        "root": DATA_ROOT / "cifar100",
    },
    "svhn": {
        "classes": 10,
        "train_size": 5000,
        "val_size": 2000,
        "official_train_size": 73257,
        "root": DATA_ROOT / "svhn",
    },
}


# ============================================================
# 2. Dataset wrapper
# ============================================================

class RelabeledSubset(Dataset):
    """
    Select original dataset indices and supply specified labels.
    The original dataset and its labels are never modified.
    """

    def __init__(self, base_dataset, indices, labels):
        self.base = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)

        if len(self.indices) != len(self.labels):
            raise ValueError("Indices and labels have different lengths.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, _ = self.base[int(self.indices[position])]
        label = int(self.labels[position])
        return image, label


def create_base_dataset(name):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    if name == "cifar100":
        dataset = datasets.CIFAR100(
            root=str(DATASETS[name]["root"]),
            train=True,
            transform=transform,
            download=False,
        )
        labels = np.asarray(dataset.targets, dtype=np.int64)

    else:
        dataset = datasets.SVHN(
            root=str(DATASETS[name]["root"]),
            split="train",
            transform=transform,
            download=False,
        )
        labels = np.asarray(dataset.labels, dtype=np.int64)
        labels[labels == 10] = 0

    expected = DATASETS[name]["official_train_size"]

    if len(dataset) != expected or len(labels) != expected:
        raise RuntimeError(
            f"{name}: expected {expected} training examples, "
            f"got {len(dataset)}"
        )

    if labels.min() < 0 or labels.max() >= DATASETS[name]["classes"]:
        raise RuntimeError(f"{name}: unexpected label values")

    return dataset, labels


def create_fixed_split(name, labels):
    """
    Create a reproducible, non-overlapping split in memory.
    The same split is reused across noise levels and initializations.
    """

    info = DATASETS[name]

    rng = np.random.default_rng(
        SPLIT_SEED + (0 if name == "cifar100" else 1)
    )
    indices = rng.permutation(len(labels))

    n_train = info["train_size"]
    n_val = info["val_size"]

    if n_train + n_val > len(indices):
        raise RuntimeError("Requested subset exceeds official train split.")

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]

    if np.intersect1d(train_indices, val_indices).size != 0:
        raise RuntimeError("Training and validation indices overlap.")

    train_labels = labels[train_indices].copy()
    val_labels = labels[val_indices].copy()

    return train_indices, train_labels, val_indices, val_labels


# ============================================================
# 3. Symmetric label noise
# ============================================================

def create_noisy_labels(clean_labels, noise_rate, num_classes, seed):
    """
    Corrupt exactly round(noise_rate * N) labels.

    Each selected label is replaced uniformly by one of the other
    classes. Reusing the same seed makes the noise sets nested:
    the 20% set includes the 0%-20% corruptions used at lower rates,
    and the 40% set includes those from the 20% set.
    """

    clean = np.asarray(clean_labels, dtype=np.int64)
    noisy = clean.copy()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clean))

    # Draw a replacement different from the original label.
    replacements = rng.integers(
        low=0,
        high=num_classes - 1,
        size=len(clean),
    )
    replacements += replacements >= clean

    n_corrupt = int(round(noise_rate * len(clean)))
    selected = order[:n_corrupt]

    noisy[selected] = replacements[selected]

    actual_changed = int(np.count_nonzero(noisy != clean))

    if actual_changed != n_corrupt:
        raise RuntimeError(
            f"Expected {n_corrupt} corrupted labels, "
            f"got {actual_changed}"
        )

    return noisy, actual_changed


# ============================================================
# 4. Data loaders
# ============================================================

def make_loader(dataset, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        generator=generator,
    )


# ============================================================
# 5. One training epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
):
    model.train()

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros(
        (), device=device, dtype=torch.long
    )
    total_examples = 0

    torch.cuda.synchronize()
    start = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(
            device, dtype=torch.long, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_count = labels.size(0)

        total_loss += loss.detach() * batch_count
        total_correct += (
            logits.detach().argmax(dim=1) == labels
        ).sum()
        total_examples += batch_count

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mean_loss = (total_loss / total_examples).item()
    accuracy = (total_correct / total_examples).item()

    return mean_loss, accuracy, elapsed


# ============================================================
# 6. Validation on clean labels
# ============================================================

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    total_correct = torch.zeros(
        (), device=device, dtype=torch.long
    )
    total_examples = 0

    torch.cuda.synchronize()
    start = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(
            device, dtype=torch.long, non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            logits = model(images)

        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum()
        total_examples += labels.size(0)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    accuracy = (total_correct / total_examples).item()

    return accuracy, elapsed


# ============================================================
# 7. Run one pilot experiment
# ============================================================

def run_experiment(
    name,
    base_dataset,
    train_indices,
    clean_train_labels,
    val_indices,
    clean_val_labels,
    noise_rate,
    initialization,
    device,
):
    num_classes = DATASETS[name]["classes"]

    # Same noise realization for both initializations and all rates.
    noise_seed = NOISE_SEED + (
        0 if name == "cifar100" else 1000
    )

    noisy_labels, n_corrupt = create_noisy_labels(
        clean_train_labels,
        noise_rate,
        num_classes,
        noise_seed,
    )

    train_dataset = RelabeledSubset(
        base_dataset,
        train_indices,
        noisy_labels,
    )

    val_dataset = RelabeledSubset(
        base_dataset,
        val_indices,
        clean_val_labels,
    )

    train_loader = make_loader(
        train_dataset,
        shuffle=True,
        seed=LOADER_SEED,
    )

    val_loader = make_loader(
        val_dataset,
        shuffle=False,
        seed=LOADER_SEED,
    )

    print("\n" + "-" * 72)
    print(
        f"Dataset={name} | Init={initialization} | "
        f"Noise={noise_rate:.0%} | Seed={MODEL_SEED}"
    )
    print(
        f"Train={len(train_dataset):,} | "
        f"Validation={len(val_dataset):,} | "
        f"Corrupted labels={n_corrupt:,}"
    )
    print("-" * 72, flush=True)

    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)

    model = build_model(
        initialization=initialization,
        num_classes=num_classes,
        seed=MODEL_SEED,
    ).to(device)

    # Fix CUDA stochastic operations, such as dropout, for paired runs.
    torch.cuda.manual_seed_all(MODEL_SEED)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = -1.0
    best_epoch = 0
    total_train_seconds = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy, train_seconds = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
        )

        val_accuracy, val_seconds = evaluate(
            model,
            val_loader,
            device,
        )

        total_train_seconds += train_seconds

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"TrainLoss={train_loss:.4f} | "
            f"TrainAcc(noisy targets)={train_accuracy:.2%} | "
            f"ValAcc(clean labels)={val_accuracy:.2%} | "
            f"TrainTime={train_seconds:.1f}s | "
            f"ValTime={val_seconds:.1f}s",
            flush=True,
        )

    result = {
        "dataset": name,
        "initialization": initialization,
        "noise_rate": noise_rate,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "final_val_accuracy": val_accuracy,
        "total_train_seconds": total_train_seconds,
    }

    print(
        f"[RUN SUMMARY] Best ValAcc={best_val_accuracy:.2%} "
        f"at epoch {best_epoch}; "
        f"Final ValAcc={val_accuracy:.2%}; "
        f"TrainingTime={total_train_seconds / 60:.1f} min",
        flush=True,
    )

    del model, optimizer, scaler, criterion
    del train_loader, val_loader
    del train_dataset, val_dataset

    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# 8. Main
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda")

    print("=" * 72)
    print("ViT-Tiny Learning-Curve Pilot")
    print("=" * 72)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Epochs per run: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Noise rates: {NOISE_RATES}")
    print(f"AMP: {USE_AMP}")
    print("No checkpoints, reports or curve files will be saved.")

    results = []

    for name in ("cifar100", "svhn"):
        print("\n" + "=" * 72)
        print(f"Preparing {name.upper()}")
        print("=" * 72)

        base_dataset, all_labels = create_base_dataset(name)

        (
            train_indices,
            clean_train_labels,
            val_indices,
            clean_val_labels,
        ) = create_fixed_split(name, all_labels)

        print(
            f"Fixed training subset: {len(train_indices):,}; "
            f"fixed clean validation subset: {len(val_indices):,}"
        )

        for noise_rate in NOISE_RATES:
            for initialization in ("random", "procedural"):
                result = run_experiment(
                    name=name,
                    base_dataset=base_dataset,
                    train_indices=train_indices,
                    clean_train_labels=clean_train_labels,
                    val_indices=val_indices,
                    clean_val_labels=clean_val_labels,
                    noise_rate=noise_rate,
                    initialization=initialization,
                    device=device,
                )
                results.append(result)

        del base_dataset
        gc.collect()

    print("\n" + "=" * 72)
    print("PILOT SUMMARY")
    print("=" * 72)

    for result in results:
        print(
            f"{result['dataset']:8s} | "
            f"{result['initialization']:10s} | "
            f"Noise={result['noise_rate']:.0%} | "
            f"BestEpoch={result['best_epoch']:02d} | "
            f"BestValAcc={result['best_val_accuracy']:.2%} | "
            f"FinalValAcc={result['final_val_accuracy']:.2%} | "
            f"TrainTime={result['total_train_seconds'] / 60:.1f} min"
        )

    print("\nPILOT_COMPLETED")
    print("No files or model checkpoints were saved.")
    print("Official test sets were not used.")


if __name__ == "__main__":
    main()
