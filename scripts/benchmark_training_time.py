
"""
Benchmark ViT-Tiny training speed on CIFAR-100 and SVHN.

- Uses existing local datasets and src/model_factory.py
- Benchmarks Random and Procedural initialization
- Runs real forward, backward and optimizer steps
- Measures validation inference time
- Prints estimates for different epoch budgets
- Does NOT save any files, plots, reports or checkpoints
"""

import gc
import math
import random
import time
import sys
from pathlib import Path
from statistics import median

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_factory import build_model


DATA_ROOT = ROOT / "data"

SEED = 11

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

WARMUP_STEPS = 20
TIMED_STEPS = 100
REPEATS = 2

LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.05

# RTX 3060: use mixed precision for this benchmark.
# Must match the eventual formal training configuration.
USE_AMP = True

# Scenarios only; NOT fixed training budgets.
EPOCH_CANDIDATES = [5, 10, 20, 50, 100, 300]

DATASET_INFO = {
    "cifar100": {
        "classes": 100,
        "train_size": 10000,
        "val_size": 5000,
        "official_train": 50000,
        "official_test": 10000,
    },
    "svhn": {
        "classes": 10,
        "train_size": 5000,
        "val_size": 2000,
        "official_train": 73257,
        "official_test": 26032,
    },
}


# ============================================================
# 2. Data loading
# ============================================================

def create_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def create_loaders(dataset_name):
    """
    Read local official dataset files.

    For this speed benchmark only, select temporary,
    non-overlapping train/validation indices in memory.

    No split files are written.
    """
    info = DATASET_INFO[dataset_name]
    transform = create_transform()

    if dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            root=str(DATA_ROOT / "cifar100"),
            train=True,
            transform=transform,
            download=False,
        )
    else:
        dataset = datasets.SVHN(
            root=str(DATA_ROOT / "svhn"),
            split="train",
            transform=transform,
            download=False,
        )

    expected = info["official_train"]

    if len(dataset) != expected:
        raise RuntimeError(
            f"{dataset_name}: expected {expected} images, "
            f"found {len(dataset)}"
        )

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(dataset))

    n_train = info["train_size"]
    n_val = info["val_size"]

    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": NUM_WORKERS > 0,
    }

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )

    val_loader = DataLoader(
        val_subset,
        shuffle=False,
        **loader_options,
    )

    return train_loader, val_loader


# ============================================================
# 3. Actual training steps
# ============================================================

def train_n_steps(
    model,
    loader_iterator,
    optimizer,
    scaler,
    criterion,
    device,
    steps,
):
    model.train()

    last_loss = None

    for _ in range(steps):
        images, labels = next(loader_iterator)

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        last_loss = loss

    return last_loss


# ============================================================
# 4. Validation speed
# ============================================================

@torch.inference_mode()
def measure_validation(model, val_loader, device):
    model.eval()

    # Warm up validation computation before measuring it.
    iterator = iter(val_loader)

    for _ in range(min(5, len(val_loader))):
        images, _ = next(iterator)

        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            model(images)

    torch.cuda.synchronize()

    start = time.perf_counter()

    # Time one complete pass over the validation subset.
    for images, _ in val_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            model(images)

    torch.cuda.synchronize()

    return time.perf_counter() - start


# ============================================================
# 5. Benchmark one model
# ============================================================

def benchmark_once(
    name,
    initialization,
    train_loader,
    val_loader,
    repeat_index,
    measure_val,
):
    info = DATASET_INFO[name]
    device = torch.device("cuda")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Reset training-batch order for a fair speed comparison.
    train_loader.generator.manual_seed(SEED)

    print(
        f"\n[{name}] "
        f"{initialization.upper()} "
        f"| repeat {repeat_index}/{REPEATS}",
        flush=True,
    )

    torch.cuda.synchronize()
    init_start = time.perf_counter()

    model = build_model(
        initialization=initialization,
        num_classes=info["classes"],
        seed=SEED,
    ).to(device)

    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - init_start

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

    train_iterator = iter(train_loader)

    # Stage A: real training, but exclude its time.
    print(
        f"  Warming up: {WARMUP_STEPS} steps...",
        flush=True,
    )

    train_n_steps(
        model,
        train_iterator,
        optimizer,
        scaler,
        criterion,
        device,
        WARMUP_STEPS,
    )

    torch.cuda.synchronize()

    # Stage B: actual timed training.
    print(
        f"  Timing: {TIMED_STEPS} steps...",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    last_loss = train_n_steps(
        model,
        train_iterator,
        optimizer,
        scaler,
        criterion,
        device,
        TIMED_STEPS,
    )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    seconds_per_step = elapsed / TIMED_STEPS
    images_per_second = BATCH_SIZE / seconds_per_step

    memory_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    loss_value = float(last_loss.detach().cpu())

    print(f"  Timed training: {elapsed:.2f} s")
    print(f"  Time per step:  {seconds_per_step:.4f} s")
    print(f"  Images/second: {images_per_second:.2f}")
    print(f"  Peak GPU memory: {memory_gb:.2f} GiB")
    print(f"  Last training loss: {loss_value:.4f}")

    # Validation is measured only once per combination.
    val_seconds = None

    if measure_val:
        print("  Measuring full validation pass...", flush=True)

        val_seconds = measure_validation(
            model,
            val_loader,
            device,
        )

        print(f"  Validation time: {val_seconds:.2f} s")

    result = {
        "step_seconds": seconds_per_step,
        "init_seconds": init_seconds,
        "memory_gb": memory_gb,
        "validation_seconds": val_seconds,
    }

    del last_loss
    del model
    del optimizer
    del scaler
    del train_iterator

    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# 6. Time estimation
# ============================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} seconds"

    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes"

    return f"{seconds / 3600:.2f} hours"


def estimate_dataset(name, results):
    info = DATASET_INFO[name]

    n_train_batches = math.ceil(
        info["train_size"] / BATCH_SIZE
    )

    n_full_batches = math.ceil(
        info["official_train"] / BATCH_SIZE
    )

    n_val_batches = math.ceil(
        info["val_size"] / BATCH_SIZE
    )

    n_test_batches = math.ceil(
        info["official_test"] / BATCH_SIZE
    )

    estimates = {}

    print("\n" + "=" * 66)
    print(f"RESULTS: {name.upper()}")
    print("=" * 66)

    for initialization in ("random", "procedural"):
        runs = results[initialization]

        step_seconds = median(
            run["step_seconds"] for run in runs
        )

        init_seconds = median(
            run["init_seconds"] for run in runs
        )

        peak_memory = max(
            run["memory_gb"] for run in runs
        )

        val_seconds = next(
            run["validation_seconds"]
            for run in runs
            if run["validation_seconds"] is not None
        )

        train_epoch_seconds = (
            n_train_batches * step_seconds
        )

        full_epoch_seconds = (
            n_full_batches * step_seconds
        )

        # Estimated, not directly measured, full-test duration.
        test_seconds = (
            val_seconds / n_val_batches * n_test_batches
        )

        estimates[initialization] = {
            "train_epoch": train_epoch_seconds,
            "val_epoch": val_seconds,
            "test_once": test_seconds,
            "init_once": init_seconds,
        }

        print(f"\nInitialization: {initialization}")
        print(f"  Median step: {step_seconds:.4f} s")
        print(f"  Peak memory: {peak_memory:.2f} GiB")

        print(
            "  Planned training subset, 1 epoch: "
            + format_time(train_epoch_seconds)
        )

        print(
            "  Full official train set, 1 epoch: "
            + format_time(full_epoch_seconds)
        )

        print(
            "  Planned validation pass: "
            + format_time(val_seconds)
        )

        print(
            "  Full official test pass (estimate): "
            + format_time(test_seconds)
        )

    print(f"\nPlanned train batches/epoch: {n_train_batches}")
    print(f"Full train batches/epoch:    {n_full_batches}")

    return estimates


def print_total_estimates(all_estimates):
    print("\n" + "=" * 66)
    print("ESTIMATED TOTAL TIME FOR ALL 20 EXPERIMENTS")
    print("=" * 66)

    print(
        "2 datasets x 2 initializations x 5 noise levels "
        "= 20 training runs"
    )
    print("Assumption: same epoch count for every run.")
    print("Assumption: one validation pass after each epoch.")
    print("Assumption: one test pass at the end of each run.")
    print()

    for epochs in EPOCH_CANDIDATES:
        total_seconds = 0

        for dataset_results in all_estimates.values():
            for item in dataset_results.values():

                one_run_seconds = (
                    epochs * (
                        item["train_epoch"]
                        + item["val_epoch"]
                    )
                    + item["test_once"]
                    + item["init_once"]
                )

                # One initialization x five noise levels.
                total_seconds += 5 * one_run_seconds

        buffered = total_seconds * 1.15

        print(
            f"{epochs:>3} epochs/run | "
            f"estimated: {format_time(total_seconds):>14} | "
            f"with 15% reserve: {format_time(buffered)}"
        )

    print("\nImportant:")
    print("- These are timing projections, not finished experiments.")
    print("- No label-noise experiment was performed.")
    print("- No training budget or optimal epoch was selected.")
    print("- Final epoch count must be judged from learning curves.")
    print("- No results or checkpoint files were written.")


# ============================================================
# 7. Main
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. Check PyTorch/CUDA installation."
        )

    print("=" * 66)
    print("ViT-Tiny Real Training Speed Benchmark")
    print("=" * 66)

    print(f"Project: {ROOT}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"AMP enabled: {USE_AMP}")
    print(f"DataLoader workers: {NUM_WORKERS}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Timed steps: {TIMED_STEPS}")
    print(f"Repeats: {REPEATS}")

    all_estimates = {}

    for name in ("cifar100", "svhn"):
        print("\n" + "-" * 66)
        print(f"LOADING {name.upper()}")
        print("-" * 66)

        train_loader, val_loader = create_loaders(name)

        print(
            f"Train subset: "
            f"{len(train_loader.dataset):,} images"
        )

        print(
            f"Validation subset: "
            f"{len(val_loader.dataset):,} images"
        )

        results = {}

        for initialization in ("random", "procedural"):
            runs = []

            for repeat_index in range(1, REPEATS + 1):
                result = benchmark_once(
                    name=name,
                    initialization=initialization,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    repeat_index=repeat_index,
                    measure_val=(repeat_index == 1),
                )

                runs.append(result)

            results[initialization] = runs

        all_estimates[name] = estimate_dataset(
            name,
            results,
        )

        del train_loader
        del val_loader

        gc.collect()
        torch.cuda.empty_cache()

    print_total_estimates(all_estimates)

    print("\nBENCHMARK_COMPLETED")


if __name__ == "__main__":
    main()
