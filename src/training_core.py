"""Training with fixed manifests, accumulation and epoch-boundary recovery."""

import os

# Must be set before CUDA initialization.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import contextlib
import copy
import hashlib
import io
import json
import math
import random
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from src.data_pipeline import seed_worker
from src.model_factory import (
    build_model,
    load_official_file,
    official_utils,
    read_lock,
)

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))

    require(cfg["seed"] == 11, "Protocol requires seed=11")
    require(cfg["epochs"] == 300, "Protocol requires 300 epochs")
    require(cfg["warmup_epochs"] == 50, "Protocol requires 50 warmup epochs")
    require(
        cfg["batch_size"] > 0
        and cfg["accumulation_steps"] > 0
        and cfg["batch_size"] * cfg["accumulation_steps"] == 512,
        "Effective batch size must be 512",
    )
    require(cfg["eval_batch_size"] > 0, "Invalid evaluation batch size")
    require(cfg["num_workers"] >= 0, "Invalid worker count")
    require(cfg["cpu_threads"] > 0, "Invalid CPU thread count")
    require(cfg["amp"] is True, "This configuration requires AMP")
    require(cfg["amp_init_scale"] > 0, "Invalid AMP initial scale")
    require(
        cfg["lr"] == 0.002
        and cfg["min_lr"] == 0.000001
        and cfg["weight_decay"] == 0.05,
        "Unexpected optimization configuration",
    )
    return cfg


def setup(cfg):
    require(torch.cuda.is_available(), "CUDA is required")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def derived_seed(seed, purpose):
    # Deterministic substreams derived from the ONE experimental seed.
    raw = hashlib.sha256(f"{seed}:{purpose}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "little") % (2**63)


class EpochSampler(Sampler):
    """Order is determined only by seed, epoch and dataset size."""

    def __init__(self, dataset, seed):
        self.count = len(dataset)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(
            derived_seed(self.seed, f"train-order:{self.epoch}")
        )
        return iter(
            torch.randperm(self.count, generator=generator).tolist()
        )

    def __len__(self):
        return self.count


def make_loader(dataset, cfg, training=True):
    # Evaluation uses the main process, avoiding a second persistent pool.
    workers = cfg["num_workers"] if training else 0
    generator = torch.Generator().manual_seed(
        derived_seed(cfg["seed"], "workers")
    )

    options = {
        "dataset": dataset,
        "batch_size": (
            cfg["batch_size"] if training else cfg["eval_batch_size"]
        ),
        "sampler": EpochSampler(dataset, cfg["seed"]) if training else None,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if workers > 0:
        options.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(**options)


def make_components(cfg, initialization, num_classes):
    model = build_model(
        initialization=initialization,
        num_classes=num_classes,
        seed=cfg["seed"],
    ).cuda()

    require(
        all(p.requires_grad for p in model.parameters()),
        "Unexpected frozen parameters",
    )

    factory = load_official_file(
        "_pwvit_training_optimizer",
        ROOT
        / read_lock()["code"]["directory"]
        / "downstream"
        / "optim_factory.py",
    )
    args = SimpleNamespace(
        opt="adamw",
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        opt_eps=1e-8,
        opt_betas=None,
    )

    # Suppress only the author's long parameter-group listing.
    with contextlib.redirect_stdout(io.StringIO()):
        optimizer = factory.create_optimizer(args, model)

    optimizer_parameters = [
        p for group in optimizer.param_groups for p in group["params"]
    ]
    require(
        len(optimizer_parameters) == len(list(model.parameters()))
        and len({id(p) for p in optimizer_parameters})
        == len(optimizer_parameters)
        and {id(p) for p in optimizer_parameters}
        == {id(p) for p in model.parameters()},
        "Optimizer must contain every model parameter exactly once",
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg["amp"],
        init_scale=cfg["amp_init_scale"],
    )
    return model, optimizer, scaler


def make_schedule(cfg, sample_count):
    effective_batch = cfg["batch_size"] * cfg["accumulation_steps"]
    steps = math.ceil(sample_count / effective_batch)

    # Use the author's scheduler, including its warmup convention.
    schedule = official_utils().cosine_scheduler(
        cfg["lr"],
        cfg["min_lr"],
        cfg["epochs"],
        steps,
        warmup_epochs=cfg["warmup_epochs"],
    )

    require(len(schedule) == cfg["epochs"] * steps, "Invalid schedule length")
    require(float(schedule[0]) == 0.0, "Expected zero-start LR warmup")
    require(
        np.isclose(schedule[cfg["warmup_epochs"] * steps - 1], cfg["lr"]),
        "Warmup endpoint mismatch",
    )
    return schedule


def train_epoch(model, optimizer, scaler, loader, cfg, epoch, schedule):
    model.train()
    loader.sampler.epoch = epoch

    device = next(model.parameters()).device
    effective_batch = cfg["batch_size"] * cfg["accumulation_steps"]
    sample_count = len(loader.dataset)
    steps = math.ceil(sample_count / effective_batch)

    seen = 0
    group_seen = 0
    attempts = 0
    applied = 0
    skipped = 0
    consecutive_skips = 0
    total_loss = 0.0
    total_correct = 0
    group_sizes = []
    trace = hashlib.sha256()

    optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        if group_seen == 0:
            group_target = min(effective_batch, sample_count - seen)
            lr = float(schedule[epoch * steps + attempts])
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_scale", 1.0)

        trace.update(batch["source_index"].numpy().tobytes())
        trace.update(batch["label"].numpy().tobytes())

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        count = len(labels)

        require(
            group_seen + count <= group_target,
            "Microbatch crosses an accumulation boundary",
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=cfg["amp"] and device.type == "cuda",
        ):
            logits = model(images)
            loss_sum = F.cross_entropy(logits, labels, reduction="sum")

            # Divide by the ACTUAL number of samples in this update.
            # This handles both the full 512-sample group and the tail.
            loss = loss_sum / group_target

        require(torch.isfinite(loss).item(), "Non-finite training loss")
        scaler.scale(loss).backward()

        total_loss += float(loss_sum.detach())
        total_correct += int((logits.argmax(1) == labels).sum())
        seen += count
        group_seen += count

        # Do not keep forward graphs or batch tensors between iterations.
        del images, labels, logits, loss, loss_sum

        if group_seen == group_target:
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.get_scale() < old_scale

            attempts += 1
            group_sizes.append(group_seen)

            if overflow:
                skipped += 1
                consecutive_skips += 1
                print(
                    f"  AMP skipped update {attempts}; "
                    f"scale={scaler.get_scale():g}",
                    flush=True,
                )
                require(consecutive_skips < 8, "Persistent AMP overflow")
            else:
                applied += 1
                consecutive_skips = 0

            optimizer.zero_grad(set_to_none=True)
            group_seen = 0

            if attempts % 8 == 0:
                print(
                    f"  epoch={epoch + 1}, samples={seen}/{sample_count}",
                    flush=True,
                )

    require(seen == sample_count, "Not all training samples were used")
    require(group_seen == 0 and attempts == steps, "Incomplete accumulation")
    require(applied > 0, "No optimizer update succeeded in this epoch")
    require(
        all(torch.isfinite(p).all().item() for p in model.parameters()),
        "Non-finite model parameters",
    )

    return {
        "epoch": epoch + 1,
        "samples": seen,
        "loss": total_loss / seen,
        "observed_label_accuracy": total_correct / seen,
        "attempted_updates": attempts,
        "applied_updates": applied,
        "amp_skipped_updates": skipped,
        "group_sizes": group_sizes,
        "last_lr": lr,
        "sample_label_order_sha256": trace.hexdigest(),
    }


@torch.inference_mode()
def evaluate_validation(model, loader, cfg):
    model.eval()
    count = 0
    total_loss = 0.0
    correct = 0

    for batch in loader:
        require(
            torch.equal(batch["label"], batch["clean_label"]),
            "Validation labels must remain clean",
        )
        images = batch["image"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")

        require(torch.isfinite(loss).item(), "Non-finite validation loss")
        total_loss += float(loss)
        correct += int((logits.argmax(1) == labels).sum())
        count += len(labels)

    require(count == len(loader.dataset), "Incomplete validation")
    return {"loss": total_loss / count, "accuracy": correct / count}


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def make_metadata(cfg, dataset, initialization, mode):
    files = [
        "config/upstream.lock.json",
        "src/model_factory.py",
        "src/data_pipeline.py",
        "src/training_core.py",
        "scripts/run_training.py",
    ]
    return {
        "schema_version": 1,
        "mode": mode,
        "config": copy.deepcopy(cfg),
        "dataset": dataset.dataset_name,
        "initialization": initialization,
        "num_classes": dataset.num_classes,
        "manifests": copy.deepcopy(dataset.manifest_hashes),
        "upstream": read_lock(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in files
        },
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torchvision": version("torchvision"),
            "timm": version("timm"),
            "numpy": np.__version__,
            "pillow": version("pillow"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }


def pack_checkpoint(meta, model, optimizer, scaler, next_epoch, history):
    return {
        "meta": copy.deepcopy(meta),
        "next_epoch": next_epoch,
        "model": cpu_copy(model.state_dict()),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "scaler": cpu_copy(scaler.state_dict()),
        "rng": cpu_copy(rng_state()),
        "history": copy.deepcopy(history),
    }


def save_checkpoint(directory, payload):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    last = directory / "last.pt"
    temporary = directory / "last.pt.tmp"

    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())

    # Keep last.pt intact until the new file has finished writing.
    if last.exists():
        previous_tmp = directory / "previous.pt.tmp"
        shutil.copyfile(last, previous_tmp)
        os.replace(previous_tmp, directory / "previous.pt")

    os.replace(temporary, last)


def read_checkpoint(path):
    # Only load checkpoints generated locally by this training script.
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_checkpoint(payload, expected_meta, model, optimizer, scaler):
    require(payload["meta"] == expected_meta, "Checkpoint protocol/source mismatch")
    require(
        len(payload["history"]) == payload["next_epoch"],
        "Checkpoint progress/history mismatch",
    )
    require(
        0 <= payload["next_epoch"] <= expected_meta["config"]["epochs"],
        "Checkpoint epoch outside configured range",
    )

    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng(payload["rng"])

    return payload["next_epoch"], copy.deepcopy(payload["history"])


def assert_same(left, right, path="state"):
    """Exact comparison for the recovery check."""
    if isinstance(left, torch.Tensor):
        require(
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right),
            f"Recovery mismatch: {path}",
        )
    elif isinstance(left, np.ndarray):
        require(
            isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and np.array_equal(left, right),
            f"Recovery mismatch: {path}",
        )
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), f"Different keys: {path}")
        for key in left:
            assert_same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        require(type(left) is type(right), f"Different types: {path}")
        require(len(left) == len(right), f"Different lengths: {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f"{path}[{index}]")
    else:
        require(left == right, f"Recovery mismatch: {path}")