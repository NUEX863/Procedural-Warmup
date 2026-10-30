"""Pet 90/10 baseline: training, clean validation and checkpoint recovery."""

import os

# Set before importing torch or initializing CUDA.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import copy
import csv
import gc
import hashlib
import io
import json
import math
from importlib.metadata import version
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.pet90_data import (
    Pet90Dataset,
    accumulation_group_sizes,
    file_sha256,
    load_config,
    make_loader,
    require,
)
from src.model_factory import read_lock
from src.training_core import (
    assert_same,
    cpu_copy,
    make_components,
    make_schedule,
    pack_checkpoint,
    read_checkpoint,
    restore_checkpoint,
    setup,
    train_epoch,
)
from run_pet_batch import queue_lock


RUN_ROOT = ROOT / "runs" / "pet90_baseline"


def checked_config():
    cfg = load_config()
    require(
        cfg["optimizer"] == "adamw"
        and cfg["scheduler"] == "cosine"
        and cfg["weight_decay"] == 0.05
        and cfg["min_lr"] == 0.000001
        and cfg["amp"] is True
        and cfg["early_stopping"] is False,
        "Unexpected baseline optimizer/schedule settings",
    )
    return cfg


def metadata(cfg, initialization, mode, train_count, validation_count):
    files = [
        "config/pet90_baseline.json",
        "config/upstream.lock.json",
        "src/pet90_data.py",
        "src/model_factory.py",
        "src/training_core.py",
        "scripts/run_pet90_training.py",
    ]
    return {
        "schema_version": 1,
        "protocol": "pet90_baseline_v1",
        "mode": mode,
        "initialization": initialization,
        "config": copy.deepcopy(cfg),
        "train_count": train_count,
        "validation_count": validation_count,
        "split_sha256": cfg["split_sha256"],
        "upstream": read_lock(),
        "source_sha256": {
            name: file_sha256(ROOT / name) for name in files
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
        "selection": (
            "Highest clean validation correct count; "
            "keep earliest epoch on ties"
        ),
        "validation_precision": "FP32; autocast and TF32 disabled",
        "augmentation": (
            "Deterministic function of seed, epoch and sample identity"
        ),
    }


def schedule_info(schedule, sample_count, cfg, next_epoch):
    steps = math.ceil(
        sample_count / (cfg["batch_size"] * cfg["accumulation_steps"])
    )
    return {
        "sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "steps_per_epoch": steps,
        "total_steps": len(schedule),
        "next_step": next_epoch * steps,
    }


def model_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def validate(model, loader):
    model.eval()
    dataset = loader.dataset
    seen = 0
    correct = 0
    loss_sum = 0.0

    for batch in loader:
        end = seen + len(batch["label"])

        require(
            list(batch["sample_id"]) == dataset.sample_ids[seen:end],
            "Validation image order mismatch",
        )
        require(
            batch["source_index"].tolist() == dataset.indices[seen:end],
            "Validation source indices mismatch",
        )
        require(
            batch["label"].tolist() == dataset.labels[seen:end]
            and torch.equal(batch["label"], batch["clean_label"])
            and not batch["is_corrupted"].any().item(),
            "Validation labels must remain clean",
        )

        images = batch["image"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)

        with torch.autocast("cuda", enabled=False):
            logits = model(images.float())
            loss = F.cross_entropy(logits, labels, reduction="sum")

        require(
            logits.shape == (end - seen, 37)
            and torch.isfinite(logits).all().item()
            and torch.isfinite(loss).item(),
            "Invalid validation output",
        )

        loss_sum += float(loss)
        correct += int((logits.argmax(1) == labels).sum())
        seen = end

    require(seen == len(dataset) and seen > 0, "Incomplete validation")
    return {
        "count": seen,
        "correct": correct,
        "loss": loss_sum / seen,
        "accuracy": correct / seen,
    }


def is_improvement(best, validation):
    if best is None:
        return True
    require(
        validation["count"] == best["validation_count"],
        "Validation denominator changed",
    )
    return validation["correct"] > best["validation_correct"]


def one_epoch(
    model, optimizer, scaler, train_loader, validation_loader,
    cfg, schedule, epoch, best, meta, check_validation=False,
):
    # The existing, previously checked accumulation implementation is reused.
    training = train_epoch(
        model, optimizer, scaler, train_loader, cfg, epoch, schedule
    )
    expected_groups = accumulation_group_sizes(len(train_loader.dataset), cfg)
    require(
        training["group_sizes"] == expected_groups,
        "Incorrect accumulation group sizes",
    )

    before_validation = (
        cpu_copy(model.state_dict()) if check_validation else None
    )
    validation = validate(model, validation_loader)

    if check_validation:
        assert_same(before_validation, cpu_copy(model.state_dict()))
        require(
            all(parameter.grad is None for parameter in model.parameters()),
            "Unexpected gradients after validation",
        )

    improved = is_improvement(best, validation)
    if improved:
        state = cpu_copy(model.state_dict())
        best = {
            "meta": copy.deepcopy(meta),
            "epoch": epoch + 1,
            "validation_count": validation["count"],
            "validation_correct": validation["correct"],
            "validation_accuracy": validation["accuracy"],
            "validation_loss": validation["loss"],
            "model": state,
            "model_sha256": model_digest(state),
        }

    steps = len(expected_groups)
    row = {
        "epoch": epoch + 1,
        "train_samples": training["samples"],
        "train_loss": training["loss"],
        "train_accuracy": training["observed_label_accuracy"],
        "validation_samples": validation["count"],
        "validation_correct": validation["correct"],
        "validation_loss": validation["loss"],
        "validation_accuracy": validation["accuracy"],
        "learning_rate_first": float(schedule[epoch * steps]),
        "learning_rate_last": training["last_lr"],
        "attempted_updates": training["attempted_updates"],
        "applied_updates": training["applied_updates"],
        "amp_skipped_updates": training["amp_skipped_updates"],
        "group_sizes": training["group_sizes"],
        "sample_label_order_sha256": training["sample_label_order_sha256"],
        "best_epoch": best["epoch"],
        "best_validation_accuracy": best["validation_accuracy"],
    }
    return row, best, improved


def make_payload(meta, model, optimizer, scaler, history, best, schedule):
    payload = pack_checkpoint(
        meta, model, optimizer, scaler, len(history), history
    )
    payload["best"] = cpu_copy(best)
    payload["schedule"] = schedule_info(
        schedule, meta["train_count"], meta["config"], len(history)
    )
    return payload


def validate_payload(payload, expected_meta, schedule):
    require(
        payload["meta"] == expected_meta,
        "Checkpoint configuration/source/runtime mismatch",
    )
    epoch = payload["next_epoch"]
    history = payload["history"]
    require(
        type(epoch) is int
        and 0 <= epoch <= expected_meta["config"]["epochs"]
        and len(history) == epoch,
        "Invalid checkpoint progress",
    )
    require(
        payload["schedule"] == schedule_info(
            schedule, expected_meta["train_count"],
            expected_meta["config"], epoch,
        ),
        "Learning-rate schedule/progress mismatch",
    )

    for number, row in enumerate(history, 1):
        require(
            row["epoch"] == number
            and row["train_samples"] == expected_meta["train_count"]
            and row["validation_samples"] == expected_meta["validation_count"],
            "Invalid history record",
        )
        require(
            0 <= row["validation_correct"] <= row["validation_samples"]
            and row["validation_accuracy"]
            == row["validation_correct"] / row["validation_samples"],
            "Invalid validation accuracy",
        )
        require(
            math.isfinite(row["train_loss"])
            and math.isfinite(row["validation_loss"]),
            "Non-finite saved loss",
        )

    best = payload["best"]
    if epoch == 0:
        require(best is None, "Epoch-zero checkpoint cannot have a best model")
    else:
        # Python max keeps the first item when values tie.
        winner = max(history, key=lambda row: row["validation_correct"])
        require(
            best is not None
            and best["meta"] == expected_meta
            and best["epoch"] == winner["epoch"]
            and best["validation_correct"] == winner["validation_correct"]
            and best["validation_count"] == winner["validation_samples"]
            and best["validation_accuracy"] == winner["validation_accuracy"]
            and best["validation_loss"] == winner["validation_loss"],
            "Best-model selection is inconsistent with history",
        )
        require(
            model_digest(best["model"]) == best["model_sha256"],
            "Best-model tensor fingerprint mismatch",
        )


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_torch_save(path, payload):
    atomic_write(path, lambda stream: torch.save(payload, stream))


def write_derived_outputs(directory, payload, update_best=True):
    """The committed latest checkpoint is the authoritative source."""
    directory = Path(directory)
    best = payload["best"]
    best_path = directory / "best_model.pth"

    if best is not None and (update_best or not best_path.exists()):
        atomic_torch_save(best_path, best)

    history = payload["history"]
    if history:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(history[0]))
        writer.writeheader()
        for row in history:
            writer.writerow({
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })
        content = buffer.getvalue().encode("utf-8-sig")
        atomic_write(
            directory / "history.csv",
            lambda stream: stream.write(content),
        )


def commit(directory, payload, update_best=True):
    directory = Path(directory)

    # Commit latest first. It also contains the selected best model.
    atomic_torch_save(directory / "latest_checkpoint.pth", payload)
    write_derived_outputs(directory, payload, update_best=update_best)


def restore(payload, meta, schedule, model, optimizer, scaler):
    validate_payload(payload, meta, schedule)
    epoch, history = restore_checkpoint(
        payload, meta, model, optimizer, scaler
    )
    return epoch, history, cpu_copy(payload["best"])


class DataView(Dataset):
    """Small real-data subsets used only by functional checks."""

    def __init__(self, base, positions):
        self.base = base
        self.positions = list(positions)
        self.split = base.split
        self.sample_ids = [base.sample_ids[i] for i in positions]
        self.indices = [base.indices[i] for i in positions]
        self.labels = [base.labels[i] for i in positions]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, key):
        if self.split == "train":
            epoch, position = key
            item = self.base[(epoch, self.positions[position])]
        else:
            position = key
            item = self.base[self.positions[position]]
        return dict(item, position=position)


def balanced_positions(labels, count):
    buckets = [
        [i for i, label in enumerate(labels) if label == class_id]
        for class_id in range(37)
    ]
    positions = []
    depth = 0
    while len(positions) < count:
        for bucket in buckets:
            if depth < len(bucket):
                positions.append(bucket[depth])
                if len(positions) == count:
                    break
        depth += 1
        require(depth <= max(map(len, buckets)), "Insufficient samples")
    return positions


def check_selection_rule():
    best = None
    decisions = []
    for epoch, correct in enumerate((5, 5, 4, 6), 1):
        validation = {"count": 37, "correct": correct}
        improved = is_improvement(best, validation)
        decisions.append(improved)
        if improved:
            best = {
                "epoch": epoch,
                "validation_count": 37,
                "validation_correct": correct,
            }
        if epoch == 2:
            require(best["epoch"] == 1, "Tie did not preserve earlier epoch")

    require(
        decisions == [True, False, False, True] and best["epoch"] == 4,
        "Incorrect best-model selection",
    )
    print("[PASS] Best selection: improve, tie, decline, improve")


def check_optimizer(model, optimizer, cfg):
    skip = set(model.no_weight_decay())
    mapping = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    require(
        all(group["lr"] == cfg["lr"] for group in optimizer.param_groups),
        "Learning rate was unexpectedly rescaled",
    )
    for name, parameter in model.named_parameters():
        expected = (
            0.0 if parameter.ndim == 1 or name.endswith(".bias") or name in skip
            else cfg["weight_decay"]
        )
        require(mapping[id(parameter)] == expected, f"Wrong decay group: {name}")
    print("[PASS] Author AdamW grouping; actual peak LR=0.0003")


def run_checks(cfg):
    # Independent float64 reference: accumulated gradients versus direct batches.
    from run_training import check_accumulation_math

    setup(cfg)
    check_accumulation_math(cfg)
    check_selection_rule()

    full_train = Pet90Dataset("train", cfg)
    full_validation = Pet90Dataset("validation", cfg)

    train_data = DataView(
        full_train, balanced_positions(full_train.labels, 112)
    )
    validation_data = DataView(
        full_validation, balanced_positions(full_validation.labels, 37)
    )

    formal_schedule = make_schedule(cfg, len(full_train))
    require(
        len(formal_schedule) == 150 * 52
        and float(formal_schedule[0]) == 0.0
        and np.isclose(formal_schedule[259], 0.0003)
        and formal_schedule[-1] >= cfg["min_lr"]
        and formal_schedule[-1] < cfg["min_lr"] * 1.01,
        "Incorrect full training schedule",
    )
    print("[PASS] Formal schedule: 150 epochs, 52 steps/epoch, warmup=260 steps")

    check_root = ROOT / "runs"
    check_root.mkdir(exist_ok=True)
    paired_traces = {}

    with tempfile.TemporaryDirectory(
        prefix="_pet90_training_check_", dir=check_root
    ) as temporary:
        temporary = Path(temporary)

        for initialization in ("random", "procedural"):
            print(f"\nChecking {initialization} ...", flush=True)
            setup(cfg)
            model, optimizer, scaler = make_components(cfg, initialization, 37)
            check_optimizer(model, optimizer, cfg)

            initial_head = model.head.weight.detach().cpu().clone()
            train_loader = make_loader(train_data, cfg)
            validation_loader = make_loader(validation_data, cfg)
            schedule = make_schedule(cfg, len(train_data))
            meta = metadata(
                cfg, initialization, "functional_check",
                len(train_data), len(validation_data),
            )
            directory = temporary / initialization

            history = []
            best = None

            row, best, improved = one_epoch(
                model, optimizer, scaler, train_loader, validation_loader,
                cfg, schedule, 0, best, meta, check_validation=True,
            )
            require(row["group_sizes"] == [64, 48], "Tail group was not exercised")
            require(
                not torch.equal(initial_head, model.head.weight.detach().cpu()),
                "No real model weight update occurred",
            )
            history.append(row)

            first = make_payload(
                meta, model, optimizer, scaler, history, best, schedule
            )
            validate_payload(first, meta, schedule)
            commit(directory, first, update_best=improved)
            atomic_torch_save(directory / "resume_input.pth", first)

            # Simulate an interrupted write before the atomic replacement.
            latest_path = directory / "latest_checkpoint.pth"
            original_hash = file_sha256(latest_path)

            def failing_writer(stream):
                stream.write(b"incomplete")
                raise OSError("Simulated interrupted checkpoint write")

            try:
                atomic_write(latest_path, failing_writer)
            except OSError:
                pass
            else:
                raise RuntimeError("Interrupted-write simulation did not fail")

            require(
                file_sha256(latest_path) == original_hash,
                "Interrupted write damaged the committed checkpoint",
            )
            print("[PASS] Interrupted write preserves the previous checkpoint")

            row, best, improved = one_epoch(
                model, optimizer, scaler, train_loader, validation_loader,
                cfg, schedule, 1, best, meta, check_validation=True,
            )
            history.append(row)
            expected = make_payload(
                meta, model, optimizer, scaler, history, best, schedule
            )
            validate_payload(expected, meta, schedule)
            commit(directory, expected, update_best=improved)
            assert_same(read_checkpoint(latest_path), expected)
            require(
                file_sha256(latest_path) != original_hash,
                "Latest checkpoint was not replaced",
            )
            print("[PASS] Latest replacement and checkpoint round trip")
            print("[PASS] FP32 validation leaves model tensors unchanged")

            paired_traces[initialization] = [
                item["sample_label_order_sha256"] for item in history
            ]

            del model, optimizer, scaler, train_loader, validation_loader, first
            gc.collect()
            torch.cuda.empty_cache()

            # Reconstruct everything, then restore epoch 1 and replay epoch 2.
            setup(cfg)
            model, optimizer, scaler = make_components(cfg, initialization, 37)
            train_loader = make_loader(train_data, cfg)
            validation_loader = make_loader(validation_data, cfg)

            first = read_checkpoint(directory / "resume_input.pth")
            epoch, history, best = restore(
                first, meta, schedule, model, optimizer, scaler
            )
            require(epoch == 1, "Unexpected resume epoch")

            # Simulate stale best_model.pth after latest was already committed.
            restart_directory = directory / "restart"
            atomic_torch_save(
                restart_directory / "latest_checkpoint.pth", first
            )
            atomic_torch_save(
                restart_directory / "best_model.pth",
                {"simulated_stale_file": True},
            )
            write_derived_outputs(restart_directory, first, update_best=True)
            assert_same(
                read_checkpoint(restart_directory / "best_model.pth"),
                first["best"],
            )
            print("[PASS] Best-model file can be repaired from latest checkpoint")

            row, best, improved = one_epoch(
                model, optimizer, scaler, train_loader, validation_loader,
                cfg, schedule, epoch, best, meta, check_validation=True,
            )
            history.append(row)
            restored = make_payload(
                meta, model, optimizer, scaler, history, best, schedule
            )

            assert_same(expected, restored)
            print("[PASS] Continuous vs restored: all saved states exactly equal")

            del (
                model, optimizer, scaler, train_loader, validation_loader,
                first, expected, restored, best,
            )
            gc.collect()
            torch.cuda.empty_cache()

    require(
        paired_traces["random"] == paired_traces["procedural"],
        "Random/Procedural training orders differ",
    )
    print("\nPET90_TRAINING_CHECKS_COMPLETED")
    print("status=PASS")
    print("initializations_checked=random,procedural")
    print("effective_batch_size=64")
    print("actual_check_groups=[64,48]")
    print("best_selection=clean_validation_top1")
    print("resume_state_equality=PASS")
    print("test_images_opened=False")
    print("formal_training_started=False")
    print("Temporary functional-check models were removed.")
    print("No speed benchmark was performed.")


def verify_pairing(directory, initialization, cfg, history):
    other = "procedural" if initialization == "random" else "random"
    other_path = RUN_ROOT / f"oxford_pet_seed11_noise00_{other}" / "latest_checkpoint.pth"
    if not other_path.exists():
        return

    payload = read_checkpoint(other_path)
    expected_meta = metadata(cfg, other, "formal", 3312, 368)
    schedule = make_schedule(cfg, 3312)
    validate_payload(payload, expected_meta, schedule)

    count = min(len(history), len(payload["history"]))
    require(
        [row["sample_label_order_sha256"] for row in history[:count]]
        == [
            row["sample_label_order_sha256"]
            for row in payload["history"][:count]
        ],
        "Random/Procedural sample-label orders differ",
    )
    print(f"[PASS] Random/Procedural paired training orders: {count} epochs")


def run_formal(cfg, initialization, resume):
    setup(cfg)
    train_data = Pet90Dataset("train", cfg)
    validation_data = Pet90Dataset("validation", cfg)
    require(len(train_data) == 3312 and len(validation_data) == 368, "Wrong split")

    directory = RUN_ROOT / f"oxford_pet_seed11_noise00_{initialization}"
    latest_path = directory / "latest_checkpoint.pth"

    if resume:
        require(latest_path.is_file(), "No latest checkpoint to resume")
    else:
        require(
            not directory.exists() or not any(directory.iterdir()),
            "Run directory is not empty. Use explicit --resume when appropriate.",
        )

    model, optimizer, scaler = make_components(cfg, initialization, 37)
    train_loader = make_loader(train_data, cfg)
    validation_loader = make_loader(validation_data, cfg)
    schedule = make_schedule(cfg, len(train_data))
    meta = metadata(cfg, initialization, "formal", 3312, 368)

    if resume:
        payload = read_checkpoint(latest_path)
        start, history, best = restore(
            payload, meta, schedule, model, optimizer, scaler
        )
        # Rebuild best/history files from the authoritative committed checkpoint.
        write_derived_outputs(directory, payload, update_best=True)
        del payload
        print(f"Resuming after completed epoch {start}", flush=True)
    else:
        start, history, best = 0, [], None
        payload = make_payload(
            meta, model, optimizer, scaler, history, best, schedule
        )
        commit(directory, payload)
        del payload

    verify_pairing(directory, initialization, cfg, history)

    for epoch in range(start, cfg["epochs"]):
        require(
            metadata(cfg, initialization, "formal", 3312, 368) == meta,
            "Configuration/source changed during training",
        )
        print(
            f"\nPet90 / {initialization} / noise=0% / "
            f"epoch={epoch + 1}/{cfg['epochs']}",
            flush=True,
        )

        row, best, improved = one_epoch(
            model, optimizer, scaler, train_loader, validation_loader,
            cfg, schedule, epoch, best, meta,
        )
        history.append(row)
        payload = make_payload(
            meta, model, optimizer, scaler, history, best, schedule
        )
        commit(directory, payload, update_best=improved)
        del payload

        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        print(f"Saved latest: {latest_path}", flush=True)
        if improved:
            print(
                f"New best: epoch={best['epoch']}, "
                f"validation={best['validation_correct']}/{best['validation_count']}"
                f" ({100 * best['validation_accuracy']:.4f}%)",
                flush=True,
            )

    payload = read_checkpoint(latest_path)
    validate_payload(payload, meta, schedule)
    assert_same(
        read_checkpoint(directory / "best_model.pth"),
        payload["best"],
    )
    verify_pairing(directory, initialization, cfg, history)

    print("\nPET90_FORMAL_TRAINING_COMPLETED")
    print(f"Completed epochs: {len(history)}")
    print(f"Best epoch: {best['epoch']}")
    print(f"Best validation accuracy: {100 * best['validation_accuracy']:.4f}%")
    print(f"Models and history: {directory}")
    print("Test images were not opened.")


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--train", action="store_true")
    parser.add_argument("--initialization", choices=("random", "procedural"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.check:
        require(
            args.initialization is None and not args.resume,
            "--check must not include training arguments",
        )
    else:
        require(args.initialization is not None, "--train needs --initialization")

    cfg = checked_config()

    # Share the lock with the existing Pet queues.
    with queue_lock():
        if args.check:
            run_checks(cfg)
        else:
            run_formal(cfg, args.initialization, args.resume)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Resume from the latest completed epoch checkpoint.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except torch.cuda.OutOfMemoryError:
        print(
            "\nCUDA OOM. Existing committed checkpoints are retained. "
            "Do not change settings or overwrite checkpoints; return the error.",
            file=sys.stderr,
        )
        raise