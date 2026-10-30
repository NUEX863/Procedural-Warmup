"""Pet90: eight noise/initialization experiments with verified recovery."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import contextlib
import copy
import csv
import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

import run_pet90_training as base
from run_pet_batch import queue_lock
from src.model_factory import build_model
from src.pet90_data import EpochKeySampler
from src.pet90_noise_data import (
    NOISE_HASHES,
    Pet90NoiseDataset,
    noise_path,
    read_noise_manifest,
)


RUN_ROOT = ROOT / "runs" / "pet90_noise_v1"
LOG_ROOT = RUN_ROOT / "logs"
JOBS = [
    (noise, initialization)
    for noise in (0, 10, 20, 30)
    for initialization in ("random", "procedural")
]


def require(condition, message):
    base.require(condition, message)


def job_name(noise, initialization):
    require((noise, initialization) in JOBS, "Unknown experiment")
    return f"oxford_pet_seed11_noise{noise:02d}_{initialization}"


def timestamp():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def config_for(noise):
    require(noise in NOISE_HASHES, "Invalid noise percentage")
    cfg = base.checked_config()
    # Read the verified common recipe, then resolve the actual experiment.
    cfg["protocol"] = "pet90_noise_training_v1"
    cfg["noise_percent"] = noise
    return cfg


def metadata(cfg, initialization, mode, train_count, validation_count):
    manifest = read_noise_manifest(cfg, cfg["noise_percent"])
    meta = base.metadata(
        cfg, initialization, mode, train_count, validation_count
    )
    meta["protocol"] = "pet90_noise_training_v1"
    path = noise_path(cfg["noise_percent"])
    extra_files = [
        ROOT / "src" / "pet90_noise_data.py",
        Path(__file__).resolve(),
        path,
        path.with_name(path.name + ".sha256"),
    ]
    for item in extra_files:
        meta["source_sha256"][item.relative_to(ROOT).as_posix()] = (
            base.file_sha256(item)
        )

    meta["noise"] = {
        "nominal_percent": cfg["noise_percent"],
        "actual_percent": manifest["actual_noise_percent"],
        "corrupted_count_full_training": manifest["corrupted_count"],
        "full_training_count": manifest["training_count"],
        "manifest": path.relative_to(ROOT).as_posix(),
        "manifest_sha256": NOISE_HASHES[cfg["noise_percent"]],
    }
    meta["training_accuracy_target"] = (
        "Observed training labels, including fixed corrupted labels"
    )
    return meta


def fingerprint(meta):
    return hashlib.sha256(
        json.dumps(meta, sort_keys=True).encode("utf-8")
    ).hexdigest()


def expected_trace(dataset, cfg, epoch):
    """Independently reconstruct the labels and indices used by each batch."""
    order = [p for _, p in EpochKeySampler(dataset, cfg["seed"], epoch)]
    digest = hashlib.sha256()
    size = cfg["batch_size"]

    for start in range(0, len(order), size):
        positions = order[start:start + size]
        indices = torch.tensor(
            [dataset.indices[p] for p in positions], dtype=torch.int64
        )
        labels = torch.tensor(
            [dataset.labels[p] for p in positions], dtype=torch.int64
        )
        digest.update(indices.numpy().tobytes())
        digest.update(labels.numpy().tobytes())
    return digest.hexdigest()


def one_epoch(
    model, optimizer, scaler, train_loader, validation_loader,
    cfg, schedule, epoch, best, meta, checking=False,
):
    trace = expected_trace(train_loader.dataset, cfg, epoch)
    row, best, improved = base.one_epoch(
        model, optimizer, scaler, train_loader, validation_loader,
        cfg, schedule, epoch, best, meta,
        check_validation=checking,
    )
    require(
        row["sample_label_order_sha256"] == trace,
        "Training did not consume the expected observed labels/order",
    )
    row["noise_percent"] = cfg["noise_percent"]
    row["actual_noise_percent"] = meta["noise"]["actual_percent"]
    return row, best, improved


def validate_payload(payload, meta, schedule):
    base.validate_payload(payload, meta, schedule)
    cfg = meta["config"]
    groups = base.accumulation_group_sizes(meta["train_count"], cfg)
    steps = len(groups)

    for index, row in enumerate(payload["history"]):
        require(
            row["noise_percent"] == cfg["noise_percent"]
            and row["actual_noise_percent"] == meta["noise"]["actual_percent"],
            "History noise setting differs from its checkpoint",
        )
        require(
            row["group_sizes"] == groups
            and row["attempted_updates"] == steps
            and 0 < row["applied_updates"] <= steps
            and row["amp_skipped_updates"] >= 0
            and row["applied_updates"] + row["amp_skipped_updates"] == steps,
            "Invalid update/accumulation history",
        )
        require(
            row["learning_rate_first"] == float(schedule[index * steps])
            and row["learning_rate_last"]
            == float(schedule[(index + 1) * steps - 1]),
            "Saved learning rates differ from the fixed schedule",
        )
        require(
            0 <= row["train_accuracy"] <= 1,
            "Invalid training accuracy",
        )


def make_payload(meta, model, optimizer, scaler, history, best, schedule):
    payload = base.make_payload(
        meta, model, optimizer, scaler, history, best, schedule
    )
    validate_payload(payload, meta, schedule)
    return payload


def restore(payload, meta, schedule, model, optimizer, scaler):
    validate_payload(payload, meta, schedule)
    return base.restore(
        payload, meta, schedule, model, optimizer, scaler
    )


def derived_outputs_match(directory, payload):
    """Best/history files can be reconstructed from the committed latest."""
    if not payload["history"]:
        return True

    try:
        base.assert_same(
            base.read_checkpoint(directory / "best_model.pth"),
            payload["best"],
        )
        with (directory / "history.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            actual = list(csv.DictReader(stream))

        expected = [
            {
                key: (
                    json.dumps(value)
                    if isinstance(value, (list, dict))
                    else str(value)
                )
                for key, value in row.items()
            }
            for row in payload["history"]
        ]
        return actual == expected
    except Exception:
        # These are derived files. The latest checkpoint is validated separately.
        return False


def inspect_job(noise, initialization):
    cfg = config_for(noise)
    meta = metadata(cfg, initialization, "formal", 3312, 368)
    directory = RUN_ROOT / job_name(noise, initialization)
    latest = directory / "latest_checkpoint.pth"

    state = {
        "name": directory.name,
        "status": "new",
        "completed": 0,
        "traces": [],
        "fingerprint": fingerprint(meta),
    }

    if not latest.is_file():
        require(
            not directory.exists() or not any(directory.iterdir()),
            f"Nonempty run directory without a checkpoint: {directory}",
        )
        return state

    schedule = base.make_schedule(cfg, 3312)
    payload = base.read_checkpoint(latest)
    validate_payload(payload, meta, schedule)

    # Check that both current and selected-best weights load into ViT-Tiny.
    reference = build_model(
        initialization=initialization, num_classes=37, seed=11
    )
    states = [payload["model"]]
    if payload["best"] is not None:
        states.append(payload["best"]["model"])

    for model_state in states:
        reference.load_state_dict(model_state, strict=True)
        require(
            all(torch.isfinite(tensor).all().item()
                for tensor in reference.state_dict().values()),
            "Saved model contains non-finite tensors",
        )

    completed = payload["next_epoch"]
    complete = (
        completed == cfg["epochs"]
        and derived_outputs_match(directory, payload)
    )
    state.update(
        status="complete" if complete else "resume",
        completed=completed,
        traces=[
            row["sample_label_order_sha256"]
            for row in payload["history"]
        ],
    )
    return state


def check_pairing(states):
    for noise in NOISE_HASHES:
        left = states.get((noise, "random"))
        right = states.get((noise, "procedural"))
        if left is None or right is None:
            continue
        common = min(left["completed"], right["completed"])
        require(
            left["traces"][:common] == right["traces"][:common],
            f"Random/Procedural input orders differ at noise={noise}%",
        )


def train_one(noise, initialization, expected_fingerprint):
    cfg = config_for(noise)
    base.setup(cfg)

    train_data = Pet90NoiseDataset("train", cfg, noise)
    validation_data = Pet90NoiseDataset("validation", cfg)
    require(
        len(train_data) == 3312 and len(validation_data) == 368,
        "Unexpected training/validation sizes",
    )

    meta = metadata(cfg, initialization, "formal", 3312, 368)
    require(
        fingerprint(meta) == expected_fingerprint,
        "Experiment configuration changed after queue planning",
    )
    schedule = base.make_schedule(cfg, len(train_data))
    directory = RUN_ROOT / job_name(noise, initialization)
    latest = directory / "latest_checkpoint.pth"

    model, optimizer, scaler = base.make_components(
        cfg, initialization, 37
    )
    train_loader = base.make_loader(train_data, cfg)
    validation_loader = base.make_loader(validation_data, cfg)

    if latest.is_file():
        payload = base.read_checkpoint(latest)
        start, history, best = restore(
            payload, meta, schedule, model, optimizer, scaler
        )
        base.write_derived_outputs(directory, payload, update_best=True)
        del payload
        print(f"RESUME: completed={start}/{cfg['epochs']}", flush=True)
    else:
        require(
            not directory.exists() or not any(directory.iterdir()),
            f"Refusing to overwrite nonempty directory: {directory}",
        )
        start, history, best = 0, [], None
        payload = make_payload(
            meta, model, optimizer, scaler, history, best, schedule
        )
        base.commit(directory, payload)
        del payload

    for epoch in range(start, cfg["epochs"]):
        require(
            metadata(cfg, initialization, "formal", 3312, 368) == meta,
            "Configuration, source or manifest changed during training",
        )
        print(
            f"\n{directory.name} / epoch={epoch + 1}/{cfg['epochs']}",
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
        base.commit(directory, payload, update_best=improved)
        del payload

        print(json.dumps(row, allow_nan=False), flush=True)
        print(
            f"Training observed-label accuracy: "
            f"{100 * row['train_accuracy']:.4f}% | "
            f"Clean validation accuracy: "
            f"{100 * row['validation_accuracy']:.4f}% | "
            f"Best epoch: {best['epoch']}",
            flush=True,
        )
        print(f"Saved: {latest}", flush=True)

    payload = base.read_checkpoint(latest)
    validate_payload(payload, meta, schedule)
    require(
        payload["next_epoch"] == cfg["epochs"]
        and derived_outputs_match(directory, payload),
        "Final checkpoint/best model/history verification failed",
    )
    print(
        f"\nVERIFIED COMPLETE: {directory.name}; "
        f"best_epoch={best['epoch']}; "
        f"best_validation_accuracy={100 * best['validation_accuracy']:.4f}%",
        flush=True,
    )

    del model, optimizer, scaler, train_loader, validation_loader, payload, best
    gc.collect()
    torch.cuda.empty_cache()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        self.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_queue(status_only):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / "queue.log").open(
        "a", encoding="utf-8", buffering=1
    ) as queue_log:
        output = Tee(sys.stdout, queue_log)
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            print(
                f"\n[{timestamp()}] "
                f"{'STATUS' if status_only else 'BATCH START'}",
                flush=True,
            )
            states = {}
            for key in JOBS:
                state = inspect_job(*key)
                states[key] = state
                print(
                    f"[PLAN] {state['name']}: {state['status']}, "
                    f"completed={state['completed']}/150",
                    flush=True,
                )

            check_pairing(states)
            if status_only:
                complete = sum(s["status"] == "complete" for s in states.values())
                print(f"Verified complete: {complete}/8")
                print("No training was started.")
                return

            fingerprints = {
                key: state["fingerprint"] for key, state in states.items()
            }
            try:
                for number, key in enumerate(JOBS, 1):
                    state = inspect_job(*key)
                    require(
                        state["fingerprint"] == fingerprints[key],
                        "Configuration/source changed while queue was running",
                    )
                    states[key] = state
                    check_pairing(states)

                    if state["status"] == "complete":
                        print(f"[{number}/8 SKIP VERIFIED] {state['name']}")
                        continue

                    print(
                        f"\n[{timestamp()}] [{number}/8] {state['name']}",
                        flush=True,
                    )
                    log_path = LOG_ROOT / f"{state['name']}.log"
                    with log_path.open(
                        "a", encoding="utf-8", buffering=1
                    ) as experiment_log:
                        tee = Tee(sys.stdout, experiment_log)
                        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                            print(f"\n[{timestamp()}] START OR RESUME", flush=True)
                            train_one(*key, fingerprints[key])

                    finished = inspect_job(*key)
                    require(
                        finished["status"] == "complete"
                        and finished["fingerprint"] == fingerprints[key],
                        "Experiment exited without verified completion",
                    )
                    states[key] = finished
                    check_pairing(states)

                require(
                    all(s["status"] == "complete" for s in states.values()),
                    "Not all eight experiments completed",
                )
                print("\nPET90_NOISE_BATCH_COMPLETED")
                print("verified_experiments=8/8")
                print("epochs_per_experiment=150")
                print("paired_sample_label_orders=PASS")
                print("test_images_opened=False")
                print(f"Results: {RUN_ROOT}")
            except BaseException as error:
                print(
                    f"\n[{timestamp()}] QUEUE STOPPED: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                raise


def run_checks():
    """Exercise all eight combinations using temporary real-data subsets."""
    cfg0 = config_for(0)
    base.setup(cfg0)
    base.check_selection_rule()

    clean = Pet90NoiseDataset("train", cfg0, 0)
    positions = base.balanced_positions(clean.clean_labels, 112)

    # Include at least one image corrupted even at 10%.
    # Keep this same subset for all four noise levels.
    ten = read_noise_manifest(cfg0, 10)["training"]
    first_corrupted = next(
        i for i, flag in enumerate(ten["is_corrupted"]) if flag
    )
    if first_corrupted not in positions:
        positions[-1] = first_corrupted

    validation_full = Pet90NoiseDataset("validation", cfg0)
    validation = base.DataView(
        validation_full,
        base.balanced_positions(validation_full.clean_labels, 37),
    )

    formal_schedule = base.make_schedule(cfg0, 3312)
    require(
        len(formal_schedule) == 150 * 52
        and abs(float(formal_schedule[259]) - 0.0003) < 1e-12,
        "Unexpected formal learning-rate schedule",
    )

    traces = {}
    check_root = ROOT / "runs"
    check_root.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="_pet90_noise_check_", dir=check_root
    ) as temporary:
        temporary = Path(temporary)

        for noise, initialization in JOBS:
            print(f"\nChecking {job_name(noise, initialization)} ...", flush=True)
            cfg = config_for(noise)
            full = Pet90NoiseDataset("train", cfg, noise)
            train_data = base.DataView(full, positions)
            changed = sum(full.is_corrupted[i] for i in positions)
            require(
                (changed == 0 if noise == 0 else 0 < changed < len(positions)),
                "Functional subset did not exercise the intended labels",
            )

            base.setup(cfg)
            model, optimizer, scaler = base.make_components(
                cfg, initialization, 37
            )
            base.check_optimizer(model, optimizer, cfg)
            initial_head = model.head.weight.detach().cpu().clone()

            train_loader = base.make_loader(train_data, cfg)
            validation_loader = base.make_loader(validation, cfg)
            schedule = base.make_schedule(cfg, len(train_data))
            meta = metadata(
                cfg, initialization, "functional_check",
                len(train_data), len(validation),
            )
            directory = temporary / job_name(noise, initialization)
            history, best = [], None

            for epoch in range(2):
                row, best, improved = one_epoch(
                    model, optimizer, scaler, train_loader, validation_loader,
                    cfg, schedule, epoch, best, meta, checking=True,
                )
                history.append(row)
                payload = make_payload(
                    meta, model, optimizer, scaler, history, best, schedule
                )
                base.commit(directory, payload, update_best=improved)
                require(
                    derived_outputs_match(directory, payload),
                    "Saved best model/history differs from latest",
                )
                if epoch == 0:
                    base.atomic_torch_save(directory / "resume_input.pth", payload)

            require(
                not torch.equal(initial_head, model.head.weight.detach().cpu()),
                "No real parameter update occurred",
            )
            expected = payload
            traces[(noise, initialization)] = [
                row["sample_label_order_sha256"] for row in history
            ]

            # A checkpoint must reject a different experiment's metadata.
            wrong_meta = copy.deepcopy(meta)
            wrong_meta["config"]["noise_percent"] = (noise + 10) % 40
            try:
                validate_payload(expected, wrong_meta, schedule)
            except RuntimeError:
                pass
            else:
                raise RuntimeError("Mismatched noise metadata was accepted")

            del model, optimizer, scaler, train_loader, validation_loader, best
            gc.collect()
            torch.cuda.empty_cache()

            # Reconstruct the entire training state, then replay epoch 2.
            base.setup(cfg)
            model, optimizer, scaler = base.make_components(
                cfg, initialization, 37
            )
            train_loader = base.make_loader(train_data, cfg)
            validation_loader = base.make_loader(validation, cfg)
            first = base.read_checkpoint(directory / "resume_input.pth")
            start, history, best = restore(
                first, meta, schedule, model, optimizer, scaler
            )
            require(start == 1, "Wrong recovery epoch")

            # Repair a stale derived model from the authoritative checkpoint.
            repair_directory = directory / "repair"
            base.atomic_torch_save(
                repair_directory / "best_model.pth", {"stale": True}
            )
            base.write_derived_outputs(repair_directory, first, update_best=True)
            require(
                derived_outputs_match(repair_directory, first),
                "Best/history recovery failed",
            )

            row, best, improved = one_epoch(
                model, optimizer, scaler, train_loader, validation_loader,
                cfg, schedule, start, best, meta, checking=True,
            )
            history.append(row)
            restored = make_payload(
                meta, model, optimizer, scaler, history, best, schedule
            )
            base.assert_same(expected, restored)

            print(
                f"[PASS] noise={noise}% / {initialization}: "
                f"observed targets; groups=[64,48]; clean validation; "
                f"best/latest/history; exact resume",
                flush=True,
            )

            del (
                model, optimizer, scaler, train_loader, validation_loader,
                first, expected, restored, payload, best,
            )
            gc.collect()
            torch.cuda.empty_cache()

    for noise in NOISE_HASHES:
        require(
            traces[(noise, "random")] == traces[(noise, "procedural")],
            f"Paired training orders differ at noise={noise}%",
        )

    print("\nPET90_NOISE_TRAINING_CHECKS_COMPLETED")
    print("status=PASS")
    print("combinations_checked=8")
    print("observed_training_labels=PASS")
    print("clean_validation=PASS")
    print("best_model_and_history_recovery=PASS")
    print("continuous_vs_restored_all_states=PASS")
    print("cross_noise_checkpoint_rejected=PASS")
    print("paired_training_orders=PASS")
    print("test_images_opened=False")
    print("formal_training_started=False")
    print("Temporary functional-check models were removed.")
    print("No speed benchmark was performed.")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    cfg = base.checked_config()
    torch.set_num_threads(cfg["cpu_threads"])

    # One shared lock protects the entire operation, including all eight runs.
    with queue_lock():
        if args.check:
            run_checks()
        else:
            run_queue(status_only=args.status)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Completed epoch checkpoints are retained.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except torch.cuda.OutOfMemoryError:
        print(
            "\nCUDA OOM. Queue stopped; committed checkpoints are retained. "
            "Return the error before changing the experiment settings.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise