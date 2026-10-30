"""Replay the original Pet training trajectory, stopping after epoch 25."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import contextlib
import copy
import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from run_pet_batch import JOBS, job_name, queue_lock, check_pairing
from src.data_pipeline import build_dataset
from src.model_factory import build_model
from src.training_core import (
    make_components,
    make_loader,
    make_metadata,
    make_schedule,
    pack_checkpoint,
    read_checkpoint,
    read_config,
    require,
    restore_checkpoint,
    save_checkpoint,
    setup,
    train_epoch,
)

STOP_EPOCH = 25
OLD_ROOT = ROOT / "runs" / "formal"
NEW_ROOT = ROOT / "runs" / "formal_pet25"
LOG_ROOT = ROOT / "runs" / "batch_logs_pet25"
CONFIG = ROOT / "config" / "training.json"
EVALUATION = ROOT / "runs" / "evaluation" / "oxford_pet_epoch300" / "evaluation.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Tee:
    """Show output in the terminal and append it to the experiment log."""

    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return False


def prepare_context(noise, initialization):
    cfg = read_config(CONFIG)

    # Deliberately keep epochs=300 and warmup_epochs=50.
    # Only the execution stopping point changes to epoch 25.
    require(
        cfg["epochs"] == 300 and cfg["warmup_epochs"] == 50,
        "Do not change the original learning-rate schedule",
    )
    torch.set_num_threads(cfg["cpu_threads"])

    dataset = build_dataset(
        "oxford_pet", "train", noise_percent=noise
    )
    require(
        len(dataset) == 3680 and dataset.num_classes == 37,
        "Unexpected Pet training dataset",
    )

    base_meta = make_metadata(
        cfg, dataset, initialization, mode="formal"
    )
    name = job_name(noise, initialization)
    original_path = OLD_ROOT / name / "last.pt"

    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    require(evaluation["status"] == "PASS", "Original evaluation is incomplete")

    matches = [
        row for row in evaluation["results"]
        if row["noise_percent"] == noise
        and row["initialization"] == initialization
    ]
    require(len(matches) == 1, "Original checkpoint identity is ambiguous")

    original_hash = sha256_file(original_path)
    require(
        original_hash == matches[0]["checkpoint_sha256"],
        f"Original epoch-300 checkpoint changed: {name}",
    )

    # Read the old history only. Its trained weights are NOT used to initialize.
    original = read_checkpoint(original_path)
    require(
        original["next_epoch"] == 300
        and len(original["history"]) == 300
        and original["meta"] == base_meta,
        f"Original protocol/environment/source mismatch: {name}",
    )
    reference_history = copy.deepcopy(original["history"][:STOP_EPOCH])
    del original
    gc.collect()

    require(
        sha256_file(original_path) == original_hash,
        "Original checkpoint changed while reading",
    )

    meta = copy.deepcopy(base_meta)
    meta["mode"] = "formal_pet_early_stop_25"
    meta["execution"] = {
        "stop_after_epoch": STOP_EPOCH,
        "schedule_epochs": 300,
        "warmup_epochs": 50,
        "initialization_policy": "fresh original initialization; not epoch-300 weights",
        "reference_checkpoint_sha256": original_hash,
        "runner_sha256": sha256_file(__file__),
    }

    return {
        "name": name,
        "noise": noise,
        "initialization": initialization,
        "cfg": cfg,
        "dataset": dataset,
        "meta": meta,
        "reference_history": reference_history,
        "directory": NEW_ROOT / name,
    }


def check_history(history, completed, context):
    require(
        type(completed) is int
        and 0 <= completed <= STOP_EPOCH
        and len(history) == completed,
        "Invalid 25-epoch checkpoint progress",
    )
    reference = context["reference_history"]

    for index, actual in enumerate(history):
        expected = reference[index]
        if actual != expected:
            differences = {
                key: {
                    "actual": actual.get(key),
                    "reference": expected.get(key),
                }
                for key in set(actual) | set(expected)
                if actual.get(key) != expected.get(key)
            }
            raise RuntimeError(
                f"{context['name']}: epoch {index + 1} differs from "
                f"the original training history:\n"
                f"{json.dumps(differences, ensure_ascii=False, indent=2)}"
            )


def validate_saved_checkpoint(path, context, expected_epoch=None):
    payload = read_checkpoint(path)
    require(
        payload["meta"] == context["meta"],
        f"Checkpoint configuration/source mismatch: {path}",
    )

    completed = payload["next_epoch"]
    check_history(payload["history"], completed, context)

    if expected_epoch is not None:
        require(completed == expected_epoch, f"Unexpected epoch: {path}")

    # Verify that all saved model tensors load into the correct architecture.
    probe = build_model("random", num_classes=37, seed=11)
    probe.load_state_dict(payload["model"], strict=True)
    require(
        all(
            torch.isfinite(value).all().item()
            for value in probe.state_dict().values()
        ),
        f"Non-finite model tensors: {path}",
    )

    traces = [
        row["sample_label_order_sha256"]
        for row in payload["history"]
    ]
    del probe, payload
    gc.collect()
    return completed, traces


def inspect_context(context):
    directory = context["directory"]
    last = directory / "last.pt"

    if not last.is_file():
        require(
            not directory.exists() or not any(directory.iterdir()),
            f"{directory} contains files but has no last.pt. "
            "Stop for inspection; do not delete or overwrite it.",
        )
        return {
            "status": "new",
            "completed_epochs": 0,
            "traces": [],
        }

    completed, traces = validate_saved_checkpoint(last, context)

    if completed == STOP_EPOCH:
        previous = directory / "previous.pt"
        require(previous.is_file(), "Epoch-24 backup is missing")
        validate_saved_checkpoint(previous, context, expected_epoch=24)

    return {
        "status": "complete" if completed == STOP_EPOCH else "resume",
        "completed_epochs": completed,
        "traces": traces,
    }


def train_one(context):
    cfg = context["cfg"]
    initialization = context["initialization"]
    directory = context["directory"]
    meta = context["meta"]

    setup(cfg)
    loader = make_loader(context["dataset"], cfg, training=True)

    # Fresh Random or official Procedural initialization.
    # No weights are loaded from the old epoch-300 model.
    model, optimizer, scaler = make_components(
        cfg, initialization, num_classes=37
    )
    schedule = make_schedule(cfg, len(context["dataset"]))

    last = directory / "last.pt"
    if last.is_file():
        payload = read_checkpoint(last)
        start_epoch, history = restore_checkpoint(
            payload, meta, model, optimizer, scaler
        )
        check_history(history, start_epoch, context)
        del payload
    else:
        require(
            not directory.exists() or not any(directory.iterdir()),
            "Nonempty run directory without a usable checkpoint",
        )
        start_epoch = 0
        history = []
        save_checkpoint(
            directory,
            pack_checkpoint(meta, model, optimizer, scaler, 0, history),
        )

    for epoch in range(start_epoch, STOP_EPOCH):
        print(
            f"\n{context['name']} / epoch={epoch + 1}/{STOP_EPOCH} "
            f"/ original schedule=300 epochs",
            flush=True,
        )

        metrics = train_epoch(
            model, optimizer, scaler, loader, cfg, epoch, schedule
        )

        candidate_history = history + [metrics]
        check_history(candidate_history, epoch + 1, context)
        history = candidate_history

        # Rolling two-checkpoint storage:
        # after epoch 25, last.pt=25 and previous.pt=24.
        save_checkpoint(
            directory,
            pack_checkpoint(
                meta, model, optimizer, scaler, epoch + 1, history
            ),
        )
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        print(
            f"[MATCH] Epoch {epoch + 1} training record matches "
            "the original run",
            flush=True,
        )
        print(f"Saved: {directory / 'last.pt'}", flush=True)

    del model, optimizer, scaler, loader
    gc.collect()
    torch.cuda.empty_cache()


def run_batch(status_only):
    # Share the existing lock with the original Pet queue.
    with queue_lock():
        contexts = {}
        states = {}

        print(
            "Checking eight Pet experiments. "
            "Stop=25; LR schedule=300; warmup=50.",
            flush=True,
        )

        # Validate every experiment before starting any new training.
        for noise, initialization in JOBS:
            key = (noise, initialization)
            context = prepare_context(noise, initialization)
            state = inspect_context(context)
            contexts[key] = context
            states[key] = state
            print(
                f"[PLAN] {context['name']}: {state['status']}, "
                f"completed={state['completed_epochs']}/25",
                flush=True,
            )

        check_pairing(states)

        if status_only:
            complete = sum(
                state["status"] == "complete"
                for state in states.values()
            )
            print(f"\nVerified complete: {complete}/8")
            print("Completed experiments retain epoch 25 and epoch 24.")
            return

        LOG_ROOT.mkdir(parents=True, exist_ok=True)

        for number, (noise, initialization) in enumerate(JOBS, start=1):
            key = (noise, initialization)

            # Recheck source/config/manifests immediately before each job.
            current = prepare_context(noise, initialization)
            require(
                current["meta"] == contexts[key]["meta"],
                "Configuration, source or manifests changed during the queue",
            )
            context = current
            state = inspect_context(context)

            if state["status"] == "complete":
                states[key] = state
                check_pairing(states)
                print(
                    f"[{number}/8 SKIP VERIFIED] {context['name']}: 25/25",
                    flush=True,
                )
                continue

            log_path = LOG_ROOT / f"{context['name']}.log"
            terminal_out = sys.stdout
            terminal_err = sys.stderr

            with log_path.open(
                "a", encoding="utf-8", buffering=1
            ) as log:
                with contextlib.redirect_stdout(Tee(terminal_out, log)):
                    with contextlib.redirect_stderr(Tee(terminal_err, log)):
                        print(
                            f"\n[{datetime.datetime.now().astimezone().isoformat()}] "
                            f"[{number}/8] {context['name']}: "
                            f"{state['status']} from "
                            f"{state['completed_epochs']}/25",
                            flush=True,
                        )
                        try:
                            train_one(context)
                            finished = inspect_context(context)
                            require(
                                finished["status"] == "complete",
                                "Epoch-24/25 verification failed",
                            )
                            states[key] = finished
                            check_pairing(states)
                            print(
                                f"[{number}/8 VERIFIED COMPLETE] "
                                f"{context['name']}: last=25, previous=24",
                                flush=True,
                            )
                        except BaseException:
                            import traceback
                            traceback.print_exc()
                            raise

            gc.collect()
            torch.cuda.empty_cache()

        require(
            all(state["status"] == "complete" for state in states.values()),
            "Not all experiments completed",
        )
        check_pairing(states)

        print("\nPET_25_BATCH_COMPLETED")
        print("verified_experiments=8")
        print("last_checkpoint_epoch=25")
        print("previous_checkpoint_epoch=24")
        print("learning_rate_schedule_epochs=300")
        print("warmup_epochs=50")
        print("first_25_epoch_records_match_original=True")
        print("test_dataset_opened=False")
        print(f"Models: {NEW_ROOT}")
        print(f"Logs: {LOG_ROOT}")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()
    run_batch(status_only=args.status)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Completed epoch checkpoints are retained. "
            "Use the same --run command to resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as error:
        print(
            f"\nQUEUE STOPPED: {type(error).__name__}: {error}\n"
            "Existing checkpoints were not deleted. "
            "Return the error output before changing any settings.",
            file=sys.stderr,
        )
        raise SystemExit(1)