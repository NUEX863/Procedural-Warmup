"""Run the eight Pet experiments sequentially, with verified resume/skip."""

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import math
import msvcrt
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PYTHON = sys.executable
CONFIG = ROOT / "config" / "training.json"
LOG_ROOT = ROOT / "runs" / "batch_logs"
FORMAL_ROOT = ROOT / "runs" / "formal"

JOBS = [
    (noise, initialization)
    for noise in (0, 10, 20, 30)
    for initialization in ("random", "procedural")
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def job_name(noise, initialization):
    return f"oxford_pet_seed11_noise{noise:02d}_{initialization}"


def timestamp():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


@contextlib.contextmanager
def queue_lock():
    """Prevent two copies of this queue from running at the same time."""
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stream = (LOG_ROOT / "pet_batch.lock").open("a+b")
    locked = False

    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()

        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            locked = True
        except OSError as error:
            raise RuntimeError(
                "Another Pet queue is running. "
                "Read pet_batch.log for progress; do not launch a second queue."
            ) from error

        yield
    finally:
        if locked:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        stream.close()


def inspect_job(noise, initialization):
    """
    Runs in a short-lived subprocess.
    The queue itself does not import torch or retain a CUDA context.
    """
    require((noise, initialization) in JOBS, "Unknown experiment")
    sys.path.insert(0, str(ROOT))

    # Keep inspection stdout machine-readable.
    with contextlib.redirect_stdout(io.StringIO()):
        from src.training_core import (
            make_metadata,
            read_checkpoint,
            read_config,
        )
        from src.data_pipeline import build_dataset
        from src.model_factory import build_model
        import torch

        cfg = read_config(CONFIG)
        torch.set_num_threads(cfg["cpu_threads"])

        dataset = build_dataset(
            "oxford_pet", "train", noise_percent=noise
        )
        require(len(dataset) == 3680, "Expected 3680 Pet training images")
        require(dataset.num_classes == 37, "Expected 37 Pet classes")

        expected_meta = make_metadata(
            cfg, dataset, initialization, mode="formal"
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                expected_meta,
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()

        # CPU construction verifies the pinned author implementation.
        # Procedural construction also verifies the warmup checkpoint.
        model = build_model(
            initialization=initialization,
            num_classes=37,
            seed=11,
        )

        directory = FORMAL_ROOT / job_name(noise, initialization)
        checkpoint_path = directory / "last.pt"

        result = {
            "name": directory.name,
            "directory": str(directory),
            "fingerprint": fingerprint,
            "status": "new",
            "completed_epochs": 0,
            "traces": [],
        }

        if not checkpoint_path.is_file():
            require(
                not directory.exists() or not any(directory.iterdir()),
                f"{directory} contains files but has no last.pt. "
                "Stop for inspection; do not overwrite or delete it.",
            )
        else:
            payload = read_checkpoint(checkpoint_path)
            require(
                isinstance(payload, dict)
                and {
                    "meta", "next_epoch", "model", "optimizer",
                    "scaler", "rng", "history",
                }.issubset(payload),
                f"Incomplete checkpoint: {checkpoint_path}",
            )
            require(
                payload["meta"] == expected_meta,
                f"Configuration, source, environment or manifest mismatch: "
                f"{checkpoint_path}",
            )

            completed = payload["next_epoch"]
            history = payload["history"]
            require(
                type(completed) is int
                and 0 <= completed <= cfg["epochs"]
                and len(history) == completed,
                f"Invalid checkpoint progress: {checkpoint_path}",
            )

            # Verify that final/partial model parameters can actually load.
            model.load_state_dict(payload["model"], strict=True)
            require(
                all(
                    torch.isfinite(value).all().item()
                    for value in model.state_dict().values()
                ),
                f"Non-finite model weights: {checkpoint_path}",
            )

            expected_groups = [512] * 7 + [96]
            for epoch, row in enumerate(history, start=1):
                require(
                    row["epoch"] == epoch
                    and row["samples"] == 3680
                    and row["group_sizes"] == expected_groups
                    and row["attempted_updates"] == 8
                    and 0 < row["applied_updates"] <= 8
                    and row["amp_skipped_updates"] >= 0
                    and (
                        row["applied_updates"]
                        + row["amp_skipped_updates"] == 8
                    ),
                    f"Invalid epoch record: {directory.name}, epoch={epoch}",
                )
                require(
                    math.isfinite(row["loss"])
                    and math.isfinite(row["last_lr"])
                    and 0 <= row["observed_label_accuracy"] <= 1,
                    f"Invalid metrics: {directory.name}, epoch={epoch}",
                )
                require(
                    "validation" not in row,
                    "Pet must not have an independent validation result",
                )
                trace = row["sample_label_order_sha256"]
                require(
                    isinstance(trace, str) and len(trace) == 64,
                    f"Invalid sample-order record: {directory.name}",
                )

            result.update(
                status=(
                    "complete" if completed == cfg["epochs"] else "resume"
                ),
                completed_epochs=completed,
                traces=[
                    row["sample_label_order_sha256"] for row in history
                ],
            )

    print(json.dumps(result, ensure_ascii=True))


def inspect_subprocess(noise, initialization):
    result = subprocess.run(
        [
            PYTHON, "-u", str(SCRIPT),
            "--inspect", str(noise), initialization,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        },
    )
    require(
        result.returncode == 0,
        f"Inspection failed: {job_name(noise, initialization)}\n"
        f"{result.stdout}\n{result.stderr}",
    )
    return json.loads(result.stdout)


def check_pairing(states):
    """Compare actual recorded image/label orders for matching epochs."""
    for noise in (0, 10, 20, 30):
        random_state = states.get((noise, "random"))
        procedural_state = states.get((noise, "procedural"))
        if random_state is None or procedural_state is None:
            continue

        shared_epochs = min(
            random_state["completed_epochs"],
            procedural_state["completed_epochs"],
        )
        require(
            random_state["traces"][:shared_epochs]
            == procedural_state["traces"][:shared_epochs],
            f"Random/Procedural image-label order differs at noise={noise}%",
        )


def run_child(noise, initialization, resume, emit):
    name = job_name(noise, initialization)
    command = [
        PYTHON, "-u", str(ROOT / "scripts" / "run_training.py"),
        "--train",
        "--config", str(CONFIG),
        "--dataset", "oxford_pet",
        "--initialization", initialization,
        "--noise", str(noise),
    ]
    if resume:
        command.extend(["--resume", "last"])

    experiment_log = LOG_ROOT / f"{name}.log"
    emit(f"[{timestamp()}] Log: {experiment_log}")

    with experiment_log.open(
        "a", encoding="utf-8", buffering=1
    ) as stream:
        stream.write(
            f"\n[{timestamp()}] "
            f"{'RESUME' if resume else 'START'} {name}\n"
        )

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            },
        )

        try:
            for line in process.stdout:
                stream.write(line)
                emit(line.rstrip("\r\n"))

            returncode = process.wait()
        except BaseException:
            # Stop this launched process tree if the queue is interrupted.
            # The training script's previous completed checkpoint is retained.
            if process.poll() is None:
                subprocess.run(
                    [
                        "taskkill", "/PID", str(process.pid), "/T", "/F"
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                process.wait()
            raise
        finally:
            process.stdout.close()

    require(
        returncode == 0,
        f"{name} exited with code {returncode}. "
        f"Queue stopped. See {experiment_log}",
    )


def run_queue(status_only):
    with queue_lock():
        with (LOG_ROOT / "pet_batch.log").open(
            "a", encoding="utf-8", buffering=1
        ) as queue_log:

            def emit(message):
                print(message, flush=True)
                queue_log.write(message + "\n")

            try:
                emit(
                    f"\n[{timestamp()}] "
                    f"{'STATUS CHECK' if status_only else 'PET BATCH START'}"
                )
                emit("Checking all eight experiments before training ...")

                states = {}
                fingerprints = {}

                for noise, initialization in JOBS:
                    state = inspect_subprocess(noise, initialization)
                    key = (noise, initialization)
                    states[key] = state
                    fingerprints[key] = state["fingerprint"]
                    emit(
                        f"[PLAN] {state['name']}: {state['status']}, "
                        f"completed={state['completed_epochs']}/300"
                    )

                check_pairing(states)

                if status_only:
                    complete_count = sum(
                        state["status"] == "complete"
                        for state in states.values()
                    )
                    emit(f"Verified complete: {complete_count}/8")
                    return

                for number, (noise, initialization) in enumerate(JOBS, start=1):
                    key = (noise, initialization)
                    state = inspect_subprocess(noise, initialization)

                    require(
                        state["fingerprint"] == fingerprints[key],
                        "Configuration/source changed while the queue was running",
                    )

                    if state["status"] == "complete":
                        states[key] = state
                        check_pairing(states)
                        emit(
                            f"[{number}/8 SKIP VERIFIED] "
                            f"{state['name']}: 300/300"
                        )
                        continue

                    emit(
                        f"\n[{timestamp()}] [{number}/8] "
                        f"{state['name']}: "
                        f"{state['status']} from {state['completed_epochs']}/300"
                    )

                    run_child(
                        noise,
                        initialization,
                        resume=state["status"] == "resume",
                        emit=emit,
                    )

                    # A zero exit code alone is not proof of completion.
                    finished = inspect_subprocess(noise, initialization)
                    require(
                        finished["fingerprint"] == fingerprints[key],
                        "Configuration/source changed during training",
                    )
                    require(
                        finished["status"] == "complete"
                        and finished["completed_epochs"] == 300,
                        "Training exited without a verified epoch-300 checkpoint",
                    )

                    states[key] = finished
                    check_pairing(states)
                    emit(
                        f"[{timestamp()}] [{number}/8 VERIFIED COMPLETE] "
                        f"{finished['name']}"
                    )

                require(
                    all(state["status"] == "complete" for state in states.values()),
                    "Not all eight experiments completed",
                )
                check_pairing(states)

                emit("\nPET_BATCH_COMPLETED")
                emit("Verified final models: 8/8")
                emit("Epochs per model: 300")
                emit("Random/Procedural sample-label orders: MATCH")
                emit("Test dataset was not opened.")
                emit(f"Model root: {FORMAL_ROOT}")

            except BaseException as error:
                emit(
                    f"\n[{timestamp()}] QUEUE STOPPED: "
                    f"{type(error).__name__}: {error}"
                )
                raise


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--inspect", nargs=2, metavar=("NOISE", "INITIALIZATION"))
    args = parser.parse_args()

    if args.inspect is not None:
        inspect_job(int(args.inspect[0]), args.inspect[1])
    else:
        run_queue(status_only=args.status)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nQueue interrupted. Resume later with the same --run command.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)