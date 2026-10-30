"""Explicitly separate functional checks from formal training."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import copy
import gc
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from src.data_pipeline import build_dataset
from src.training_core import (
    assert_same,
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
    evaluate_validation,
)


class ToyDataset(Dataset):
    """A tiny CPU fixture for checking accumulation mathematics."""

    def __init__(self):
        self.images = torch.arange(20, dtype=torch.float64).reshape(5, 4) / 20
        self.labels = torch.arange(5, dtype=torch.long) % 3

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "source_index": index,
        }


def check_accumulation_math(cfg):
    small_cfg = dict(cfg)
    small_cfg.update(
        batch_size=2,
        accumulation_steps=2,
        num_workers=0,
        amp=False,
    )
    dataset = ToyDataset()
    loader = make_loader(dataset, small_cfg, training=True)
    order = list(iter(loader.sampler))

    model = torch.nn.Linear(4, 3).double()
    reference = copy.deepcopy(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    result = train_epoch(
        model,
        optimizer,
        scaler,
        loader,
        small_cfg,
        epoch=0,
        schedule=np.array([0.01, 0.01]),
    )

    # Reference: directly average over groups of 4 and 1 samples.
    for indices in (order[:4], order[4:]):
        reference_optimizer.zero_grad(set_to_none=True)
        logits = reference(dataset.images[indices])
        loss = torch.nn.functional.cross_entropy(
            logits, dataset.labels[indices]
        )
        loss.backward()
        reference_optimizer.step()

    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    require(result["group_sizes"] == [4, 1], "Tail-group math check failed")
    print("[PASS] Accumulation equals direct batch averaging, including tail")


def check_training(cfg):
    setup(cfg)
    check_accumulation_math(cfg)

    work_root = ROOT / "runs"
    work_root.mkdir(exist_ok=True)

    # Temporary checkpoints are necessary to test real saving/loading.
    # They are removed automatically when the check finishes.
    with tempfile.TemporaryDirectory(
        prefix="_training_check_", dir=work_root
    ) as temporary:
        check_root = Path(temporary)
        paired_traces = {}

        for dataset_name in ("cifar100", "oxford_pet"):
            # Use a noisy manifest to exercise observed-label training.
            base = build_dataset(dataset_name, "train", noise_percent=30)

            # One full effective batch and one incomplete tail.
            subset = Subset(base, range(529))

            for initialization in ("random", "procedural"):
                print(
                    f"\nChecking {dataset_name} / {initialization} ...",
                    flush=True,
                )
                setup(cfg)
                loader = make_loader(subset, cfg, training=True)
                model, optimizer, scaler = make_components(
                    cfg, initialization, base.num_classes
                )
                schedule = make_schedule(cfg, len(subset))
                meta = make_metadata(cfg, base, initialization, mode="check")

                initial_head = model.head.weight.detach().cpu().clone()
                torch.cuda.reset_peak_memory_stats()

                first = train_epoch(
                    model, optimizer, scaler, loader, cfg, 0, schedule
                )

                require(first["samples"] == 529, "Incorrect check sample count")
                require(
                    first["group_sizes"] == [512, 17],
                    "Incorrect full/tail accumulation groups",
                )
                require(
                    not torch.equal(
                        initial_head, model.head.weight.detach().cpu()
                    ),
                    "Classification head did not actually change",
                )

                trace = first["sample_label_order_sha256"]
                if initialization == "random":
                    paired_traces[dataset_name] = trace
                else:
                    require(
                        trace == paired_traces[dataset_name],
                        "Random/Procedural sample-label order differs",
                    )

                allocated = torch.cuda.max_memory_allocated() / (1024**3)
                reserved = torch.cuda.max_memory_reserved() / (1024**3)
                print(
                    f"[PASS] {dataset_name} / {initialization}: "
                    f"real update; groups=[512, 17]; "
                    f"peak_allocated={allocated:.3f} GiB; "
                    f"peak_reserved={reserved:.3f} GiB",
                    flush=True,
                )

                # One complete recovery comparison is enough to exercise
                # the shared checkpoint implementation.
                if dataset_name == "cifar100" and initialization == "random":
                    folder = check_root / "recovery"

                    first_payload = pack_checkpoint(
                        meta, model, optimizer, scaler, 1, [first]
                    )
                    save_checkpoint(folder, first_payload)
                    del first_payload

                    second = train_epoch(
                        model, optimizer, scaler, loader, cfg, 1, schedule
                    )
                    continuous = pack_checkpoint(
                        meta, model, optimizer, scaler, 2, [first, second]
                    )

                    # Release the old model and optimizer before recreating.
                    del model, optimizer, scaler, loader
                    gc.collect()
                    torch.cuda.empty_cache()

                    setup(cfg)
                    loader = make_loader(subset, cfg, training=True)
                    model, optimizer, scaler = make_components(
                        cfg, initialization, base.num_classes
                    )
                    saved = read_checkpoint(folder / "last.pt")
                    next_epoch, history = restore_checkpoint(
                        saved, meta, model, optimizer, scaler
                    )
                    require(next_epoch == 1, "Incorrect resume epoch")
                    del saved

                    resumed_second = train_epoch(
                        model, optimizer, scaler, loader,
                        cfg, next_epoch, schedule
                    )
                    history.append(resumed_second)
                    resumed = pack_checkpoint(
                        meta, model, optimizer, scaler, 2, history
                    )

                    # Includes weights, optimizer, scaler, random states,
                    # losses, sample order and progress.
                    assert_same(continuous, resumed)
                    print(
                        "[PASS] Continuous vs restored training: "
                        "all saved states exactly equal",
                        flush=True,
                    )

                    # Also test rotation of last.pt to previous.pt.
                    save_checkpoint(folder, resumed)
                    previous = read_checkpoint(folder / "previous.pt")
                    latest = read_checkpoint(folder / "last.pt")
                    require(previous["next_epoch"] == 1, "Invalid backup")
                    require(latest["next_epoch"] == 2, "Invalid latest checkpoint")
                    print("[PASS] Checkpoint replacement and previous backup")

                    del continuous, resumed, previous, latest

                del model, optimizer, scaler, loader
                gc.collect()
                torch.cuda.empty_cache()

            del subset, base

    print("\nTRAINING_FUNCTIONAL_CHECKS_COMPLETED")
    print("status=PASS")
    print("effective_batch_size=512")
    print("all_model_parameters_trainable=True")
    print("test_dataset_opened=False")
    print("formal_training_started=False")
    print("No speed benchmark was performed.")


def formal_training(cfg, dataset_name, initialization, noise, resume):
    setup(cfg)

    train_data = build_dataset(
        dataset_name, "train", noise_percent=noise
    )
    loader = make_loader(train_data, cfg, training=True)

    validation_loader = None
    if dataset_name == "cifar100":
        validation_data = build_dataset(dataset_name, "validation")
        validation_loader = make_loader(
            validation_data, cfg, training=False
        )

    directory = (
        ROOT / "runs" / "formal"
        / f"{dataset_name}_seed11_noise{noise:02d}_{initialization}"
    )
    if resume is None:
        require(
            not directory.exists() or not any(directory.iterdir()),
            "Run directory already contains files; explicit resume required",
        )
    else:
        require(
            (directory / f"{resume}.pt").is_file(),
            "Requested checkpoint does not exist",
        )

    model, optimizer, scaler = make_components(
        cfg, initialization, train_data.num_classes
    )
    schedule = make_schedule(cfg, len(train_data))
    meta = make_metadata(cfg, train_data, initialization, mode="formal")

    if resume is None:
        start_epoch = 0
        history = []
        # Protect the starting point before the first training epoch.
        save_checkpoint(
            directory,
            pack_checkpoint(meta, model, optimizer, scaler, 0, history),
        )
    else:
        payload = read_checkpoint(directory / f"{resume}.pt")
        start_epoch, history = restore_checkpoint(
            payload, meta, model, optimizer, scaler
        )
        del payload

    for epoch in range(start_epoch, cfg["epochs"]):
        print(
            f"\n{dataset_name} / {initialization} / noise={noise}% "
            f"/ epoch={epoch + 1}/{cfg['epochs']}",
            flush=True,
        )
        metrics = train_epoch(
            model, optimizer, scaler, loader, cfg, epoch, schedule
        )

        if validation_loader is not None:
            metrics["validation"] = evaluate_validation(
                model, validation_loader, cfg
            )

        history.append(metrics)
        save_checkpoint(
            directory,
            pack_checkpoint(
                meta, model, optimizer, scaler, epoch + 1, history
            ),
        )
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        print(f"Saved: {directory / 'last.pt'}", flush=True)

    print("\nFORMAL_TRAINING_COMPLETED")
    print(f"Completed epochs: {cfg['epochs']}")
    print(f"Final checkpoint: {directory / 'last.pt'}")
    print("Test dataset was not opened.")
    print("Final test evaluation is a separate, explicitly authorized step.")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--train", action="store_true")

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "training.json",
    )
    parser.add_argument("--dataset", choices=("cifar100", "oxford_pet"))
    parser.add_argument("--initialization", choices=("random", "procedural"))
    parser.add_argument("--noise", type=int, choices=(0, 10, 20, 30))
    parser.add_argument("--resume", choices=("last", "previous"))
    args = parser.parse_args()

    cfg = read_config(args.config)

    try:
        if args.check:
            require(
                args.dataset is None
                and args.initialization is None
                and args.noise is None
                and args.resume is None,
                "--check does not accept formal-training arguments",
            )
            check_training(cfg)
        else:
            require(
                args.dataset is not None
                and args.initialization is not None
                and args.noise is not None,
                "--train requires dataset, initialization and noise",
            )
            formal_training(
                cfg,
                args.dataset,
                args.initialization,
                args.noise,
                args.resume,
            )
    except torch.cuda.OutOfMemoryError:
        print(
            "\nOOM: stopped without saving an incomplete epoch.\n"
            "During formal training, the last completed checkpoint remains.\n"
            "Do not automatically change batch size or continue this process.\n"
            "Please return this output for review.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()