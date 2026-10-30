"""Evaluate eight validation-selected Pet90 models on the official test set."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import contextlib
import csv
import datetime
import gc
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import OxfordIIITPet

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

import run_pet90_noise_batch as training
from run_pet_batch import queue_lock
from src.model_factory import build_model
from src.pet90_data import Pet90Dataset, read_manifest
from src.pet90_noise_data import noise_path
from src.figure_style import (
    apply_paper_style,
    COLORS,
    LABELS,
    LINESTYLES,
)

base = training.base
require = base.require

EVALUATION_BATCH_SIZE = 16
FIGURE_ROOT = ROOT / "paper" / "figures"


def write_json(path, value):
    content = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    base.atomic_write(path, lambda stream: stream.write(content))


def write_csv(path, rows):
    require(bool(rows), f"No rows to save: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8-sig")
    base.atomic_write(path, lambda stream: stream.write(content))


class CleanTestDataset(Dataset):
    def __init__(self, cfg):
        partition = read_manifest(cfg)["partitions"]["test"]
        self.indices = list(partition["indices"])
        self.sample_ids = list(partition["sample_ids"])
        self.labels = list(partition["labels_clean"])

        # Reuse the exact deterministic validation preprocessing.
        # Constructing this object does not open its validation images.
        validation = Pet90Dataset("validation", cfg)
        self.transform = validation.transform

        official = OxfordIIITPet(
            root=str(ROOT / "data"),
            split="test",
            target_types="category",
            download=False,
        )
        self.classes = list(official.classes)
        require(
            self.classes == validation.classes,
            "Training/validation/test class ordering differs",
        )
        require(
            len(official) == len(self.indices) == 3669
            and set(self.indices) == set(range(3669))
            and len(set(self.sample_ids)) == 3669,
            "Test split does not cover all official test images exactly once",
        )

        pet_root = ROOT / "data" / "oxford-iiit-pet"
        self.paths = []
        for index, sample_id, label in zip(
            self.indices, self.sample_ids, self.labels
        ):
            path = Path(official._images[index])
            require(
                path.relative_to(pet_root).as_posix() == sample_id,
                "Test image identity differs from the fixed manifest",
            )
            require(
                int(official._labels[index]) == label,
                "Test label differs from the official clean label",
            )
            breed = (
                Path(sample_id).stem.rsplit("_", 1)[0]
                .replace("_", " ").lower()
            )
            require(
                breed == self.classes[label].lower(),
                "Breed name and class index disagree",
            )
            self.paths.append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, position):
        with Image.open(self.paths[position]) as opened:
            image = self.transform(opened.convert("RGB"))
        return {
            "image": image,
            "label": self.labels[position],
            "sample_id": self.sample_ids[position],
            "source_index": self.indices[position],
        }


@torch.inference_mode()
def predict(model, loader, dataset):
    model.eval()
    collected = []
    seen = 0

    for batch in loader:
        end = seen + len(batch["label"])
        require(
            list(batch["sample_id"]) == dataset.sample_ids[seen:end]
            and batch["source_index"].tolist() == dataset.indices[seen:end]
            and batch["label"].tolist() == dataset.labels[seen:end],
            "Actual test batch differs from the fixed clean test data",
        )

        with torch.autocast("cuda", enabled=False):
            logits = model(batch["image"].cuda(non_blocking=True).float())

        require(
            logits.shape == (end - seen, 37)
            and logits.dtype == torch.float32
            and torch.isfinite(logits).all().item(),
            "Invalid model output",
        )
        collected.append(logits.cpu().numpy())
        seen = end

        if seen % 1024 == 0 or seen == len(dataset):
            print(f"  Evaluated {seen}/{len(dataset)}", flush=True)

    require(seen == 3669, "Incomplete test evaluation")
    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "Evaluation unexpectedly created gradients",
    )
    return np.concatenate(collected, axis=0)


def draw_accuracy(summaries, output, report_path, tag, registry, font_path):
    lookup = {
        (row["noise_percent"], row["initialization"]): row
        for row in summaries
    }
    noises = (0, 10, 20, 30)
    figure_id = f"pet90_best_test_accuracy_{tag}"

    fig, ax = plt.subplots(figsize=(7.16, 3.6))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.78)

    for initialization in ("random", "procedural"):
        values = [
            lookup[(noise, initialization)]["test_accuracy_percent"]
            for noise in noises
        ]
        ax.plot(
            noises,
            values,
            color=COLORS[initialization],
            linestyle=LINESTYLES[initialization],
            marker="o" if initialization == "random" else "s",
            markersize=4,
            label=LABELS[initialization],
        )

    ax.set(
        xlabel="Training label noise (%)",
        ylabel="Test accuracy (%)",
        xlim=(-2, 32),
        ylim=(0, 100),
    )
    ax.set_xticks(noises)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="#dddddd", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Oxford-IIIT Pet: validation-selected models, seed 11",
        y=0.97,
    )

    fig.canvas.draw()
    for item in fig.findobj(match=Text):
        if item.get_text():
            require(
                abs(item.get_fontsize() - 8) < 1e-8
                and "Times New Roman" in item.get_fontfamily(),
                "Figure text violates the required font style",
            )

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for extension in ("svg", "png"):
        path = FIGURE_ROOT / f"{figure_id}.{extension}"
        require(not path.exists(), f"Figure already exists: {path}")
        base.atomic_write(
            path,
            lambda stream, fmt=extension: fig.savefig(
                stream, format=fmt, dpi=600
            ),
        )
        paths[extension] = path

    plt.close(fig)
    ET.parse(paths["svg"])
    with Image.open(paths["png"]) as image:
        image.verify()

    actual_rates = [
        lookup[(noise, "random")]["actual_noise_percent"]
        for noise in noises
    ]
    caption = (
        "Top-one accuracy on the 3669 official Oxford-IIIT Pet test images "
        "for random and procedural initialization, with training seed 11. "
        "Each model is selected by its highest clean validation accuracy "
        "among 150 epochs, retaining the earliest epoch on ties. "
        "Training and validation use 3312 and 368 images, respectively. "
        "Test labels remain clean. Nominal training label-noise rates are "
        "0%, 10%, 20%, and 30%; actual rates are "
        + ", ".join(f"{rate:.6f}%" for rate in actual_rates)
        + ". Lines connect measured conditions as visual guides. "
        "This single-seed exploratory comparison follows earlier test-set "
        "inspection; it does not measure variation across training seeds "
        "or directly measure memorization of incorrect training labels."
    )

    entry = {
        "id": figure_id,
        "caption_english": caption,
        "evaluation_report": str(report_path),
        "evaluation_report_sha256": base.file_sha256(report_path),
        "generator": "scripts/evaluate_pet90_best.py",
        "generator_sha256": base.file_sha256(__file__),
        "style_module": "src/figure_style.py",
        "style_sha256": base.file_sha256(ROOT / "src" / "figure_style.py"),
        "font_family": "Times New Roman",
        "font_size_pt": 8,
        "font_file": str(font_path),
        "language": "English",
        "png_dpi": 600,
        "figure_size_inches": [7.16, 3.6],
    }
    for extension, path in paths.items():
        entry[extension] = str(path)
        entry[f"{extension}_relative"] = path.relative_to(ROOT).as_posix()
        entry[f"{extension}_sha256"] = base.file_sha256(path)

    registry["figures"].append(entry)
    write_json(FIGURE_ROOT / "figure_manifest.json", registry)
    write_json(output / "figures.json", {"status": "PASS", "figures": [entry]})
    return paths


def run(output, tag, registry, font_path):
    cfg = training.config_for(0)
    base.setup(cfg)

    # Snapshot inputs before verification and inference.
    protected = {
        Path(__file__).resolve(),
        ROOT / "src" / "figure_style.py",
        ROOT / cfg["split_manifest"],
        (ROOT / cfg["split_manifest"]).with_name(
            Path(cfg["split_manifest"]).name + ".sha256"
        ),
    }
    for noise, initialization in training.JOBS:
        directory = training.RUN_ROOT / training.job_name(noise, initialization)
        protected.update({
            directory / "best_model.pth",
            directory / "latest_checkpoint.pth",
        })
        path = noise_path(noise)
        protected.update({path, path.with_name(path.name + ".sha256")})

        meta = training.metadata(
            training.config_for(noise), initialization, "formal", 3312, 368
        )
        protected.update(ROOT / name for name in meta["source_sha256"])

    input_hashes = {path: base.file_sha256(path) for path in protected}

    print("Verifying all eight validation-selected best models ...", flush=True)
    states = {}
    for key in training.JOBS:
        state = training.inspect_job(*key)
        require(
            state["status"] == "complete" and state["completed"] == 150,
            f"Experiment is not verified complete: {state['name']}",
        )
        states[key] = state
        print(f"[PASS] {state['name']}", flush=True)
    training.check_pairing(states)

    dataset = CleanTestDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(11),
    )
    labels = np.asarray(dataset.labels, dtype=np.int64)
    summaries = []
    prediction_rows = []

    for number, (noise, initialization) in enumerate(training.JOBS, 1):
        name = training.job_name(noise, initialization)
        path = training.RUN_ROOT / name / "best_model.pth"
        require(
            base.file_sha256(path) == input_hashes[path],
            "Best checkpoint changed before evaluation",
        )

        best = base.read_checkpoint(path)
        expected_meta = training.metadata(
            training.config_for(noise), initialization, "formal", 3312, 368
        )
        require(best["meta"] == expected_meta, "Best model identity mismatch")
        require(
            base.model_digest(best["model"]) == best["model_sha256"],
            "Best model tensor fingerprint mismatch",
        )

        model = build_model(
            initialization=initialization, num_classes=37, seed=11
        )
        model.load_state_dict(best["model"], strict=True)
        require(
            base.model_digest(model.state_dict()) == best["model_sha256"],
            "Model weights were not loaded exactly",
        )

        print(
            f"\n[{number}/8] {name}: selected epoch={best['epoch']}",
            flush=True,
        )
        logits = predict(model.cuda(), loader, dataset)
        model.cpu()
        require(
            base.model_digest(model.state_dict()) == best["model_sha256"],
            "Evaluation changed model tensors",
        )

        predicted = logits.argmax(axis=1)
        correct = predicted == labels
        correct_count = int(correct.sum())
        accuracy = 100.0 * correct_count / len(labels)

        summary = {
            "noise_percent": noise,
            "actual_noise_percent": best["meta"]["noise"]["actual_percent"],
            "initialization": initialization,
            "seed": 11,
            "best_epoch": best["epoch"],
            "best_validation_accuracy_percent": 100 * best["validation_accuracy"],
            "test_samples": len(labels),
            "test_correct": correct_count,
            "test_accuracy_percent": accuracy,
            "checkpoint_path": str(path),
            "checkpoint_sha256": input_hashes[path],
            "model_sha256": best["model_sha256"],
        }
        summaries.append(summary)

        # Preserve raw outputs so later metrics do not require another GPU pass.
        np.savez_compressed(
            output / f"noise{noise:02d}_{initialization}_outputs.npz",
            logits=logits,
            true_labels=labels,
            predicted_labels=predicted,
            sample_ids=np.asarray(dataset.sample_ids),
            class_names=np.asarray(dataset.classes),
        )

        for position, sample_id in enumerate(dataset.sample_ids):
            prediction_rows.append({
                "noise_percent": noise,
                "initialization": initialization,
                "best_epoch": best["epoch"],
                "sample_id": sample_id,
                "source_index": dataset.indices[position],
                "true_label": int(labels[position]),
                "true_class": dataset.classes[int(labels[position])],
                "predicted_label": int(predicted[position]),
                "predicted_class": dataset.classes[int(predicted[position])],
                "correct": int(correct[position]),
            })

        # Save accumulated results after each model.
        write_csv(output / "summary.csv", summaries)
        write_csv(output / "predictions.csv", prediction_rows)

        print(
            f"[RESULT] noise={noise}% / {initialization}: "
            f"{correct_count}/3669, accuracy={accuracy:.4f}%",
            flush=True,
        )

        del model, best, logits
        gc.collect()
        torch.cuda.empty_cache()

    lookup = {
        (row["noise_percent"], row["initialization"]): row
        for row in summaries
    }
    comparisons = []
    for noise in (0, 10, 20, 30):
        random = lookup[(noise, "random")]
        procedural = lookup[(noise, "procedural")]
        comparisons.append({
            "noise_percent": noise,
            "actual_noise_percent": random["actual_noise_percent"],
            "random_best_epoch": random["best_epoch"],
            "procedural_best_epoch": procedural["best_epoch"],
            "random_correct": random["test_correct"],
            "procedural_correct": procedural["test_correct"],
            "test_samples": 3669,
            "random_accuracy_percent": random["test_accuracy_percent"],
            "procedural_accuracy_percent": procedural["test_accuracy_percent"],
            "procedural_minus_random_percentage_points": (
                procedural["test_accuracy_percent"]
                - random["test_accuracy_percent"]
            ),
        })

    require(
        len(summaries) == 8 and len(prediction_rows) == 8 * 3669,
        "Incomplete evaluation results",
    )
    for path, digest in input_hashes.items():
        require(
            base.file_sha256(path) == digest,
            f"Input file changed during evaluation: {path}",
        )

    write_csv(output / "comparison.csv", comparisons)
    report_path = output / "evaluation.json"
    report = {
        "status": "PASS",
        "completed_at": datetime.datetime.now().astimezone().isoformat(),
        "seed": 11,
        "evaluated_models": 8,
        "test_samples_per_model": 3669,
        "prediction_rows": len(prediction_rows),
        "selection": "Highest clean validation accuracy; earliest epoch on ties",
        "test_used_for_epoch_selection": False,
        "precision": "FP32 inference; autocast and TF32 disabled",
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "preprocessing": (
            "Exact validation transform: RGB; bilinear shorter-side resize "
            "256; center crop 224; tensor; ImageNet normalization"
        ),
        "analysis_scope": (
            "Single training seed; exploratory follow-up after earlier "
            "test-set inspection"
        ),
        "model_tensors_unchanged": True,
        "checkpoint_files_unchanged": True,
        "class_names": dataset.classes,
        "input_sha256": {str(path): digest for path, digest in input_hashes.items()},
        "results": summaries,
        "comparisons": comparisons,
        "output_sha256": {
            path.name: base.file_sha256(path)
            for path in sorted(output.iterdir())
            if path.suffix in {".csv", ".npz"}
        },
    }
    write_json(report_path, report)

    paths = draw_accuracy(
        summaries, output, report_path, tag, registry, font_path
    )

    print("\nNoise    Random       Procedural    Difference (percentage points)")
    for row in comparisons:
        print(
            f"{row['noise_percent']:>3}%"
            f"    {row['random_accuracy_percent']:>9.4f}%"
            f"    {row['procedural_accuracy_percent']:>9.4f}%"
            f"    {row['procedural_minus_random_percentage_points']:>+10.4f}"
        )

    print("\nPET90_BEST_TEST_EVALUATION_COMPLETED")
    print("status=PASS")
    print("evaluated_models=8")
    print("test_samples_per_model=3669")
    print("prediction_rows=29352")
    print("model_tensors_unchanged=True")
    print("checkpoint_files_unchanged=True")
    print("figure_formats=SVG,PNG")
    print(f"Results: {output}")
    print(f"SVG: {paths['svg']}")
    print(f"PNG: {paths['png']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()

    with queue_lock():
        font_path = apply_paper_style()
        registry_path = FIGURE_ROOT / "figure_manifest.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        else:
            registry = {"schema_version": 1, "figures": []}
        require(
            isinstance(registry.get("figures"), list),
            "Invalid figure registry",
        )

        tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = (
            ROOT / "runs" / "evaluation" / f"pet90_best_test_{tag}"
        )
        output.mkdir(parents=True, exist_ok=False)

        with (output / "console.log").open(
            "w", encoding="utf-8", buffering=1
        ) as log:
            tee = training.Tee(sys.stdout, log)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                try:
                    run(output, tag, registry, font_path)
                except BaseException:
                    import traceback
                    traceback.print_exc()
                    raise


if __name__ == "__main__":
    main()