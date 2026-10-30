"""Evaluate the eight fixed epoch-300 Pet models on the official test set."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import csv
import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from run_pet_batch import (
    JOBS,
    FORMAL_ROOT,
    check_pairing,
    inspect_subprocess,
    job_name,
    queue_lock,
)
from src.data_pipeline import build_dataset
from src.model_factory import build_model
from src.training_core import (
    make_loader,
    read_checkpoint,
    read_config,
    require,
    setup,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_fingerprint(meta):
    return hashlib.sha256(
        json.dumps(
            meta,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def write_csv(path, rows):
    require(len(rows) > 0, f"No rows to save: {path}")
    temporary = path.with_name(path.name + ".tmp")

    # UTF-8 with BOM makes Chinese/Excel handling easier on Windows.
    with temporary.open(
        "x", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())

    os.replace(temporary, path)


def write_json(path, content):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(
            content,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    os.replace(temporary, path)


@torch.inference_mode()
def predict(model, loader, dataset, noise, initialization):
    model.eval()

    rows = []
    seen = 0
    correct_count = 0

    for batch in loader:
        count = len(batch["label"])
        end = seen + count

        # Verify identity, order and clean labels for every actual batch.
        require(
            list(batch["sample_id"]) == dataset.sample_ids[seen:end],
            "Test sample order differs from the fixed manifest",
        )
        require(
            batch["source_index"].tolist()
            == dataset.indices[seen:end].tolist(),
            "Test source indices differ from the fixed manifest",
        )
        require(
            torch.equal(batch["label"], batch["clean_label"])
            and batch["label"].tolist()
            == dataset.clean_labels[seen:end].tolist()
            and not batch["is_corrupted"].any().item(),
            "Test labels must be the original clean labels",
        )

        images = batch["image"].cuda(non_blocking=True)

        # Deliberately use FP32 for every model's final evaluation.
        with torch.autocast("cuda", enabled=False):
            logits = model(images.float())

        require(
            logits.shape == (count, 37)
            and logits.dtype == torch.float32
            and torch.isfinite(logits).all().item(),
            "Invalid test logits",
        )

        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)

        predictions = predictions.cpu().tolist()
        confidence = confidence.cpu().tolist()
        true_labels = batch["label"].tolist()

        for sample_id, true_label, prediction, probability in zip(
            batch["sample_id"],
            true_labels,
            predictions,
            confidence,
        ):
            is_correct = int(prediction == true_label)
            correct_count += is_correct

            rows.append({
                "noise_percent": noise,
                "initialization": initialization,
                "sample_id": sample_id,
                "true_label": true_label,
                "predicted_label": prediction,
                "correct": is_correct,
                "predicted_probability": probability,
            })

        seen = end
        del images, logits, probabilities

    require(
        seen == len(dataset) == 3669
        and len(rows) == 3669
        and len({row["sample_id"] for row in rows}) == 3669,
        "Each of the 3669 test samples must appear exactly once",
    )
    require(
        correct_count == sum(row["correct"] for row in rows),
        "Prediction/accuracy accounting mismatch",
    )
    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "Unexpected gradients during evaluation",
    )

    return correct_count, rows


def evaluate(output):
    output = output.resolve()
    require(
        not output.exists(),
        f"Output directory already exists: {output}\n"
        "Existing evaluation results will not be overwritten.",
    )

    cfg = read_config(ROOT / "config" / "training.json")
    script_hash = sha256_file(__file__)

    # This lock also prevents the Pet training queue from starting.
    with queue_lock():
        print("Verifying all eight epoch-300 checkpoints ...", flush=True)

        states = {}
        checkpoint_hashes = {}

        # Do not open the test dataset until every model passes preflight.
        for noise, initialization in JOBS:
            key = (noise, initialization)
            name = job_name(noise, initialization)
            path = FORMAL_ROOT / name / "last.pt"

            digest_before = sha256_file(path)
            state = inspect_subprocess(noise, initialization)
            digest_after = sha256_file(path)

            require(
                digest_before == digest_after,
                f"Checkpoint changed during inspection: {name}",
            )
            require(
                state["status"] == "complete"
                and state["completed_epochs"] == 300,
                f"Expected a completed epoch-300 model: {name}",
            )

            states[key] = state
            checkpoint_hashes[key] = digest_after
            print(f"[PASS] {name}: verified epoch 300", flush=True)

        check_pairing(states)

        setup(cfg)

        test_data = build_dataset(
            "oxford_pet",
            "test",
            allow_test=True,
        )
        require(
            len(test_data) == 3669 and test_data.num_classes == 37,
            "Unexpected Pet test dataset",
        )
        require(
            not test_data.is_corrupted.any(),
            "Test dataset contains corrupted labels",
        )

        loader = make_loader(test_data, cfg, training=False)

        # Reserve a separate results directory; never write into model folders.
        output.mkdir(parents=True, exist_ok=False)

        summaries = []
        all_predictions = []
        training_sources = None
        runtime = None

        for number, (noise, initialization) in enumerate(JOBS, start=1):
            key = (noise, initialization)
            name = job_name(noise, initialization)
            checkpoint_path = FORMAL_ROOT / name / "last.pt"

            print(f"\n[{number}/8] Evaluating {name} ...", flush=True)

            require(
                sha256_file(checkpoint_path) == checkpoint_hashes[key],
                f"Checkpoint changed before evaluation: {name}",
            )

            payload = read_checkpoint(checkpoint_path)
            require(
                payload["next_epoch"] == 300
                and len(payload["history"]) == 300
                and payload["meta"]["config"] == cfg
                and metadata_fingerprint(payload["meta"])
                == states[key]["fingerprint"],
                f"Checkpoint identity/protocol mismatch: {name}",
            )
            require(
                payload["meta"]["manifests"]["split_sha256"]
                == test_data.manifest_hashes["split_sha256"],
                "Training and test partitions refer to different manifests",
            )

            trained_state = payload["model"]
            training_sources = payload["meta"]["source_sha256"]
            runtime = payload["meta"]["runtime"]
            noise_hash = payload["meta"]["manifests"]["noise_sha256"]
            del payload

            # Build only the architecture, then replace EVERY state tensor
            # with the trained epoch-300 checkpoint.
            model = build_model(
                initialization="random",
                num_classes=37,
                seed=11,
            )
            model.load_state_dict(trained_state, strict=True)

            require(
                all(
                    torch.equal(value, trained_state[name])
                    for name, value in model.state_dict().items()
                ),
                "Some trained model tensors were not loaded exactly",
            )

            model = model.cuda()
            correct, predictions = predict(
                model, loader, test_data, noise, initialization
            )

            # Prove evaluation did not change model tensors.
            model = model.cpu()
            require(
                all(
                    torch.equal(value, trained_state[name])
                    for name, value in model.state_dict().items()
                ),
                "Model parameters/buffers changed during evaluation",
            )
            require(
                sha256_file(checkpoint_path) == checkpoint_hashes[key],
                "Checkpoint file changed during evaluation",
            )

            accuracy = 100.0 * correct / len(test_data)
            summaries.append({
                "dataset": "oxford_pet",
                "seed": 11,
                "noise_percent": noise,
                "initialization": initialization,
                "epoch": 300,
                "test_samples": len(test_data),
                "correct": correct,
                "accuracy_percent": accuracy,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hashes[key],
                "noise_manifest_sha256": noise_hash,
            })
            all_predictions.extend(predictions)

            print(
                f"[RESULT] noise={noise}% / {initialization}: "
                f"{correct}/3669, accuracy={accuracy:.4f}%",
                flush=True,
            )

            del model, trained_state, predictions
            gc.collect()
            torch.cuda.empty_cache()

        require(
            len(summaries) == 8
            and len(all_predictions) == 8 * 3669,
            "Incomplete evaluation results",
        )

        lookup = {
            (row["noise_percent"], row["initialization"]): row
            for row in summaries
        }
        comparisons = []

        for noise in (0, 10, 20, 30):
            random_accuracy = lookup[(noise, "random")]["accuracy_percent"]
            procedural_accuracy = lookup[
                (noise, "procedural")
            ]["accuracy_percent"]

            comparisons.append({
                "noise_percent": noise,
                "random_accuracy_percent": random_accuracy,
                "procedural_accuracy_percent": procedural_accuracy,
                "procedural_minus_random_percentage_points": (
                    procedural_accuracy - random_accuracy
                ),
            })

        # Final check before declaring success.
        for noise, initialization in JOBS:
            path = FORMAL_ROOT / job_name(noise, initialization) / "last.pt"
            require(
                sha256_file(path)
                == checkpoint_hashes[(noise, initialization)],
                f"Checkpoint changed: {path}",
            )
        require(
            sha256_file(__file__) == script_hash,
            "Evaluation script changed while running",
        )

        write_csv(output / "summary.csv", summaries)
        write_csv(output / "comparison.csv", comparisons)
        write_csv(output / "predictions.csv", all_predictions)

        report = {
            "status": "PASS",
            "completed_at": datetime.datetime.now().astimezone().isoformat(),
            "dataset": "oxford_pet",
            "split": "official_test",
            "test_samples_per_model": 3669,
            "num_classes": 37,
            "label_encoding": "zero-based integers 0 through 36",
            "seed": 11,
            "checkpoint_epoch": 300,
            "model_count": 8,
            "precision": "FP32; autocast disabled; TF32 disabled",
            "batch_size": cfg["eval_batch_size"],
            "preprocessing": (
                "Existing IMAGE_TRANSFORM: RGB, bilinear resize to 224x224, "
                "ToTensor, ImageNet normalization; no random augmentation"
            ),
            "model_selection": "Fixed epoch 300 for all eight models",
            "metric": "Top-1 accuracy = correct / 3669",
            "comparison_unit": "percentage points; Procedural minus Random",
            "model_tensors_unchanged": True,
            "checkpoint_files_unchanged": True,
            "test_split_manifest_sha256": (
                test_data.manifest_hashes["split_sha256"]
            ),
            "evaluation_script_sha256": script_hash,
            "training_source_sha256": training_sources,
            "runtime": runtime,
            "results": summaries,
            "comparisons": comparisons,
            "output_sha256": {
                filename: sha256_file(output / filename)
                for filename in (
                    "summary.csv",
                    "comparison.csv",
                    "predictions.csv",
                )
            },
        }

        # Written last: this is the completion marker.
        write_json(output / "evaluation.json", report)

        print("\nNoise    Random       Procedural    Difference (pp)")
        for row in comparisons:
            print(
                f"{row['noise_percent']:>3}%     "
                f"{row['random_accuracy_percent']:>9.4f}%   "
                f"{row['procedural_accuracy_percent']:>9.4f}%   "
                f"{row['procedural_minus_random_percentage_points']:>+10.4f}"
            )

        print("\nPET_TEST_EVALUATION_COMPLETED")
        print("status=PASS")
        print("evaluated_models=8")
        print("test_samples_per_model=3669")
        print("prediction_rows=29352")
        print("model_tensors_unchanged=True")
        print("checkpoint_files_unchanged=True")
        print(f"Results: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "evaluation" / "oxford_pet_epoch300",
    )
    args = parser.parse_args()
    evaluate(args.output)


if __name__ == "__main__":
    main()