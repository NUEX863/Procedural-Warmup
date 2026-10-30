"""Plot saved epoch-300 Pet test accuracy; no model inference."""

import argparse
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_pet import sha256_file, write_json
from src.figure_style import (
    apply_paper_style, COLORS, LABELS, LINESTYLES,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "runs/evaluation/oxford_pet_epoch300",
    )
    args = parser.parse_args()
    directory = args.results.resolve()

    report_path = directory / "evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(
        report["status"] == "PASS"
        and report["checkpoint_epoch"] == 300
        and report["model_count"] == 8
        and report["test_samples_per_model"] == 3669,
        "Expected a completed eight-model epoch-300 evaluation",
    )

    for filename, digest in report["output_sha256"].items():
        require(
            sha256_file(directory / filename) == digest,
            f"Saved evaluation file changed: {filename}",
        )

    noises = (0, 10, 20, 30)
    initializations = ("random", "procedural")
    rows = report["results"]
    lookup = {
        (row["noise_percent"], row["initialization"]): row
        for row in rows
    }
    require(
        len(rows) == 8
        and set(lookup) == {
            (noise, initialization)
            for noise in noises
            for initialization in initializations
        },
        "Missing or repeated experimental conditions",
    )
    for row in rows:
        require(
            row["epoch"] == 300
            and row["seed"] == 11
            and row["test_samples"] == 3669
            and 0 <= row["correct"] <= 3669
            and abs(
                row["accuracy_percent"] - 100 * row["correct"] / 3669
            ) < 1e-10,
            "Inconsistent test accuracy record",
        )

    font_path = apply_paper_style()
    figure_root = ROOT / "paper" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    registry_path = figure_root / "figure_manifest.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else {"schema_version": 1, "figures": []}
    )
    require(isinstance(registry.get("figures"), list), "Invalid figure registry")

    # Match the new Pet90 figure's size, colors, lines and percentage axis.
    fig, ax = plt.subplots(figsize=(7.16, 3.6))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.78)

    for initialization in initializations:
        values = [
            lookup[(noise, initialization)]["accuracy_percent"]
            for noise in noises
        ]
        ax.plot(
            noises, values,
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
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=False,
    )
    fig.suptitle("Oxford-IIIT Pet: epoch 300, seed 11", y=0.97)

    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    figure_id = f"pet_test_epoch300_accuracy_{tag}"
    entry = {
        "id": figure_id,
        "checkpoint_epoch": 300,
        "evaluation_report": str(report_path),
        "evaluation_report_sha256": sha256_file(report_path),
        "generator": "scripts/plot_pet_accuracy_300.py",
        "generator_sha256": sha256_file(__file__),
        "style_module": "src/figure_style.py",
        "style_sha256": sha256_file(ROOT / "src/figure_style.py"),
        "font_family": "Times New Roman",
        "font_size_pt": 8,
        "font_file": str(font_path),
        "language": "English",
        "png_dpi": 600,
        "figure_size_inches": [7.16, 3.6],
        "caption_english": (
            "Top-one accuracy on the 3669 official Oxford-IIIT Pet test "
            "images at training label-noise rates of 0%, 10%, 20%, and 30%. "
            "All eight models use training seed 11 and the fixed epoch-300 "
            "checkpoint. Training uses all 3680 official trainval images, "
            "without an independent validation partition. Test labels are "
            "clean. Evaluation uses RGB images resized directly to 224 by "
            "224 pixels and ImageNet normalization, following the original "
            "experiment protocol. Lines connect measured conditions as "
            "visual guides. This single-seed comparison does not quantify "
            "variation across training seeds."
        ),
    }

    for extension in ("svg", "png"):
        path = figure_root / f"{figure_id}.{extension}"
        require(not path.exists(), f"Figure already exists: {path}")
        fig.savefig(path, format=extension, dpi=600)
        entry[extension] = str(path)
        entry[f"{extension}_relative"] = path.relative_to(ROOT).as_posix()
        entry[f"{extension}_sha256"] = sha256_file(path)
    plt.close(fig)

    registry["figures"].append(entry)
    write_json(registry_path, registry)
    write_json(
        directory / f"{figure_id}.json",
        {"status": "PASS", "figures": [entry]},
    )

    print("Noise    Random       Procedural    Difference (percentage points)")
    for noise in noises:
        random_accuracy = lookup[(noise, "random")]["accuracy_percent"]
        procedural_accuracy = lookup[(noise, "procedural")]["accuracy_percent"]
        print(
            f"{noise:>3}%    {random_accuracy:>9.4f}%"
            f"    {procedural_accuracy:>9.4f}%"
            f"    {procedural_accuracy - random_accuracy:>+10.4f}"
        )

    print("\nPET_300_ACCURACY_FIGURE_COMPLETED")
    print("status=PASS")
    print("Test inference was not rerun.")
    print(f"SVG: {entry['svg']}")
    print(f"PNG: {entry['png']}")


if __name__ == "__main__":
    main()