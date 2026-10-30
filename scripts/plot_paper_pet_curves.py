"""Export full 300-epoch Pet curves with the project's manuscript style."""

import gc
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import torch

from src.figure_style import (
    apply_paper_style, COLORS, FIGURE_SIZE_INCHES,
    FONT_FAMILY, FONT_SIZE_PT, LABELS, LINESTYLES,
)

REPORT = ROOT / "runs/evaluation/oxford_pet_epoch300/evaluation.json"
OUTPUT = ROOT / "paper/figures"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_histories():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    require(report["status"] == "PASS", "Original evaluation incomplete")
    histories, sources = {}, []
    for result in report["results"]:
        key = (result["noise_percent"], result["initialization"])
        require(key not in histories, "Duplicate experiment")
        path = Path(result["checkpoint_path"])
        require(sha256(path) == result["checkpoint_sha256"], f"Changed checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        history = checkpoint["history"]
        require(checkpoint["next_epoch"] == 300 and len(history) == 300, "Incomplete history")
        require(checkpoint["meta"]["config"]["seed"] == 11, "Wrong seed")
        for epoch, row in enumerate(history, 1):
            require(row["epoch"] == epoch and row["samples"] == 3680, "Incorrect epoch")
            require(math.isfinite(row["loss"]) and row["loss"] > 0, "Invalid loss")
            require(0 <= row["observed_label_accuracy"] <= 1, "Invalid accuracy")
        histories[key] = history
        sources.append({"noise_percent": key[0], "initialization": key[1],
                        "path": str(path), "sha256": result["checkpoint_sha256"]})
        del checkpoint
        gc.collect()
        require(sha256(path) == result["checkpoint_sha256"], "Checkpoint changed during read")
    require(set(histories) == {(n, i) for n in (0, 10, 20, 30) for i in COLORS}, "Expected eight histories")
    for noise in (0, 10, 20, 30):
        require([r["sample_label_order_sha256"] for r in histories[(noise, "random")]]
                == [r["sample_label_order_sha256"] for r in histories[(noise, "procedural")]],
                "Paired sample orders differ")
    return histories, sources


def check_figure_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text():
            continue
        require(item.get_fontfamily() == [FONT_FAMILY], f"Unexpected font: {item.get_text()}")
        require(abs(item.get_fontsize() - FONT_SIZE_PT) < 1e-8, "Unexpected font size")
        require(item.get_text().isascii(), f"Non-English/non-ASCII text: {item.get_text()}")
        box = item.get_window_extent(renderer)
        require(box.x0 >= -1 and box.y0 >= -1
                and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1,
                f"Text outside figure: {item.get_text()}")


def main():
    torch.set_num_threads(4)
    font_path = apply_paper_style()
    histories, sources = load_histories()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    entries = []
    captions = {
        "accuracy": (
            "Training label accuracy over all 300 epochs on Oxford-IIIT Pet with seed 11. "
            "Panels (a)-(d) correspond to label-noise rates of 0%, 10%, 20%, and 30%. "
            "Random and procedural initialization use paired images and observed labels. "
            "Accuracy is accumulated during training against the observed labels, including "
            "corrupted labels in the noisy conditions; it is not clean test accuracy. "
            "The original per-epoch records are plotted without smoothing."
        ),
        "loss": (
            "Training cross-entropy loss over all 300 epochs on Oxford-IIIT Pet with seed 11. "
            "Panels (a)-(d) correspond to label-noise rates of 0%, 10%, 20%, and 30%. "
            "The vertical axis uses a logarithmic scale. Loss is accumulated during each "
            "training epoch against the observed training labels. The original per-epoch "
            "records are plotted without smoothing. No validation or test curve is shown."
        ),
    }
    for metric in ("accuracy", "loss"):
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_INCHES)
        handles = [Line2D([0], [0], color=COLORS[i], linestyle=LINESTYLES[i],
                          label=LABELS[i]) for i in COLORS]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(.53, .995), handlelength=2.8, columnspacing=2.5)
        for index, (ax, noise) in enumerate(zip(axes.flat, (0, 10, 20, 30))):
            ax.set_title(f"({chr(97 + index)}) Label noise: {noise}%", loc="left", pad=5)
            for initialization in COLORS:
                history = histories[(noise, initialization)]
                values = [r["observed_label_accuracy"] * 100 if metric == "accuracy"
                          else r["loss"] for r in history]
                ax.plot(range(1, 301), values, color=COLORS[initialization],
                        linestyle=LINESTYLES[initialization])
            ax.set_xlim(1, 300)
            ax.set_xticks([1, 50, 100, 150, 200, 250, 300])
            ax.set_xlabel("Training epoch", labelpad=3)
            ax.grid(True, which="major", color="#D6D6D6", linewidth=.35)
            ax.set_axisbelow(True)
            if metric == "accuracy":
                ax.set_ylim(0, 103)
                ax.set_yticks([0, 20, 40, 60, 80, 100])
                ax.set_ylabel("Training label accuracy (%)", labelpad=4)
            else:
                ax.set_yscale("log")
                ax.set_ylim(.00015, 5)
                # Plain numeric labels avoid smaller mathematical superscripts.
                ax.yaxis.set_major_locator(FixedLocator([.001, .01, .1, 1]))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_ylabel("Training cross-entropy loss", labelpad=4)
        fig.subplots_adjust(left=.105, right=.98, bottom=.105, top=.89,
                            wspace=.34, hspace=.48)
        check_figure_text(fig)
        stem = f"pet_training_{metric}_epochs001_300"
        svg_path, png_path = OUTPUT / f"{stem}.svg", OUTPUT / f"{stem}.png"
        # Preserve the physical figure dimensions; do not use bbox_inches='tight'.
        fig.savefig(svg_path, metadata={"Title": captions[metric].split(". ")[0]})
        fig.savefig(png_path, dpi=600)
        plt.close(fig)
        entries.append({
            "id": stem, "metric": metric, "epoch_range": [1, 300],
            "svg": str(svg_path), "png": str(png_path),
            "svg_relative": svg_path.relative_to(ROOT).as_posix(),
            "png_relative": png_path.relative_to(ROOT).as_posix(),
            "svg_sha256": sha256(svg_path), "png_sha256": sha256(png_path),
            "caption_english": captions[metric],
            "logarithmic_vertical_axis": metric == "loss",
        })
        print(f"SAVED: {svg_path}\nSAVED: {png_path}")

    manifest_path = OUTPUT / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"schema_version": 1, "figures": []}
    new_ids = {entry["id"] for entry in entries}
    manifest["figures"] = [entry for entry in manifest["figures"] if entry["id"] not in new_ids] + entries
    manifest["style"] = {
        "language": "English", "font_family": FONT_FAMILY, "font_size_pt": FONT_SIZE_PT,
        "resolved_font_path": font_path, "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "output_formats": ["svg", "png"], "png_dpi": 600,
        "svg_text": "Outlined glyphs from Times New Roman at 8 pt; no viewer font substitution required.",
        "axis_labels": "Descriptive words, with any units in parentheses; never units alone or slash-separated quantity/unit labels.",
        "placement": "Insert at native width 7.16 inches (181.864 mm) to preserve 8 pt labels. Regenerate at the intended width rather than shrinking the entire figure.",
        "style_module": "src/figure_style.py",
    }
    manifest["pet_curve_sources"] = sources
    manifest["generator"] = {"path": "scripts/plot_paper_pet_curves.py", "sha256": sha256(Path(__file__)),
                             "style_sha256": sha256(ROOT / "src/figure_style.py")}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MANIFEST: {manifest_path}")
    print("PASS: eight unchanged checkpoints, 2400 epochs, English figure text, Times New Roman, 8 pt")


if __name__ == "__main__":
    main()
