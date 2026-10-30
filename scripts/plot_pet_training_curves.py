"""Read existing checkpoint histories and plot them; never train or evaluate."""

import gc
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs/evaluation/oxford_pet_epoch300"
OUTPUT = RESULTS / "curves"
COLORS = {"random": "#2463A6", "procedural": "#CF651E"}
LABELS = {"random": "Random 随机初始化", "procedural": "Procedural 预热初始化"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    report = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))
    require(report["status"] == "PASS", "Evaluation is not complete")
    records = {}
    statistics = []
    for result in report["results"]:
        key = (result["noise_percent"], result["initialization"])
        require(key not in records, "Duplicate model in evaluation")
        path = Path(result["checkpoint_path"])
        require(sha256(path) == result["checkpoint_sha256"], f"Checkpoint changed: {path}")
        # These are the locally generated, trusted training checkpoints.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        history = checkpoint["history"]
        require(checkpoint["next_epoch"] == 300 and len(history) == 300, "Incomplete history")
        require(checkpoint["meta"]["config"]["seed"] == 11, "Unexpected seed")
        for epoch, row in enumerate(history, 1):
            require(row["epoch"] == epoch and row["samples"] == 3680, "Invalid epoch record")
            require(math.isfinite(row["loss"]) and row["loss"] > 0, "Invalid loss")
            require(0 <= row["observed_label_accuracy"] <= 1, "Invalid accuracy")
            require("validation" not in row, "Unexpected Pet validation record")
        records[key] = {"history": history, "test_accuracy": result["accuracy_percent"]}
        stats = {
            "noise_percent": key[0], "initialization": key[1],
            "final_train_accuracy_percent": 100 * history[-1]["observed_label_accuracy"],
            "final_train_loss": history[-1]["loss"],
            "final_test_accuracy_percent": result["accuracy_percent"],
            "first_epoch_train_ge_95_percent": next((r["epoch"] for r in history if r["observed_label_accuracy"] >= .95), None),
            "first_epoch_train_ge_99_percent": next((r["epoch"] for r in history if r["observed_label_accuracy"] >= .99), None),
            "total_amp_skipped_updates": sum(r["amp_skipped_updates"] for r in history),
            "train_accuracy_at_50_100_150_200": [100 * history[e-1]["observed_label_accuracy"] for e in (50, 100, 150, 200)],
        }
        statistics.append(stats)
        del checkpoint
        gc.collect()
        require(sha256(path) == result["checkpoint_sha256"], "Checkpoint changed while reading")

    expected = {(n, i) for n in (0, 10, 20, 30) for i in COLORS}
    require(set(records) == expected, "Expected eight model histories")
    for noise in (0, 10, 20, 30):
        require(
            [r["sample_label_order_sha256"] for r in records[(noise, "random")]["history"]]
            == [r["sample_label_order_sha256"] for r in records[(noise, "procedural")]["history"]],
            "Paired training orders differ",
        )

    plt.rcParams.update({
        "font.family": "Microsoft YaHei", "axes.unicode_minus": False,
        "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 11,
        "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FAFBFD",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, 301))
    for metric in ("accuracy", "loss"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        title = "Oxford-IIIT Pet｜完整训练准确率曲线" if metric == "accuracy" else "Oxford-IIIT Pet｜完整训练损失曲线"
        fig.suptitle(title, fontsize=21, y=.975, fontweight="bold")
        fig.text(.5, .928, "seed = 11 · 300 epochs · 每个面板对比同一噪声率下的两种初始化 · 原始逐轮记录，未平滑", ha="center", color="#536071", fontsize=11)
        handles = [Line2D([0], [0], color=COLORS[i], lw=2.4, label=LABELS[i]) for i in COLORS]
        if metric == "accuracy":
            handles.append(Line2D([0], [0], color="#4D5663", marker="D", ls="None", markersize=6, label="仅第 300 轮的干净测试结果"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .908), ncol=len(handles), frameon=False)

        for ax, noise in zip(axes.flat, (0, 10, 20, 30)):
            ax.set_title(f"训练标签噪声：{noise}%", loc="left", pad=10)
            ax.grid(True, color="#DDE3EB", lw=.65, alpha=.8)
            ax.set_axisbelow(True)
            for initialization in COLORS:
                record = records[(noise, initialization)]
                values = [100 * r["observed_label_accuracy"] if metric == "accuracy" else r["loss"] for r in record["history"]]
                ax.plot(epochs, values, color=COLORS[initialization], lw=1.9)
                if metric == "accuracy":
                    test = record["test_accuracy"]
                    ax.scatter([300], [test], color=COLORS[initialization], marker="D", s=42, zorder=5)
                    offset = 7 if initialization == "random" else -5
                    ax.annotate(f"测试 {test:.2f}%", xy=(300, test), xytext=(215, test + offset),
                                color=COLORS[initialization], fontsize=10,
                                arrowprops={"arrowstyle": "-", "color": COLORS[initialization], "lw": .8})
            ax.set_xlim(1, 310)
            ax.set_xticks([1, 50, 100, 150, 200, 250, 300])
            ax.set_xlabel("训练轮次（epoch）")
            if metric == "accuracy":
                ax.set_ylim(-3, 105)
                ax.set_yticks([0, 20, 40, 60, 80, 100])
                ax.set_ylabel("准确率（%）")
            else:
                ax.set_yscale("log")
                ax.set_ylabel("训练交叉熵损失（对数刻度）")

        footer = (
            "实线：训练过程中对所用训练标签的准确率（含人为错误标签）；菱形：仅最终测试结果，并非逐轮测试曲线。\n没有独立验证曲线，无法由此确定最佳泛化轮次或过拟合开始轮次。"
            if metric == "accuracy" else
            "损失按实际训练标签计算（噪声组含人为错误标签）。纵轴使用对数刻度，以显示后期很小的损失。\n记录来自每轮训练过程中的累计值，不是该轮结束后重新评估整个训练集的结果。"
        )
        fig.text(.065, .028, footer, fontsize=10, color="#536071", linespacing=1.6)
        fig.subplots_adjust(top=.82, bottom=.13, left=.075, right=.97, hspace=.36, wspace=.20)
        path = OUTPUT / f"pet_training_{metric}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        print(f"SAVED: {path}")

    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print("CURVE_CHECKS_PASS: 8 histories, 2400 epoch records, unchanged checkpoints; no training or test evaluation")


if __name__ == "__main__":
    # Route future command-line exports through the manuscript-wide style.
    from plot_paper_pet_curves import main as paper_main
    paper_main()
