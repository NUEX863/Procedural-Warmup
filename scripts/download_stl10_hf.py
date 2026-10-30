from pathlib import Path
import os
import tarfile

# Set timeouts before importing huggingface_hub.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "stl10_hf"

REPO_ID = "Shu1L0n9/CleanSTL-10"
REVISION = "a7b816d9fe66372a8aff6428d14b12dcbeefb736"


def main():
    print(f"Dataset: {REPO_ID}", flush=True)
    print(f"Revision: {REVISION}", flush=True)
    print(f"Destination: {DESTINATION}", flush=True)

    api = HfApi()
    files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
    )

    # Select only labeled training and test archives.
    selected = sorted(
        name for name in files
        if name.endswith(".tar")
        and (name.startswith("train-") or name.startswith("test-"))
    )

    if not any(name.startswith("train-") for name in selected):
        raise RuntimeError("No labeled training archives found.")
    if not any(name.startswith("test-") for name in selected):
        raise RuntimeError("No test archives found.")

    print("\nFiles selected:", flush=True)
    for name in selected:
        print(f"  {name}", flush=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=selected,
        local_dir=str(DESTINATION),
        max_workers=2,
    )

    # Inspect archive contents without extracting or decoding images.
    totals = {
        "train": {"images": 0, "labels": 0},
        "test": {"images": 0, "labels": 0},
    }

    for name in selected:
        split = "train" if name.startswith("train-") else "test"
        path = DESTINATION / name

        with tarfile.open(path, mode="r|*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if member.name.endswith(".png"):
                    totals[split]["images"] += 1
                elif member.name.endswith(".cls"):
                    totals[split]["labels"] += 1

    print("\nArchive inventory:")
    for split, counts in totals.items():
        print(
            f"{split}: images={counts['images']}, "
            f"label_files={counts['labels']}"
        )

    print("\nHF_DOWNLOAD_COMPLETED")
    print(f"Saved to: {DESTINATION}")
    print("unlabeled_archives_downloaded=False")
    print("official_data_equivalence=NOT_CHECKED")
    print("formal_training_started=False")


if __name__ == "__main__":
    main()