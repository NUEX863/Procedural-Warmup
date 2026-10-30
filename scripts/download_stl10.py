"""Download the specified Kaggle STL-10 dataset and inspect its layout."""

import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Keep the Kaggle download cache inside this project's data directory.
CACHE_ROOT = ROOT / "data" / "kaggle_cache"
os.environ["KAGGLEHUB_CACHE"] = str(CACHE_ROOT)

import kagglehub


HANDLE = "jessicali9530/stl10/versions/3"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
METADATA_SUFFIXES = {".csv", ".txt", ".json", ".bin", ".mat", ".npy", ".npz"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Kaggle dataset: {HANDLE}", flush=True)
    print(f"Cache directory: {CACHE_ROOT}", flush=True)
    print("Downloading through Kaggle ...", flush=True)

    # Reuse Kaggle's existing cache when available.
    # No forced deletion or overwrite of previous dataset files.
    dataset_root = Path(
        kagglehub.dataset_download(HANDLE)
    ).resolve()

    require(dataset_root.is_dir(), "Downloaded dataset directory not found")

    files = sorted(
        path for path in dataset_root.rglob("*")
        if path.is_file()
    )
    require(bool(files), "Downloaded dataset contains no files")

    image_files = [
        path for path in files
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    metadata_files = [
        path for path in files
        if path.suffix.lower() in METADATA_SUFFIXES
    ]

    print(f"\nDataset directory: {dataset_root}")
    print(f"Total files: {len(files)}")
    print(f"Image files: {len(image_files)}")

    print("\nFile counts by extension:")
    extension_counts = Counter(
        path.suffix.lower() or "(no extension)" for path in files
    )
    for extension, count in sorted(extension_counts.items()):
        print(f"  {extension}: {count}")

    print("\nImage counts by top-level directory:")
    directory_counts = Counter()
    for path in image_files:
        relative = path.relative_to(dataset_root)
        directory = relative.parts[0] if len(relative.parts) > 1 else "(root)"
        directory_counts[directory] += 1

    for directory, count in sorted(directory_counts.items()):
        print(f"  {directory}: {count}")

    print("\nExample image paths:")
    for path in image_files[:5]:
        print(f"  {path.relative_to(dataset_root).as_posix()}")

    print("\nPotential label/metadata files:")
    if not metadata_files:
        print("  No recognized label/metadata files found.")
    else:
        for path in metadata_files[:40]:
            print(
                f"  {path.relative_to(dataset_root).as_posix()} "
                f"({path.stat().st_size} bytes)"
            )
        if len(metadata_files) > 40:
            print(f"  ... and {len(metadata_files) - 40} additional files")

    # Preview small text metadata only; do not guess its interpretation.
    previews = [
        path for path in metadata_files
        if path.suffix.lower() in {".csv", ".txt", ".json"}
    ]
    for path in previews[:8]:
        print(
            f"\nPreview: {path.relative_to(dataset_root).as_posix()}"
        )
        with path.open(
            "r", encoding="utf-8-sig", errors="replace"
        ) as stream:
            for _ in range(4):
                line = stream.readline(500)
                if not line:
                    break
                print("  " + line.rstrip())

    print("\nSTL10_KAGGLE_DOWNLOAD_COMPLETED")
    print("download_status=PASS")
    print("official_binary_equivalence=NOT_CHECKED")
    print("image_label_alignment=NOT_CHECKED")
    print("formal_training_ready=False")
    print("model_training_started=False")
    print("test_accuracy_evaluated=False")
    print(f"Dataset directory: {dataset_root}")


if __name__ == "__main__":
    import time
    import requests

    max_attempts = 12

    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"\nDownload attempt {attempt}/{max_attempts}",
                flush=True,
            )
            main()
            break

        except requests.exceptions.SSLError:
            # Certificate problems should not be treated as temporary outages.
            raise

        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as error:
            archive = (
                CACHE_ROOT / "datasets" / "jessicali9530"
                / "stl10" / "3.archive"
            )
            if archive.is_file():
                saved_mib = archive.stat().st_size / (1024 ** 2)
                print(
                    f"Partial download retained: {saved_mib:.2f} MiB",
                    flush=True,
                )

            print(
                f"Network interruption: {type(error).__name__}",
                flush=True,
            )
            if attempt == max_attempts:
                print(
                    "Retry limit reached. Partial files are retained.",
                    flush=True,
                )
                raise

            print(
                "Retrying in 20 seconds; requesting resume when supported.",
                flush=True,
            )
            time.sleep(20)