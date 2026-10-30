import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_factory import build_model


SEEDS = (11,)
NUM_CLASSES_LIST = (100, 37)

TRANSFERRED_NORM_KEYS = {
    "norm.weight",
    "norm.bias",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print("=" * 68)
    print("Checkpoint Transfer Verification")
    print("=" * 68)

    lock_path = ROOT / "config" / "upstream.lock.json"

    if not lock_path.is_file():
        raise FileNotFoundError(f"Lock file not found: {lock_path}")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    spec = lock["checkpoint"]
    checkpoint_path = ROOT / spec["path"]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Verify the original checkpoint file before loading.
    actual_size = checkpoint_path.stat().st_size
    actual_sha256 = sha256_file(checkpoint_path)

    if actual_size != spec["size_bytes"]:
        raise RuntimeError(
            f"Checkpoint size mismatch: "
            f"expected {spec['size_bytes']}, got {actual_size}"
        )

    if actual_sha256 != spec["sha256"]:
        raise RuntimeError("Checkpoint SHA256 mismatch")

    print("[PASS] Original checkpoint size and SHA256 verified.")
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("Unexpected checkpoint structure")

    source = checkpoint["model"]

    if not all(isinstance(v, torch.Tensor) for v in source.values()):
        raise RuntimeError("Checkpoint contains non-tensor model entries")

    # Expected transfer: all Transformer Blocks + final LayerNorm.
    source_keys = {
        key for key in source
        if key.startswith("blocks.")
        or key in TRANSFERRED_NORM_KEYS
    }

    block_keys = {
        key for key in source_keys
        if key.startswith("blocks.")
    }

    norm_keys = source_keys & TRANSFERRED_NORM_KEYS

    if len(block_keys) != 144:
        raise RuntimeError(
            f"Expected 144 Block tensors, found {len(block_keys)}"
        )

    if norm_keys != TRANSFERRED_NORM_KEYS:
        raise RuntimeError(
            f"Expected both final LayerNorm tensors, found {norm_keys}"
        )

    if len(source_keys) != 146:
        raise RuntimeError(
            f"Expected 146 transferred tensors, found {len(source_keys)}"
        )

    print("[PASS] Checkpoint contains 144 Block tensors + 2 LayerNorm tensors.")

    all_passed = True

    for seed in SEEDS:
        for num_classes in NUM_CLASSES_LIST:
            print(
                f"\nChecking seed={seed}, classes={num_classes}"
            )

            model = build_model(
                initialization="procedural",
                num_classes=num_classes,
                seed=seed,
            )

            target = model.state_dict()

            target_keys = {
                key for key in target
                if key.startswith("blocks.")
                or key in TRANSFERRED_NORM_KEYS
            }

            if target_keys != source_keys:
                missing = sorted(source_keys - target_keys)
                extra = sorted(target_keys - source_keys)

                print("[FAIL] Transfer key sets differ.")
                print("Missing in target:", missing[:10])
                print("Extra in target:", extra[:10])
                all_passed = False
                del model
                continue

            mismatches = []

            # Exact element-by-element comparison.
            for key in sorted(source_keys):
                expected = source[key]
                actual = target[key]

                if expected.shape != actual.shape:
                    mismatches.append(
                        f"{key}: shape {actual.shape} != {expected.shape}"
                    )
                elif expected.dtype != actual.dtype:
                    mismatches.append(
                        f"{key}: dtype {actual.dtype} != {expected.dtype}"
                    )
                elif not torch.equal(actual.cpu(), expected.cpu()):
                    mismatches.append(f"{key}: tensor values differ")

            if mismatches:
                print(
                    f"[FAIL] {len(mismatches)} checkpoint tensor "
                    "mismatch(es):"
                )

                for item in mismatches[:10]:
                    print("  -", item)

                all_passed = False
            else:
                print(
                    "[PASS] All 146 transferred tensors exactly "
                    "match the checkpoint."
                )
                print("[PASS] Transformer Blocks: 144/144")
                print("[PASS] Final LayerNorm: 2/2")

            del model

    print("\n" + "=" * 68)

    if all_passed:
        print("[FINAL RESULT] ALL CHECKS PASSED")
        print(
            "Procedural initialization matches the checkpoint "
            "for Blocks and final LayerNorm."
        )
    else:
        print("[FINAL RESULT] VERIFICATION FAILED")
        print("Do not start formal experiments yet.")

    print("=" * 68)

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()