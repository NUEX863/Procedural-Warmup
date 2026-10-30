
from pathlib import Path
import sys

import torch

# Add project root to Python import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_factory import build_model


SEEDS = (11,)
NUM_CLASSES_LIST = (100, 37)

# According to the intended transfer protocol:
# Transformer Blocks and the final LayerNorm are transferred.
TRANSFERRED_EXACT_KEYS = {
    "norm.weight",
    "norm.bias",
}


def is_transferred(name):
    return (
        name.startswith("blocks.")
        or name in TRANSFERRED_EXACT_KEYS
    )


def verify_pair(num_classes, seed):
    print("\n" + "=" * 65)
    print(f"Classes: {num_classes} | Seed: {seed}")
    print("=" * 65)

    # Construct both models on CPU.
    random_model = build_model(
        initialization="random",
        num_classes=num_classes,
        seed=seed,
    )

    procedural_model = build_model(
        initialization="procedural",
        num_classes=num_classes,
        seed=seed,
    )

    random_params = dict(random_model.named_parameters())
    procedural_params = dict(procedural_model.named_parameters())

    passed = True

    # Check 1: both models have the same parameter names.
    if random_params.keys() != procedural_params.keys():
        print("[FAIL] Parameter names differ between models.")

        missing = random_params.keys() - procedural_params.keys()
        extra = procedural_params.keys() - random_params.keys()

        print("Missing in procedural:", sorted(missing)[:10])
        print("Extra in procedural:", sorted(extra)[:10])

        passed = False
    else:
        print("[PASS] Parameter names match.")

    # Check 2: verify the expected transfer keys exist.
    transferred_names = {
        name for name in random_params
        if is_transferred(name)
    }

    non_transferred_names = {
        name for name in random_params
        if not is_transferred(name)
    }

    required_keys = {
        "norm.weight",
        "norm.bias",
    }

    missing_norm = required_keys - random_params.keys()

    if missing_norm:
        print(
            "[FAIL] Expected final LayerNorm parameters are missing:",
            sorted(missing_norm),
        )
        passed = False
    elif not any(
        name.startswith("blocks.")
        for name in transferred_names
    ):
        print("[FAIL] No Transformer Block parameters were found.")
        passed = False
    else:
        block_count = sum(
            name.startswith("blocks.")
            for name in transferred_names
        )

        print(
            f"[INFO] Expected transfer region: "
            f"{block_count} Block parameter tensors + final LayerNorm."
        )

    # Check 3: all non-transferred initial parameters must be identical.
    if random_params.keys() == procedural_params.keys():
        mismatches = []

        for name in sorted(non_transferred_names):
            p_random = random_params[name]
            p_procedural = procedural_params[name]

            if (
                p_random.shape != p_procedural.shape
                or p_random.dtype != p_procedural.dtype
                or not torch.equal(p_random, p_procedural)
            ):
                mismatches.append(name)

        if mismatches:
            print(
                "[FAIL] Non-transferred parameters differ:",
                mismatches[:20],
            )
            passed = False
        else:
            print(
                f"[PASS] All {len(non_transferred_names)} "
                "non-transferred parameter tensors are identical."
            )

    # Check 4: all parameters in both models must be trainable.
    random_frozen = [
        name for name, parameter in random_params.items()
        if not parameter.requires_grad
    ]

    procedural_frozen = [
        name for name, parameter in procedural_params.items()
        if not parameter.requires_grad
    ]

    if random_frozen:
        print(
            "[FAIL] Random model has frozen parameters:",
            random_frozen[:20],
        )
        passed = False
    else:
        print(
            f"[PASS] All {len(random_params)} Random model "
            "parameter tensors require gradients."
        )

    if procedural_frozen:
        print(
            "[FAIL] Procedural model has frozen parameters:",
            procedural_frozen[:20],
        )
        passed = False
    else:
        print(
            f"[PASS] All {len(procedural_params)} Procedural model "
            "parameter tensors require gradients."
        )

    # Report how many parameters are outside the transfer region.
    print(
        f"[INFO] Non-transferred parameter tensors: "
        f"{len(non_transferred_names)}"
    )

    print(
        "[RESULT]",
        "PASS" if passed else "FAIL",
    )

    del random_model, procedural_model

    return passed


def main():
    print("=" * 65)
    print("Model Initialization Verification")
    print("=" * 65)
    print("Device: CPU")
    print("Checks: same-seed initialization + trainability")
    print("Note: checkpoint weight equality is NOT checked here.")

    all_passed = True

    for seed in SEEDS:
        for num_classes in NUM_CLASSES_LIST:
            try:
                passed = verify_pair(
                    num_classes=num_classes,
                    seed=seed,
                )
                all_passed = all_passed and passed

            except Exception as exc:
                all_passed = False
                print(
                    f"\n[ERROR] classes={num_classes}, seed={seed}: "
                    f"{type(exc).__name__}: {exc}"
                )

    print("\n" + "=" * 65)

    if all_passed:
        print("[FINAL RESULT] ALL CHECKS PASSED")
        print(
            "Next: verify that Blocks and final LayerNorm "
            "equal the checkpoint tensors."
        )
    else:
        print("[FINAL RESULT] VERIFICATION FAILED")
        print("Do not start formal experiments until failures are resolved.")

    print("=" * 65)

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
