"""Official ViT-Tiny construction and the author's checkpoint loading procedure.

Transfer all 12 blocks and terminal norm. The author's loader ignores symbolic
embedding keys absent from the vision model. Verify every loaded tensor and every
fresh tensor after loading, since the upstream loader only prints discrepancies.
Same-seed Random/Procedural models share all non-transferred initial parameters.
"""

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "config" / "upstream.lock.json"
UNUSED_SOURCE_KEYS = {"tok.emb.weight", "pos.emb.weight"}
FRESH_MODEL_KEYS = {
    "cls_token", "pos_embed", "patch_embed.proj.weight", "patch_embed.proj.bias",
    "head.weight", "head.bias",
}


def read_lock():
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def official_module():
    code = read_lock()["code"]
    repo = ROOT / code["directory"]
    git = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(git + ["status", "--porcelain"], text=True).strip()
    if commit != code["commit"] or dirty:
        raise RuntimeError("Official source must match the clean, pinned upstream commit")
    return load_official_file("_pwvit_official_model", repo / "downstream/models/vision_transformer.py")


def load_official_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@lru_cache(maxsize=1)
def official_utils():
    official_module()  # Check the pinned checkout and register its model first.
    return load_official_file("_pwvit_official_utils", ROOT / read_lock()["code"]["directory"] / "utils.py")


def read_checkpoint():
    spec = read_lock()["checkpoint"]
    path = ROOT / spec["path"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != spec["size_bytes"] or digest.hexdigest() != spec["sha256"]:
        raise RuntimeError("Official checkpoint size/SHA256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model"}:
        raise RuntimeError("Unexpected checkpoint container")
    state = checkpoint["model"]
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise RuntimeError("Checkpoint must contain a tensor state dictionary")
    return state


def validate_transfer(model, state):
    target = model.state_dict()
    expected = {key for key in target if key.startswith("blocks.")} | {"norm.weight", "norm.bias"}
    if len(expected) != 146 or set(state) != expected | UNUSED_SOURCE_KEYS:
        raise RuntimeError("Unexpected checkpoint transfer or symbolic embedding keys")
    if set(target) - expected != FRESH_MODEL_KEYS:
        raise RuntimeError("Unexpected vision model parameters outside the transfer")
    for key in expected:
        value = state[key]
        if value.shape != target[key].shape or value.dtype != target[key].dtype:
            raise RuntimeError(f"Checkpoint shape/dtype mismatch: {key}")
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f"Non-finite checkpoint tensor: {key}")
    return {key: state[key] for key in sorted(expected)}


def build_model(initialization, num_classes, seed):
    if initialization not in {"random", "procedural"}:
        raise ValueError("initialization must be random or procedural")
    if num_classes not in {10, 37, 100}:
        raise ValueError("This protocol uses only 10, 37 or 100 classes")
    utils = official_utils()
    # Model construction occurs on CPU; preserve the caller's CPU random stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = utils.build_model(SimpleNamespace(
            model="vit_tiny", nb_classes=num_classes, drop_path=0.0,
        ))
    if (len(model.blocks) != 12 or model.embed_dim != 192
            or model.patch_embed.patch_size != (16, 16)
            or model.patch_embed.img_size != (224, 224)
            or any(block.attn.num_heads != 3 for block in model.blocks)):
        raise RuntimeError("Unexpected official ViT-Tiny architecture")
    if initialization == "procedural":
        state = read_checkpoint()
        transferred = validate_transfer(model, state)
        fresh = {key: model.state_dict()[key].clone() for key in FRESH_MODEL_KEYS}
        # Same full state dictionary and loader as downstream/main.py. The two
        # symbolic embeddings are reported unused; terminal norm IS loaded.
        utils.load_state_dict(model, state, prefix="")
        loaded = model.state_dict()
        for key, value in transferred.items():
            if not torch.equal(loaded[key], value):
                raise RuntimeError(f"Upstream loader did not transfer {key} exactly")
        for key, value in fresh.items():
            if not torch.equal(loaded[key], value):
                raise RuntimeError(f"Upstream loader unexpectedly overwrote {key}")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("All model parameters must remain trainable")
    return model
