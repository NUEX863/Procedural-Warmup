"""Verify imports and CUDA training primitives using disposable random tensors.

No dataset or pretrained checkpoint is downloaded. This is an environment check,
not an experiment or an audit of the author's model/checkpoint.
"""

import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".cache" / "matplotlib"))
import huggingface_hub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io
import timm
import torch
import torchvision
import yaml

# These are the legacy timm APIs imported by the official ViT implementation.
from timm.models.helpers import (
    adapt_input_conv,
    build_model_with_cfg,
    checkpoint_seq,
    named_apply,
    resolve_pretrained_cfg,
)
from timm.models.layers import DropPath, Mlp, PatchEmbed, lecun_normal_, trunc_normal_
from timm.models.registry import register_model


def main():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("The validated environment requires Python 3.10.x")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this check with the project .venv Python")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; a CPU-only environment is not sufficient")
    torch.manual_seed(11)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()

    # Exercise the same resize/tensor path that will be needed for 32x32 images.
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    sample = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    assert preprocess(sample).shape == (3, 224, 224)
    buffer = io.BytesIO()
    scipy.io.savemat(buffer, {"labels": np.arange(10)})
    buffer.seek(0)
    assert scipy.io.loadmat(buffer)["labels"].size == 10
    assert yaml.safe_load("classes: 100")["classes"] == 100
    assert len(pd.read_csv(io.StringIO("epoch,loss\n1,1.0\n"))) == 1
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, format="png")
    plt.close(figure)
    assert image_buffer.getbuffer().nbytes > 0

    checks = []
    for num_classes in (100, 10):
        model = timm.create_model(
            "vit_tiny_patch16_224", pretrained=False, num_classes=num_classes
        ).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")
        inputs = torch.randn(2, 3, 224, 224, device="cuda")
        labels = torch.arange(2, device="cuda")
        weight = model.blocks[0].attn.qkv.weight
        before = weight.detach().clone()
        # Establish that ordinary FP32 gradients are finite before testing AMP.
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        assert torch.isfinite(loss).item()
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all().item()
            for p in model.parameters() if p.grad is not None
        ), "Non-finite FP32 gradients"
        initial_scale = scaler.get_scale()
        # GradScaler may skip initial updates while calibrating its default scale.
        # Verify both safe skipping and an eventual finite, real optimizer update.
        for skipped_steps in range(12):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
            assert outputs.shape == (2, num_classes)
            assert outputs.dtype == torch.float16
            assert torch.isfinite(loss).item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite_gradients = all(
                torch.isfinite(p.grad).all().item()
                for p in model.parameters() if p.grad is not None
            )
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if finite_gradients:
                break
            assert torch.equal(before, weight.detach()), "Overflow update was not skipped"
            assert scaler.get_scale() < previous_scale, "AMP scale did not decrease"
        else:
            raise RuntimeError("FP16 gradients remained non-finite after scale calibration")
        torch.cuda.synchronize()
        assert not torch.equal(before, weight.detach()), "Optimizer did not update weights"
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
        checks.append({
            "classes": num_classes,
            "fp32_forward_backward": "PASS",
            "fp16_forward_backward_update": "PASS",
            "amp_initial_scale": initial_scale,
            "amp_final_scale": scaler.get_scale(),
            "amp_skipped_steps": skipped_steps,
        })
        del model, optimizer, scaler, inputs, labels, outputs, loss, weight, before
        torch.cuda.empty_cache()

    packages = [
        "torch", "torchvision", "timm", "numpy", "scipy", "Pillow",
        "PyYAML", "matplotlib", "pandas", "huggingface-hub", "pip",
    ]
    print(json.dumps({
        "status": "PASS",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "isolated_environment": sys.prefix != sys.base_prefix,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "legacy_timm_imports": "PASS",
        "data_and_plot_dependencies": "PASS",
        "random_tensor_checks": checks,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "scope": "Environment only; official model and checkpoint are not yet validated.",
    }, indent=2))


if __name__ == "__main__":
    main()
