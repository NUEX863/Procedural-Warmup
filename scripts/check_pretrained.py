"""Compare initialization with unmodified upstream code; no dataset training."""

import argparse
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.model_factory import (
    build_model, read_checkpoint, validate_transfer, read_lock, official_utils,
    UNUSED_SOURCE_KEYS, FRESH_MODEL_KEYS,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def author_reference(classes, seed, initialization):
    """Execute actual build/initialize AST statements from upstream main.py.

    The extracted nodes are unchanged. This avoids dataset loading and training
    while retaining the author's checkpoint selection, head handling and loader.
    """
    lock = read_lock()
    path = ROOT / lock["code"]["directory"] / "downstream/main.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    constructors = [n for n in main.body if isinstance(n, ast.Assign)
                    and ast.unparse(n) == "model = utils.build_model(args)"]
    loaders = [n for n in main.body if isinstance(n, ast.If)
               and ast.unparse(n.test) == "args.initialize"]
    require(len(constructors) == len(loaders) == 1, "Upstream initialization code changed")
    program = ast.Module(body=[constructors[0], loaders[0]], type_ignores=[])
    args = SimpleNamespace(
        model="vit_tiny", nb_classes=classes, drop_path=0.0,
        initialize=str(ROOT / lock["checkpoint"]["path"]) if initialization == "procedural" else "",
        model_key="model|module", model_prefix="",
    )
    namespace = {"args": args, "utils": official_utils(), "torch": torch}
    messages = io.StringIO()
    with torch.random.fork_rng(devices=[]), redirect_stdout(messages):
        torch.random.default_generator.manual_seed(seed)
        exec(compile(program, str(path), "exec"), namespace)
    return namespace["model"], messages.getvalue()


def check_pretraining_norm():
    """Exercise the published pretraining graph/optimizer with dummy tokens."""
    repo = ROOT / read_lock()["code"]["directory"]
    sys.path.insert(0, str(repo))
    try:
        from procedural_warmup.cfg import load_cfg
        from procedural_warmup.vocab import FrozenTokenEmbedding
        from procedural_warmup.pos_embed import FrozenPositionalEmbedding
        from procedural_warmup.model_factory import build_model as build_warmup
        from procedural_warmup.optim.optimizer import make_optimizer
        from procedural_warmup import process
    finally:
        sys.path.pop(0)
    cfg = load_cfg(str(repo / "procedural_warmup/configs/vit-t.yaml"))
    torch.manual_seed(cfg.seed)
    tok = FrozenTokenEmbedding(cfg.vocab.K, cfg.model.embed_dim)
    pos = FrozenPositionalEmbedding(cfg.grid.H * cfg.grid.W, cfg.model.embed_dim)
    model, head = build_warmup(cfg, tok, pos)

    # Execute the author's exact optimizer-parameter selection from Trainer.
    trainer_path = repo / "procedural_warmup/train/trainer.py"
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    init = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    selected = [n for n in init.body if isinstance(n, ast.Assign)
                and any(ast.unparse(t) in {"params", "self.opt"} for t in n.targets)]
    require(len(selected) == 2, "Upstream optimizer construction changed")
    instance = SimpleNamespace(model=model, mlm_head=head)
    namespace = {"self": instance, "cfg": cfg, "make_optimizer": make_optimizer}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(trainer_path), "exec"), namespace)
    optimizer = instance.opt
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    before = {}
    for name, param in model.vit.norm.named_parameters():
        require(param.requires_grad and id(param) in optimizer_ids, f"Norm {name} not optimized")
        before[name] = param.detach().clone()

    tokens = torch.randint(2, cfg.vocab.K, (2, cfg.grid.H * cfg.grid.W))
    logits = head(model.forward_tokens(tokens))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab.K), tokens.flatten())
    loss.backward()
    for name, param in model.vit.norm.named_parameters():
        require(param.grad is not None and torch.isfinite(param.grad).all().item()
                and param.grad.abs().max().item() > 0, f"Norm {name} gradient missing/invalid")
    optimizer.step()
    for name, param in model.vit.norm.named_parameters():
        require(not torch.equal(param, before[name]), f"Norm {name} did not update")

    retained = [key for key in sorted(UNUSED_SOURCE_KEYS | {"norm.weight", "norm.bias"})
                if key not in process._SKIP_EXACT
                and not any(key.startswith(p) for p in process._SKIP_PREFIXES)]
    require(set(retained) == UNUSED_SOURCE_KEYS | {"norm.weight", "norm.bias"},
            "Official checkpoint processing rules changed")
    return {"norm_in_optimizer": True, "norm_nonzero_gradients": True,
            "norm_updated": True, "process_retains_actual_keys": retained}


def gpu_update(model, classes):
    model.cuda().train()
    torch.manual_seed(123)
    inputs = torch.randn(2, 3, 224, 224, device="cuda")
    labels = torch.tensor([0, classes - 1], device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    before = {f"block{i}": block.attn.qkv.weight.detach().clone()
              for i, block in enumerate(model.blocks)}
    before.update({f"norm.{name}": p.detach().clone() for name, p in model.norm.named_parameters()})
    logits = model(inputs)
    require(logits.shape == (2, classes), "Output shape mismatch")
    loss = torch.nn.functional.cross_entropy(logits, labels)
    require(torch.isfinite(loss).item(), "Non-finite loss")
    loss.backward()
    require(all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all().item()
                for p in model.parameters()), "Frozen parameter or missing/non-finite gradient")
    optimizer.step()
    torch.cuda.synchronize()
    for i, block in enumerate(model.blocks):
        require(not torch.equal(before[f"block{i}"], block.attn.qkv.weight), "Block did not update")
    for name, param in model.norm.named_parameters():
        require(not torch.equal(before[f"norm.{name}"], param), "Terminal norm did not update")
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Non-finite updated parameter")
    model.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-checkpoint-keys", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    require(torch.cuda.is_available(), "CUDA unavailable")
    state = read_checkpoint()  # Verify hash before the author's loader reads it.
    if args.list_checkpoint_keys:
        for key, tensor in state.items():
            print(f"{key}: {list(tensor.shape)}")
    official_utils()  # Verify the source before exercising any upstream code.
    warmup = check_pretraining_norm()
    results = []
    last_messages = ""
    for seed in (11,):
        for classes in (100,37):
            random_model = build_model("random", classes, seed)
            with redirect_stdout(io.StringIO()):
                procedural = build_model("procedural", classes, seed)
            transferred = validate_transfer(procedural, state)
            for key in transferred:
                require(torch.equal(procedural.state_dict()[key], state[key]), f"Incorrect transfer: {key}")
            for key in FRESH_MODEL_KEYS:
                require(torch.equal(random_model.state_dict()[key], procedural.state_dict()[key]),
                        f"Unpaired fresh parameter: {key}")
            for name, model in (("random", random_model), ("procedural", procedural)):
                reference, messages = author_reference(classes, seed, name)
                if messages:
                    last_messages = messages
                actual, expected = model.state_dict(), reference.state_dict()
                require(set(actual) == set(expected) and len(actual) == 152, "State schema differs")
                require(repr(model) == repr(reference), "Module configuration differs")
                mismatches = [key for key in expected if not torch.equal(actual[key], expected[key])]
                require(not mismatches, f"Differs from author initialization: {mismatches}")
                model.eval()
                reference.eval()
                generator = torch.Generator().manual_seed(321)
                inputs = torch.randn(2, 3, 224, 224, generator=generator)
                with torch.no_grad():
                    output = model(inputs)
                    expected_output = reference(inputs)
                require(torch.equal(output, expected_output), "Author/reference outputs differ")
                del actual, expected, reference, inputs, output, expected_output
                gpu_update(model, classes)
                results.append({"seed": seed, "classes": classes, "initialization": name,
                                "all_152_state_tensors_bitwise_equal_to_author": True,
                                "cpu_eval_output_bitwise_equal_to_author": True,
                                "gpu_forward_backward_update": "PASS"})
            del random_model, procedural, model
            torch.cuda.empty_cache()
    print(json.dumps({
        "status": "PASS", "upstream": read_lock(), "warmup_norm_evidence": warmup,
        "checkpoint_tensor_count": len(state), "transferred_tensors": len(transferred),
        "unused_checkpoint_keys": sorted(UNUSED_SOURCE_KEYS),
        "fresh_parameters_equal_between_R_P": sorted(FRESH_MODEL_KEYS),
        "checkpoint_norm_max_deviation_from_default": {
            "weight": (state["norm.weight"] - 1).abs().max().item(),
            "bias": state["norm.bias"].abs().max().item(),
        },
        "author_loader_output": last_messages.strip(), "checks": results,
        "scope": "Pinned published source/checkpoint, same seed/classes/settings/environment; no real-data training.",
    }, indent=2))


if __name__ == "__main__":
    main()

