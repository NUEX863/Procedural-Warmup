"""Fetch the exact upstream code and ViT-Tiny checkpoint in upstream.lock.json."""

import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "config" / "upstream.lock.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo, *args):
    # Scope the trust exception to this explicitly selected clone and invocation.
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args],
        text=True,
    ).strip()


def check_checkpoint(path, spec):
    if path.stat().st_size != spec["size_bytes"] or sha256(path) != spec["sha256"]:
        raise RuntimeError(f"Checkpoint size/SHA256 mismatch: {path}")


def main():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    code = lock["code"]
    repo = ROOT / code["directory"]
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", code["url"], str(repo)], check=True)
    if git(repo, "remote", "get-url", "origin") != code["url"]:
        raise RuntimeError("Unexpected upstream remote")
    if git(repo, "status", "--porcelain"):
        raise RuntimeError("Upstream checkout has changes; inspect them before proceeding")
    git(repo, "checkout", "--detach", code["commit"])
    if git(repo, "rev-parse", "HEAD") != code["commit"]:
        raise RuntimeError("Upstream commit mismatch")
    print(f"Code pinned: {code['commit']}", flush=True)

    spec = lock["checkpoint"]
    destination = ROOT / spec["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        url = (f"https://huggingface.co/{spec['repo_id']}/resolve/"
               f"{spec['revision']}/{spec['filename']}")
        temporary = destination.with_suffix(".pth.part")
        # Exclusive creation avoids overwriting any previous incomplete download.
        print(f"Downloading {spec['filename']} ({spec['size_bytes']} bytes)", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("xb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        check_checkpoint(temporary, spec)
        temporary.replace(destination)
    check_checkpoint(destination, spec)
    print(f"Checkpoint verified: {destination}\nSHA256: {spec['sha256']}")


if __name__ == "__main__":
    main()
