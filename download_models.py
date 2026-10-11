"""Download the GGUF models listed in config.yaml and verify size + SHA256.

This is the only script in the project that touches the network.
Usage: python download_models.py [model_key ...]
"""
import hashlib
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    models_dir = ROOT / cfg["paths"]["models_dir"]
    models_dir.mkdir(exist_ok=True)
    wanted = set(sys.argv[1:])
    api = HfApi()
    ok = True

    for m in cfg["models"]:
        if wanted and m["key"] not in wanted:
            continue
        # Confirm the repo and file exist before downloading anything.
        info = api.model_info(m["repo"], files_metadata=True)
        remote = {s.rfilename: s for s in info.siblings}
        if m["filename"] not in remote:
            print(f"[{m['key']}] {m['filename']} not found in {m['repo']}")
            ok = False
            continue
        remote_sha = remote[m["filename"]].lfs.sha256

        path = models_dir / m["filename"]
        if not path.exists() or path.stat().st_size != m["size_bytes"]:
            print(f"[{m['key']}] downloading {m['repo']}/{m['filename']} ...")
            hf_hub_download(m["repo"], m["filename"], local_dir=models_dir)

        size, digest = path.stat().st_size, sha256(path)
        good = size == m["size_bytes"] and digest == m["sha256"] == remote_sha
        ok &= good
        print(f"[{m['key']}] {size / 2**20:.0f} MB sha256={digest} {'OK' if good else 'MISMATCH'}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
