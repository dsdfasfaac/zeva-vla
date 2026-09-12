"""Hash a complete RoboTwin adapter dataset for explicit host relocation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def component(path: Path) -> dict:
    if path.is_file():
        return {"sha256": sha256(path), "files": 1, "bytes": path.stat().st_size}
    if not path.is_dir():
        raise ValueError(f"Missing component: {path}")
    digest = hashlib.sha256()
    count = total = 0
    for item in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if item.is_symlink() and item.is_dir():
            raise ValueError(f"Refusing an incompletely traversed directory symlink: {item}")
        if item.is_file():
            size = item.stat().st_size
            record = [item.relative_to(path).as_posix(), size, sha256(item)]
            digest.update((json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n").encode())
            count += 1
            total += size
    if not count:
        raise ValueError(f"Empty component: {path}")
    return {"sha256": digest.hexdigest(), "files": count, "bytes": total}


def audit(dataset_root: Path) -> dict:
    dataset_root = dataset_root.resolve()
    adapter_path = dataset_root / "adapter.json"
    adapter = json.loads(adapter_path.read_text())
    semantic = dict(adapter)
    components = {}
    for field, name in (("dataset_root", "source"), ("eef_cache_root", "eef-index"),
                        ("joint_cache_root", "joint14-index"), ("stats_path", "stats")):
        path = Path(adapter[field])
        if not path.is_absolute():
            path = dataset_root / path
        components[name] = component(path)
        semantic[field] = "@" + name
        print(f"verified {name}: {components[name]}", flush=True)
    return {"schema": "robotwin-dataset-content-identity-v1", "dataset_root": str(dataset_root),
            "adapter_sha256": sha256(adapter_path), "semantic_adapter": semantic,
            "components": components}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.dataset_root)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
