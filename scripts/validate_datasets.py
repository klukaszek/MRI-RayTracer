#!/usr/bin/env python3
"""
Validate docs/datasets.yaml for required fields and types.
Requires PyYAML. If unavailable, prints a helpful message.
"""
from __future__ import annotations
import sys
import os

DATASETS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "datasets.yaml")

def fail(msg: str) -> int:
    print(f"[ERROR] {msg}")
    return 1

def main() -> int:
    try:
        import yaml  # type: ignore
    except Exception:
        print("[ERROR] PyYAML not installed. Install with: pip install pyyaml")
        print(f"         Expected file at: {DATASETS_PATH}")
        return 1

    if not os.path.exists(DATASETS_PATH):
        return fail(f"datasets.yaml not found at {DATASETS_PATH}")

    with open(DATASETS_PATH, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except Exception as e:
            return fail(f"Failed to parse YAML: {e}")

    errors = []

    # Basic keys
    for k in ("version", "resolution"):
        if k not in data:
            errors.append(f"Missing top-level key: {k}")

    # Resolution must be 1920x1080
    res = data.get("resolution")
    if not (isinstance(res, (list, tuple)) and len(res) == 2 and all(isinstance(x, int) for x in res)):
        errors.append("resolution must be a list of two integers, e.g., [1920, 1080]")
    else:
        if res != [1920, 1080]:
            errors.append(f"resolution is {res}, expected [1920, 1080] for this project")

    # Volumes
    vols = data.get("volumes", [])
    if not isinstance(vols, list) or not vols:
        errors.append("volumes must be a non-empty list")
    else:
        for i, v in enumerate(vols):
            prefix = f"volumes[{i}]"
            if not isinstance(v, dict):
                errors.append(f"{prefix} must be a mapping")
                continue
            for req in ("name", "source", "target_resolution", "preprocessing", "splits"):
                if req not in v:
                    errors.append(f"{prefix}: missing '{req}'")
            tr = v.get("target_resolution")
            if not (isinstance(tr, list) and len(tr) == 3 and all(isinstance(x, int) for x in tr)):
                errors.append(f"{prefix}.target_resolution must be [Dx, Dy, Dz] integers")
            splits = v.get("splits", {})
            if not (isinstance(splits, dict) and all(k in splits for k in ("train_views", "test_views"))):
                errors.append(f"{prefix}.splits must contain train_views and test_views")
            else:
                tv, tev = splits.get("train_views"), splits.get("test_views")
                if not (isinstance(tv, int) and tv > 0 and isinstance(tev, int) and tev > 0):
                    errors.append(f"{prefix}.splits train/test views must be positive integers")

    # Point clouds (optional)
    pcs = data.get("point_clouds", [])
    if pcs is not None:
        if not isinstance(pcs, list):
            errors.append("point_clouds must be a list if present")
        else:
            for i, p in enumerate(pcs):
                prefix = f"point_clouds[{i}]"
                if not isinstance(p, dict):
                    errors.append(f"{prefix} must be a mapping")
                    continue
                for req in ("name", "source", "provenance", "preprocessing", "splits"):
                    if req not in p:
                        errors.append(f"{prefix}: missing '{req}'")

    if errors:
        print("\nValidation failed:")
        for e in errors:
            print(f" - {e}")
        return 2

    print("datasets.yaml looks good ✔")
    # Short summary
    print(f"Resolution: {res[0]}x{res[1]}")
    print(f"Volumes: {[v['name'] for v in vols]}")
    if pcs:
        print(f"Point clouds: {[p['name'] for p in pcs]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

