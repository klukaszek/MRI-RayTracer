import json
import pathlib
from typing import List

import jax
import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "brats-inr-segmentation"


def glorot(key, shape):
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def init_mlp(key, in_dim: int, hidden_dims: List[int], out_dim: int):
    params = []
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    for i in range(len(dims) - 1):
        key, k1, _ = jax.random.split(key, 3)
        W = glorot(k1, (dims[i], dims[i + 1]))
        b = jnp.zeros((dims[i + 1],))
        params.append({"W": W, "b": b})
    return key, params


def apply_mlp(params, x):
    *hidden, last = params
    h = x
    for layer in hidden:
        h = jnp.dot(h, layer["W"]) + layer["b"]
        h = jax.nn.relu(h)
    out = jnp.dot(h, last["W"]) + last["b"]
    return out


def load_run(run_name: str):
    """Load config + best checkpoint for a given W&B run name.

    Returns (config, params, metadata_dict).
    """
    save_path = ARTIFACTS_ROOT / run_name
    json_path = save_path / "training_config_and_results.json"

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cfg = payload.get("config", {})
    arch = payload.get("metadata", {}).get("architecture", {})
    # Fallback to top-level keys if not present in metadata
    input_dim = arch.get("input_dim", cfg.get("input_dim"))
    hidden_dims = arch.get("hidden_dims") or cfg.get("hidden_dims")
    num_classes = arch.get("num_classes", cfg.get("num_classes", 4))

    artifacts = payload.get("artifacts", {})
    ckpt_best_name = artifacts.get("checkpoint_best", "checkpoint_best.npz")
    ckpt_path = save_path / ckpt_best_name

    ckpt = np.load(ckpt_path)

    # Reconstruct parameter list in the same ordering convention
    layer_indices = sorted({int(k.split("_")[1]) for k in ckpt.files})
    params = []
    for i in layer_indices:
        W = jnp.array(ckpt[f"W_{i}"])
        b = jnp.array(ckpt[f"b_{i}"])
        params.append({"W": W, "b": b})

    meta = {
        "results": payload.get("results", {}),
        "artifacts": artifacts,
    }

    return {
        "config": cfg,
        "input_dim": int(input_dim),
        "hidden_dims": hidden_dims,
        "num_classes": int(num_classes),
        "params": params,
        "meta": meta,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load best INR checkpoint for a run")
    parser.add_argument("run_name", help="W&B run name (folder under artifacts/brats-inr-segmentation)")
    args = parser.parse_args()

    info = load_run(args.run_name)
    print("Loaded run", args.run_name)
    print("Hidden dims:", info["hidden_dims"])
    print("Num classes:", info["num_classes"])
    print("Best val dice:", info["meta"]["results"].get("best_val_dice"))
