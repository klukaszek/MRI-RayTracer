#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import time
from typing import List, Tuple, Dict, Sequence

import numpy as np
import nibabel as nib

import jax
import jax.numpy as jnp
import optax


MODALITY_SUFFIXES = ["t1n", "t1c", "t2w", "t2f"]
SEG_SUFFIX = "seg"
NUM_CLASSES = 4


def find_cases(root: str) -> List[str]:
    root = os.fspath(root)
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        if any(os.path.exists(os.path.join(p, f"{name}-{m}.nii.gz")) for m in MODALITY_SUFFIXES):
            out.append(p)
    return out


def _load_case(case_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    base = os.path.basename(case_dir)
    mods = []
    for suf in MODALITY_SUFFIXES:
        fp = os.path.join(case_dir, f"{base}-{suf}.nii.gz")
        arr = nib.load(fp).get_fdata().astype(np.float32)
        m = arr != 0
        if m.any():
            mu = arr[m].mean(); sigma = arr[m].std() + 1e-6
            arr = (arr - mu) / sigma
        mods.append(arr)
    seg = nib.load(os.path.join(case_dir, f"{base}-{SEG_SUFFIX}.nii.gz")).get_fdata().astype(np.int16)
    return np.stack(mods, axis=0), seg


def load_case_list(case_paths: Sequence[str]):
    cache: List[Dict[str, np.ndarray]] = []
    for cp in case_paths:
        mods, seg = _load_case(cp)
        cache.append({"mods": mods, "seg": seg})
    return cache


def sample_from_single_case_np(rng_key, case: Dict[str, np.ndarray], batch_size: int):
    mods = case["mods"]  # (M,H,W,D)
    seg = case["seg"]
    M, H, W, D = mods.shape
    kx, ky, kz = jax.random.split(rng_key, 3)
    xs = np.array(jax.random.randint(kx, (batch_size,), 0, H))
    ys = np.array(jax.random.randint(ky, (batch_size,), 0, W))
    zs = np.array(jax.random.randint(kz, (batch_size,), 0, D))
    intens = mods[:, xs, ys, zs].transpose(1, 0)
    labels = seg[xs, ys, zs].astype(np.int32)
    coords = np.stack([xs, ys, zs], axis=-1)
    norm_coords = (coords / np.array([H - 1, W - 1, D - 1])) * 2.0 - 1.0
    return jnp.array(norm_coords), jnp.array(intens), jnp.array(labels)


def fourier_features(coords: jnp.ndarray, k: int) -> jnp.ndarray:
    # coords: (B,3) in [-1,1]
    B, dim = coords.shape
    freqs = jnp.arange(1, k + 1)
    ang = coords[..., None] * freqs[None, None, :] * math.pi
    ff = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1).reshape(B, dim * 2 * k)
    return ff


def build_input(coords: jnp.ndarray, intensities: jnp.ndarray, k: int) -> jnp.ndarray:
    ff = fourier_features(coords, k)
    return jnp.concatenate([coords, ff, intensities], axis=-1)


def glorot(key, shape):
    fan_in, fan_out = shape[0], shape[1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def init_mlp(key, in_dim: int, hidden: List[int], out_dim: int):
    params = []
    dims = [in_dim] + hidden + [out_dim]
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
    return jnp.dot(h, last["W"]) + last["b"]


def one_hot(labels, num_classes):
    return jax.nn.one_hot(labels, num_classes)


def loss_fn(params, coords, intensities, labels, k):
    x = build_input(coords, intensities, k)
    logits = apply_mlp(params, x)
    y = one_hot(labels, NUM_CLASSES)
    return optax.softmax_cross_entropy(logits, y).mean()


def predict_volume(params, mods: np.ndarray, seg_true: np.ndarray, k: int, chunk: int = 200_000):
    M, H, W, D = mods.shape
    xs, ys, zs = np.arange(H), np.arange(W), np.arange(D)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    intens = mods.transpose(1, 2, 3, 0).reshape(-1, M)
    norm_coords = (grid / np.array([H - 1, W - 1, D - 1])) * 2.0 - 1.0
    preds = []
    for i in range(0, len(grid), chunk):
        x_in = build_input(jnp.array(norm_coords[i:i+chunk]), jnp.array(intens[i:i+chunk]), k)
        cls = jnp.argmax(apply_mlp(params, x_in), axis=-1)
        preds.append(np.array(cls, dtype=np.int16))
    pred = np.concatenate(preds, axis=0).reshape(H, W, D)
    return pred, seg_true


def dice_score(pred: np.ndarray, true: np.ndarray, num_classes: int = NUM_CLASSES):
    out = {}
    for c in range(num_classes):
        p = pred == c; t = true == c
        inter = (p & t).sum(); denom = p.sum() + t.sum()
        out[c] = float((2 * inter + 1e-6) / (denom + 1e-6)) if denom > 0 else np.nan
    return out


def save_params_npz(params, path: str):
    flat = {}
    for i, layer in enumerate(params):
        flat[f"W_{i}"] = np.array(layer["W"])
        flat[f"b_{i}"] = np.array(layer["b"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **flat)


def maybe_load_params(path: str):
    d = np.load(path)
    layers = []
    i = 0
    while f"W_{i}" in d:
        layers.append({"W": jnp.array(d[f"W_{i}"]), "b": jnp.array(d[f"b_{i}"])})
        i += 1
    if not layers:
        raise RuntimeError(f"No layers found in {path}")
    return layers


def main():
    ap = argparse.ArgumentParser(description="Baseline Fourier INR (JAX) for BraTS 2023")
    ap.add_argument("--data-root", default="data/BraTS-2023")
    ap.add_argument("--case-limit", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--fourier-freqs", type=int, default=16)
    ap.add_argument("--hidden", type=int, nargs="*", default=[256, 256, 256, 256])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-index", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=120_000)
    ap.add_argument("--out", default="artifacts/inr_brats23.npz")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("JAX devices:", jax.devices())

    all_cases = find_cases(args.data_root)
    if not all_cases:
        raise RuntimeError(f"No cases under {args.data_root}")
    train_cases = all_cases[: args.case_limit]
    print(f"Using {len(train_cases)} training cases")

    train_cache = load_case_list(train_cases)
    M, H, W, D = train_cache[0]["mods"].shape
    print("Data dims (M,H,W,D)=", (M, H, W, D))

    key = jax.random.PRNGKey(args.seed)
    coords, feats, _ = sample_from_single_case_np(key, train_cache[0], 2)
    in_dim = coords.shape[-1] + feats.shape[-1] + coords.shape[-1] * 2 * args.fourier_freqs
    print("Input dimension:", in_dim)

    # init or resume
    if args.resume and os.path.exists(args.resume):
        params = maybe_load_params(args.resume)
        assert params[0]["W"].shape[0] == in_dim and params[-1]["W"].shape[1] == NUM_CLASSES, \
            "Checkpoint arch mismatch"
        print(f"Resumed from {args.resume}")
    else:
        key, params = init_mlp(key, in_dim, args.hidden, NUM_CLASSES)
    total_params = sum(p["W"].size + p["b"].size for p in params)
    print("Total parameters:", total_params)

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    micro_bs = args.micro_batch_size or args.batch_size
    accum_steps = max(1, math.ceil(args.batch_size / micro_bs))
    micro_bs = int(math.ceil(args.batch_size / accum_steps))
    print(f"Microbatching: global={args.batch_size}, micro={micro_bs}, accum_steps={accum_steps}")

    def _loss_batch(p, c, f, y):
        return loss_fn(p, c, f, y, args.fourier_freqs)
    loss_and_grad = jax.jit(jax.value_and_grad(_loss_batch))

    def train_step_host(params, opt_state, rng_key):
        grads_acc = jax.tree_util.tree_map(jnp.zeros_like, params)
        loss_acc = 0.0
        key = rng_key
        for _ in range(accum_steps):
            key, sub_case, sub = jax.random.split(key, 3)
            case_idx = int(jax.random.randint(sub_case, (), 0, len(train_cache)))
            c, f, y = sample_from_single_case_np(sub, train_cache[case_idx], micro_bs)
            loss_val, grads = loss_and_grad(params, c, f, y)
            grads_acc = jax.tree_util.tree_map(lambda a, b: a + b, grads_acc, grads)
            loss_acc += float(loss_val)
        grads_mean = jax.tree_util.tree_map(lambda x: x / accum_steps, grads_acc)
        updates, opt_state2 = optimizer.update(grads_mean, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss_acc / accum_steps, key

    # warmup
    key, step_key = jax.random.split(key)
    params, opt_state, warm_loss, key = train_step_host(params, opt_state, step_key)
    print(f"Warm-up loss: {warm_loss:.4f}")

    # train
    start = time.time()
    for step in range(1, args.steps + 1):
        key, step_key = jax.random.split(key)
        params, opt_state, loss_val, key = train_step_host(params, opt_state, step_key)
        if step % 20 == 0 or step == 1:
            print(f"Step {step}/{args.steps} | loss={loss_val:.4f}")
    print("Training time: {:.2f}s".format(time.time() - start))

    save_params_npz(params, args.out)
    print("Saved parameters ->", args.out)

    # eval on one training case
    val_case = train_cases[min(args.val_index, len(train_cases) - 1)]
    val_mods, val_seg = _load_case(val_case)
    pred, true = predict_volume(params, val_mods, val_seg, args.fourier_freqs, chunk=args.chunk)
    print("Validation Dice:", dice_score(pred, true))

    # eval on first hold-out if exists
    extra = all_cases[args.case_limit:]
    if extra:
        hmods, hseg = _load_case(extra[0])
        hpred, htrue = predict_volume(params, hmods, hseg, args.fourier_freqs, chunk=args.chunk)
        print("Hold-out Dice:", dice_score(hpred, htrue))
    else:
        print("No hold-out cases beyond training subset.")


if __name__ == "__main__":
    main()

