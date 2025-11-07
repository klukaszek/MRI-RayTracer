#!/usr/bin/env python3
"""
Fourier-feature INR (JAX) for BraTS 2023 segmentation.

Train a coordinate MLP with Fourier features using JAX + Optax,
then evaluate Dice on a validation case and an optional hold-out case.

Example:
  python scripts/jax_inr_brats.py \
    --data-root data/BraTS-2023 \
    --case-limit 8 \
    --batch-size 8192 \
    --fourier-freqs 16 \
    --hidden 256 256 256 256 \
    --steps 200 \
    --lr 2e-3 \
    --val-index 0 \
    --chunk 120000 \
    --out artifacts/inr_brats23.npz
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Sequence, Optional

import numpy as np
import nibabel as nib

import jax
import jax.numpy as jnp
import optax
from scipy.ndimage import gaussian_filter


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
        # Heuristic: expect at least one modality present
        if any(os.path.exists(os.path.join(p, f"{name}-{m}.nii.gz")) for m in MODALITY_SUFFIXES):
            out.append(p)
    return out


def _load_case(case_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    base = os.path.basename(case_dir)
    mods = []
    for suf in MODALITY_SUFFIXES:
        fp = os.path.join(case_dir, f"{base}-{suf}.nii.gz")
        img = nib.load(fp)
        arr = img.get_fdata().astype(np.float32)
        mask = arr != 0
        if mask.any():
            mu = arr[mask].mean(); sigma = arr[mask].std() + 1e-6
            arr = (arr - mu) / sigma
        mods.append(arr)
    seg_fp = os.path.join(case_dir, f"{base}-{SEG_SUFFIX}.nii.gz")
    seg = nib.load(seg_fp).get_fdata().astype(np.int16)
    mods_arr = np.stack(mods, axis=0)  # (M,H,W,D)
    return mods_arr, seg


def load_case_list(case_paths: Sequence[str]):
    """Load cases into a lightweight Python list to avoid a single giant device array."""
    cache: List[Dict[str, np.ndarray]] = []
    for cp in case_paths:
        mods, seg = _load_case(cp)
        cache.append({"mods": mods, "seg": seg})
    return cache


def fourier_features(coords: jnp.ndarray, k: int, rff_B: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Fourier features of coordinates.

    If rff_B is provided (shape (3, F)), uses random Fourier features with
    projection matrix B ~ N(0, sigma^2), producing features sin(2pi xB), cos(...)
    Otherwise uses deterministic sinusoidal features with harmonics 1..k per axis.
    """
    Bsz, dim = coords.shape
    if rff_B is not None:
        proj = coords @ rff_B  # (B,F)
        ang = 2 * math.pi * proj
        return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)  # (B,2F)
    else:
        freqs = jnp.arange(1, k + 1)
        ang = coords[..., None] * freqs[None, None, :] * math.pi  # (B,3,k)
        sin = jnp.sin(ang)
        cos = jnp.cos(ang)
        ff = jnp.concatenate([sin, cos], axis=-1).reshape(Bsz, dim * 2 * k)
        return ff


def build_input(coords: jnp.ndarray, intensities: jnp.ndarray, k: int, rff_B: Optional[jnp.ndarray]) -> jnp.ndarray:
    ff = fourier_features(coords, k, rff_B)
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
    return jnp.dot(h, last["W"]) + last["b"]  # logits


def one_hot(labels, num_classes):
    return jax.nn.one_hot(labels, num_classes)


def sample_from_single_case_np(rng_key, case: Dict[str, np.ndarray], batch_size: int):
    """Sample coordinates within a single case (NumPy), minimal device memory."""
    mods = case["mods"]  # (M,H,W,D)
    seg = case["seg"]     # (H,W,D)
    M, H, W, D = mods.shape
    # Use JAX RNG for reproducibility, convert to NumPy arrays (small)
    key_x, key_y, key_z = jax.random.split(rng_key, 3)
    xs = np.array(jax.random.randint(key_x, (batch_size,), 0, H))
    ys = np.array(jax.random.randint(key_y, (batch_size,), 0, W))
    zs = np.array(jax.random.randint(key_z, (batch_size,), 0, D))
    # Gather intensities/labels via NumPy advanced indexing
    intens = mods[:, xs, ys, zs].transpose(1, 0)  # (B,M)
    labels = seg[xs, ys, zs].astype(np.int32)     # (B,)
    coords = np.stack([xs, ys, zs], axis=-1)      # (B,3)
    norm_coords = (coords / np.array([H - 1, W - 1, D - 1])) * 2.0 - 1.0
    return jnp.array(norm_coords), jnp.array(intens), jnp.array(labels)


def soft_dice_loss(probs: jnp.ndarray, onehot: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    # probs/onehot: (B,C)
    inter = jnp.sum(probs * onehot, axis=0)
    sums = jnp.sum(probs, axis=0) + jnp.sum(onehot, axis=0)
    dice = (2 * inter + eps) / (sums + eps)
    return 1.0 - jnp.mean(dice)


def loss_fn(params, coords, intensities, labels, k, rff_B, class_weights=None, dice_weight: float = 0.0):
    x = build_input(coords, intensities, k, rff_B)
    logits = apply_mlp(params, x)
    y = one_hot(labels, NUM_CLASSES)
    ce = optax.softmax_cross_entropy(logits, y)
    if class_weights is not None:
        w = jnp.take(class_weights, labels)
        ce = ce * w
    ce = jnp.mean(ce)
    if dice_weight > 0.0:
        probs = jax.nn.softmax(logits, axis=-1)
        dl = soft_dice_loss(probs, y)
        return (1.0 - dice_weight) * ce + dice_weight * dl
    return ce


def predict_volume(params, case_data: Dict[str, np.ndarray], k: int, chunk: int = 200_000, rff_B: Optional[jnp.ndarray] = None):
    mods = case_data["mods"]  # (M,H,W,D)
    seg_true = case_data["seg"]
    M, H, W, D = mods.shape
    # Optional light denoising to improve stability (signal processing)
    mods_proc = np.empty_like(mods)
    for m in range(M):
        # Apply a small Gaussian filter slice-wise to reduce spikes
        mods_proc[m] = gaussian_filter(mods[m], sigma=0.5)
    xs, ys, zs = np.arange(H), np.arange(W), np.arange(D)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)  # (N,3)
    intens = mods_proc.transpose(1, 2, 3, 0).reshape(-1, M)
    norm_coords = (grid / np.array([H - 1, W - 1, D - 1])) * 2.0 - 1.0
    preds = []
    for i in range(0, len(grid), chunk):
        c_chunk = jnp.array(norm_coords[i : i + chunk])
        f_chunk = jnp.array(intens[i : i + chunk])
        x_in = build_input(c_chunk, f_chunk, k, rff_B)
        logits = apply_mlp(params, x_in)
        cls = jnp.argmax(logits, axis=-1)
        preds.append(np.array(cls, dtype=np.int16))
    pred_flat = np.concatenate(preds, axis=0)
    pred_vol = pred_flat.reshape(H, W, D)
    return pred_vol, seg_true


def dice_score(pred: np.ndarray, true: np.ndarray, num_classes: int = NUM_CLASSES):
    scores = {}
    for c in range(num_classes):
        pred_c = pred == c
        true_c = true == c
        inter = (pred_c & true_c).sum()
        denom = pred_c.sum() + true_c.sum()
        dice = (2 * inter + 1e-6) / (denom + 1e-6) if denom > 0 else np.nan
        scores[c] = float(dice)
    return scores


def save_params_npz(params, path: str):
    flat = {}
    for i, layer in enumerate(params):
        flat[f"W_{i}"] = np.array(layer["W"])
        flat[f"b_{i}"] = np.array(layer["b"])
    np.savez_compressed(path, **flat)


def main():
    ap = argparse.ArgumentParser(description="Fourier INR (JAX) for BraTS 2023")
    ap.add_argument("--data-root", default="data/BraTS-2023")
    ap.add_argument("--case-limit", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8192, help="Global batch size")
    ap.add_argument("--micro-batch-size", type=int, default=None, help="Per-step micro batch (for gradient accumulation)")
    ap.add_argument("--fourier-freqs", type=int, default=16)
    ap.add_argument("--hidden", type=int, nargs="*", default=[256, 256, 256, 256])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-lr", type=float, default=2e-4, help="Cosine decay minimum learning rate")
    ap.add_argument("--warmup-steps", type=int, default=0, help="Linear warmup steps")
    ap.add_argument("--val-index", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=120_000)
    ap.add_argument("--out", default="artifacts/inr_brats23.npz")
    ap.add_argument("--resume", default=None, help="Path to .npz weights to resume training from")
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=0.3, help="Weight for soft-Dice in loss (0 means pure CE)")
    ap.add_argument("--class-weights", type=str, default=None, help="Comma-separated weights for classes 0..3, e.g., 0.2,1,1,1")
    ap.add_argument("--tumor-ratio", type=float, default=0.4, help="Fraction of tumour voxels in each micro-batch [0,1]")
    ap.add_argument("--rff-dim", type=int, default=0, help="Random Fourier feature dim (0 to disable)")
    ap.add_argument("--rff-sigma", type=float, default=0.0, help="Stddev for RFF Gaussian B; >0 enables RFF")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("JAX devices:", jax.devices())

    all_cases = find_cases(args.data_root)
    if len(all_cases) == 0:
        raise RuntimeError(f"No cases found under {args.data_root}")
    train_cases = all_cases[: args.case_limit]
    print(f"Using {len(train_cases)} training cases")

    train_cache = load_case_list(train_cases)
    # Use first case to infer dims
    M, H, W, D = train_cache[0]["mods"].shape
    print("Data dims (M,H,W,D)=", (M, H, W, D))

    # Dummy input to determine input dimension
    key = jax.random.PRNGKey(args.seed)
    coords, feats, _ = sample_from_single_case_np(key, train_cache[0], 2)
    # Random Fourier feature matrix (if enabled)
    rff_B = None
    if args.rff_sigma > 0.0 and args.rff_dim > 0:
        key, rkey = jax.random.split(key)
        rff_B = jax.random.normal(rkey, (coords.shape[-1], args.rff_dim)) * args.rff_sigma
        print(f"Using Random Fourier Features: dim={args.rff_dim}, sigma={args.rff_sigma}")

    if rff_B is not None:
        ff_dim = 2 * args.rff_dim
    else:
        ff_dim = coords.shape[-1] * 2 * args.fourier_freqs
    in_dim = coords.shape[-1] + feats.shape[-1] + ff_dim
    print("Input dimension:", in_dim)

    # Initialize or resume parameters
    if args.resume is not None and os.path.exists(args.resume):
        def load_params_npz(path: str):
            d = np.load(path)
            layers = []
            i = 0
            while f"W_{i}" in d:
                layers.append({"W": jnp.array(d[f"W_{i}"]), "b": jnp.array(d[f"b_{i}"])})
                i += 1
            if not layers:
                raise RuntimeError(f"No layers found in checkpoint {path}")
            return layers

        params = load_params_npz(args.resume)
        ckpt_in = int(params[0]["W"].shape[0])
        ckpt_out = int(params[-1]["W"].shape[1])
        if ckpt_in != int(in_dim) or ckpt_out != NUM_CLASSES:
            raise RuntimeError(
                f"Checkpoint shape mismatch. ckpt_in={ckpt_in}, ckpt_out={ckpt_out} vs current in_dim={in_dim}, num_classes={NUM_CLASSES}. "
                f"Ensure FOURIER_FREQS, modalities, and classes match."
            )
        print(f"Resumed parameters from {args.resume}")
    else:
        key, params = init_mlp(key, in_dim, args.hidden, NUM_CLASSES)
    total_params = sum(p["W"].size + p["b"].size for p in params)
    print("Total parameters:", total_params)

    # Optimizer with cosine decay, warmup, clipping
    if args.warmup_steps > 0:
        sched = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.lr,
            warmup_steps=args.warmup_steps,
            decay_steps=max(1, args.steps - args.warmup_steps),
            end_value=args.min_lr,
        )
    else:
        sched = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=max(1, args.steps), alpha=args.min_lr/args.lr)

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.clip_norm),
        optax.adamw(learning_rate=sched)
    )
    opt_state = optimizer.init(params)

    micro_bs = args.micro_batch_size or args.batch_size
    accum_steps = max(1, math.ceil(args.batch_size / micro_bs))
    micro_bs = int(math.ceil(args.batch_size / accum_steps))
    print(f"Microbatching configured: global={args.batch_size}, micro={micro_bs}, accum_steps={accum_steps}")

    # JIT a single-batch value_and_grad for reuse
    class_weights = None
    if args.class_weights is not None:
        cw = np.array([float(x) for x in args.class_weights.split(',')], dtype=np.float32)
        if cw.size != NUM_CLASSES:
            raise ValueError(f"--class-weights must have {NUM_CLASSES} values")
        class_weights = jnp.array(cw)

    def _loss_batch(p, c, f, y):
        return loss_fn(p, c, f, y, args.fourier_freqs, rff_B, class_weights, args.dice_weight)
    loss_and_grad = jax.jit(jax.value_and_grad(_loss_batch))

    def train_step_host(params, opt_state, rng_key):
        grads_acc = jax.tree_util.tree_map(jnp.zeros_like, params)
        loss_acc = 0.0
        key = rng_key
        for i in range(accum_steps):
            key, sub_case, sub = jax.random.split(key, 3)
            # pick one random case for this micro-step
            case_idx = int(jax.random.randint(sub_case, (), 0, len(train_cache)))
            # Enforce tumour/background mix per micro-batch via rejection sampling
            tumour_ratio = float(np.clip(args.tumor_ratio, 0.0, 1.0))
            tb = int(micro_bs * tumour_ratio)
            rb = micro_bs - tb
            case = train_cache[case_idx]
            M, H, W, D = case["mods"].shape
            # Background/uniform samples
            cx = np.array(jax.random.randint(sub, (rb,), 0, H));
            key, sy, sz = jax.random.split(key, 3)
            cy = np.array(jax.random.randint(sy, (rb,), 0, W));
            cz = np.array(jax.random.randint(sz, (rb,), 0, D));
            # Tumour-biased samples (rejection)
            txs, tys, tzs = [], [], []
            tries = 0
            while len(txs) < tb and tries < 20:
                tries += 1
                key, kx, ky, kz = jax.random.split(key, 4)
                xs = np.array(jax.random.randint(kx, (tb,), 0, H))
                ys = np.array(jax.random.randint(ky, (tb,), 0, W))
                zs = np.array(jax.random.randint(kz, (tb,), 0, D))
                mask = case["seg"][xs, ys, zs] > 0
                if mask.any():
                    txs.extend(xs[mask].tolist()); tys.extend(ys[mask].tolist()); tzs.extend(zs[mask].tolist())
            if len(txs) < tb:
                # fallback to fill remainder uniformly
                need = tb - len(txs)
                key, kx, ky, kz = jax.random.split(key, 4)
                txs += np.array(jax.random.randint(kx, (need,), 0, H)).tolist()
                tys += np.array(jax.random.randint(ky, (need,), 0, W)).tolist()
                tzs += np.array(jax.random.randint(kz, (need,), 0, D)).tolist()
            xs = np.concatenate([np.array(txs[:tb]), cx])
            ys = np.concatenate([np.array(tys[:tb]), cy])
            zs = np.concatenate([np.array(tzs[:tb]), cz])

            # Gather batch
            coords = np.stack([xs, ys, zs], axis=-1)
            norm_coords = (coords / np.array([H - 1, W - 1, D - 1])) * 2.0 - 1.0
            intens = case["mods"][:, xs, ys, zs].transpose(1, 0)
            labels = case["seg"][xs, ys, zs].astype(np.int32)
            c = jnp.array(norm_coords); f = jnp.array(intens); y = jnp.array(labels)
            loss_val, grads = loss_and_grad(params, c, f, y)
            grads_acc = jax.tree_util.tree_map(lambda a, b: a + b, grads_acc, grads)
            loss_acc += float(loss_val)
        grads_mean = jax.tree_util.tree_map(lambda x: x / accum_steps, grads_acc)
        updates, opt_state2 = optimizer.update(grads_mean, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss_acc / accum_steps, key

    # Warm-up
    key, step_key = jax.random.split(key)
    params, opt_state, warm_loss, key = train_step_host(params, opt_state, step_key)
    print(f"Warm-up loss: {float(warm_loss):.4f}")

    # Train
    start = time.time()
    for step in range(1, args.steps + 1):
        key, step_key = jax.random.split(key)
        params, opt_state, loss_val, key = train_step_host(params, opt_state, step_key)
        if step % 20 == 0 or step == 1:
            print(f"Step {step}/{args.steps} | loss={float(loss_val):.4f}")
    print("Training time: {:.2f}s".format(time.time() - start))

    # Save parameters
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_params_npz(params, args.out)
    print("Saved parameters ->", args.out)

    # Evaluate on validation case from training set (by index)
    val_case = train_cases[min(args.val_index, len(train_cases) - 1)]
    val_mods, val_seg = _load_case(val_case)
    pred_vol, true_vol = predict_volume(params, {"mods": val_mods, "seg": val_seg}, args.fourier_freqs, chunk=args.chunk, rff_B=rff_B)
    scores = dice_score(pred_vol, true_vol)
    print("Validation Dice:", scores)

    # Evaluate on first hold-out (if available)
    extra_cases = all_cases[args.case_limit :]
    if len(extra_cases) > 0:
        hold_mods, hold_seg = _load_case(extra_cases[0])
        hold_pred, hold_true = predict_volume(params, {"mods": hold_mods, "seg": hold_seg}, args.fourier_freqs, chunk=args.chunk, rff_B=rff_B)
        hold_scores = dice_score(hold_pred, hold_true)
        print("Hold-out Dice:", hold_scores)
    else:
        print("No hold-out cases beyond training subset.")


if __name__ == "__main__":
    main()
