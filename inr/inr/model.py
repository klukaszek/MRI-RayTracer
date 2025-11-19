import math
import json
import pathlib as _pl
from typing import Dict, Any, Tuple

import numpy as np
import jax
import jax.numpy as jnp


def fourier_features(coords: jnp.ndarray, k: int) -> jnp.ndarray:
    B, dim = coords.shape
    freqs = jnp.arange(1, k + 1)
    ang = coords[..., None] * freqs[None, None, :] * math.pi
    sin = jnp.sin(ang)
    cos = jnp.cos(ang)
    ff = jnp.concatenate([sin, cos], axis=-1).reshape(B, dim * 2 * k)
    return ff


def build_input(coords: jnp.ndarray, intensities: jnp.ndarray, fourier_freqs: int) -> jnp.ndarray:
    ff = fourier_features(coords, fourier_freqs)
    return jnp.concatenate([coords, ff, intensities], axis=-1)


def glorot(key, shape):
    fan_in, fan_out = shape[0], shape[1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def init_mlp(key, in_dim: int, hidden_dims, out_dim: int):
    params = []
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    for i in range(len(dims) - 1):
        key, k1, k2 = jax.random.split(key, 3)
        W = glorot(k1, (dims[i], dims[i + 1]))
        b = jnp.zeros((dims[i + 1],))
        params.append({"W": W, "b": b})
    return key, params


def apply_mlp(params, x: jnp.ndarray) -> jnp.ndarray:
    *hidden, last = params
    h = x
    for layer in hidden:
        h = jnp.dot(h, layer["W"]) + layer["b"]
        h = jax.nn.relu(h)
    out = jnp.dot(h, last["W"]) + last["b"]
    return out


def one_hot(labels, num_classes):
    return jax.nn.one_hot(labels, num_classes)


def soft_dice_per_class(probs, onehot, eps: float = 1e-6):
    inter = jnp.sum(probs * onehot, axis=0)
    sums = jnp.sum(probs, axis=0) + jnp.sum(onehot, axis=0)
    dice_k = (2 * inter + eps) / (sums + eps)
    return dice_k


def make_loss_and_grad(num_classes, class_weights, dice_weight: float, fourier_freqs: int):
    cw = jnp.array(class_weights)

    def loss_fn(params, coords, intensities, labels):
        x = build_input(coords, intensities, fourier_freqs)
        logits = apply_mlp(params, x)
        y = one_hot(labels, num_classes)
        ce_vec = jax.nn.log_softmax(logits, axis=-1)
        ce_vec = -jnp.sum(y * ce_vec, axis=-1)
        w = jnp.take(cw, labels)
        ce_scalar = (ce_vec * w).mean()

        probs = jax.nn.softmax(logits, axis=-1)
        dice_k = soft_dice_per_class(probs, y)
        if dice_weight > 0:
            dice_mean = dice_k.mean()
            loss = (1 - dice_weight) * ce_scalar + dice_weight * (1 - dice_mean)
        else:
            loss = ce_scalar

        counts = jnp.sum(y, axis=0)
        ce_sum_k = jnp.sum(ce_vec[:, None] * y, axis=0)
        ce_mean_k = ce_sum_k / jnp.maximum(counts, 1.0)
        aux = {"ce_per_class": ce_mean_k, "dice_per_class": dice_k}
        return loss, aux

    return jax.jit(jax.value_and_grad(loss_fn, has_aux=True))


def predict_slice(params, cache, vol_shape, z: int, fourier_freqs: int, case_index: int = 0):
    H, W, D = vol_shape
    xs = jnp.arange(H)
    ys = jnp.arange(W)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    x_flat = X.reshape(-1)
    y_flat = Y.reshape(-1)
    z_flat = jnp.full_like(x_flat, z)

    coords = jnp.stack([x_flat, y_flat, z_flat], axis=-1)
    norm_coords = (coords / jnp.array([H - 1, W - 1, D - 1])) * 2.0 - 1.0

    x_np = np.array(x_flat)
    y_np = np.array(y_flat)
    z_np = np.full(len(x_flat), z, dtype=np.int32)
    case_indices = np.full(len(x_flat), case_index, dtype=np.int32)

    intens_np, _ = cache.sample_voxels(case_indices, x_np, y_np, z_np)
    intens = jnp.array(intens_np)

    x_in = build_input(norm_coords, intens, fourier_freqs)
    logits = apply_mlp(params, x_in)
    pred = jnp.argmax(logits, axis=-1)
    return pred.reshape(H, W)


def predict_volume(params, case_data: Dict[str, Any], fourier_freqs: int, chunk: int = 200000):
    mods = case_data["mods"]
    seg_true = case_data["seg"]
    M, H, W, D = mods.shape

    xs, ys, zs = np.arange(H), np.arange(W), np.arange(D)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    intens = mods.transpose(1, 2, 3, 0).reshape(-1, M)

    norm_coords = (grid / np.array([H - 1, W - 1, D - 1])) * 2.0 - 1.0

    preds = []
    for i in range(0, len(grid), chunk):
        c_chunk = jnp.array(norm_coords[i : i + chunk])
        f_chunk = jnp.array(intens[i : i + chunk])
        x_in = build_input(c_chunk, f_chunk, fourier_freqs)
        logits = apply_mlp(params, x_in)
        cls = jnp.argmax(logits, axis=-1)
        preds.append(np.array(cls, dtype=np.int16))

    pred_flat = np.concatenate(preds, axis=0)
    pred_vol = pred_flat.reshape(H, W, D)
    return pred_vol, seg_true


def dice_score(pred, true, num_classes: int):
    scores = {}
    for c in range(num_classes):
        pred_c = pred == c
        true_c = true == c
        inter = (pred_c & true_c).sum()
        denom = pred_c.sum() + true_c.sum()
        dice = (2 * inter + 1e-6) / (denom + 1e-6) if denom > 0 else np.nan
        scores[c] = dice
    return scores


def coverage_dice(pred, true):
    pred_any = pred > 0
    true_any = true > 0
    inter = (pred_any & true_any).sum()
    denom = pred_any.sum() + true_any.sum()
    return (2 * inter + 1e-6) / (denom + 1e-6) if denom > 0 else 0.0


def hausdorff_distance(pred, true, spacing=(1.0, 1.0, 1.0), num_classes: int = 4):
    from scipy.spatial import cKDTree

    H, W, D = true.shape
    coords = np.stack(np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing="ij"), axis=-1)
    coords = coords.astype(np.float32)
    coords[..., 0] *= spacing[0]
    coords[..., 1] *= spacing[1]
    coords[..., 2] *= spacing[2]

    hd_per_class = {}
    for c in range(num_classes):
        pred_mask = pred == c
        true_mask = true == c

        if not pred_mask.any() or not true_mask.any():
            hd_per_class[c] = np.nan
            continue

        pred_pts = coords[pred_mask]
        true_pts = coords[true_mask]

        tree_true = cKDTree(true_pts)
        tree_pred = cKDTree(pred_pts)

        dists_pred_true, _ = tree_true.query(pred_pts, k=1)
        dists_true_pred, _ = tree_pred.query(true_pts, k=1)

        hd = float(max(dists_pred_true.max(), dists_true_pred.max()))
        hd_per_class[c] = hd

    return hd_per_class


def evaluate_single_case(case_idx: int, case_data: Dict[str, Any], params, num_classes: int, fourier_freqs: int):
    pred_vol, true_vol = predict_volume(params, case_data, fourier_freqs, chunk=120000)
    scores = dice_score(pred_vol, true_vol, num_classes)
    hausdorff_scores = hausdorff_distance(pred_vol, true_vol, num_classes=num_classes)
    cov_dice = coverage_dice(pred_vol, true_vol)
    valid_scores = [s for s in scores.values() if not np.isnan(s)]
    mean_dice = float(np.mean(valid_scores)) if valid_scores else 0.0
    return {
        "case_idx": case_idx,
        "pred_vol": pred_vol,
        "true_vol": true_vol,
        "case_data": case_data,
        "class_scores": scores,
        "coverage_dice": cov_dice,
        "mean_dice": mean_dice,
        "hausdorff_scores": hausdorff_scores,
    }


def model_load(
    npz_path: str | _pl.Path,
    config_override: Dict[str, Any] | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load INR model parameters and associated config from disk.

    This helper allows notebooks or scripts to load a trained model
    directly from a checkpoint without re-running the training cells
    to recreate `params` and `config`.

    Expectations:
      - `npz_path` points to an `.npz` checkpoint file that contains
        at least one of the following keys:
            * "params": the saved model parameters
            * otherwise, if the file has a single array, that array is
              treated as the parameters (fallback behavior).
      - Next to the NPZ file there is a JSON config file with the
        same stem, for example:
            "inr_final_step000001.npz" -> "inr_final_step000001.json".

    Args:
        npz_path: Path to the NPZ checkpoint file.
        config_override: Optional mapping whose entries override or
            extend the loaded config.

    Returns:
        A tuple `(params, config)` where `params` are ready to pass to
        `predict_volume` and `config` provides metadata such as
        `FOURIER_FREQS`, `NUM_CLASSES`, etc.
    """

    npz_path = _pl.Path(npz_path).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    # Expect a sidecar JSON named like "{checkpoint}_info.json"
    # e.g., "inr_final_step000001.npz" -> "inr_final_step000001_info.json".
    cfg_path = npz_path.with_name(f"{npz_path.stem}_info.json")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config JSON not found next to NPZ: {cfg_path}")

    # Load params from NPZ; allow pickled or raw arrays
    npz = np.load(str(npz_path), allow_pickle=True)
    if "params" in npz.files:
        arr = npz["params"]
        if arr.dtype == object:
            # Accept scalar or length-1 object arrays
            if arr.ndim == 0 or arr.size == 1:
                params = arr.item()
            else:
                raise ValueError(
                    f"'params' in {npz_path} is an object array with shape {arr.shape}; "
                    "expected a single serialized object."
                )
        else:
            params = arr
    else:
        # Fallback: if a single key exists, treat its value as params
        if len(npz.files) == 1:
            key = npz.files[0]
            arr = npz[key]
            if arr.dtype == object and (arr.ndim == 0 or arr.size == 1):
                params = arr.item()
            elif arr.dtype == object:
                raise ValueError(
                    f"Single array '{key}' in {npz_path} is object with shape {arr.shape}, "
                    "expected a single serialized object."
                )
            else:
                params = arr
        else:
            raise KeyError(
                f"Could not find 'params' key in {npz_path}; "
                f"available keys: {list(npz.files)}"
            )

    # Load JSON config stored alongside the checkpoint
    with cfg_path.open("r") as f:
        config = json.load(f)

    # Apply optional overrides (e.g., different DATA_ROOT during testing)
    if config_override is not None:
        config = {**config, **config_override}

    return params, config


