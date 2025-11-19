import pathlib
from typing import List, Tuple, Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
import nibabel as nib


MODALITY_SUFFIXES = ["t1n", "t1c", "t2w", "t2f"]
SEG_SUFFIX = "seg"


def find_cases(root: pathlib.Path) -> List[pathlib.Path]:
    cases: List[pathlib.Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            if any((p / f"{p.name}-{m}.nii.gz").exists() for m in MODALITY_SUFFIXES):
                cases.append(p)
    return cases


def load_case(case_dir: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    base = case_dir.name
    mods = []
    for suf in MODALITY_SUFFIXES:
        fp = case_dir / f"{base}-{suf}.nii.gz"
        img = nib.load(str(fp))
        arr = img.get_fdata().astype(np.float32)
        mask = arr != 0
        if mask.any():
            mu = arr[mask].mean()
            sigma = arr[mask].std() + 1e-6
            arr = (arr - mu) / sigma
        mods.append(arr)
    seg_fp = case_dir / f"{base}-{SEG_SUFFIX}.nii.gz"
    seg = nib.load(str(seg_fp)).get_fdata().astype(np.int16)
    mods_arr = np.stack(mods, axis=0)
    return mods_arr, seg


def load_mu_glioma_manifest(manifest_path: pathlib.Path):
    """Load MU-Glioma-Post manifest as a pandas DataFrame.

    The manifest is expected to contain at least `case_id` and `relative_path`
    columns, where `relative_path` points to the case directory relative to
    the manifest's parent folder.
    """
    import pandas as pd

    manifest_path = pathlib.Path(manifest_path)
    df = pd.read_csv(manifest_path)
    return df


def load_mu_glioma_case(case_dir: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single MU-Glioma-Post case.

    This currently reuses the BraTS-style `load_case` logic, assuming the same
    file naming convention (base name + modality suffixes + seg suffix).
    """
    return load_case(case_dir)


class StreamingBraTSCache:
    def __init__(self, case_paths, name: str = "cache"):
        self.case_paths = list(case_paths)
        self.name = name
        self.n_cases = len(self.case_paths)
        self.cache: List[Dict[str, Any]] = []

        print(f"Building {name} cache: {self.n_cases} cases...")
        for i, cp in enumerate(self.case_paths):
            if i % 20 == 0 and i > 0:
                print(f"  Loaded {i}/{self.n_cases}...")
            mods, seg = load_case(cp)
            self.cache.append({"mods": mods, "seg": seg})

        self.vol_shape = self.cache[0]["mods"].shape[1:]
        self.n_modalities = self.cache[0]["mods"].shape[0]

        bytes_per_case = self.cache[0]["mods"].nbytes + self.cache[0]["seg"].nbytes
        total_gb = (bytes_per_case * self.n_cases) / 1e9
        print(f"{name} complete: {self.n_cases} cases, {total_gb:.2f} GB")

    def sample_voxels(self, case_indices, h_coords, w_coords, d_coords):
        N = len(case_indices)
        M = self.n_modalities
        mods_out = np.zeros((N, M), dtype=np.float32)
        segs_out = np.zeros(N, dtype=np.int16)
        for i in range(N):
            c_idx = case_indices[i]
            h, w, d = h_coords[i], w_coords[i], d_coords[i]
            mods_out[i] = self.cache[c_idx]["mods"][:, h, w, d]
            segs_out[i] = self.cache[c_idx]["seg"][h, w, d]
        return mods_out, segs_out


def build_train_val_caches(
    data_root: pathlib.Path,
    case_limit: int,
    num_folds: int,
    fold_index: int,
    rng_seed: int,
):
    all_cases_full = find_cases(data_root)
    subset_cases = all_cases_full[:case_limit]
    print("Total discovered:", len(all_cases_full), "Subset used:", len(subset_cases))

    rng = np.random.default_rng(rng_seed)
    shuffled = list(subset_cases)
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, num_folds)
    assert 0 <= fold_index < len(folds), "FOLD_INDEX out of range"
    val_cases = list(folds[fold_index])
    train_cases = [c for i, f in enumerate(folds) if i != fold_index for c in f]
    print(f"Fold sizes: {[len(f) for f in folds]} | Train={len(train_cases)} Val={len(val_cases)}")

    train_cache = StreamingBraTSCache(train_cases, name="train")
    val_cache = StreamingBraTSCache(val_cases, name="val") if val_cases else None

    vol_shape = train_cache.vol_shape

    info = {
        "all_cases_full": all_cases_full,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "folds": folds,
    }
    return train_cache, val_cache, vol_shape, info


def sample_batch(rng_key, batch_size: int, cache: StreamingBraTSCache, vol_shape):
    H, W, D = vol_shape
    key_case, key_x, key_y, key_z = jax.random.split(rng_key, 4)

    ci = jax.random.randint(key_case, (batch_size,), 0, cache.n_cases)
    xs = jax.random.randint(key_x, (batch_size,), 0, H)
    ys = jax.random.randint(key_y, (batch_size,), 0, W)
    zs = jax.random.randint(key_z, (batch_size,), 0, D)

    ci_np = np.array(ci)
    xs_np = np.array(xs)
    ys_np = np.array(ys)
    zs_np = np.array(zs)

    intens_np, labels_np = cache.sample_voxels(ci_np, xs_np, ys_np, zs_np)

    coords = jnp.stack([xs, ys, zs], axis=-1)
    norm_coords = (coords / jnp.array([H - 1, W - 1, D - 1])) * 2.0 - 1.0

    intens = jnp.array(intens_np)
    labels = jnp.array(labels_np, dtype=jnp.int32)

    return norm_coords, intens, labels
