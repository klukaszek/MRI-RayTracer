import os
import json
import time
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
import wandb

from .dataloader import build_train_val_caches, sample_batch
from .model import init_mlp, build_input, make_loss_and_grad, predict_slice, evaluate_single_case


def train_inr(config: Dict[str, Any], use_wandb: bool = True, resume_from: str | pathlib.Path | None = None):
    data_root = pathlib.Path(config["DATA_ROOT"])
    case_limit = int(config["CASE_LIMIT"])
    num_folds = int(config["NUM_FOLDS"])
    fold_index = int(config["FOLD_INDEX"])
    global_batch_size = int(config["GLOBAL_BATCH_SIZE"])
    micro_batch_size = int(config["MICRO_BATCH_SIZE"])
    fourier_freqs = int(config["FOURIER_FREQS"])
    hidden_dims = list(config["HIDDEN_DIMS"])
    lr = float(config["LR"])
    min_lr = float(config["MIN_LR"])
    warmup_steps = int(config["WARMUP_STEPS"])
    train_steps = int(config["TRAIN_STEPS"])
    rng_seed = int(config["RNG_SEED"])
    num_classes = int(config["NUM_CLASSES"])
    dice_weight = float(config["DICE_WEIGHT"])
    class_weights = list(config["CLASS_WEIGHTS"])
    clip_norm = float(config["CLIP_NORM"])
    optimizer_choice = config.get("OPTIMIZER_CHOICE", "adamw")

    jax_key = jax.random.PRNGKey(rng_seed)

    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    train_cache, val_cache, vol_shape, info = build_train_val_caches(
        data_root=data_root,
        case_limit=case_limit,
        num_folds=num_folds,
        fold_index=fold_index,
        rng_seed=rng_seed,
    )

    H, W, D = vol_shape

    all_cases_full = info["all_cases_full"]
    train_cases = info["train_cases"]
    val_cases = info["val_cases"]
    folds = info["folds"]

    accum_steps = int((global_batch_size + micro_batch_size - 1) // micro_batch_size)

    log_config = dict(config)
    log_config.update(
        {
            "accum_steps": accum_steps,
            "volume_shape": vol_shape,
            "num_modalities": train_cache.n_modalities,
            "total_cases": len(all_cases_full),
            "train_cases": len(train_cases),
            "val_cases": len(val_cases),
            "fold_sizes": [len(f) for f in folds],
        }
    )

    run = None
    save_path = None
    if use_wandb:
        run = wandb.init(
            project=config.get("WANDB_PROJECT", "brats-inr-segmentation"),
            entity=config.get("WANDB_ENTITY"),
            name=config.get("WANDB_RUN_NAME"),
            config=log_config,
            tags=config.get("WANDB_TAGS"),
            notes=config.get("WANDB_NOTES"),
        )
        save_path = pathlib.Path(f"./artifacts/brats-inr-segmentation/{run.name}/")
    else:
        save_path = pathlib.Path("./artifacts/brats-inr-segmentation/offline/")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    training_json_path = save_path / "training_config_and_results.json"
    initial_payload = {
        "config": log_config,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(training_json_path, "w", encoding="utf-8") as f:
        json.dump(initial_payload, f, indent=2)

    jax_key, sample_key = jax.random.split(jax_key)
    coords_s, feats_s, _ = sample_batch(sample_key, 4, train_cache, vol_shape)
    in_dim = build_input(coords_s, feats_s, fourier_freqs).shape[-1]

    jax_key, params_key = jax.random.split(jax_key)
    jax_key, params = init_mlp(params_key, in_dim, hidden_dims, num_classes)

    sum_params = sum(p["W"].size + p["b"].size for p in params)
    print("Total parameters:", sum_params)
    if run is not None:
        run.config.update({"input_dim": in_dim, "total_parameters": sum_params})
        run.summary["model_parameters"] = sum_params

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=max(1, train_steps - warmup_steps),
        end_value=min_lr,
    )

    if optimizer_choice == "adamw":
        optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(schedule))
    else:
        optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(schedule))

    opt_state = optimizer.init(params)
    loss_and_grad = make_loss_and_grad(num_classes, class_weights, dice_weight, fourier_freqs)

    def microbatch_step(local_params, local_opt_state, rng_key_inner):
        grads_acc = [
            {"W": jnp.zeros_like(p["W"]), "b": jnp.zeros_like(p["b"])} for p in local_params
        ]
        loss_acc = 0.0
        ce_pc_acc = jnp.zeros((num_classes,))
        dice_pc_acc = jnp.zeros((num_classes,))
        key_inner = rng_key_inner
        for _ in range(accum_steps):
            key_inner, sub = jax.random.split(key_inner)
            coords, feats, labels = sample_batch(sub, micro_batch_size, train_cache, vol_shape)
            (loss_val, aux), grads = loss_and_grad(local_params, coords, feats, labels)
            loss_acc += float(loss_val)
            ce_pc_acc = ce_pc_acc + aux["ce_per_class"]
            dice_pc_acc = dice_pc_acc + aux["dice_per_class"]
            grads_acc = [
                {"W": ga["W"] + g["W"], "b": ga["b"] + g["b"]}
                for ga, g in zip(grads_acc, grads)
            ]
        grads_mean = [
            {"W": g["W"] / accum_steps, "b": g["b"] / accum_steps} for g in grads_acc
        ]
        updates, local_opt_state = optimizer.update(grads_mean, local_opt_state, local_params)
        local_params = optax.apply_updates(local_params, updates)
        aux_mean = {
            "ce_per_class": ce_pc_acc / accum_steps,
            "dice_per_class": dice_pc_acc / accum_steps,
        }
        return local_params, local_opt_state, loss_acc / accum_steps, aux_mean

    loss_history = []
    dice_history = [[] for _ in range(num_classes)]
    ce_history = [[] for _ in range(num_classes)]

    start_step = 0
    if resume_from is not None:
        ckpt_path = pathlib.Path(resume_from)
        if ckpt_path.is_file():
            npz = np.load(str(ckpt_path), allow_pickle=True)
            if "params" in npz.files:
                arr = npz["params"]
                if arr.dtype == object and (arr.ndim == 0 or arr.size == 1):
                    params = arr.item()
                else:
                    params = arr
            else:
                layer_keys = sorted({int(k.split("_")[1]) for k in npz.files if k.startswith("W_")})
                loaded = []
                for i in layer_keys:
                    W = npz[f"W_{i}"]
                    b = npz[f"b_{i}"]
                    loaded.append({"W": jnp.array(W), "b": jnp.array(b)})
                params = loaded
            print(f"Resuming training from checkpoint {ckpt_path}")
        else:
            print(f"Warning: resume_from path not found: {ckpt_path}")

    best_val_dice = None
    best_step = None

    best_json_path = save_path / "best_results.json"
    checkpoint_best_path = save_path / "checkpoint_best.npz"
    checkpoint_periodic_basename = "checkpoint_step{step:06d}.npz"
    checkpoint_every_steps = int(config.get("CHECKPOINT_EVERY_STEPS", 200))

    start = time.time()
    mid_z = D // 2
    vis_cache = val_cache if val_cache else train_cache
    vis_case_index = 0

    for step in range(1, train_steps + 1):
        jax_key, step_key = jax.random.split(jax_key)
        params, opt_state, loss_val, aux = microbatch_step(params, opt_state, step_key)
        loss_history.append(float(loss_val))
        dice_k = aux["dice_per_class"]
        ce_k = aux["ce_per_class"]

        metrics = {"train/loss": float(loss_val), "train/step": step}
        for k in range(num_classes):
            dice_history[k].append(float(dice_k[k]))
            ce_history[k].append(float(ce_k[k]))
            metrics[f"train/dice_class_{k}"] = float(dice_k[k])
            metrics[f"train/ce_class_{k}"] = float(ce_k[k])
        metrics["train/dice_mean"] = float(dice_k.mean())
        metrics["train/ce_mean"] = float(ce_k.mean())

        if run is not None:
            wandb.log(metrics, step=step)

        if step % checkpoint_every_steps == 0:
            flat_params_step = {}
            for i, layer in enumerate(params):
                flat_params_step[f"W_{i}"] = np.array(layer["W"])
                flat_params_step[f"b_{i}"] = np.array(layer["b"])
            ckpt_path = save_path / checkpoint_periodic_basename.format(step=step)
            np.savez_compressed(ckpt_path, **flat_params_step)
            print(f"Saved periodic checkpoint to {ckpt_path}")

        if step % max(train_steps // 10, 1) == 0:
            pred_slice = predict_slice(
                params,
                cache=vis_cache,
                vol_shape=vol_shape,
                z=mid_z,
                fourier_freqs=fourier_freqs,
                case_index=vis_case_index,
            )
            print(
                f"Step {step}/{train_steps} loss={loss_val:.4f} dice_mean={float(dice_k.mean()):.4f}",
                f"slice_pred_shape={np.array(pred_slice).shape}",
            )

    training_time = time.time() - start
    print(f"Training time: {training_time:.2f}s")
    if run is not None:
        run.summary["training_time_seconds"] = training_time

    state = {
        "params": params,
        "train_cache": train_cache,
        "val_cache": val_cache,
        "vol_shape": vol_shape,
        "loss_history": loss_history,
        "dice_history": dice_history,
        "ce_history": ce_history,
        "best_val_dice": best_val_dice,
        "best_step": best_step,
        "save_path": save_path,
        "training_json_path": training_json_path,
        "checkpoint_best_path": checkpoint_best_path,
        "checkpoint_periodic_basename": checkpoint_periodic_basename,
    }
    return params, state


def evaluate_inr(params, state, config: Dict[str, Any], use_wandb: bool = True):
    num_classes = int(config["NUM_CLASSES"])
    fourier_freqs = int(config["FOURIER_FREQS"])
    train_steps = int(config["TRAIN_STEPS"])

    train_cache = state["train_cache"]
    val_cache = state["val_cache"] or train_cache
    save_path = state["save_path"]
    training_json_path = state["training_json_path"]
    checkpoint_best_path = state["checkpoint_best_path"]
    checkpoint_periodic_basename = state["checkpoint_periodic_basename"]

    if val_cache and val_cache.n_cases > 0:
        eval_cache = val_cache
        eval_set = "validation"
        n_cases = val_cache.n_cases
    else:
        eval_cache = train_cache
        eval_set = "training"
        n_cases = min(5, train_cache.n_cases)

    print(f"Evaluating {n_cases} {eval_set} cases in parallel...")

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                evaluate_single_case,
                idx,
                eval_cache.cache[idx],
                params,
                num_classes,
                fourier_freqs,
            ): idx
            for idx in range(n_cases)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    results.sort(key=lambda x: x["case_idx"])

    all_dice_scores = {c: [] for c in range(num_classes)}
    all_coverage_dice = []
    case_mean_dice = []
    all_hausdorff_scores = {c: [] for c in range(num_classes)}

    for result in results:
        scores = result["class_scores"]
        cov_dice = result["coverage_dice"]
        mean_dice = result["mean_dice"]
        hd_scores = result["hausdorff_scores"]
        for c, score in scores.items():
            all_dice_scores[c].append(score)
        for c, score in hd_scores.items():
            all_hausdorff_scores[c].append(score)
        all_coverage_dice.append(cov_dice)
        case_mean_dice.append(mean_dice)

    mean_dice_per_class = {}
    mean_hausdorff_per_class = {}
    for c in range(num_classes):
        valid_scores = [s for s in all_dice_scores[c] if not np.isnan(s)]
        if valid_scores:
            mean_dice_per_class[c] = float(np.mean(valid_scores))
        else:
            mean_dice_per_class[c] = 0.0
        valid_hd = [h for h in all_hausdorff_scores[c] if not np.isnan(h)]
        if valid_hd:
            mean_hausdorff_per_class[c] = float(np.mean(valid_hd))
        else:
            mean_hausdorff_per_class[c] = float("nan")

    overall_mean_dice = float(np.mean(case_mean_dice)) if case_mean_dice else 0.0
    overall_coverage_dice = float(np.mean(all_coverage_dice)) if all_coverage_dice else 0.0

    best_val_dice = state.get("best_val_dice") or overall_mean_dice
    best_step = state.get("best_step") or train_steps

    try:
        with open(training_json_path, "r", encoding="utf-8") as f:
            training_payload = json.load(f)
    except FileNotFoundError:
        training_payload = {}

    training_payload.setdefault("results", {})
    training_payload["results"].update(
        {
            "final_step": int(train_steps),
            "final_loss": float(state["loss_history"][-1]) if state["loss_history"] else None,
            "val_dice_mean": overall_mean_dice,
            "val_dice_per_class": {int(c): float(v) for c, v in mean_dice_per_class.items()},
            "val_coverage_dice": overall_coverage_dice,
            "val_hausdorff_per_class": {
                int(c): (float(v) if not np.isnan(v) else None)
                for c, v in mean_hausdorff_per_class.items()
            },
            "num_eval_cases": int(n_cases),
            "best_val_dice": float(best_val_dice),
            "best_step": int(best_step),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )

    # Save a final, consolidated checkpoint that matches the `model_load`
    # convention: `{name}.npz` for parameters and `{name}_info.json` for
    # the associated metadata/config.

    # Derive a run name from the wandb run (if any) or fall back to a timestamp.
    if use_wandb and wandb.run is not None and wandb.run.name:
        run_name = wandb.run.name
    else:
        run_name = f"offline_run_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"

    final_npz_name = f"{run_name}.npz"
    final_info_name = f"{run_name}_info.json"
    final_npz_path = save_path / final_npz_name
    final_info_path = save_path / final_info_name

    # Pack params in a simple structure that `model_load` can consume.
    # Store as a single 0-D object array so `.item()` works reliably.
    # `params` is a Python list of layer dicts, so we wrap it in a
    # scalar object array directly instead of reshaping an array of
    # length N to shape ().
    params_array = np.empty((), dtype=object)
    params_array[()] = params
    flat_params_final = {"params": params_array}
    np.savez_compressed(final_npz_path, **flat_params_final)

    # The info JSON is a compact, run-specific view that pairs with
    # the NPZ for easy loading in notebooks.
    info_payload = {
        "config": config,
        "results": {
            "val_dice_mean": overall_mean_dice,
            "val_dice_per_class": {int(c): float(v) for c, v in mean_dice_per_class.items()},
            "val_coverage_dice": overall_coverage_dice,
            "val_hausdorff_per_class": {
                int(c): (float(v) if not np.isnan(v) else None)
                for c, v in mean_hausdorff_per_class.items()
            },
            "num_eval_cases": int(n_cases),
            "best_val_dice": float(best_val_dice),
            "best_step": int(best_step),
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(final_info_path, "w", encoding="utf-8") as f_info:
        json.dump(info_payload, f_info, indent=2)

    training_payload.setdefault("artifacts", {})
    training_payload["artifacts"].update(
        {
            "final_model": final_npz_name,
            "final_model_info": final_info_name,
            "checkpoint_best": checkpoint_best_path.name,
            "checkpoint_periodic_pattern": checkpoint_periodic_basename,
        }
    )

    with open(training_json_path, "w", encoding="utf-8") as f:
        json.dump(training_payload, f, indent=2)

    metrics = {
        **{f"val/dice_class_{c}": mean_dice_per_class[c] for c in range(num_classes)},
        "val/dice_mean": overall_mean_dice,
        "val/coverage_dice": overall_coverage_dice,
        **{f"val/hd_class_{c}": mean_hausdorff_per_class[c] for c in range(num_classes)},
    }

    if use_wandb and wandb.run is not None:
        wandb.log(metrics)
        wandb.run.summary.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        # Log the final model NPZ and its sidecar JSON as a W&B artifact
        try:
            artifact = wandb.Artifact(name=f"{run_name}-final-model", type="model")
            artifact.add_file(str(final_npz_path), name=final_npz_name)
            artifact.add_file(str(final_info_path), name=final_info_name)
            wandb.run.log_artifact(artifact)
        except Exception as e:
            # Don't fail evaluation if artifact logging has issues
            print(f"Warning: failed to log W&B artifact: {e}")

        # Also log the best checkpoint (if it exists) as a separate artifact
        try:
            if checkpoint_best_path.is_file():
                best_artifact = wandb.Artifact(
                    name=f"{run_name}-best-model",
                    type="model",
                )
                best_artifact.add_file(str(checkpoint_best_path), name=checkpoint_best_path.name)
                # Attach the best_results.json if present for metadata.
                best_json_path = save_path / "best_results.json"
                if best_json_path.is_file():
                    best_artifact.add_file(str(best_json_path), name=best_json_path.name)
                wandb.run.log_artifact(best_artifact)
        except Exception as e:
            print(f"Warning: failed to log W&B best-model artifact: {e}")

    return metrics, {"training_json_path": training_json_path, "save_path": save_path}

