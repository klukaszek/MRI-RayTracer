#!/usr/bin/env python
# coding: utf-8

# # nnU-Net v2: Train and Export Weights (BraTS23 example)
# 
# This notebook prepares a dataset in nnU-Net format, runs planning + preprocessing, trains a model, and exports
# the trained weights to a zip. You can then use the checkpoint for INR distillation.
# 
# Notes:
# - Ensure this kernel uses the project virtual environment.
# - Set the GPU/compute configuration as needed. On macOS without CUDA, nnU-Net will use MPS if available.
# - BraTS23 labels are 0,1,2,4. We remap 4→3 to make labels continuous.
# 

# ### BraTS-2023 naming fix
# 
# Your BraTS-2023 files appear named as `-t1n`, `-t1c`, `-t2f`, `-t2w`, and `-seg` (e.g., `BraTS-GLI-XXXXX-XXX-t1n.nii.gz`).
# This cell overrides the conversion utilities to use those suffixes and re-generates `Dataset900_BraTS2023`.

# In[ ]:


# Environment and path setup: set nnU-Net directories BEFORE importing nnunetv2
import os, sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent  # assumes notebook is run from 'notebooks' directory
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'nnUNet_raw'
PP_DIR = DATA_DIR / 'nnUNet_preprocessed'
RES_DIR = DATA_DIR / 'nnUNet_results'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'

RAW_DIR.mkdir(parents=True, exist_ok=True)
PP_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ['nnUNet_raw'] = str(RAW_DIR)
os.environ['nnUNet_preprocessed'] = str(PP_DIR)
os.environ['nnUNet_results'] = str(RES_DIR)

# Prefer local nnUNet sources if present (so docs & APIs line up)
LOCAL_NNUNET = PROJECT_ROOT / 'nnUNet'
if LOCAL_NNUNET.exists():
    sys.path.insert(0, str(LOCAL_NNUNET))
    print('Using local nnUNet source:', LOCAL_NNUNET)
else:
    print('Using installed nnunetv2 package')

print('nnUNet_raw       =', os.environ['nnUNet_raw'])
print('nnUNet_preprocessed =', os.environ['nnUNet_preprocessed'])
print('nnUNet_results   =', os.environ['nnUNet_results'])


# ## Convert BraTS23 to nnU-Net format
# 
# We search in `data/BraTS-2023` for training cases and construct an nnU-Net dataset:
# - Dataset ID/Name: `Dataset900_BraTS2023` (custom)
# - Channels: FLAIR, T1w, T1ce, T2w → `_0000.._0003`
# - Labels: remap 4 → 3 to ensure continuous labels 0..3
# - Writes `dataset.json`, `imagesTr/`, `labelsTr/`
# 
# If your raw dataset is already in nnU-Net format, skip this cell and set `DATASET_ID` and `DATASET_NAME` accordingly.

# In[ ]:


import json
import shutil
import nibabel as nib
import numpy as np
from glob import glob

# Configure dataset source and target
BRATS23_ROOT = DATA_DIR / 'BraTS-2023'  # adjust if your location is different
DATASET_ID = 900
DATASET_NAME = f'Dataset{DATASET_ID:03d}_BraTS2023'
DS_RAW = RAW_DIR / DATASET_NAME
IMAGES_TR = DS_RAW / 'imagesTr'
LABELS_TR = DS_RAW / 'labelsTr'

IMAGES_TR.mkdir(parents=True, exist_ok=True)
LABELS_TR.mkdir(parents=True, exist_ok=True)

def find_cases_brats23(root: Path):
    # Accept common patterns; adjust if your structure differs
    case_dirs = []
    if (root / 'BraTS-GLI-00000-000').exists():
        # flat dir with cases
        case_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith('BraTS-')])
    else:
        # recursive
        case_dirs = sorted([Path(p).parent for p in glob(str(root / '**/*_flair.nii*'), recursive=True)])
    return case_dirs

def has_all_modalities(case_dir: Path):
    flair = list(case_dir.glob('*_flair.nii*'))
    t1    = list(case_dir.glob('*_t1.nii*'))
    t1ce  = list(case_dir.glob('*_t1ce.nii*'))
    t2    = list(case_dir.glob('*_t2.nii*'))
    seg   = list(case_dir.glob('*_seg.nii*'))
    return len(flair)==1 and len(t1)==1 and len(t1ce)==1 and len(t2)==1 and len(seg)==1

def case_id_from_dir(case_dir: Path):
    # Use folder name as case_id, strip non-alnum for safety
    return case_dir.name

def remap_segmentation_to_continuous(src_seg: Path, dst_seg: Path):
    img = nib.load(str(src_seg))
    data = img.get_fdata().astype(np.int16)
    # BraTS labels are typically 0,1,2,4 -> map 4->3
    if (data==4).any():
        data[data==4] = 3
    # ensure int dtype
    out = nib.Nifti1Image(data.astype(np.int16), img.affine, img.header)
    nib.save(out, str(dst_seg))

def dataset_exists(ds_raw: Path) -> bool:
    if not ds_raw.exists():
        return False
    ds_json = ds_raw / 'dataset.json'
    images_tr = ds_raw / 'imagesTr'
    labels_tr = ds_raw / 'labelsTr'
    if not (ds_json.exists() and images_tr.exists() and labels_tr.exists()):
        return False
    imgs = list(images_tr.glob('*.nii*'))
    labs = list(labels_tr.glob('*.nii*'))
    # Expect at least one label and roughly 4x images for 4 modalities
    if len(labs) == 0:
        return False
    if len(imgs) < 4 * len(labs):
        return False
    return True

def prepare_brats23_dataset():
    # Early exit if dataset already prepared
    if dataset_exists(DS_RAW):
        print('Dataset already exists at:', DS_RAW)
        print('Skipping creation/filtering.')
        return

    cases = []
    for cdir in find_cases_brats23(BRATS23_ROOT):
        if not has_all_modalities(cdir):
            continue
        cid = case_id_from_dir(cdir)
        cases.append((cid, cdir))
    print(f'Found {len(cases)} cases')

    for cid, cdir in cases:
        # channel order: FLAIR, T1, T1ce, T2
        mapping = {
            '_flair': 0,
            '_t1': 1,
            '_t1ce': 2,
            '_t2': 3,
        }
        for suffix, ch in mapping.items():
            src = list(cdir.glob(f'*{suffix}.nii*'))[0]
            dst = IMAGES_TR / f'{cid}_{ch:04d}.nii.gz'
            if not dst.exists():
                # Use symlink to save space; fallback to copy if needed
                try:
                    os.symlink(src, dst)
                except Exception:
                    shutil.copy2(src, dst)
        # labels
        src_seg = list(cdir.glob('*_seg.nii*'))[0]
        dst_seg = LABELS_TR / f'{cid}.nii.gz'
        if not dst_seg.exists():
            remap_segmentation_to_continuous(src_seg, dst_seg)

    # dataset.json
    dataset = {
        'name': 'BraTS2023',
        'description': 'BraTS 2023 converted to nnU-Net format',
        'reference': 'https://www.synapse.org/#!Synapse:syn51156910/wiki/',
        'licence': 'see original dataset licence',
        'release': '1.0',
        'modality': {
            '0': 'FLAIR', '1': 'T1w', '2': 'T1gd', '3': 'T2w'
        },
        'labels': {
            'background': 0, 'edema': 1, 'non_enhancing': 2, 'enhancing': 3
        },
        'numTraining': len(list(LABELS_TR.glob('*.nii*'))),
        'file_ending': '.nii.gz'
    }
    with open(DS_RAW / 'dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    print('Wrote', DS_RAW / 'dataset.json')

prepare_brats23_dataset()
print('Raw dataset prepared at:', DS_RAW)


# In[ ]:


# Override helpers for BraTS-2023 filename scheme (-t1n, -t1c, -t2f, -t2w, -seg)
import os, shutil, json
import nibabel as nib
import numpy as np
from pathlib import Path
from glob import glob

def find_cases_brats23(root: Path):
    # Flat directories named BraTS-GLI-... exist in your tree
    case_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith('BraTS-')])
    if not case_dirs:
        # fallback to recursive search
        case_dirs = sorted([Path(p).parent for p in glob(str(root / '**/*-t1n.nii*'), recursive=True)])
    return case_dirs

def has_all_modalities(case_dir: Path):
    t2f = list(case_dir.glob('*-t2f.nii*'))  # FLAIR
    t1n = list(case_dir.glob('*-t1n.nii*'))  # T1 native
    t1c = list(case_dir.glob('*-t1c.nii*'))  # T1 contrast
    t2w = list(case_dir.glob('*-t2w.nii*'))  # T2 weighted
    seg = list(case_dir.glob('*-seg.nii*'))
    return len(t2f)==1 and len(t1n)==1 and len(t1c)==1 and len(t2w)==1 and len(seg)==1

def case_id_from_dir(case_dir: Path):
    return case_dir.name

def remap_segmentation_to_continuous(src_seg: Path, dst_seg: Path):
    img = nib.load(str(src_seg))
    data = img.get_fdata().astype(np.int16)
    if (data==4).any():
        data[data==4] = 3
    out = nib.Nifti1Image(data.astype(np.int16), img.affine, img.header)
    nib.save(out, str(dst_seg))

def prepare_brats23_dataset():
    cases = []
    for cdir in find_cases_brats23(BRATS23_ROOT):
        if not has_all_modalities(cdir):
            continue
        cid = case_id_from_dir(cdir)
        cases.append((cid, cdir))
    print(f'Found {len(cases)} cases')
    if not cases:
        # help debug
        print('No cases found. Check BRATS23_ROOT:', BRATS23_ROOT)
        print('Sample entries under root:')
        for p in list(BRATS23_ROOT.iterdir())[:5]:
            print(' ', p)
        return

    # ensure clean imagesTr/labelsTr exist
    if IMAGES_TR.exists():
        shutil.rmtree(IMAGES_TR)
    if LABELS_TR.exists():
        shutil.rmtree(LABELS_TR)
    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)

    # mapping: channel index -> suffix
    mapping = {0: '-t2f', 1: '-t1n', 2: '-t1c', 3: '-t2w'}

    for cid, cdir in cases:
        for ch, suffix in mapping.items():
            src_list = list(cdir.glob(f'*{suffix}.nii*'))
            assert len(src_list)==1, (cid, suffix, src_list)
            src = src_list[0]
            dst = IMAGES_TR / f'{cid}_{ch:04d}.nii.gz'
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                except Exception:
                    shutil.copy2(src, dst)
        # labels
        src_seg = list(cdir.glob('*-seg.nii*'))[0]
        dst_seg = LABELS_TR / f'{cid}.nii.gz'
        remap_segmentation_to_continuous(src_seg, dst_seg)

    dataset = {
        'name': 'BraTS2023',
        'description': 'BraTS 2023 converted to nnU-Net format',
        'reference': 'https://www.synapse.org/#!Synapse:syn51156910/wiki/',
        'licence': 'see original dataset licence',
        'release': '1.0',
        'modality': {'0': 'FLAIR', '1': 'T1w', '2': 'T1gd', '3': 'T2w'},
        'labels': {'background': 0, 'edema': 1, 'non_enhancing': 2, 'enhancing': 3},
        'numTraining': len(list(LABELS_TR.glob('*.nii*'))),
        'file_ending': '.nii.gz'
    }
    with open(DS_RAW / 'dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    print('Wrote', DS_RAW / 'dataset.json')
    print('imagesTr count:', len(list(IMAGES_TR.glob('*.nii*'))))
    print('labelsTr count:', len(list(LABELS_TR.glob('*.nii*'))))

# Rebuild with corrected mapping
prepare_brats23_dataset()
print('Raw dataset prepared at:', DS_RAW)


# ## Plan and preprocess
# 
# We run the standard pipeline via nnU-Net v2 APIs. You can choose configurations; for a first pass `3d_fullres`
# is typical for BraTS-sized volumes. Adjust process counts based on your CPU.

# In[ ]:


from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprints, plan_experiments, preprocess
)

# 1) Fingerprint (optionally verify integrity)
extract_fingerprints([DATASET_ID], check_dataset_integrity=False, clean=True, verbose=True)

# 2) Plan experiments (returns plans identifier string)
plans_identifier = plan_experiments([DATASET_ID])
print('Using plans:', plans_identifier)

# 3) Preprocess selected configurations
configs = ('3d_fullres',)
preprocess([DATASET_ID], plans_identifier=plans_identifier, configurations=configs, num_processes=(4,), verbose=False)
print('Preprocessing complete')


# ## Train
# 
# Train one fold. For a quick smoke test, set `FAST_TRAINING=True` to use a short trainer variant.
# For a proper run, use the default `nnUNetTrainer` and more epochs.

# In[ ]:


from nnunetv2.run.run_training import run_training

CONFIG = '3d_fullres'
FOLD = 0  # 0..4 or 'all'
FAST_TRAINING = True  # set False for real training
trainer_name = 'nnUNetTrainer_5epochs' if FAST_TRAINING else 'nnUNetTrainer'

# Use string dataset identifier to avoid AttributeError in get_trainer_from_args
dataset_arg = DATASET_NAME  # or use str(DATASET_ID)
print('Training with dataset_arg =', dataset_arg)

# Choose device: CUDA if available, else MPS (Apple), else CPU
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print('Using device:', device)

run_training(
    dataset_arg,
    CONFIG,
    FOLD,
    trainer_class_name=trainer_name,
    plans_identifier=plans_identifier,
    num_gpus=1,
    export_validation_probabilities=False,
    continue_training=False,
    only_run_validation=False,
    disable_checkpointing=False,
    val_with_best=False,
    device=device,
)
print('Training complete')


# ## Export trained model to zip
# 
# This bundles `plans.json`, folds, checkpoints (e.g., `checkpoint_final.pth`), and validation summaries.
# You can extract the checkpoint for INR distillation or downstream use.

# In[ ]:


from datetime import datetime
from nnunetv2.model_sharing.model_export import export_pretrained_model

export_path = ARTIFACTS_DIR / f'{DATASET_NAME}_{CONFIG}_fold{FOLD}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip'
export_pretrained_model(
    DATASET_ID,
    str(export_path),
    configurations=(CONFIG,),
    trainer=trainer_name,
    plans_identifier=plans_identifier,
    folds=(FOLD,),
    strict=False,
    save_checkpoints=('checkpoint_final.pth',),
    export_crossval_predictions=False,
)
print('Exported to:', export_path)


# ## Locate checkpoint for INR distillation
# 
# Finds the saved `checkpoint_final.pth` for the trained fold.

# In[ ]:


from nnunetv2.utilities.file_path_utilities import get_output_folder
from os.path import join

out_dir = get_output_folder(f'Dataset{DATASET_ID:03d}_BraTS2023', trainer_name, plans_identifier, CONFIG)
ckpt_path = Path(join(out_dir, f'fold_{FOLD}', 'checkpoint_final.pth'))
print('Checkpoint:', ckpt_path)
print('Exists:', ckpt_path.exists())


# ## (Optional) Quick inference sanity check
# 
# Run a single-case prediction to validate the pipeline. Requires raw/test data prepared similarly.
# Skip on first run if time-constrained.

# In[ ]:


# Example (disabled by default)
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True)
# pred.initialize_from_trained_model_folder(
#     str(Path(out_dir) / f'fold_{FOLD}'), use_folds=(FOLD,), checkpoint_name='checkpoint_final.pth'
# )
# pred.predict_from_files(...)
pass

