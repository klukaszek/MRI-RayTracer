import numpy as np
import argparse

ap = argparse.ArgumentParser()
path = ap.add_argument("--npz", default="artifacts/brats-inr-segmentation/quiet-snowball-221/quiet-snowball-221.npz")
args = ap.parse_args()
path = args.npz

def summarize_tree(tree, prefix=""):
    """
    Recursively walk nested dicts / lists / tuples and numpy arrays.
    Prints key paths + shape/dtype for arrays, without values.
    """
    if isinstance(tree, dict):
      for k, v in tree.items():
          new_prefix = f"{prefix}.{k}" if prefix else str(k)
          summarize_tree(v, new_prefix)
    elif isinstance(tree, (list, tuple)):
      for i, v in enumerate(tree):
          new_prefix = f"{prefix}[{i}]"
          summarize_tree(v, new_prefix)
    else:
      # Leaf node: print info if ndarray, else just type
      if isinstance(tree, np.ndarray):
          print(f"{prefix}: ndarray, shape={tree.shape}, dtype={tree.dtype}")
      else:
          print(f"{prefix}: {type(tree)}")

with np.load(path, allow_pickle=True) as data:
  print("Top-level keys in npz file:")
  for key in data.files:
      arr = data[key]
      print(f"- {key}: type={type(arr)}", end="")
      if isinstance(arr, np.ndarray):
          print(f", shape={arr.shape}, dtype={arr.dtype}")
      else:
          print()

  print("\nInspecting contents of 'params' (if present):")
  for key in data.files:
      if key == "params":
          params_obj = data[key]
          # Often this is a 0-d or 1-element object array containing a dict/tree.
          if isinstance(params_obj, np.ndarray) and params_obj.dtype == object:
              # Try common cases: scalar object array or array with a single element
              if params_obj.shape == ():
                  tree = params_obj.item()
              elif params_obj.size == 1:
                  tree = params_obj.reshape(()).item()
              else:
                  # If it’s a bigger object array, just walk it as-is
                  tree = params_obj
          else:
              tree = params_obj

          print("Nested keys under 'params':")
          summarize_tree(tree)
          break
  else:
      print("No top-level key named 'params' was found.")
