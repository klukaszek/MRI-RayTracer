#!/usr/bin/env python3
"""
Validate docs/experiments.csv schema and basic value ranges.
"""
from __future__ import annotations
import csv
import os
import sys

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "experiments.csv")

REQUIRED_HEADERS = [
    "scene",
    "view_id",
    "alpha",
    "baseline",
    "policy_variant",
    "psnr",
    "ssim",
    "ms_per_frame",
    "rays_per_sec",
    "steps_per_ray_p50",
    "steps_per_ray_p90",
    "notes",
]

def main() -> int:
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] experiments.csv not found at {CSV_PATH}")
        return 1

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("[ERROR] CSV has no header row")
            return 2

        missing = [h for h in REQUIRED_HEADERS if h not in reader.fieldnames]
        if missing:
            print("[ERROR] CSV missing required headers:", ", ".join(missing))
            return 2

        rownum = 1
        errs = 0
        for row in reader:
            rownum += 1
            try:
                alpha = float(row["alpha"]) if row["alpha"] else 0.0
                if not (0.0 <= alpha <= 2.0):
                    print(f"[ERROR] row {rownum}: alpha out of range [0,2]")
                    errs += 1
                psnr = float(row["psnr"]) if row["psnr"] else 0.0
                if psnr <= 0 or psnr > 100:
                    print(f"[ERROR] row {rownum}: psnr suspicious: {psnr}")
                ssim = float(row["ssim"]) if row["ssim"] else 0.0
                if not (0.0 <= ssim <= 1.0):
                    print(f"[ERROR] row {rownum}: ssim out of [0,1]: {ssim}")
                    errs += 1
                ms = float(row["ms_per_frame"]) if row["ms_per_frame"] else 0.0
                if ms <= 0:
                    print(f"[ERROR] row {rownum}: ms_per_frame must be > 0")
                    errs += 1
                rps = float(row["rays_per_sec"]) if row["rays_per_sec"] else 0.0
                if rps < 0:
                    print(f"[ERROR] row {rownum}: rays_per_sec negative")
                    errs += 1
                p50 = int(float(row["steps_per_ray_p50"])) if row["steps_per_ray_p50"] else 0
                p90 = int(float(row["steps_per_ray_p90"])) if row["steps_per_ray_p90"] else 0
                if p50 < 0 or p90 < 0 or p90 < p50:
                    print(f"[ERROR] row {rownum}: invalid steps per ray p50/p90")
                    errs += 1
            except Exception as e:
                print(f"[ERROR] row {rownum}: {e}")
                errs += 1

        if errs:
            print(f"Validation finished with {errs} error(s)")
            return 3

    print("experiments.csv looks good ✔")
    return 0

if __name__ == "__main__":
    sys.exit(main())

