#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from alignment.dtw import dtw_warp_mel  # noqa: E402


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DTW-align a source mel to a target mel, exporting warped mel and alignment metadata."
    )
    parser.add_argument("--source-mel", type=Path, required=True, help="Path to source mel .npy")
    parser.add_argument("--target-mel", type=Path, required=True, help="Path to target mel .npy")
    parser.add_argument("--out-aligned-mel", type=Path, required=True, help="Output path for aligned source mel .npy")
    parser.add_argument("--out-meta", type=Path, required=True, help="Output path for alignment metadata .json")
    parser.add_argument(
        "--out-path-npz",
        type=Path,
        default=None,
        help="Optional output path for the DTW path as .npz (contains `path` and `cost`).",
    )
    parser.add_argument(
        "--mel-time-axis",
        type=int,
        default=1,
        choices=(0, 1),
        help="Time axis in mel arrays: 1 for (n_mels, T) [default], 0 for (T, n_mels).",
    )
    parser.add_argument("--metric", type=str, default="l2", choices=("l2", "l1", "cosine"), help="Frame distance metric.")
    parser.add_argument(
        "--band-radius",
        type=int,
        default=None,
        help="Optional DTW band radius (frames) around the scaled diagonal; smaller is faster but riskier.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting output files.")
    args = parser.parse_args()

    for p in (args.out_aligned_mel, args.out_meta, args.out_path_npz):
        if p is None:
            continue
        if p.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {p} (pass --overwrite)")

    source_mel = np.load(args.source_mel)
    target_mel = np.load(args.target_mel)

    aligned_mel, alignment = dtw_warp_mel(
        source_mel,
        target_mel,
        mel_time_axis=args.mel_time_axis,
        metric=args.metric,
        band_radius=args.band_radius,
    )

    args.out_aligned_mel.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_aligned_mel, aligned_mel.astype(np.float32))

    path_npz_rel = None
    if args.out_path_npz is not None:
        args.out_path_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_path_npz, path=alignment.path, cost=np.asarray(alignment.cost, dtype=np.float32))
        try:
            path_npz_rel = str(args.out_path_npz.relative_to(REPO_ROOT))
        except ValueError:
            path_npz_rel = str(args.out_path_npz)

    meta = {
        "use_dtw": True,
        "source_mel_path": str(args.source_mel),
        "target_mel_path": str(args.target_mel),
        "aligned_source_mel_path": str(args.out_aligned_mel),
        "mel_time_axis": int(args.mel_time_axis),
        "metric": str(alignment.metric),
        "band_radius": alignment.band_radius,
        "dtw_cost": float(alignment.cost),
        "num_source_frames": int(alignment.num_source_frames),
        "num_target_frames": int(alignment.num_target_frames),
        "path_npz_path": path_npz_rel,
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
    }
    _write_json(args.out_meta, meta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

