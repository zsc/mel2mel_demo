#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image


def _stable_rand01(*, seed: int, sample_id: str) -> float:
    h = hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _ensure_empty_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _list_images_flat(dir_path: Path, *, exts: tuple[str, ...]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(dir_path.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") not in exts:
            continue
        out[p.stem] = p
    return out


def _symlink_or_copy(src: Path, dst: Path, *, mode: Literal["symlink", "copy"]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        dst.write_bytes(src.read_bytes())
    else:
        raise ValueError(f"unsupported mode={mode!r}")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def _load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _extract_column_features_gray(
    img: Image.Image,
    *,
    pool_h: int,
) -> np.ndarray:
    """
    Return time-major features shaped (W, F) from a mel-spectrum PNG.

    We use grayscale intensities and average-pool along the frequency axis
    (height) to keep DTW fast and robust.
    """
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0  # (H, W)
    h, w = gray.shape
    feats = gray.T  # (W, H)
    if pool_h <= 1:
        return np.ascontiguousarray(feats, dtype=np.float32)
    if h < pool_h:
        raise ValueError(f"image too short for pool_h={pool_h}: H={h}")
    h2 = h - (h % pool_h)
    if h2 <= 0:
        raise ValueError(f"invalid pooled height: H={h}, pool_h={pool_h}")
    feats = feats[:, :h2].reshape(w, h2 // pool_h, pool_h).mean(axis=2)
    return np.ascontiguousarray(feats, dtype=np.float32)


def _warp_source_rgb_to_target_width(
    src_rgb: Image.Image,
    *,
    alignment_path: np.ndarray,
    target_width: int,
    mel2mel_root: Path,
) -> Image.Image:
    sys.path.insert(0, str(mel2mel_root / "src"))
    from alignment.dtw import warp_source_to_target  # noqa: E402

    src_arr = np.asarray(src_rgb, dtype=np.float32)  # (H, W, 3)
    if src_arr.ndim != 3 or src_arr.shape[2] != 3:
        raise ValueError(f"expected RGB array, got shape={src_arr.shape!r}")
    h, w, c = src_arr.shape
    src_tm = np.transpose(src_arr, (1, 0, 2)).reshape(w, h * c)  # (W, H*C)
    aligned_tm = warp_source_to_target(src_tm, alignment_path, target_len=target_width)  # (W_tgt, H*C)
    aligned = aligned_tm.reshape(target_width, h, c).transpose(1, 0, 2)  # (H, W_tgt, 3)
    aligned_u8 = np.clip(np.rint(aligned), 0, 255).astype(np.uint8)
    return Image.fromarray(aligned_u8, mode="RGB")


@dataclass(frozen=True)
class PairTask:
    sample_id: str
    src_path: Path
    tgt_path: Path
    out_src_path: Path
    tgt_w: int
    tgt_h: int
    ratio: float
    use_dtw: bool


def _align_one(task: PairTask, *, mel2mel_root: Path, metric: str, band_radius: int | None, pool_h: int) -> dict[str, Any]:
    sys.path.insert(0, str(mel2mel_root / "src"))
    from alignment.dtw import dtw_path  # noqa: E402

    # Use a temp file then atomic rename to avoid partial outputs.
    # Keep a non-image extension so training scans won't pick up leftovers.
    tmp_out = task.out_src_path.with_suffix(task.out_src_path.suffix + ".tmp")

    src_rgb = _load_rgb(task.src_path)
    tgt_rgb = _load_rgb(task.tgt_path)

    if src_rgb.size[1] != tgt_rgb.size[1]:
        raise ValueError(f"height mismatch: src={src_rgb.size}, tgt={tgt_rgb.size}")
    if tgt_rgb.size[0] != task.tgt_w or tgt_rgb.size[1] != task.tgt_h:
        raise ValueError("target size changed during task execution")

    src_feats = _extract_column_features_gray(src_rgb, pool_h=pool_h)
    tgt_feats = _extract_column_features_gray(tgt_rgb, pool_h=pool_h)
    alignment = dtw_path(src_feats, tgt_feats, metric=metric, band_radius=band_radius)

    aligned_img = _warp_source_rgb_to_target_width(
        src_rgb,
        alignment_path=alignment.path,
        target_width=task.tgt_w,
        mel2mel_root=mel2mel_root,
    )
    task.out_src_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_img.save(tmp_out, format="PNG")
    tmp_out.replace(task.out_src_path)

    return {
        "id": task.sample_id,
        "use_dtw": True,
        "dtw_cost": float(alignment.cost),
        "metric": str(alignment.metric),
        "band_radius": alignment.band_radius,
        "num_source_frames": int(alignment.num_source_frames),
        "num_target_frames": int(alignment.num_target_frames),
    }


def _iter_tasks(
    src_dir: Path,
    tgt_dir: Path,
    out_src_dir: Path,
    *,
    exts: tuple[str, ...],
    seed: int,
    p_dtw: float,
    ratio_min: float,
    ratio_max: float,
    max_width: int | None,
    max_samples: int | None,
    max_dtw: int | None,
) -> tuple[list[PairTask], list[dict[str, Any]]]:
    src_map = _list_images_flat(src_dir, exts=exts)
    tgt_map = _list_images_flat(tgt_dir, exts=exts)
    common_ids = sorted(set(src_map.keys()) & set(tgt_map.keys()))

    skipped: list[dict[str, Any]] = []
    tasks: list[PairTask] = []

    dtw_budget = max_dtw
    for sample_id in common_ids:
        if max_samples is not None and len(tasks) >= max_samples:
            break
        sp = src_map[sample_id]
        tp = tgt_map[sample_id]
        try:
            with Image.open(sp) as im_s:
                src_w, src_h = im_s.size
            with Image.open(tp) as im_t:
                tgt_w, tgt_h = im_t.size
        except Exception as e:
            skipped.append({"id": sample_id, "reason": "open_failed", "error": str(e)})
            continue
        if src_w <= 0 or tgt_w <= 0:
            skipped.append({"id": sample_id, "reason": "bad_width", "src_w": src_w, "tgt_w": tgt_w})
            continue
        if src_h != tgt_h:
            skipped.append({"id": sample_id, "reason": "height_mismatch", "src_h": src_h, "tgt_h": tgt_h})
            continue
        ratio = float(tgt_w) / float(src_w)
        if ratio < ratio_min or ratio > ratio_max:
            skipped.append(
                {"id": sample_id, "reason": "ratio_outlier", "src_w": src_w, "tgt_w": tgt_w, "ratio": ratio}
            )
            continue
        if max_width is not None and (src_w > max_width or tgt_w > max_width):
            skipped.append(
                {"id": sample_id, "reason": "width_too_large", "src_w": src_w, "tgt_w": tgt_w, "ratio": ratio}
            )
            continue

        use_dtw = _stable_rand01(seed=seed, sample_id=sample_id) < p_dtw
        if use_dtw and dtw_budget is not None:
            if dtw_budget <= 0:
                use_dtw = False
            else:
                dtw_budget -= 1

        tasks.append(
            PairTask(
                sample_id=sample_id,
                src_path=sp,
                tgt_path=tp,
                out_src_path=out_src_dir / f"{sample_id}{sp.suffix.lower()}",
                tgt_w=tgt_w,
                tgt_h=tgt_h,
                ratio=ratio,
                use_dtw=use_dtw,
            )
        )

    skipped.append({"reason": "summary", "num_common_ids": len(common_ids), "num_tasks": len(tasks)})
    return tasks, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Build mel-spectrum A/B PNG pairs with optional DTW time-axis alignment.")
    parser.add_argument("--src-dir", type=Path, required=True, help="Source dir (A): samples10k_lab_mel_png")
    parser.add_argument("--tgt-dir", type=Path, required=True, help="Target dir (B): samples10k_mel_png")
    parser.add_argument("--out-src-dir", type=Path, required=True, help="Output source dir (mixed aligned/symlinked).")
    parser.add_argument("--out-tgt-dir", type=Path, required=True, help="Output target dir (symlinked + prompt .txt).")
    parser.add_argument("--prompt", type=str, default="mel-spectrum", help='Prompt text (default: "mel-spectrum").')
    parser.add_argument("--seed", type=int, default=42, help="Seed for DTW selection (deterministic per-id).")
    parser.add_argument("--p-dtw", type=float, default=0.5, help="Probability to DTW-align a sample.")
    parser.add_argument("--metric", type=str, default="l2", choices=("l2", "l1", "cosine"), help="DTW frame metric.")
    parser.add_argument("--band-radius", type=int, default=32, help="DTW band radius (frames). Use -1 for no band.")
    parser.add_argument("--feature-pool-h", type=int, default=4, help="Average pooling along height for DTW features.")
    parser.add_argument("--ratio-min", type=float, default=0.5, help="Min allowed tgt/src width ratio; outliers are skipped.")
    parser.add_argument("--ratio-max", type=float, default=2.0, help="Max allowed tgt/src width ratio; outliers are skipped.")
    parser.add_argument("--max-width", type=int, default=2048, help="Skip pairs if either width exceeds this. Use -1 to disable.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for processed samples (debug).")
    parser.add_argument("--max-dtw", type=int, default=None, help="Optional cap for DTW-aligned samples (debug).")
    parser.add_argument("--copy-mode", type=str, default="symlink", choices=("symlink", "copy"), help="How to handle non-DTW src.")
    parser.add_argument("--jobs", type=int, default=0, help="DTW worker processes (0/1 = sequential).")
    parser.add_argument("--mel2mel-root", type=Path, default=Path("/root/autodl-tmp/mel2mel_demo"), help="Path to mel2mel_demo repo.")
    args = parser.parse_args()

    exts = ("png", "jpg", "jpeg", "webp")
    if args.band_radius == -1:
        band_radius = None
    else:
        band_radius = int(args.band_radius)
        if band_radius < 0:
            raise SystemExit("--band-radius must be >= -1")

    if not (0.0 <= args.p_dtw <= 1.0):
        raise SystemExit("--p-dtw must be in [0, 1]")
    if args.feature_pool_h < 1:
        raise SystemExit("--feature-pool-h must be >= 1")
    if args.ratio_min <= 0 or args.ratio_max <= 0 or args.ratio_min > args.ratio_max:
        raise SystemExit("--ratio-min/--ratio-max invalid")
    if args.max_width == -1:
        max_width = None
    else:
        max_width = int(args.max_width)
        if max_width <= 0:
            raise SystemExit("--max-width must be > 0 or -1")

    mel2mel_root = Path(args.mel2mel_root)
    if not (mel2mel_root / "src" / "alignment" / "dtw.py").exists():
        raise SystemExit(f"mel2mel_root does not look valid: {mel2mel_root}")

    _ensure_empty_dir(args.out_src_dir)
    _ensure_empty_dir(args.out_tgt_dir)
    meta_dir = args.out_tgt_dir / "__meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tasks, skipped = _iter_tasks(
        args.src_dir,
        args.tgt_dir,
        args.out_src_dir,
        exts=exts,
        seed=args.seed,
        p_dtw=args.p_dtw,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        max_width=max_width,
        max_samples=args.max_samples,
        max_dtw=args.max_dtw,
    )

    num_dtw = sum(1 for t in tasks if t.use_dtw)
    num_no = len(tasks) - num_dtw
    print(f"paired: {len(tasks)} (dtw={num_dtw}, no_dtw={num_no}), skipped={len(skipped)-1}")

    # Always materialize target symlinks + prompts first.
    for t in tasks:
        _symlink_or_copy(t.tgt_path, args.out_tgt_dir / t.tgt_path.name, mode="symlink")
        _write_text(args.out_tgt_dir / f"{t.sample_id}.txt", args.prompt)

    manifest_path = meta_dir / "manifest.jsonl"
    skipped_path = meta_dir / "skipped.jsonl"
    config_path = meta_dir / "config.json"

    cfg = {
        "src_dir": str(args.src_dir),
        "tgt_dir": str(args.tgt_dir),
        "out_src_dir": str(args.out_src_dir),
        "out_tgt_dir": str(args.out_tgt_dir),
        "prompt": args.prompt,
        "seed": int(args.seed),
        "p_dtw": float(args.p_dtw),
        "metric": str(args.metric),
        "band_radius": band_radius,
        "feature_pool_h": int(args.feature_pool_h),
        "ratio_min": float(args.ratio_min),
        "ratio_max": float(args.ratio_max),
        "max_width": max_width,
        "max_samples": args.max_samples,
        "max_dtw": args.max_dtw,
        "copy_mode": str(args.copy_mode),
        "jobs": int(args.jobs),
        "mel2mel_root": str(mel2mel_root),
    }
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Process samples: DTW-align selected ones; otherwise symlink/copy.
    kept_meta: list[dict[str, Any]] = []
    dtw_errors = 0

    def build_base_meta(t: PairTask) -> dict[str, Any]:
        return {
            "id": t.sample_id,
            "src_path": str(t.src_path),
            "tgt_path": str(t.tgt_path),
            "out_src_path": str(t.out_src_path),
            "tgt_w": int(t.tgt_w),
            "tgt_h": int(t.tgt_h),
            "ratio": float(t.ratio),
            "prompt": args.prompt,
        }

    # Fast path: materialize all non-DTW sources immediately.
    dtw_tasks = [t for t in tasks if t.use_dtw]
    no_dtw_tasks = [t for t in tasks if not t.use_dtw]
    for t in no_dtw_tasks:
        _symlink_or_copy(t.src_path, t.out_src_path, mode=args.copy_mode)
        m = build_base_meta(t)
        m.update({"use_dtw": False})
        kept_meta.append(m)

    if args.jobs is None or int(args.jobs) <= 1 or len(dtw_tasks) == 0:
        # Sequential DTW.
        for idx, t in enumerate(dtw_tasks, 1):
            if idx % 200 == 0 or idx == len(dtw_tasks):
                print(f"[dtw {idx}/{len(dtw_tasks)}] ...")
            base_meta = build_base_meta(t)
            try:
                dtw_meta = _align_one(
                    t,
                    mel2mel_root=mel2mel_root,
                    metric=args.metric,
                    band_radius=band_radius,
                    pool_h=int(args.feature_pool_h),
                )
                base_meta.update(dtw_meta)
                kept_meta.append(base_meta)
            except Exception as e:
                dtw_errors += 1
                _symlink_or_copy(t.src_path, t.out_src_path, mode=args.copy_mode)
                base_meta.update({"use_dtw": False, "dtw_error": str(e)})
                kept_meta.append(base_meta)
    else:
        # Parallel DTW workers.
        jobs = int(args.jobs)
        print(f"dtw: running with jobs={jobs}")
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fut_to_task = {
                ex.submit(
                    _align_one,
                    t,
                    mel2mel_root=mel2mel_root,
                    metric=args.metric,
                    band_radius=band_radius,
                    pool_h=int(args.feature_pool_h),
                ): t
                for t in dtw_tasks
            }
            done = 0
            for fut in as_completed(fut_to_task):
                t = fut_to_task[fut]
                done += 1
                if done % 200 == 0 or done == len(fut_to_task):
                    print(f"[dtw {done}/{len(fut_to_task)}] ...")
                base_meta = build_base_meta(t)
                try:
                    dtw_meta = fut.result()
                    base_meta.update(dtw_meta)
                    kept_meta.append(base_meta)
                except Exception as e:
                    dtw_errors += 1
                    _symlink_or_copy(t.src_path, t.out_src_path, mode=args.copy_mode)
                    base_meta.update({"use_dtw": False, "dtw_error": str(e)})
                    kept_meta.append(base_meta)

    kept_meta.sort(key=lambda d: d.get("id", ""))

    with skipped_path.open("w", encoding="utf-8") as f:
        for item in skipped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in kept_meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"done. kept={len(kept_meta)}, dtw_errors={dtw_errors}")
    print(f"meta: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
