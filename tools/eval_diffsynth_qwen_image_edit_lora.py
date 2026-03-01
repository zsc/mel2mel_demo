#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch

from fid_torchvision import compute_fid


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_split_ids(dataset_dir: Path, split: str) -> set[str] | None:
    if split == "all":
        return None
    split_path = dataset_dir / "split.json"
    if not split_path.exists():
        return None
    obj = json.loads(split_path.read_text(encoding="utf-8"))
    return set(obj.get(split, []))


def _crop_resize_like_training(image: Image.Image, *, max_pixels: int) -> Image.Image:
    # Mirror DiffSynth ImageCropAndResize (dynamic size, division_factor=16).
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f"bad image size: {image.size}")

    if w * h > max_pixels:
        scale = (w * h / max_pixels) ** 0.5
        h = int(h / scale)
        w = int(w / scale)

    # floor to division factor (16)
    h = (h // 16) * 16
    w = (w // 16) * 16
    h = max(h, 16)
    w = max(w, 16)

    # Resize then center crop to (h, w)
    ow, oh = image.size
    scale = max(w / ow, h / oh)
    rh = int(round(oh * scale))
    rw = int(round(ow * scale))
    resized = image.resize((rw, rh), resample=Image.BILINEAR)
    left = max(0, (rw - w) // 2)
    top = max(0, (rh - h) // 2)
    return resized.crop((left, top, left + w, top + h))


def _crop_resize_to(image: Image.Image, *, target_w: int, target_h: int) -> Image.Image:
    ow, oh = image.size
    if ow <= 0 or oh <= 0:
        raise ValueError(f"bad image size: {image.size}")
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"bad target size: {(target_w, target_h)}")
    scale = max(target_w / ow, target_h / oh)
    rw = int(round(ow * scale))
    rh = int(round(oh * scale))
    resized = image.resize((rw, rh), resample=Image.BILINEAR)
    left = max(0, (rw - target_w) // 2)
    top = max(0, (rh - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))

def _to_float_rgb(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a - b)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a DiffSynth Qwen-Image-Edit LoRA checkpoint on a dataset.")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Prepared dataset dir (qwen_edit_demo prepare output).")
    parser.add_argument("--lora", type=Path, required=True, help="LoRA checkpoint (.safetensors), e.g. step-2000.safetensors")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for eval artifacts.")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "all"), help="Which split to sample from.")
    parser.add_argument("--num_samples", type=int, default=8, help="How many samples to run.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling + inference seed.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Diffusion steps for inference.")
    parser.add_argument("--max_pixels", type=int, default=1048576, help="Max pixels (match training).")
    parser.add_argument(
        "--low_vram",
        type=int,
        default=1,
        choices=(0, 1),
        help="Use DiffSynth VRAM management (recommended if training is running on the same GPU).",
    )
    parser.add_argument("--compute_fid", type=int, default=1, choices=(0, 1), help="Compute FID on generated preds.")
    parser.add_argument("--fid_device", type=str, default="cuda", help="Device for Inception FID feature extraction.")
    parser.add_argument("--fid_batch_size", type=int, default=32, help="Batch size for Inception feature extraction.")
    args = parser.parse_args()

    ds_root = Path("/root/autodl-tmp/DiffSynth-Studio")
    if not ds_root.exists():
        raise SystemExit(f"DiffSynth-Studio not found at {ds_root}")
    sys.path.insert(0, str(ds_root))

    # Ensure offline/local model loading.
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/.autodl-model/data")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig  # noqa: E402

    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = args.dataset_dir / "metadata.diffsynth.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"missing: {meta_path}")

    rows = _load_jsonl(meta_path)
    split_ids = _load_split_ids(args.dataset_dir, args.split)
    if split_ids is not None:
        rows = [r for r in rows if Path(r["image"]).stem in split_ids]
    if not rows:
        raise SystemExit("no samples after split filtering")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.num_samples]

    float8 = getattr(torch, "float8_e4m3fn", torch.float16)
    vram_config = None
    if args.low_vram:
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": float8,
            "onload_device": "cpu",
            "preparing_dtype": float8,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2511",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                **(vram_config or {}),
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="text_encoder/model*.safetensors",
                **(vram_config or {}),
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **(vram_config or {}),
            ),
        ],
        processor_config=ModelConfig("/.autodl-model/data/Qwen/Qwen-Image-Edit-2511/processor"),
    )
    pipe.load_lora(pipe.dit, str(args.lora))

    results: list[dict[str, Any]] = []
    for idx, r in enumerate(rows):
        sample_id = Path(r["image"]).stem
        tgt_path = args.dataset_dir / r["image"]
        ctrl_path = args.dataset_dir / r["edit_image"]

        tgt = Image.open(tgt_path).convert("RGBA")
        ctrl = Image.open(ctrl_path).convert("RGB")

        tgt_p = _crop_resize_like_training(tgt, max_pixels=args.max_pixels)
        ctrl_p = _crop_resize_like_training(ctrl, max_pixels=args.max_pixels)
        w, h = tgt_p.size
        ctrl_to_tgt = _crop_resize_to(ctrl, target_w=w, target_h=h)

        pred = pipe(
            prompt=r.get("prompt", "mel-spectrum"),
            negative_prompt="",
            cfg_scale=1,
            edit_image=[ctrl_p],
            seed=args.seed + idx,
            num_inference_steps=args.num_inference_steps,
            height=h,
            width=w,
            edit_image_auto_resize=False,
            zero_cond_t=True,
            progress_bar_cmd=(lambda x: x),
        )

        # Save artifacts
        sample_dir = args.out_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        ctrl_p.save(sample_dir / "control.png")
        ctrl_to_tgt.save(sample_dir / "control_to_target.png")
        tgt_p.save(sample_dir / "target.png")
        pred.save(sample_dir / "pred.png")

        # Quick pixel-space metrics
        l1_ctrl = _l1(_to_float_rgb(ctrl_to_tgt), _to_float_rgb(tgt_p))
        l1_pred = _l1(_to_float_rgb(pred), _to_float_rgb(tgt_p))
        results.append(
            {
                "id": sample_id,
                "width": int(w),
                "height": int(h),
                "control_w": int(ctrl_p.size[0]),
                "control_h": int(ctrl_p.size[1]),
                "l1_control_to_target": l1_ctrl,
                "l1_pred_to_target": l1_pred,
                "improve_l1": float(l1_ctrl - l1_pred),
            }
        )

    metrics: dict[str, Any] = {"results": results}

    if args.compute_fid:
        real_paths = [args.out_dir / r["id"] / "target.png" for r in results]
        fake_paths = [args.out_dir / r["id"] / "pred.png" for r in results]
        fid_res = compute_fid(real_paths, fake_paths, device=args.fid_device, batch_size=args.fid_batch_size)
        metrics["fid"] = {
            "value": fid_res.fid,
            "n_real": fid_res.n_real,
            "n_fake": fid_res.n_fake,
            "feature_dim": fid_res.feature_dim,
            "impl": fid_res.impl,
        }

    out_json = args.out_dir / "metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mean_improve = float(np.mean([r["improve_l1"] for r in results])) if results else 0.0
    print(f"wrote: {out_json}")
    print(f"mean improve_l1: {mean_improve:.6f} (positive is better)")
    if "fid" in metrics:
        print(f"fid: {metrics['fid']['value']:.6f} ({metrics['fid']['impl']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
