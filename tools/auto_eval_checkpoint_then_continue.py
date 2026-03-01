#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_int(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def _safe_killpg(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _load_split_ids(dataset_dir: Path, split: str) -> list[str] | None:
    if split == "all":
        return None
    split_path = dataset_dir / "split.json"
    if not split_path.exists():
        return None
    obj = json.loads(split_path.read_text(encoding="utf-8"))
    ids = obj.get(split)
    if not isinstance(ids, list):
        return None
    return [str(x) for x in ids]


@dataclass(frozen=True)
class EvalConfig:
    dataset_dir: Path
    lora_ckpt: Path
    out_dir: Path
    split: str
    num_samples: int
    seed: int
    num_inference_steps: int
    low_vram: int
    vram_limit_gb: float
    computation_dtype: str
    compute_fid: int
    fid_device: str
    fid_batch_size: int


def _run_eval(cfg: EvalConfig) -> None:
    script = Path(__file__).resolve().parent / "eval_diffsynth_qwen_image_edit_lora.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset_dir",
        str(cfg.dataset_dir),
        "--lora",
        str(cfg.lora_ckpt),
        "--out_dir",
        str(cfg.out_dir),
        "--split",
        cfg.split,
        "--num_samples",
        str(cfg.num_samples),
        "--seed",
        str(cfg.seed),
        "--num_inference_steps",
        str(cfg.num_inference_steps),
        "--low_vram",
        str(cfg.low_vram),
        "--vram_limit_gb",
        str(cfg.vram_limit_gb),
        "--computation_dtype",
        str(cfg.computation_dtype),
        "--compute_fid",
        str(cfg.compute_fid),
        "--fid_device",
        cfg.fid_device,
        "--fid_batch_size",
        str(cfg.fid_batch_size),
    ]
    print("eval cmd:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Wait for a training checkpoint, pause training, run eval (FID/L1), then resume training.",
    )
    p.add_argument("--run_dir", type=Path, required=True, help="Training run dir containing pid.bash.txt and step-*.safetensors")
    p.add_argument("--dataset_dir", type=Path, required=True, help="Prepared dataset dir containing metadata.diffsynth.jsonl")
    p.add_argument("--step", type=int, default=2000, help="Checkpoint step to wait for, e.g. 2000")
    p.add_argument("--pause_pgid_file", type=str, default="pid.bash.txt", help="File in run_dir containing the PGID to SIGSTOP/SIGCONT")
    p.add_argument("--poll_sec", type=float, default=10.0, help="Polling interval while waiting for checkpoint")

    p.add_argument("--split", type=str, default="val", choices=("train", "val", "all"))
    p.add_argument("--num_samples", type=int, default=0, help="0 = use full split size if available")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--low_vram", type=int, default=1, choices=(0, 1))
    p.add_argument("--vram_limit_gb", type=float, default=90.0, help="See eval script; only applies if --low_vram=1")
    p.add_argument("--computation_dtype", type=str, default="bf16", choices=("bf16", "fp16", "fp8"))
    p.add_argument("--compute_fid", type=int, default=1, choices=(0, 1))
    p.add_argument("--fid_device", type=str, default="cpu")
    p.add_argument("--fid_batch_size", type=int, default=32)

    args = p.parse_args()

    run_dir: Path = args.run_dir
    dataset_dir: Path = args.dataset_dir
    ckpt = run_dir / f"step-{int(args.step)}.safetensors"
    if not run_dir.exists():
        raise SystemExit(f"missing run_dir: {run_dir}")
    if not dataset_dir.exists():
        raise SystemExit(f"missing dataset_dir: {dataset_dir}")

    # Resolve eval sample count.
    split_ids = _load_split_ids(dataset_dir, args.split)
    if args.num_samples and args.num_samples > 0:
        num_samples = int(args.num_samples)
    elif split_ids is not None:
        num_samples = len(split_ids)
    else:
        # Fallback: cap to something reasonable if split.json is missing.
        num_samples = 96 if args.split == "val" else 256

    print(f"waiting for checkpoint: {ckpt}", flush=True)
    while not ckpt.exists():
        time.sleep(float(args.poll_sec))

    # Pause training process group.
    pgid_path = run_dir / args.pause_pgid_file
    if not pgid_path.exists():
        raise SystemExit(f"missing pause pgid file: {pgid_path}")
    pgid = _read_int(pgid_path)

    print(f"pausing training pgid={pgid} ...", flush=True)
    _safe_killpg(pgid, signal.SIGSTOP)
    time.sleep(2.0)

    # Run eval while training is paused.
    out_dir = run_dir / f"eval_step-{int(args.step)}_{args.split}_s{int(args.num_inference_steps)}_auto_{_now_stamp()}"
    cfg = EvalConfig(
        dataset_dir=dataset_dir,
        lora_ckpt=ckpt,
        out_dir=out_dir,
        split=args.split,
        num_samples=num_samples,
        seed=int(args.seed),
        num_inference_steps=int(args.num_inference_steps),
        low_vram=int(args.low_vram),
        vram_limit_gb=float(args.vram_limit_gb),
        computation_dtype=str(args.computation_dtype),
        compute_fid=int(args.compute_fid),
        fid_device=str(args.fid_device),
        fid_batch_size=int(args.fid_batch_size),
    )

    try:
        print(f"running eval into: {out_dir}", flush=True)
        _run_eval(cfg)
        print("eval finished", flush=True)
    finally:
        # Always resume training.
        print(f"resuming training pgid={pgid} ...", flush=True)
        _safe_killpg(pgid, signal.SIGCONT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
