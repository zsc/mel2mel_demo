#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fid_torchvision import compute_fid


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute torchvision-Inception FID from an eval output directory.")
    parser.add_argument("--eval_dir", type=Path, required=True, help="Eval dir containing per-sample subdirs with target.png/pred.png.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for Inception feature extraction.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for Inception feature extraction.")
    parser.add_argument("--out_json", type=Path, default=None, help="Optional output json path (default: <eval_dir>/fid.json).")
    args = parser.parse_args()

    eval_dir = args.eval_dir
    if not eval_dir.exists():
        raise SystemExit(f"not found: {eval_dir}")

    real_paths: list[Path] = []
    fake_paths: list[Path] = []
    for sub in sorted(eval_dir.iterdir()):
        if not sub.is_dir():
            continue
        tgt = sub / "target.png"
        pred = sub / "pred.png"
        if tgt.exists() and pred.exists():
            real_paths.append(tgt)
            fake_paths.append(pred)

    if len(real_paths) < 2:
        raise SystemExit(f"need at least 2 pairs, found {len(real_paths)} in {eval_dir}")

    res = compute_fid(real_paths, fake_paths, device=args.device, batch_size=args.batch_size)
    out = {
        "fid": {
            "value": res.fid,
            "n_real": res.n_real,
            "n_fake": res.n_fake,
            "feature_dim": res.feature_dim,
            "impl": res.impl,
        }
    }
    out_path = args.out_json or (eval_dir / "fid.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"fid: {res.fid:.6f}")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

