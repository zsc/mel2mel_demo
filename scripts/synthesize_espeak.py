#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pypinyin import Style, lazy_pinyin

PUNCT_MAP: dict[str, str] = {
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "、": ",",
    "；": ";",
    "：": ":",
    "…": ".",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "《": "<",
    "》": ">",
    "“": "\"",
    "”": "\"",
    "‘": "'",
    "’": "'",
}


def text_to_pinyin_for_espeak(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    tokens = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, errors="default")
    out: list[str] = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        out.append(PUNCT_MAP.get(t, t))

    # Collapse consecutive punctuation spacing a bit.
    return " ".join(out)


def synthesize_wav_espeak_ng(
    *,
    text: str,
    out_wav_path: Path,
    voice: str,
    speed: int | None,
    amplitude: int | None,
) -> None:
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    out_wav_path = out_wav_path.resolve()

    with tempfile.TemporaryDirectory(prefix="espeak_ng_") as td:
        td_path = Path(td)
        tmp_wav = td_path / "out.wav"

        cmd = ["espeak-ng", "-v", voice]
        if speed is not None:
            cmd += ["-s", str(speed)]
        if amplitude is not None:
            cmd += ["-a", str(amplitude)]
        cmd += ["-w", str(tmp_wav), text]

        proc = subprocess.run(
            cmd,
            cwd=str(td_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-4000:]
            raise RuntimeError(f"espeak-ng failed (code={proc.returncode}). stderr_tail={tail!r}")

        if not tmp_wav.exists() or tmp_wav.stat().st_size == 0:
            raise RuntimeError(f"espeak-ng produced no wav at {tmp_wav}")

        tmp_out = out_wav_path.with_suffix(out_wav_path.suffix + ".tmp")
        shutil.copyfile(tmp_wav, tmp_out)
        os.replace(tmp_out, out_wav_path)


def iter_input_files(in_dir: Path, *, glob: str) -> list[Path]:
    files = sorted(in_dir.glob(glob))
    if not files:
        raise SystemExit(f"No input files matched {glob!r} under {in_dir}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize Chinese .lab text files into wav using espeak-ng.")
    parser.add_argument("--in-file", type=Path, default=None, help="Single input .lab file (UTF-8).")
    parser.add_argument("--in-dir", type=Path, default=None, help="Directory containing .lab files.")
    parser.add_argument("--glob", type=str, default="*.lab", help="Glob pattern under --in-dir (default: *.lab).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for wav files.")
    parser.add_argument(
        "--voice",
        type=str,
        default="cmn-latn-pinyin",
        help="espeak-ng voice (default: cmn-latn-pinyin).",
    )
    parser.add_argument("--speed", type=int, default=None, help="Optional espeak-ng -s speed (words/min).")
    parser.add_argument("--amplitude", type=int, default=None, help="Optional espeak-ng -a amplitude (0-200).")
    parser.add_argument("--limit", type=int, default=None, help="If set, only synthesize first N files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing wavs.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 8),
        help="Parallel workers (default: min(8, cpu_count)).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N completed files (default: 50).",
    )
    parser.add_argument("--verbose", action="store_true", help="Log every file result (very noisy for 10k).")
    parser.add_argument(
        "--write-pinyin",
        action="store_true",
        help="Also write the pinyin string to a .pinyin.txt next to each wav.",
    )
    args = parser.parse_args()

    if (args.in_file is None) == (args.in_dir is None):
        raise SystemExit("Provide exactly one of --in-file or --in-dir")

    if args.in_file is not None:
        input_files = [args.in_file]
    else:
        input_files = iter_input_files(args.in_dir, glob=args.glob)

    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be > 0")
        input_files = input_files[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.jobs <= 0:
        raise SystemExit("--jobs must be > 0")
    if args.log_every <= 0:
        raise SystemExit("--log-every must be > 0")

    total = len(input_files)
    started = time.time()

    def process_one(in_path: Path) -> tuple[str, str]:
        text = in_path.read_text(encoding="utf-8").strip()
        pinyin = text_to_pinyin_for_espeak(text)
        if not pinyin:
            return "skip_empty", in_path.name

        out_wav = args.out_dir / (in_path.stem + ".wav")
        if out_wav.exists() and not args.overwrite:
            return "exists", out_wav.name

        if args.write_pinyin:
            (args.out_dir / (in_path.stem + ".pinyin.txt")).write_text(pinyin + "\n", encoding="utf-8")

        synthesize_wav_espeak_ng(
            text=pinyin,
            out_wav_path=out_wav,
            voice=args.voice,
            speed=args.speed,
            amplitude=args.amplitude,
        )
        return "ok", out_wav.name

    counts: dict[str, int] = {"ok": 0, "exists": 0, "skip_empty": 0, "failed": 0}
    completed = 0

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        future_to_file = {ex.submit(process_one, p): p for p in input_files}
        for fut in as_completed(future_to_file):
            p = future_to_file[fut]
            completed += 1
            try:
                status, msg = fut.result()
                counts[status] = counts.get(status, 0) + 1
                if args.verbose:
                    print(f"[{completed}/{total}] {status}: {p.name} -> {msg}")
                elif completed % args.log_every == 0 or completed == total:
                    elapsed = time.time() - started
                    rate = completed / max(elapsed, 1e-6)
                    print(
                        f"[{completed}/{total}] ok={counts['ok']} exists={counts['exists']} "
                        f"skip={counts['skip_empty']} failed={counts['failed']} rate={rate:.2f}/s"
                    )
            except Exception as e:
                counts["failed"] += 1
                print(f"[{completed}/{total}] failed: {p.name}: {e}", flush=True)

    elapsed = time.time() - started
    print(
        f"done: total={total} ok={counts['ok']} exists={counts['exists']} "
        f"skip={counts['skip_empty']} failed={counts['failed']} elapsed={elapsed:.1f}s"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
