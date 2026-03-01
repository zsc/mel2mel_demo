#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbedStats:
    embedded: int
    skipped: int
    missing: int
    total_bytes: int


MIME_OVERRIDES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
}


def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in MIME_OVERRIDES:
        return MIME_OVERRIDES[ext]
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def embed_src_assets(html: str, *, base_dir: Path, allowed_exts: set[str] | None) -> tuple[str, EmbedStats]:
    # src='...' or src="..."
    pattern = re.compile(r"""src=(['"])([^'"]+)\1""", flags=re.IGNORECASE)

    embedded = 0
    skipped = 0
    missing = 0
    total_bytes = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal embedded, skipped, missing, total_bytes
        quote = match.group(1)
        url = match.group(2)

        if url.startswith(("data:", "http://", "https://", "//")):
            skipped += 1
            return match.group(0)

        rel = url.split("?", 1)[0].split("#", 1)[0]
        path = (base_dir / rel).resolve()
        if not path.exists():
            missing += 1
            return match.group(0)
        if not path.is_file():
            skipped += 1
            return match.group(0)

        if allowed_exts is not None and path.suffix.lower() not in allowed_exts:
            skipped += 1
            return match.group(0)

        data = path.read_bytes()
        total_bytes += len(data)
        b64 = base64.b64encode(data).decode("ascii")
        mime = guess_mime(path)
        embedded += 1
        return f"src={quote}data:{mime};base64,{b64}{quote}"

    out = pattern.sub(repl, html)
    return out, EmbedStats(embedded=embedded, skipped=skipped, missing=missing, total_bytes=total_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert an HTML report into an all-in-one HTML by embedding src assets.")
    parser.add_argument("--input", type=Path, required=True, help="Input HTML file (e.g. outputs/ningguang_eval/report.html).")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: <input stem>_all_in_one.html).")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory to resolve relative src paths (default: input file parent dir).",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".png,.wav",
        help="Comma-separated extensions to embed (default: .png,.wav). Use empty to embed all.",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    out_path = args.output
    if out_path is None:
        out_path = input_path.with_name(input_path.stem + "_all_in_one.html")

    base_dir = args.base_dir if args.base_dir is not None else input_path.parent
    base_dir = base_dir.resolve()

    allowed_exts: set[str] | None
    if args.exts.strip() == "":
        allowed_exts = None
    else:
        allowed_exts = {("." + e.strip().lstrip(".")).lower() for e in args.exts.split(",") if e.strip()}
        if not allowed_exts:
            allowed_exts = None

    html = input_path.read_text(encoding="utf-8")
    out_html, stats = embed_src_assets(html, base_dir=base_dir, allowed_exts=allowed_exts)
    out_path.write_text(out_html, encoding="utf-8")

    print(
        f"wrote {out_path} (embedded={stats.embedded}, skipped={stats.skipped}, missing={stats.missing}, "
        f"payload_bytes={stats.total_bytes})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

