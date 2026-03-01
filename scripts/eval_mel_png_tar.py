#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import tarfile
import unicodedata
from dataclasses import dataclass
from fnmatch import fnmatch
from html import escape as html_escape
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image


def _maybe_add_audio_avif_to_syspath() -> None:
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    audio_avif_repo = (repo_root.parent / "audio_avif").resolve()
    if (audio_avif_repo / "audio_avif" / "__init__.py").exists():
        sys.path.insert(0, str(audio_avif_repo))


_maybe_add_audio_avif_to_syspath()

import audio_avif  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
import matplotlib  # noqa: E402
import torch  # noqa: E402
import whisper  # noqa: E402


@dataclass(frozen=True)
class Sample:
    tar_member_name: str
    relpath: str
    eval_tag: str
    utt_id: str
    kind: str  # pred / target / control_to_target


def parse_member_name(name: str) -> Sample:
    norm_name = name
    while norm_name.startswith("./"):
        norm_name = norm_name[2:]
    norm_name = norm_name.lstrip("/")

    p = Path(norm_name)
    if p.suffix.lower() != ".png":
        raise ValueError(f"expected .png, got {norm_name!r}")
    if len(p.parts) < 2:
        raise ValueError(f"unexpected member path: {norm_name!r}")
    if len(p.parts) >= 3:
        eval_tag = p.parts[-3]
        utt_id = p.parts[-2]
    else:
        eval_tag = ""
        utt_id = p.parts[-2]
    kind = p.stem
    return Sample(tar_member_name=name, relpath=norm_name, eval_tag=eval_tag, utt_id=utt_id, kind=kind)


def magma_lut_u8() -> tuple[np.ndarray, dict[tuple[int, int, int], int]]:
    # Matplotlib colormap quantization matches ImageMagick 'file' output we observed:
    # PNGs use exact 8-bit RGB values from the 256-entry colormap LUT.
    cmap = matplotlib.colormaps.get_cmap("magma")
    lut = (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)
    mapping: dict[tuple[int, int, int], int] = {}
    for i, rgb in enumerate(lut):
        mapping[(int(rgb[0]), int(rgb[1]), int(rgb[2]))] = i
    return lut, mapping


def rgb_to_norm_via_magma(rgb_u8: np.ndarray, *, lut: np.ndarray, mapping: dict[tuple[int, int, int], int]) -> np.ndarray:
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"expected RGB image array (H, W, 3), got shape={rgb_u8.shape!r}")

    h, w, _ = rgb_u8.shape
    flat = rgb_u8.reshape(-1, 3)
    unique, inv = np.unique(flat, axis=0, return_inverse=True)

    idx_unique = np.empty((unique.shape[0],), dtype=np.int16)
    missing = []
    for i, c in enumerate(unique):
        key = (int(c[0]), int(c[1]), int(c[2]))
        j = mapping.get(key)
        if j is None:
            missing.append(i)
            idx_unique[i] = -1
        else:
            idx_unique[i] = j

    if missing:
        # Fall back to nearest colormap entry for out-of-lut colors (e.g., model outputs).
        # This is only 256-way and only for unique colors, so it's still fast.
        for i in missing:
            c = unique[i].astype(np.int16)
            d = lut.astype(np.int16) - c[None, :]
            d2 = (d * d).sum(axis=1)
            idx_unique[i] = int(np.argmin(d2))

    idx = idx_unique[inv].astype(np.float32)
    norm = (idx / 255.0).reshape(h, w)
    return norm


def image_to_logmel(
    image: Image.Image,
    *,
    min_db: float,
    max_db: float,
    assume_magma: bool,
    lut: np.ndarray,
    mapping: dict[tuple[int, int, int], int],
) -> np.ndarray:
    if assume_magma:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        norm = rgb_to_norm_via_magma(rgb, lut=lut, mapping=mapping)
    else:
        gray = np.array(image.convert("L"), dtype=np.float32)
        norm = gray / 255.0

    if norm.ndim != 2:
        raise ValueError(f"expected 2D norm image, got shape={norm.shape!r}")
    if norm.shape[0] != 80:
        raise ValueError(f"expected mel image height=80, got {norm.shape[0]}")

    # The PNGs are saved with low-freq at the bottom (matplotlib origin='lower'),
    # so we follow audio_avif's decode convention: flip vertically then transpose.
    logmel = np.flipud(norm).T * (max_db - min_db) + min_db  # (T, 80)
    return logmel.astype(np.float32)


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    out: list[str] = []
    for ch in s:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("Z") or cat.startswith("C"):
            continue
        out.append(ch)
    return "".join(out)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure b is the shorter one for less memory.
    if len(b) > len(a):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    dist = levenshtein(ref, hyp)
    return dist / float(len(ref))


def load_labels_from_tar(labels_tar: Path, utt_ids: set[str]) -> dict[str, str]:
    needed = {f"samples10k_lab/{u}.lab" for u in utt_ids}
    out: dict[str, str] = {}
    with tarfile.open(labels_tar, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            if m.name not in needed:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            text = f.read().decode("utf-8", errors="replace").strip()
            out[Path(m.name).stem] = text
    return out


def pick_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def iter_png_members(
    samples_tar: Path,
    *,
    pattern: str | None,
    eval_tag_override: str | None,
    allowed_kinds: set[str] | None,
) -> list[Sample]:
    items: list[Sample] = []
    with tarfile.open(samples_tar, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.lower().endswith(".png"):
                continue
            if pattern is not None and not fnmatch(m.name, pattern):
                continue
            try:
                s = parse_member_name(m.name)
            except ValueError:
                continue
            if eval_tag_override is not None:
                s = Sample(
                    tar_member_name=s.tar_member_name,
                    relpath=s.relpath,
                    eval_tag=eval_tag_override,
                    utt_id=s.utt_id,
                    kind=s.kind,
                )
            if allowed_kinds is not None and s.kind not in allowed_kinds:
                continue
            items.append(s)
    items.sort(key=lambda s: (s.eval_tag, s.utt_id, s.kind, s.relpath))
    return items


def mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.mean(np.asarray(xs, dtype=np.float32)))


def write_report_html(out_dir: Path, results: list[dict]) -> Path:
    by_eval: dict[str, dict[str, dict[str, dict]]] = {}
    for r in results:
        by_eval.setdefault(r["eval_tag"], {}).setdefault(r["utt_id"], {})[r["kind"]] = r

    eval_tags = sorted(by_eval.keys())
    kind_set = {r["kind"] for r in results}
    preferred = ["control", "target", "control_to_target", "pred"]
    kinds = [k for k in preferred if k in kind_set] + sorted(kind_set - set(preferred))

    def rel(p: str) -> str:
        try:
            return str(Path(p).resolve().relative_to(out_dir.resolve()))
        except Exception:
            return p

    # Summary stats
    summary_rows = []
    for eval_tag in eval_tags:
        all_rows = []
        for utt_id in sorted(by_eval[eval_tag].keys()):
            all_rows.append((utt_id, by_eval[eval_tag][utt_id]))
        per_kind = {}
        for k in kinds:
            scores = [row.get(k, {}).get("cer") for _, row in all_rows if row.get(k, {}).get("cer") is not None]
            scores_f = [float(s) for s in scores if isinstance(s, (float, int))]
            per_kind[k] = {"mean": mean(scores_f), "count": len(scores_f)}
        summary_rows.append((eval_tag, per_kind))

    overall_per_kind = {}
    for k in kinds:
        scores = [r["cer"] for r in results if r["kind"] == k]
        overall_per_kind[k] = {"mean": mean([float(s) for s in scores]), "count": len(scores)}

    report_path = out_dir / "report.html"
    css = """
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }
h1, h2 { margin: 0.6em 0 0.4em; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
th { background: #f6f6f6; position: sticky; top: 0; }
.small { color: #444; font-size: 12px; }
.cer { font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
.cell { min-width: 280px; }
.img { max-width: 360px; height: auto; display: block; margin-bottom: 6px; background: #111; }
.audio { width: 360px; }
.ref { font-size: 14px; margin: 4px 0; }
.hyp { font-size: 14px; margin: 4px 0; }
.muted { color: #777; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 6px; background: #eee; font-size: 12px; }
.ok { background: #e6ffed; }
.warn { background: #fff5cc; }
.bad { background: #ffe6e6; }
details { margin: 10px 0; }
summary { cursor: pointer; font-weight: 600; }
"""

    def cer_badge(v: float) -> str:
        if not np.isfinite(v):
            return "badge"
        if v <= 0.05:
            return "badge ok"
        if v <= 0.20:
            return "badge warn"
        return "badge bad"

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<style>{css}</style>")
    parts.append("<title>mel-png eval report</title></head><body>")
    parts.append("<h1>mel PNG → vocoder → whisper CER</h1>")
    parts.append("<p class='small'>Generated by <code>scripts/eval_mel_png_tar.py</code></p>")

    parts.append("<h2>Summary</h2>")
    parts.append("<table><thead><tr><th>eval_tag</th>")
    for k in kinds:
        parts.append(f"<th>{html_escape(k)}</th>")
    parts.append("</tr></thead><tbody>")
    # overall row
    parts.append("<tr><td><b>OVERALL</b></td>")
    for k in kinds:
        v = overall_per_kind[k]["mean"]
        c = overall_per_kind[k]["count"]
        parts.append(f"<td class='cer'>{v:.3f} <span class='muted'>({c})</span></td>")
    parts.append("</tr>")
    for eval_tag, per_kind in summary_rows:
        parts.append(f"<tr><td>{html_escape(eval_tag)}</td>")
        for k in kinds:
            v = per_kind[k]["mean"]
            c = per_kind[k]["count"]
            parts.append(f"<td class='cer'>{v:.3f} <span class='muted'>({c})</span></td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    parts.append("<h2>Details</h2>")
    for eval_tag in eval_tags:
        # counts
        n_utts = len(by_eval[eval_tag])
        stats = {k: mean([row.get(k, {}).get("cer") for row in by_eval[eval_tag].values() if k in row]) for k in kinds}
        parts.append("<details open>")
        parts.append(
            "<summary>"
            f"{html_escape(eval_tag)} "
            f"<span class='small muted'>utts={n_utts} "
            + " ".join([f"{k}:{stats[k]:.3f}" for k in kinds])
            + "</span></summary>"
        )
        parts.append("<table><thead><tr>")
        parts.append("<th>utt_id</th><th>ref</th>")
        for k in kinds:
            parts.append(f"<th>{html_escape(k)}</th>")
        parts.append("</tr></thead><tbody>")

        for utt_id in sorted(by_eval[eval_tag].keys()):
            row = by_eval[eval_tag][utt_id]
            ref = ""
            for k in preferred:
                if k in row and "ref" in row[k]:
                    ref = row[k]["ref"]
                    break
            if not ref and row:
                ref = next(iter(row.values())).get("ref", "")
            parts.append("<tr>")
            parts.append(f"<td><code>{html_escape(utt_id)}</code></td>")
            parts.append(f"<td class='ref'>{html_escape(ref)}</td>")
            for k in kinds:
                r = row.get(k)
                if r is None:
                    parts.append("<td class='cell muted'>missing</td>")
                    continue
                png_rel = rel(r["png_path"]) if "png_path" in r else ""
                wav_rel = rel(r["wav_path"])
                hyp = r.get("hyp", "")
                v = float(r.get("cer", float("nan")))
                badge_cls = cer_badge(v)
                parts.append("<td class='cell'>")
                if png_rel:
                    parts.append(f"<img class='img' loading='lazy' src='{html_escape(png_rel)}'/>")
                parts.append(f"<audio class='audio' controls preload='none' src='{html_escape(wav_rel)}'></audio>")
                parts.append(f"<div class='hyp'><span class='{badge_cls} cer'>CER={v:.3f}</span></div>")
                parts.append(f"<div class='hyp'>{html_escape(hyp)}</div>")
                parts.append("</td>")
            parts.append("</tr>")

        parts.append("</tbody></table>")
        parts.append("</details>")

    parts.append("</body></html>")

    report_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct audio from mel PNGs in a tarball and compute CER via whisper.")
    parser.add_argument("--samples-tar", type=Path, default=Path("ningguang_samples.tar.gz"), help="Input tar(.gz) with mel PNGs.")
    parser.add_argument(
        "--labels-tar",
        type=Path,
        default=Path("/Users/georgezhou/Downloads/try_genshinimpact/samples10k_lab.tar"),
        help="Label tar containing samples10k_lab/*.lab",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ningguang_eval"), help="Where to write wavs and transcripts.")
    parser.add_argument("--pattern", type=str, default=None, help="Optional fnmatch pattern on tar member name (e.g. '*/*/target.png').")
    parser.add_argument("--eval-tag", type=str, default=None, help="Override eval_tag for all samples (useful when tar has no eval subdir).")
    parser.add_argument(
        "--kinds",
        type=str,
        default=None,
        help="Comma-separated kinds to process (e.g. 'target,pred,control_to_target'). Default: all kinds in the tar.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N (after sorting).")
    parser.add_argument("--assume-magma", action="store_true", help="Assume PNGs are magma-colormap renders and invert them.")
    parser.add_argument("--min-db", type=float, default=-11.0, help="Log-mel min for de-normalization.")
    parser.add_argument("--max-db", type=float, default=4.0, help="Log-mel max for de-normalization.")
    parser.add_argument("--vocoder-device", type=str, default=None, help="Torch device for vocoder (cpu/cuda/mps). Default: auto.")
    parser.add_argument(
        "--whisper-device",
        type=str,
        default="cpu",
        help="Torch device for whisper (default: cpu; MPS may fail for some models).",
    )
    parser.add_argument("--whisper-model", type=str, default="turbo", help="Whisper model name (default: turbo).")
    parser.add_argument("--language", type=str, default="zh", help="Whisper language (default: zh).")
    parser.add_argument("--write-jsonl", action="store_true", help="Write per-file results to results.jsonl in output-dir.")
    parser.add_argument("--no-html", action="store_true", help="Disable generating report.html")
    args = parser.parse_args()

    vocoder_device = pick_device(args.vocoder_device)
    whisper_device = args.whisper_device
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_kinds = None
    if args.kinds is not None:
        kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
        if not kinds:
            raise SystemExit("--kinds is empty after parsing")
        allowed_kinds = kinds

    samples = iter_png_members(
        args.samples_tar,
        pattern=args.pattern,
        eval_tag_override=args.eval_tag,
        allowed_kinds=allowed_kinds,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be > 0")
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No PNG members matched.")

    utt_ids = {s.utt_id for s in samples}
    labels = load_labels_from_tar(args.labels_tar, utt_ids)
    missing = sorted(utt_ids - set(labels.keys()))
    if missing:
        raise SystemExit(f"Missing labels for utt_ids: {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    lut, mapping = magma_lut_u8()

    print(f"loading vocoder on {vocoder_device} ...", flush=True)
    vocoder = audio_avif.load_vocoder(vocoder_device)

    print(f"loading whisper model {args.whisper_model!r} on {whisper_device} ...", flush=True)
    try:
        wmodel = whisper.load_model(args.whisper_model, device=whisper_device)
    except NotImplementedError as e:
        if whisper_device == "mps":
            print(f"warning: whisper failed on mps, falling back to cpu: {e}", flush=True)
            whisper_device = "cpu"
            wmodel = whisper.load_model(args.whisper_model, device=whisper_device)
        else:
            raise

    results_path = out_dir / "results.jsonl"
    jsonl_f = results_path.open("w", encoding="utf-8") if args.write_jsonl else None
    results: list[dict] = []

    try:
        with tarfile.open(args.samples_tar, "r:*") as tf:
            for i, s in enumerate(samples, start=1):
                m = tf.getmember(s.tar_member_name)
                f = tf.extractfile(m)
                if f is None:
                    raise RuntimeError(f"failed to extract {s.tar_member_name}")
                data = f.read()
                png_out = out_dir / s.relpath
                png_out.parent.mkdir(parents=True, exist_ok=True)
                if not png_out.exists() or png_out.stat().st_size != len(data):
                    png_out.write_bytes(data)
                img = Image.open(io.BytesIO(data))

                logmel = image_to_logmel(
                    img,
                    min_db=args.min_db,
                    max_db=args.max_db,
                    assume_magma=args.assume_magma,
                    lut=lut,
                    mapping=mapping,
                )

                wav = audio_avif.reconstruct_wav(logmel, vocoder, vocoder_device)
                wav_out = out_dir / Path(s.relpath).with_suffix(".wav")
                wav_out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav_out, wav, audio_avif.TARGET_SR)

                wres = wmodel.transcribe(
                    str(wav_out),
                    task="transcribe",
                    language=args.language,
                    fp16=(whisper_device != "cpu"),
                    verbose=None,
                )
                hyp = (wres.get("text") or "").strip()
                ref = labels[s.utt_id]
                score = cer(ref, hyp)

                item = {
                    "i": i,
                    "eval_tag": s.eval_tag,
                    "utt_id": s.utt_id,
                    "kind": s.kind,
                    "tar_member": s.tar_member_name,
                    "relpath": s.relpath,
                    "png_path": str(png_out),
                    "wav_path": str(wav_out),
                    "ref": ref,
                    "hyp": hyp,
                    "cer": score,
                }
                if jsonl_f is not None:
                    jsonl_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                results.append(item)

                print(f"[{i}/{len(samples)}] {s.eval_tag}/{s.utt_id}/{s.kind}: CER={score:.3f}", flush=True)

    finally:
        if jsonl_f is not None:
            jsonl_f.close()

    if not args.no_html:
        report_path = write_report_html(out_dir, results)
        print(f"wrote {report_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
