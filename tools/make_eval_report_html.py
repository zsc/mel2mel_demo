#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


@dataclass(frozen=True)
class EvalSummary:
    step: int | None
    eval_dir: Path
    metrics_path: Path
    n: int
    fid: float | None
    fid_impl: str | None
    mean_l1_control: float
    mean_l1_pred: float
    mean_improve: float
    pos_frac: float
    min_improve: float
    p50_improve: float
    max_improve: float
    by_id: dict[str, dict[str, Any]]


_STEP_RE = re.compile(r"(?:^|/)eval_step-(\d+)(?:_|$)")


def _try_parse_step(path: Path) -> int | None:
    m = _STEP_RE.search(str(path))
    return int(m.group(1)) if m else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(mean(xs)) if xs else 0.0


def _safe_median(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(median(xs)) if xs else 0.0


def _summarize(metrics_path: Path) -> EvalSummary:
    metrics = _load_json(metrics_path)
    results = metrics.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"bad metrics format (results not list): {metrics_path}")
    if not results:
        raise ValueError(f"empty results: {metrics_path}")

    l1_control = [float(r["l1_control_to_target"]) for r in results]
    l1_pred = [float(r["l1_pred_to_target"]) for r in results]
    improve = [float(r["improve_l1"]) for r in results]

    fid_obj = metrics.get("fid") or {}
    fid = float(fid_obj["value"]) if isinstance(fid_obj, dict) and "value" in fid_obj else None
    fid_impl = str(fid_obj.get("impl")) if isinstance(fid_obj, dict) and "impl" in fid_obj else None

    by_id = {str(r["id"]): r for r in results}

    eval_dir = metrics_path.parent
    step = _try_parse_step(eval_dir)

    return EvalSummary(
        step=step,
        eval_dir=eval_dir,
        metrics_path=metrics_path,
        n=len(results),
        fid=fid,
        fid_impl=fid_impl,
        mean_l1_control=_safe_mean(l1_control),
        mean_l1_pred=_safe_mean(l1_pred),
        mean_improve=_safe_mean(improve),
        pos_frac=float(sum(1 for x in improve if x > 0) / len(improve)),
        min_improve=float(min(improve)),
        p50_improve=_safe_median(improve),
        max_improve=float(max(improve)),
        by_id=by_id,
    )


def _fmt_float(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "-"
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{nd}f}"


def _relpath(from_dir: Path, to_path: Path) -> str:
    try:
        return str(to_path.relative_to(from_dir))
    except Exception:
        return str(to_path)


def _sparkline_svg(points: list[tuple[float, float]], *, width: int = 520, height: int = 120) -> str:
    # points: list of (x, y). x and y are in data space.
    if len(points) < 2:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_y = (max_y - min_y) * 0.08 or 1.0
    min_y -= pad_y
    max_y += pad_y

    def sx(x: float) -> float:
        if max_x == min_x:
            return width / 2
        return (x - min_x) / (max_x - min_x) * (width - 8) + 4

    def sy(y: float) -> float:
        if max_y == min_y:
            return height / 2
        # Higher y goes up visually
        return (1 - (y - min_y) / (max_y - min_y)) * (height - 8) + 4

    poly = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" '
        f'aria-label="metric trend">'
        f'<polyline points="{poly}" fill="none" stroke="var(--accent)" stroke-width="2" />'
        f"</svg>"
    )


def _write_report(
    out_path: Path,
    *,
    run_dir: Path,
    summaries: list[EvalSummary],
    final_gallery_k: int = 12,
) -> None:
    run_dir = run_dir.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summaries_sorted = sorted(summaries, key=lambda s: (s.step is None, s.step or 0))
    best_by_fid = min((s for s in summaries_sorted if s.fid is not None), key=lambda s: s.fid, default=None)
    final = max((s for s in summaries_sorted if s.step is not None), key=lambda s: s.step, default=summaries_sorted[-1])

    fid_points = [(float(s.step), float(s.fid)) for s in summaries_sorted if s.step is not None and s.fid is not None]
    imp_points = [(float(s.step), float(s.mean_improve)) for s in summaries_sorted if s.step is not None]

    # Build a small gallery from the final checkpoint.
    gallery = []
    if final and final.by_id:
        items = list(final.by_id.items())
        items.sort(key=lambda kv: float(kv[1]["improve_l1"]), reverse=True)
        top_k = max(0, final_gallery_k // 2)
        bot_k = max(0, final_gallery_k - top_k)
        gallery = items[:top_k] + list(reversed(items[-bot_k:]))

    title = f"Eval Summary: {run_dir.name}"
    fid_note = ""
    if any(s.fid_impl for s in summaries_sorted):
        impls = sorted({s.fid_impl for s in summaries_sorted if s.fid_impl})
        fid_note = " / ".join(impls)

    rows_html = []
    for s in summaries_sorted:
        step = str(s.step) if s.step is not None else "-"
        eval_rel = html.escape(_relpath(run_dir, s.eval_dir))
        metrics_rel = html.escape(_relpath(run_dir, s.metrics_path))
        row = f"""
          <tr>
            <td class="mono">{step}</td>
            <td class="num">{_fmt_float(s.fid, 4)}</td>
            <td class="num">{_fmt_float(s.mean_l1_control, 4)}</td>
            <td class="num">{_fmt_float(s.mean_l1_pred, 4)}</td>
            <td class="num">{_fmt_float(s.mean_improve, 4)}</td>
            <td class="num">{_fmt_float(s.pos_frac, 3)}</td>
            <td class="num">{_fmt_float(s.min_improve, 4)}</td>
            <td class="num">{_fmt_float(s.p50_improve, 4)}</td>
            <td class="num">{_fmt_float(s.max_improve, 4)}</td>
            <td class="mono"><a href="{eval_rel}/">eval dir</a></td>
            <td class="mono"><a href="{metrics_rel}">metrics</a></td>
          </tr>
        """
        rows_html.append(row)

    best_badge = ""
    if best_by_fid and best_by_fid.step is not None:
        best_badge = (
            f"<span class=\"pill\">best FID: step-{best_by_fid.step} ({_fmt_float(best_by_fid.fid, 4)})</span>"
        )

    fid_chart = _sparkline_svg(fid_points) if fid_points else ""
    imp_chart = _sparkline_svg(imp_points) if len(imp_points) >= 2 else ""

    gallery_html = []
    if gallery:
        for sample_id, r in gallery:
            sample_id_s = html.escape(sample_id)
            sample_rel = html.escape(_relpath(run_dir, final.eval_dir / sample_id))
            ctrl_to_tgt = html.escape(_relpath(run_dir, final.eval_dir / sample_id / "control_to_target.png"))
            pred = html.escape(_relpath(run_dir, final.eval_dir / sample_id / "pred.png"))
            tgt = html.escape(_relpath(run_dir, final.eval_dir / sample_id / "target.png"))
            improve = float(r["improve_l1"])
            l1c = float(r["l1_control_to_target"])
            l1p = float(r["l1_pred_to_target"])
            dims = f'{int(r.get("width", 0))}x{int(r.get("height", 0))}'
            gallery_html.append(
                f"""
              <article class="card">
                <header>
                  <div class="card-title mono"><a href="{sample_rel}/">{sample_id_s}</a></div>
                  <div class="card-meta mono">improve {improve:+.4f} · l1 {l1p:.4f} (ctrl {l1c:.4f}) · {dims}</div>
                </header>
                <div class="triptych">
                  <figure>
                    <figcaption class="mono">control→target</figcaption>
                    <img src="{ctrl_to_tgt}" loading="lazy" alt="control to target"/>
                  </figure>
                  <figure>
                    <figcaption class="mono">pred</figcaption>
                    <img src="{pred}" loading="lazy" alt="prediction"/>
                  </figure>
                  <figure>
                    <figcaption class="mono">target</figcaption>
                    <img src="{tgt}" loading="lazy" alt="target"/>
                  </figure>
                </div>
              </article>
                """
            )

    html_text = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
      :root {{
        --bg0: #0b1114;
        --bg1: #0f1c1e;
        --card: rgba(255, 255, 255, 0.06);
        --card2: rgba(255, 255, 255, 0.08);
        --text: rgba(255, 255, 255, 0.92);
        --muted: rgba(255, 255, 255, 0.65);
        --grid: rgba(255, 255, 255, 0.12);
        --accent: #49d1b3;
        --accent2: #ffd166;
        --bad: #ef476f;
      }}
      body {{
        margin: 0;
        color: var(--text);
        background: radial-gradient(1200px 500px at 10% 0%, rgba(73, 209, 179, 0.14), transparent 60%),
                    radial-gradient(900px 420px at 85% 10%, rgba(255, 209, 102, 0.12), transparent 55%),
                    linear-gradient(180deg, var(--bg0), var(--bg1));
        font-family: "IBM Plex Sans", "Noto Sans", "DejaVu Sans", sans-serif;
      }}
      a {{ color: var(--accent); text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      .wrap {{ max-width: 1200px; margin: 0 auto; padding: 28px 18px 64px; }}
      h1 {{ margin: 0 0 10px; font-size: 26px; font-weight: 650; letter-spacing: 0.2px; }}
      .sub {{ color: var(--muted); margin-bottom: 18px; }}
      .mono {{ font-family: "JetBrains Mono", "Iosevka", "Fira Code", ui-monospace, monospace; }}
      .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(73, 209, 179, 0.14);
        border: 1px solid rgba(73, 209, 179, 0.22);
        color: var(--text);
        font-size: 12px;
        margin-right: 8px;
      }}
      .pill.warn {{
        background: rgba(255, 209, 102, 0.14);
        border-color: rgba(255, 209, 102, 0.26);
      }}
      .grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 14px;
      }}
      .panel {{
        padding: 14px 14px;
        background: var(--card);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        backdrop-filter: blur(8px);
      }}
      .panel h2 {{
        margin: 0 0 10px;
        font-size: 15px;
        font-weight: 650;
        letter-spacing: 0.25px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }}
      thead th {{
        text-align: left;
        color: var(--muted);
        font-weight: 600;
        border-bottom: 1px solid var(--grid);
        padding: 10px 10px;
        white-space: nowrap;
      }}
      tbody td {{
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 9px 10px;
        vertical-align: top;
      }}
      tbody tr:hover {{
        background: rgba(255, 255, 255, 0.035);
      }}
      .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
      .spark {{
        width: 100%;
        height: 110px;
        margin-top: 10px;
        opacity: 0.95;
      }}
      .charts {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
      }}
      @media (max-width: 860px) {{
        .charts {{ grid-template-columns: 1fr; }}
      }}
      .cards {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }}
      @media (max-width: 920px) {{
        .cards {{ grid-template-columns: 1fr; }}
      }}
      .card {{
        padding: 12px 12px;
        background: var(--card2);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 12px;
      }}
      .card header {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 10px;
      }}
      .card-title {{ font-size: 13px; }}
      .card-meta {{ color: var(--muted); font-size: 12px; }}
      .triptych {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        align-items: start;
      }}
      figure {{ margin: 0; }}
      figcaption {{ color: var(--muted); font-size: 11px; margin-bottom: 6px; }}
      img {{
        width: 100%;
        height: auto;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(0, 0, 0, 0.22);
      }}
      .foot {{
        margin-top: 16px;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>{html.escape(title)}</h1>
      <div class="sub mono">
        run_dir: {html.escape(str(run_dir))}
      </div>
      <div>
        {best_badge}
        <span class="pill warn">FID impl: {html.escape(fid_note or "unknown")}</span>
      </div>

      <div class="grid" style="margin-top: 14px">
        <section class="panel">
          <h2>Checkpoint Summary</h2>
          <table>
            <thead>
              <tr>
                <th class="mono">step</th>
                <th class="num">fid</th>
                <th class="num">mean l1(ctrl→tgt)</th>
                <th class="num">mean l1(pred→tgt)</th>
                <th class="num">mean improve</th>
                <th class="num">pos frac</th>
                <th class="num">min imp</th>
                <th class="num">p50 imp</th>
                <th class="num">max imp</th>
                <th class="mono">artifacts</th>
                <th class="mono">metrics</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows_html)}
            </tbody>
          </table>
          <div class="foot mono">
            Notes: lower FID is better. positive improve_l1 means prediction is closer to target than the resized control.
          </div>
        </section>

        <section class="panel">
          <h2>Trends</h2>
          <div class="charts">
            <div>
              <div class="mono" style="color: var(--muted); font-size: 12px">fid vs step</div>
              {fid_chart}
            </div>
            <div>
              <div class="mono" style="color: var(--muted); font-size: 12px">mean improve_l1 vs step</div>
              {imp_chart}
            </div>
          </div>
        </section>

        <section class="panel">
          <h2>Final Checkpoint Gallery (top/bottom improve_l1)</h2>
          <div class="sub mono" style="margin-bottom: 10px">
            final eval: <a href="{html.escape(_relpath(run_dir, final.eval_dir))}/">{html.escape(final.eval_dir.name)}</a>
          </div>
          <div class="cards">
            {''.join(gallery_html) if gallery_html else '<div class="mono">No gallery samples.</div>'}
          </div>
        </section>

        <section class="panel">
          <h2>How To Open</h2>
          <div class="foot mono">
            If file links do not work in your browser, serve the run dir and open this page via HTTP:
            <br/>
            <span class="mono">cd {html.escape(str(run_dir))} &amp;&amp; python -m http.server 8000</span>
            <br/>
            then open: <span class="mono">http://127.0.0.1:8000/{html.escape(out_path.name)}</span>
          </div>
        </section>
      </div>
    </div>
  </body>
</html>
"""

    out_path.write_text(html_text, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a self-contained HTML report from eval metrics.json files.")
    p.add_argument("--run_dir", type=Path, required=True, help="Run directory containing eval_* subdirs.")
    p.add_argument(
        "--glob",
        type=str,
        default="eval_step-*_val*_s*_*/metrics.json",
        help="Glob (relative to run_dir) to find metrics.json files.",
    )
    p.add_argument("--out", type=Path, default=None, help="Output HTML path. Default: <run_dir>/eval_summary.html")
    p.add_argument("--gallery_k", type=int, default=12, help="Number of samples to include in the gallery.")
    args = p.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"missing run_dir: {run_dir}")

    metrics_paths = sorted(run_dir.glob(args.glob))
    if not metrics_paths:
        raise SystemExit(f"no metrics found under {run_dir} with glob {args.glob!r}")

    summaries = [_summarize(p) for p in metrics_paths]
    out = args.out or (run_dir / "eval_summary.html")
    _write_report(out, run_dir=run_dir, summaries=summaries, final_gallery_k=args.gallery_k)
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

