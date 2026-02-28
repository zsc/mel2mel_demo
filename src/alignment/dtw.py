from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Metric = Literal["l2", "l1", "cosine"]


@dataclass(frozen=True)
class DtwAlignment:
    path: np.ndarray  # int32, shape (K, 2), each row is (src_index, tgt_index)
    cost: float
    metric: Metric
    band_radius: int | None
    num_source_frames: int
    num_target_frames: int


def _to_time_major(mel: np.ndarray, *, time_axis: int, name: str) -> np.ndarray:
    mel = np.asarray(mel)
    if mel.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={mel.shape!r}")
    if time_axis not in (0, 1):
        raise ValueError(f"{name}: time_axis must be 0 or 1, got {time_axis}")

    if time_axis == 0:
        time_major = np.ascontiguousarray(mel, dtype=np.float32)
    else:
        time_major = np.ascontiguousarray(mel.T, dtype=np.float32)

    if time_major.shape[0] == 0 or time_major.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape={mel.shape!r}")
    return time_major


def _from_time_major(time_major: np.ndarray, *, time_axis: int) -> np.ndarray:
    if time_axis == 0:
        return np.ascontiguousarray(time_major, dtype=np.float32)
    return np.ascontiguousarray(time_major.T, dtype=np.float32)


def _band_bounds(i: int, *, nx: int, ny: int, band_radius: int | None) -> tuple[int, int]:
    if band_radius is None:
        return 0, ny - 1
    if band_radius < 0:
        raise ValueError(f"band_radius must be >= 0, got {band_radius}")
    if nx <= 1:
        center = 0
    else:
        center = int(round(i * (ny - 1) / (nx - 1)))
    j_min = max(0, center - band_radius)
    j_max = min(ny - 1, center + band_radius)
    return j_min, j_max


def dtw_path(
    source_frames: np.ndarray,
    target_frames: np.ndarray,
    *,
    metric: Metric = "l2",
    band_radius: int | None = None,
) -> DtwAlignment:
    """
    Compute a monotonic DTW alignment path between two time-major feature sequences.

    Args:
      source_frames: (T_src, F)
      target_frames: (T_tgt, F)
      metric: frame distance metric ("l2", "l1", "cosine")
      band_radius: optional Sakoe–Chiba-style band radius around the scaled diagonal.

    Returns:
      DtwAlignment with `path` as int32 array of (src_index, tgt_index).
    """
    x = np.ascontiguousarray(source_frames, dtype=np.float32)
    y = np.ascontiguousarray(target_frames, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"source_frames/target_frames must be 2D; got {x.shape!r} and {y.shape!r}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"feature dim mismatch: source_frames.shape={x.shape!r}, target_frames.shape={y.shape!r}"
        )
    nx, feat_dim = x.shape
    ny = y.shape[0]
    if nx == 0 or ny == 0:
        raise ValueError(f"empty sequence: source_frames.shape={x.shape!r}, target_frames.shape={y.shape!r}")

    eps = 1e-8
    if metric == "l2":
        y_norm2 = np.einsum("ij,ij->i", y, y)
        y_t = y.T
        x_norm2 = np.einsum("ij,ij->i", x, x)

        def cost_row(i: int) -> np.ndarray:
            row = x_norm2[i] + y_norm2 - 2.0 * (x[i] @ y_t)
            np.maximum(row, 0.0, out=row)
            return row

    elif metric == "cosine":
        y_norm = np.sqrt(np.einsum("ij,ij->i", y, y) + eps)
        y_t = y.T
        x_norm = np.sqrt(np.einsum("ij,ij->i", x, x) + eps)

        def cost_row(i: int) -> np.ndarray:
            dot = x[i] @ y_t
            denom = x_norm[i] * y_norm + eps
            cos = dot / denom
            return 1.0 - cos

    elif metric == "l1":

        def cost_row(i: int) -> np.ndarray:
            return np.abs(y - x[i]).sum(axis=1)

    else:
        raise ValueError(f"unsupported metric {metric!r}; expected one of: 'l2', 'l1', 'cosine'")

    dp = np.full((nx, ny), np.inf, dtype=np.float32)
    ptr = np.zeros((nx, ny), dtype=np.int8)  # 1=up, 2=left, 3=diag; 0=unset

    for i in range(nx):
        row = cost_row(i)
        j_min, j_max = _band_bounds(i, nx=nx, ny=ny, band_radius=band_radius)
        for j in range(j_min, j_max + 1):
            c = float(row[j])
            if i == 0 and j == 0:
                dp[0, 0] = c
                continue

            best_prev = np.inf
            best_dir = 0
            if i > 0 and dp[i - 1, j] < best_prev:
                best_prev = dp[i - 1, j]
                best_dir = 1
            if j > 0 and dp[i, j - 1] < best_prev:
                best_prev = dp[i, j - 1]
                best_dir = 2
            if i > 0 and j > 0 and dp[i - 1, j - 1] < best_prev:
                best_prev = dp[i - 1, j - 1]
                best_dir = 3

            if not np.isfinite(best_prev):
                continue
            dp[i, j] = c + best_prev
            ptr[i, j] = best_dir

    total_cost = float(dp[nx - 1, ny - 1])
    if not np.isfinite(total_cost):
        raise ValueError(
            "DTW failed to find a valid path (band_radius too small or sequences too different). "
            f"nx={nx}, ny={ny}, metric={metric!r}, band_radius={band_radius}"
        )

    path: list[tuple[int, int]] = []
    i, j = nx - 1, ny - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        d = int(ptr[i, j])
        if d == 3:
            i -= 1
            j -= 1
        elif d == 1:
            i -= 1
        elif d == 2:
            j -= 1
        else:
            raise ValueError(
                f"DTW backtrace failed at (i={i}, j={j}). "
                "This usually means band_radius is too small for a valid monotonic path."
            )

    path.reverse()
    path_arr = np.asarray(path, dtype=np.int32)

    return DtwAlignment(
        path=path_arr,
        cost=total_cost,
        metric=metric,
        band_radius=band_radius,
        num_source_frames=nx,
        num_target_frames=ny,
    )


def warp_source_to_target(
    source_frames: np.ndarray,
    alignment_path: np.ndarray,
    *,
    target_len: int,
    reduce: Literal["mean"] = "mean",
) -> np.ndarray:
    """
    Warp time-major `source_frames` (T_src, F) onto target timeline of length `target_len`.
    """
    if reduce != "mean":
        raise ValueError(f"unsupported reduce={reduce!r}; currently only 'mean' is implemented")

    x = np.asarray(source_frames, dtype=np.float32)
    path = np.asarray(alignment_path)
    if x.ndim != 2:
        raise ValueError(f"source_frames must be 2D (T, F), got shape={x.shape!r}")
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"alignment_path must be shape (K, 2), got shape={path.shape!r}")
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")

    aligned = np.zeros((target_len, x.shape[1]), dtype=np.float32)
    counts = np.zeros((target_len,), dtype=np.int32)
    for src_i, tgt_j in path.astype(np.int64, copy=False):
        aligned[int(tgt_j)] += x[int(src_i)]
        counts[int(tgt_j)] += 1

    if np.any(counts == 0):
        missing = int(np.sum(counts == 0))
        raise ValueError(f"alignment_path does not cover all target frames (missing={missing}/{target_len})")

    aligned /= counts[:, None].astype(np.float32)
    return aligned


def dtw_warp_mel(
    source_mel: np.ndarray,
    target_mel: np.ndarray,
    *,
    mel_time_axis: int = 1,
    metric: Metric = "l2",
    band_radius: int | None = None,
) -> tuple[np.ndarray, DtwAlignment]:
    """
    DTW-align `source_mel` to `target_mel` and return the warped mel plus alignment metadata.

    mel_time_axis:
      - 1 (default): mel is (n_mels, T), time on axis=1 (image-like)
      - 0: mel is (T, n_mels), time on axis=0
    """
    src_tm = _to_time_major(source_mel, time_axis=mel_time_axis, name="source_mel")
    tgt_tm = _to_time_major(target_mel, time_axis=mel_time_axis, name="target_mel")

    if src_tm.shape[1] != tgt_tm.shape[1]:
        raise ValueError(
            "source_mel/target_mel mel bins mismatch after orientation handling: "
            f"source={src_tm.shape!r}, target={tgt_tm.shape!r}"
        )

    alignment = dtw_path(src_tm, tgt_tm, metric=metric, band_radius=band_radius)
    aligned_tm = warp_source_to_target(src_tm, alignment.path, target_len=tgt_tm.shape[0])
    aligned_mel = _from_time_major(aligned_tm, time_axis=mel_time_axis)
    return aligned_mel, alignment

