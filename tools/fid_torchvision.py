#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass(frozen=True)
class FidResult:
    fid: float
    n_real: int
    n_fake: int
    feature_dim: int
    impl: str  # for reproducibility / comparability notes


def _default_transform() -> transforms.Compose:
    # We use torchvision InceptionV3 weights and its expected normalization.
    # For FID-style metrics the exact transform matters. Keep it stable.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_inception_feature_extractor(*, device: str) -> tuple[torch.nn.Module, transforms.Compose]:
    """
    Returns a model that outputs `avgpool` features (N, 2048).
    Note: This is *torchvision* Inception-V3 weights, not the TF-FID Inception.
    """
    weights = Inception_V3_Weights.IMAGENET1K_V1
    # torchvision enforces aux_logits=True for official pretrained weights.
    model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
    model.eval()
    extractor = create_feature_extractor(model, return_nodes={"avgpool": "avgpool"})
    extractor.to(device=device)
    return extractor, _default_transform()


def _iter_batches(items: Sequence[Path], *, batch_size: int) -> Iterable[list[Path]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


@torch.no_grad()
def compute_inception_features_with_extractor(
    extractor: torch.nn.Module,
    tfm: transforms.Compose,
    paths: Sequence[Path],
    *,
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    for batch_paths in _iter_batches(paths, batch_size=batch_size):
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(tfm(img))
        x = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)
        out = extractor(x)["avgpool"]  # (N, 2048, 1, 1)
        out = out.flatten(1)
        feats.append(out.detach().cpu())
    return torch.cat(feats, dim=0) if feats else torch.empty((0, 2048), dtype=torch.float32)


@torch.no_grad()
def compute_inception_features(
    paths: Sequence[Path],
    *,
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    extractor, tfm = load_inception_feature_extractor(device=device)
    return compute_inception_features_with_extractor(extractor, tfm, paths, device=device, batch_size=batch_size)


def frechet_distance_from_features(real: torch.Tensor, fake: torch.Tensor) -> float:
    """
    Compute Frechet distance between two feature sets without scipy.

    Uses a low-rank SVD formulation that is fast when n << d.
    """
    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError(f"expected 2D tensors, got {real.shape} and {fake.shape}")
    if real.shape[1] != fake.shape[1]:
        raise ValueError(f"feature dim mismatch: {real.shape} vs {fake.shape}")
    n1, d = real.shape
    n2, _ = fake.shape
    if n1 < 2 or n2 < 2:
        return float("nan")

    x = real.to(dtype=torch.float64)
    y = fake.to(dtype=torch.float64)

    mu1 = x.mean(dim=0)
    mu2 = y.mean(dim=0)
    diff = mu1 - mu2
    mean_term = float(diff.dot(diff).item())

    x0 = x - mu1
    y0 = y - mu2
    tr_cov1 = float((x0.pow(2).sum() / (n1 - 1)).item())
    tr_cov2 = float((y0.pow(2).sum() / (n2 - 1)).item())

    # SVD: X = U S V^T
    # Cov = V (S^2/(n-1)) V^T  -> sqrt(Cov) = V (S/sqrt(n-1)) V^T
    # Trace(sqrt(sqrt(C1) C2 sqrt(C1))) can be computed in the low-rank subspace.
    _, sx, vhx = torch.linalg.svd(x0, full_matrices=False)
    _, sy, vhy = torch.linalg.svd(y0, full_matrices=False)

    # A = Vx^T Vy = Vhx @ Vhy^T
    a = vhx @ vhy.transpose(0, 1)  # (r1, r2)
    left = sx / math.sqrt(n1 - 1)  # (r1,)
    d2 = (sy * sy) / (n2 - 1)      # (r2,)

    # M' = diag(left) @ A @ diag(d2) @ A^T @ diag(left)
    # Compute efficiently without materializing diag matrices.
    b = (a * d2.unsqueeze(0)) @ a.transpose(0, 1)  # (r1, r1)
    m = (left.unsqueeze(1) * b) * left.unsqueeze(0)
    m = (m + m.transpose(0, 1)) * 0.5

    eigvals = torch.linalg.eigvalsh(m)
    eigvals = torch.clamp(eigvals, min=0.0)
    tr_sqrt = float(torch.sqrt(eigvals).sum().item())

    fid = mean_term + tr_cov1 + tr_cov2 - 2.0 * tr_sqrt
    return float(fid)


def compute_fid(
    real_paths: Sequence[Path],
    fake_paths: Sequence[Path],
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> FidResult:
    if len(real_paths) != len(fake_paths):
        raise ValueError(f"real/fake length mismatch: {len(real_paths)} vs {len(fake_paths)}")
    extractor, tfm = load_inception_feature_extractor(device=device)
    real_feats = compute_inception_features_with_extractor(extractor, tfm, real_paths, device=device, batch_size=batch_size)
    fake_feats = compute_inception_features_with_extractor(extractor, tfm, fake_paths, device=device, batch_size=batch_size)
    fid = frechet_distance_from_features(real_feats, fake_feats)
    return FidResult(
        fid=float(fid),
        n_real=int(real_feats.shape[0]),
        n_fake=int(fake_feats.shape[0]),
        feature_dim=int(real_feats.shape[1]) if real_feats.ndim == 2 else 0,
        impl="torchvision.inception_v3(avgpool)+svd_frechet",
    )
