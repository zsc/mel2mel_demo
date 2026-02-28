#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WAVs to SpeechT5-style log-mel PNGs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("samples10k"),
        help="Input directory containing WAV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("samples10k_mel_png"),
        help="Output directory for PNGs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNGs (default: skip existing).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root: Path = args.input
    output_root: Path = args.output
    overwrite: bool = args.overwrite
    progress_every: int = max(1, args.progress_every)

    # Parameters per ../audio_avif/mel.md (SpeechT5 HiFi-GAN log-mel)
    target_sr = 16000
    n_fft = 1024
    win_length = 1024
    hop_length = 256
    n_mels = 80
    fmin = 80.0
    fmax = 7600.0
    mel_floor = 1e-10

    mel_basis = librosa.filters.mel(
        sr=target_sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=False,
        norm="slaney",
    )

    wav_paths = sorted(input_root.rglob("*.wav"))
    total = len(wav_paths)
    print(f"Total wav files: {total}", flush=True)

    for idx, wav_path in enumerate(wav_paths, 1):
        out_path = output_root / wav_path.relative_to(input_root).with_suffix(".png")
        if out_path.exists() and not overwrite:
            continue

        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) == 2:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

        stft = librosa.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
        )
        magnitude = np.abs(stft)

        mel = mel_basis @ magnitude
        mel = np.maximum(mel, mel_floor)
        log_mel = np.log10(mel).T  # (T, 80)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(out_path, log_mel.T, origin="lower", cmap="magma")

        if idx % progress_every == 0 or idx == total:
            print(f"[{idx}/{total}] {wav_path.name} -> {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
