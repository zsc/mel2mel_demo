# Lessons Learnt (Qwen-Image-Edit-2511 LoRA for "mel-spectrum")

This repo/workflow ended up being a fairly standard "paired image-to-image" LoRA setup, with a few gotchas that are easy to forget. These notes capture what worked, what was brittle, and what to monitor.

## Data Preparation

- Keep the task definition simple and stable:
  - Source/control: `samples10k_lab_mel_png` (or the DTW-mixed variant).
  - Target: `samples10k_mel_png`.
  - Prompt: constant `"mel-spectrum"` for every sample.
- Filter obvious outliers early:
  - Width ratio (`tgt_w / src_w`) outliers create bad resizing/cropping behavior and poison training.
  - If they are a small fraction, skipping is usually better than trying to "fix" them mid-pipeline.
- Alignment can help, but should be optional:
  - DTW-aligning mel spectra can improve correspondence.
  - Mixing DTW-aligned and raw pairs (e.g. 50/50) reduces overfitting to DTW artifacts.
- Make everything reproducible:
  - Save a metadata sidecar for each pair (ratio, whether DTW was applied, original sizes, etc.).
  - Fix the train/val split once (a stable `split.json` is worth it).

Relevant tool:
- `tools/build_mel_spectrum_dtw_pairs.py` builds a filtered dataset and can create a DTW-mixed control set.

## Training (DiffSynth + LoRA)

- Dynamic resolution is fine for this task:
  - Let the dataloader crop/resize by `max_pixels` and divisibility (16).
  - Keep `height/width` unset unless you have a strong reason to force a fixed resolution.
- Match training and eval preprocessing:
  - Eval should mirror DiffSynth `ImageCropAndResize` so metrics are comparable over time.
- Loss curves are not automatically logged in DiffSynth by default:
  - Checkpoints saving works, but "loss vs step" is not written to disk unless you add it.
  - In practice, rely on periodic eval + artifact inspection.
- VRAM management:
  - Run eval with `--low_vram 1` when training is using the GPU.
  - When training is done and GPU is free, eval on GPU is faster.

## Evaluation and Monitoring

### What to monitor

- Pixel-space L1 metrics:
  - `l1_control_to_target` is the baseline after resizing control to target.
  - `l1_pred_to_target` should go down versus baseline.
  - `improve_l1 = l1_control_to_target - l1_pred_to_target` (positive is better).
- FID:
  - Useful as a "distributional sanity check" across a fixed val set.
  - Always compare to a baseline, e.g. FID(`control_to_target`, `target`) vs FID(`pred`, `target`).

### FID implementation note

- The current FID is **not** the canonical `pytorch-fid`/`torch-fidelity` implementation.
- We used a lightweight implementation based on `torchvision` `inception_v3(avgpool)` features and an SVD-based Frechet computation (no SciPy dependency).
- Treat FID values as *relative* within this project/config (same preprocessing, same feature extractor), not as numbers to compare across unrelated projects.

Relevant tools:
- `tools/eval_diffsynth_qwen_image_edit_lora.py` generates per-sample artifacts and `metrics.json` (L1 + optional FID).
- `tools/compute_fid_from_eval_dir.py` computes FID from an eval directory containing `*/target.png` and `*/pred.png`.
- `tools/make_eval_report_html.py` creates an HTML report from multiple eval runs (metrics + a small gallery).

## What Worked (Observed Trends)

- On the fixed `val=97` set (20 inference steps), later checkpoints consistently improved:
  - FID decreased across steps (lower is better).
  - Mean `improve_l1` increased (more samples improved over the resized-control baseline).
- The baseline FID (`control_to_target` vs `target`) was much worse than model predictions, which is a good sign that the model is learning a non-trivial transform.

## Practical Workflow (Recommended)

1. Build dataset once (with ratio filtering and optional DTW mix).
2. Train with periodic checkpoint saves.
3. For each checkpoint you care about:
   - Run eval on a fixed val set (`split=val`, `num_samples=val_size`) with a fixed `num_inference_steps`.
   - Track `metrics.json` and sample images (`pred.png` vs `target.png`).
4. Generate a single HTML report to compare checkpoints at a glance.

