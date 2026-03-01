# Lessons Learnt (Qwen-Image-Edit-2511 LoRA, Paired Mel PNG Editing)

This repo/workflow is a practical "paired image-to-image" LoRA setup on Qwen-Image-Edit-2511 using DiffSynth. It worked well overall, but a few details were surprisingly brittle (and caused hard crashes / misleading evals).

Two main experiments were run:

- `mel-spectrum`: source `samples10k_lab_mel_png` (optionally DTW-mixed) -> target `samples10k_mel_png`, prompt `"mel-spectrum"`.
- `improve-mel` (no DTW): source `samples10k_say_mel_png` -> target `samples10k_mel_png_trim_end`, prompt `"improve-mel"`.

## Data Preparation

- Keep the task definition simple and stable:
  - Always build explicit A/B pairs (by filename stem or a manifest).
  - Keep prompt constant per experiment (the "magic word").
- Filter obvious outliers early:
  - Width ratio outliers (`tgt_w / src_w`) produce extreme crops and poison supervision.
  - If outliers are a small fraction, skip them instead of "fixing" them later.
- DTW alignment:
  - Helpful for some mel-spectrum transforms, but keep it optional.
  - Mixing DTW-aligned and raw pairs (for example 50/50) can reduce overfitting to DTW artifacts.
  - For `improve-mel` the policy was explicitly "no DTW".
- Dynamic-resolution safety (this caused a real training crash):
  - DiffSynth `ImageCropAndResize` uses dynamic sizing with a division factor of 16.
  - If an input dimension is smaller than 16, `floor(dim/16)*16` can become `0`, which later crashes the VAE/Conv stack.
  - Fix applied: clamp dynamic `height/width` to at least the division factor in `/root/autodl-tmp/DiffSynth-Studio/diffsynth/core/data/operators.py`.
  - Alternative policy: pre-filter or pre-resize any image with `w < 16` or `h < 16`, or force a fixed `--height/--width`.
- Make everything reproducible:
  - Keep `metadata.diffsynth.jsonl` and `split.json` in the prepared dataset dir.
  - Keep a record of filtering decisions (ratio thresholds, DTW usage).

Relevant tool:
- `tools/build_mel_spectrum_dtw_pairs.py` builds a filtered dataset and can create a DTW-mixed control set.

## Training (DiffSynth + LoRA)

- Use the model-specific flags:
  - Qwen-Image-Edit-2511 needs `--zero_cond_t`.
  - Disable edit-image internal resizing: `--edit_image_auto_resize 0`.
    - Important: this flag is not "match target size". It resizes edit images toward ~1MP, which changes the task.
- Dynamic resolution works, but know the pitfalls:
  - Leaving `--height/--width` unset lets DiffSynth crop/resize per-sample by `--max_pixels` and divisibility (16).
  - Tiny-width samples can crash if you do not clamp or filter them (see above).
- Throughput knobs:
  - `--dataset_num_workers` matters; too low under-utilizes GPU.
  - Gradient checkpointing trades speed for memory. Disabling it improved it/s on this machine, but increases VRAM pressure.
- Resume training is not "exact resume":
  - DiffSynth default runner does not save/load optimizer state, scheduler state, or dataloader state.
  - Restarting with `--lora_checkpoint <step-*.safetensors>` is effectively "load weights and continue with fresh AdamW".
  - Step naming is per-run: if you restart from `step-2000`, the next save is still called `step-2000` unless you account for offsets. Track offsets in notes or encode them in the run name.
- Loss curves:
  - Loss is not persisted to disk by default; the tqdm bar alone is not a curve.
  - Practical workaround: rely on periodic eval + visual inspection, or patch the runner to log `loss` to a CSV/TensorBoard.

## Evaluation and Monitoring

### What to monitor

- Pixel-space L1 metrics:
  - `l1_control_to_target`: baseline after resizing control to target size.
  - `l1_pred_to_target`: model quality proxy.
  - `improve_l1 = l1_control_to_target - l1_pred_to_target` (positive is better).
- FID:
  - Useful as a "distribution sanity check" on a fixed val set.
  - Only compare within the same preprocessing + implementation.

### Practical pitfalls (VRAM + mid-training eval)

- Pausing training with SIGSTOP does not free VRAM.
  - The process keeps its allocations; eval on the same GPU can still OOM even while "paused".
- Options that actually work:
  - Run eval after training finishes (GPU free).
  - Run eval on a different GPU.
  - Reduce eval load: fewer samples, fewer steps, smaller FID batch size; put FID on CPU (`--fid_device cpu`).

### Tooling used

- `tools/eval_diffsynth_qwen_image_edit_lora.py`:
  - Saves per-sample `control.png`, `control_to_target.png`, `pred.png`, `target.png`.
  - Writes `metrics.json` (L1 + optional FID).
- `tools/auto_eval_checkpoint_then_continue.py`:
  - Watches for a checkpoint then runs eval.
  - Limitation: if training holds most VRAM, eval on the same GPU can still OOM even if you pause the process.
- `tools/make_eval_report_html.py`:
  - Builds a single HTML page to compare checkpoints and browse a small gallery.
  - Serve via HTTP for reliable relative image loading:
    - `cd <run_dir> && python -m http.server 8000`
- FID implementation note:
  - The current FID is not canonical `pytorch-fid`/`torch-fidelity`.
  - It's a lightweight `torchvision inception_v3(avgpool)` feature extractor + SVD-based Frechet.
  - Treat values as relative within this repo/config.

## `improve-mel` Results (No DTW)

Dataset (after ratio filtering):
- Total paired samples: 9641
- Split: train 9545 / val 96
- Prompt: constant `"improve-mel"`

Eval protocol:
- split `val`, `num_samples=96`, `num_inference_steps=20`, `seed=0`

Checkpoint metrics (val=96):
- step-2000: FID 67.0661, mean_improve_l1 0.015635
- step-4000: FID 63.4878, mean_improve_l1 0.013300
- step-6000: FID 70.2898, mean_improve_l1 0.015436
- step-8000: FID 67.5858, mean_improve_l1 0.018874 (best mean_improve_l1)
- step-9641: FID 63.3708, mean_improve_l1 0.014633 (best FID)

Takeaways:
- Metrics were not monotonic. Pick checkpoints by the metric you care about, and always sanity check visually.
- "Best FID" and "best pixel improve" did not coincide in this run.

## `mel-spectrum` Notes (DTW Optional)

- DTW alignment can make supervision "easier", but it also changes the underlying transform.
- A small DTW mix-in (instead of 100% DTW) tended to be a safer default when you are not sure if DTW artifacts will hurt.

## Packaging and Sharing

- For checkpoint comparisons, it is useful to package only eval PNG artifacts (not checkpoints):
  - One tarball per checkpoint, then a "big tar" of those tars.
  - This makes offline browsing and external sharing easy without moving `.safetensors`.

## Operational Notes

- Keep everything offline/local:
  - `DIFFSYNTH_SKIP_DOWNLOAD=true`, `DIFFSYNTH_MODEL_BASE_PATH=/.autodl-model/data`
  - `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`
- Make runs self-describing:
  - Save the exact train cmd (a `cmd.sh`) inside each run dir.
  - Save PID files so you can monitor/kill the correct process.
  - Set `PYTHONPATH=/root/autodl-tmp/DiffSynth-Studio` when running helper scripts so `import diffsynth` is reliable.
