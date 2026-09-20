# Ego action encoder

This package contains the model, objective, and inference logic for ZeVA's
two-stage visual action encoder. It is intentionally independent of the ICCL
implementation in `src/openpi/zeva`.

The public boundary is code-only. This directory does not include video
datasets, dataset acquisition or cleaning tools, checkpoints, optimizer state,
or machine-specific launch files.

## Interfaces

- Stage 1 consumes an RGB transition `[B,2,3,H,W]`, an action chunk
  `[B,L,D]`, and boolean step/dimension masks. It learns an environment
  bottleneck by reconstructing the future visual features.
- Stage 2 consumes triplets `[B,3,3,H,W]` and three aligned action chunks
  `[B,3,L,D]`. It learns task tokens while preserving the Stage-1 environment
  representation.
- Inference consumes already-sampled RGB frame pairs and returns task tokens
  plus environment tokens. Frame selection, temporal slowdown, and resizing
  remain the caller's responsibility.

For Ego videos, we recommend a one-second interval **after temporal slowdown**.
DODO is not slowed down, so one second is one second in the original clip.
These tokens are learned transition representations, not robot commands.

## Installation

```bash
python -m pip install -e 'pipelines/ego_action_encoder[vision,test]'
```

## Encode RGB pairs

Prepare a uint8 NPY array shaped `[N,2,H,W,3]` or `[N,2,3,H,W]`, then run:

```bash
python pipelines/ego_action_encoder/scripts/encode_pairs.py \
  --input /path/to/rgb_pairs.npy \
  --checkpoint /path/to/stage2_checkpoint.pt \
  --output /path/to/new_output
```

The checkpoint must contain `model` and `stage2_model_config`. DINOv2 is
loaded by the vision wrapper; applications that need offline loading can set
PyTorch's model cache or inject a backbone through the Python API.

## Training integration

The package deliberately does not prescribe a dataset format. Convert each
prepared batch to `Stage1Batch` or `Stage2Batch`, call `stage1_train_step` or
`stage2_train_step`, then pass the returned `.total` tensor to
`optimize_one_step`. This keeps the learned method reproducible without
embedding private storage layouts or data engineering infrastructure.

```python
from zeva_action_encoder.training import optimize_one_step, stage1_train_step

losses = stage1_train_step(
    model=model,
    visual_encoder=vision,
    batch=batch,
    loss_config=loss_config,
)
value = optimize_one_step(loss=losses.total, model=model, optimizer=optimizer)
```
