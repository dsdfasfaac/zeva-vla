# RoboTwin clean-only post-training

This package isolates the code-level contract used for clean-only RoboTwin
post-training while leaving ICCL code untouched.

The released boundary assumes that RoboTwin episodes are already prepared. It
does not download data, generate trajectories, decode a private storage layout,
or reproduce cluster launch infrastructure. Applications provide a dataset
factory and reuse LeRobot's native optimization, checkpoint, and resume loop.

## Frozen sample contract

- training split: RoboTwin `Clean` only;
- observation state: Joint14;
- images: high, left-wrist, and right-wrist RGB cameras;
- action: 50-step, chunk-start-relative EEF16;
- EEF order: left xyz/xyzw, right xyz/xyzw, left/right gripper;
- pose deltas: spatial deltas on the fixed main-camera axes;
- gripper: absolute command with model-side `1=open`;
- sampling: natural prepared-data sampling, without task augmentation.

`absolute_to_chunk_start_eef16` and
`align_grippers_to_model_convention` expose the two action-boundary transforms.
`PreparedRoboTwinDatasetAdapter` validates an existing dataset without changing
its sampling distribution.

## Training hook

Install the package alongside the LeRobot version used by the repository:

```bash
python -m pip install -e 'pipelines/robotwin_clean[train,test]'
```

Expose a function with LeRobot's dataset-factory signature:

```python
def make_datasets(train_config):
    return prepared_train_dataset, prepared_validation_dataset
```

Then run the upstream trainer through the isolated entry point. All regular
LeRobot CLI arguments continue to be handled by LeRobot itself.

```bash
export ZEVA_ROBOTWIN_DATASET_FACTORY=my_project.robotwin:make_datasets
python pipelines/robotwin_clean/scripts/train.py --config_path=/path/to/train_config.json
```

The example JSON records the method-level choices only. Dataset paths,
checkpoints, normalization artifacts, and compute-specific launch arguments
belong to the calling application.
