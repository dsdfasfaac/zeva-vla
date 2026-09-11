"""Measure actual cross-episode positives without decoding any training video."""
import json
from pathlib import Path

from scripts.train_robotwin_zte_v2 import Args, GroupedTaskSampler, PairedTaskSampler
from scripts.train_robotwin_zte import RobotWinZTEEpisodeDataset


def audit(dataset, sampler_type, *, replicas=4, batch_size=8, seed=1000):
    task_of = {index: task for task, group in enumerate(dataset.indices_by_task) for index in group}
    visited = []
    anchors = positives = negatives = 0
    for rank in range(replicas):
        options = dict(seed=seed, replicas=replicas, rank=rank)
        if sampler_type is PairedTaskSampler:
            options["batch_size"] = batch_size
        indices = list(sampler_type(dataset, **options))
        for start in range(0, len(indices), batch_size):
            batch = [index for index in indices[start:start + batch_size] if index >= 0]
            visited.extend(batch)
            for index in batch:
                anchors += 1
                positives += any(other != index and task_of[other] == task_of[index] for other in batch)
                negatives += any(task_of[other] != task_of[index] for other in batch)
    return {
        "episode_count": anchors,
        "exactly_once_complete_coverage": sorted(visited) == list(range(len(dataset))),
        "cross_episode_positive_fraction": positives / max(1, anchors),
        "different_task_negative_fraction": negatives / max(1, anchors),
        "replicas": replicas,
        "batch_size_per_rank": batch_size,
    }


def main():
    args = Args()
    dataset = RobotWinZTEEpisodeDataset(
        Path(args.dataset_root) / "adapter.json", subset="train",
        transition_stride=15, effect_steps=15, executed_action_steps=15,
        goal_embeddings=args.goal_embeddings,
    )
    print(json.dumps({
        "schema": "zte-v2-task-positive-sampler-audit-v1",
        "task_count": dataset.task_count,
        "interleaved": audit(dataset, GroupedTaskSampler),
        "paired": audit(dataset, PairedTaskSampler),
    }, indent=2))


if __name__ == "__main__":
    main()
