"""NON-PROMOTABLE early-CTE/real-data/full-PI integration smoke.

Constructs a tiny ten-task TRAIN-only fixture bank (one episode/task), not a
formal exported artifact. Tests two validation decision boundaries with actual
expert history. No optimizer, model checkpoint, validation gate or success-rate
report is emitted. Formal loaders/exporters retain the strict epoch80 gate.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import tyro

from openpi.zeva.behavior_effect import SCHEMA
from openpi.zeva.behavior_effect_policy import ZevaBehaviorEffectPolicy, load_cte, file_sha
from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS, prepare_robotwin_pi_image
from scripts.train_robotwin_behavior_effect_cte import Episodes


@dataclass
class Args:
    checkpoint: str
    output: str
    retrieval: str = "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911/task_retrieval.pth"
    handoff: str = "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    dataset: str = "/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data/adapter.json"
    tasks: str = "configs/robotwin_zeva_advantage10.json"


@torch.no_grad()
def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(1000)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # The ONLY relaxed loader is inside this explicitly non-promotable smoke.
    cte = load_cte(args.checkpoint, require_gate=False)
    wrapper = RobotWinZevaPolicy.from_handoff(
        args.handoff, foundation_checkpoint=Path(args.handoff)/"checkpoint/pretrained_model-best-v1",
        goal_embedding_checkpoint=Path(args.handoff)/"checkpoint/pretrained_model",
        install_injection_hooks=False, device="cuda")
    tasks = json.loads(Path(args.tasks).read_text())["task_names"]
    train = Episodes(args.dataset, "train", tasks)
    keys, values, train_record_ids = [], [], []
    for task in tasks:
        index = next(i for i, ri in enumerate(train.indices)
                     if train.source.dataset._records[ri]["key"][1] == task)
        sample = train[index]
        images = sample["images"].unsqueeze(0).cuda()
        actions = wrapper.action_normalizer.normalize(sample["actions"].unsqueeze(0).cuda())
        cache, phases = None, []
        for t in range(images.shape[1] - 1):
            phase, _, cache = cte.step(images[:, t], None if t == 0 else actions[:, t-1], cache)
            phases.append(phase[0])
        phase = torch.stack(phases)
        keys.append(F.normalize(cte.task_head(phase).mean(0), dim=-1).cpu())
        values.append(F.normalize(phase.mean(0), dim=-1).cpu())
        record = train.source.dataset._records[sample["record_index"]]
        train_record_ids.append(f"{record['key'][0]}:{record['key'][1]}:{record['episode_index']}")
        del sample, images, actions, cache, phases, phase
    bank = {"schema": SCHEMA + "-artifacts", "bank_subset": "train", "tasks": tasks,
            "keys": torch.stack(keys), "values": torch.stack(values), "task_ids": torch.arange(len(tasks))}
    retrieval = torch.load(args.retrieval, map_location="cpu", weights_only=False)
    policy = ZevaBehaviorEffectPolicy(wrapper, cte, bank, retrieval).cuda().eval().requires_grad_(False)
    validation = Episodes(args.dataset, "validation", tasks)
    index = next(i for i, ri in enumerate(validation.indices)
                 if validation.source.dataset._records[ri]["key"][1] == tasks[0])
    sample = validation[index]
    record_index = sample["record_index"]
    record = validation.source.dataset._records[record_index]
    record_id = f"{record['key'][0]}:{record['key'][1]}:{record['episode_index']}"
    assert record_id not in train_record_ids
    images = sample["images"][:2].unsqueeze(0).cuda()
    raw_previous = sample["actions"][:1].cuda()
    previous = policy.action_normalizer.normalize(raw_previous)
    phase0, effect0, cache = cte.step(images[:, 0], None, None)
    phase1, effect1, _ = cte.step(images[:, 1], previous, cache)
    phase_cache, effect_cache = (phase0, phase1), (effect0, effect1)
    captured = []
    original_step = cte.step

    def traced_step(*positional, **keywords):
        result = original_step(*positional, **keywords)
        captured.append((result[0].clone(), result[1].clone()))
        return result

    cte.step = traced_step
    checks = []
    policy.reset()
    try:
        for t, frame in enumerate((0, 15)):
            raw = dict(validation.source.dataset[int(validation.source.dataset._cumulative[record_index])+frame])
            decoded = validation.source.read_images(record, [frame])
            for key in ROBOTWIN_CAMERA_KEYS:
                raw[key] = prepare_robotwin_pi_image(decoded[key][0], name=key)
            processed = policy.preprocessor(raw)
            torch.manual_seed(500+t)
            torch.cuda.manual_seed_all(500+t)
            online = policy.predict_action_chunk(processed, task=raw["task"],
                                                 executed_actions=None if t == 0 else raw_previous)
            torch.testing.assert_close(captured[-1][0], phase_cache[t], rtol=2e-4, atol=2e-4)
            torch.testing.assert_close(captured[-1][1], effect_cache[t], rtol=2e-4, atol=2e-4)
            global_token = policy.memory(policy.language(raw["task"]))
            policy.foundation.reset()
            policy.pbd.activate(global_token, phase_cache[t], effect_cache[t])
            torch.manual_seed(500+t)
            torch.cuda.manual_seed_all(500+t)
            try:
                cached = policy.foundation.predict_action_chunk(processed)
            finally:
                policy.pbd.clear()
            torch.testing.assert_close(online, cached, rtol=3e-4, atol=3e-4)
            assert online.shape == (1, 50, 16) and torch.isfinite(online).all()
            physical = policy.postprocessor(online)
            assert physical.shape == (1, 50, 16) and torch.isfinite(physical).all()
            # Exercise the exact explicit-noise and effect-off API used by the
            # new validation reporter, independently of global RNG state.
            noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(987+t)).cuda()
            policy.pbd.activate(global_token, phase_cache[t], effect_cache[t], include_effect=False)
            try:
                off = policy.foundation.predict_action_chunk(processed, noise=noise.clone(), num_steps=10)
                torch.manual_seed(9191)
                off_repeat = policy.foundation.predict_action_chunk(processed, noise=noise.clone(), num_steps=10)
            finally:
                policy.pbd.clear()
            torch.testing.assert_close(off, off_repeat, rtol=0, atol=0)
            checks.append({"frame":frame, "max_phase_error":float((captured[-1][0]-phase_cache[t]).abs().max()),
                           "max_action_error":float((online-cached).abs().max()),
                           "explicit_noise_bit_exact":True})
        policy.reset()
        assert policy._cache is None and policy._global_token is None
    finally:
        cte.step = original_step
    report = {"status":"PASS", "promotable":False, "scope":"early CTE, tiny train-only fixture bank, real PI integration only",
              "cte_checkpoint":args.checkpoint, "cte_sha256":file_sha(args.checkpoint),
              "retrieval_sha256":file_sha(args.retrieval), "train_fixture_record_ids":train_record_ids,
              "validation_record_id":record_id, "checks":checks, "episode_reset":True,
              "source_sha256":file_sha(__file__), "elapsed_seconds":time.monotonic()-started,
              "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
