"""Bounded real-data/real-PI PBD test. Does NOT promote or save a policy.

Random conditioning vectors deliberately test the integration, not CTE utility.
Run on an independently verified idle GPU, never Stage1's active card.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import torch
import tyro

from openpi.zeva.behavior_effect import ZevaPBD
from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS, prepare_robotwin_pi_image
from scripts.train_robotwin_zte import TorchCodecRoboTwinDataset


@dataclass
class Args:
    output: str
    handoff: str = "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    dataset: str = "/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data/adapter.json"
    distributed_steps: int = 0
    capacity_global256: bool = False


class TrainingHarness(torch.nn.Module):
    def __init__(self, foundation, pbd):
        super().__init__()
        self.foundation, self.pbd = foundation, pbd

    def forward(self, processed, features):
        prior = self.pbd.activate(*features)
        try:
            result = self.foundation(processed)
            flow = result[0] if isinstance(result, tuple) else result
            return flow.mean() + self.pbd.prior_loss(prior, processed["action"])
        finally:
            self.pbd.clear()


def distributed_smoke(foundation, processed, output, steps, started, capacity_global256=False):
    """Bounded two-rank capacity test; repeated real sample, no quality claims."""
    from torch.nn.parallel import DistributedDataParallel
    from contextlib import nullcontext

    if steps != 2 or torch.distributed.get_world_size() != 2:
        raise ValueError("This bounded DDP test requires two ranks and two optimizer steps.")
    batch, accumulation = (8, 16) if capacity_global256 else (1, 2)
    if capacity_global256:
        # Replication deliberately isolates full-PI memory/compute capacity.
        # It is not a full data-loader or generalization test.
        processed = {key: value.repeat(batch, *([1] * (value.ndim - 1)))
                     if isinstance(value, torch.Tensor) and value.shape[0] == 1 else value
                     for key, value in processed.items()}
    foundation.requires_grad_(True).train()
    foundation.model.gradient_checkpointing_enable()
    pbd = ZevaPBD().cuda().train()
    pbd.install(foundation.model)
    harness = DistributedDataParallel(TrainingHarness(foundation, pbd),
                                     device_ids=[torch.cuda.current_device()],
                                     find_unused_parameters=True, broadcast_buffers=False)
    optimizer = torch.optim.AdamW([
        {"params":foundation.parameters(), "lr":5e-6},
        {"params":pbd.parameters(), "lr":5e-5},
    ], weight_decay=1e-4)
    initial = pbd.effect_projector.weight.detach().clone()
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            features = [torch.randn(batch,256,device="cuda") for _ in range(3)]
            with harness.no_sync() if micro < accumulation - 1 else nullcontext():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = harness(processed, features)/accumulation
                loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(harness.parameters(),1.,error_if_nonfinite=True)
        assert norm > 0
        optimizer.step()
        losses.append(float(loss.detach())*accumulation)
        if torch.distributed.get_rank() == 0:
            print(json.dumps({"capacity_global256":capacity_global256,"optimizer_step":step+1,
                              "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3}),flush=True)
    assert not torch.equal(initial, pbd.effect_projector.weight)
    weight = pbd.effect_projector.weight.detach().clone()
    reference = weight.clone()
    torch.distributed.broadcast(reference, src=0)
    torch.testing.assert_close(weight,reference,rtol=0,atol=0)
    report = {"status":"PASS", "scope":"bounded real PI DDP/AdamW smoke; no promotable checkpoint",
              "ranks":2, "per_rank_batch":batch, "accumulation":accumulation,
              "smoke_global_batch":2*batch*accumulation, "repeated_real_sample":True,
              "formal_eight_rank_topology_tested":False,
              "optimizer_steps":steps, "losses_last_microbatch_rank0":losses,
              "effect_weights_updated_and_rank_identical":True,
              "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
              "elapsed_seconds":time.monotonic()-started}
    if torch.distributed.get_rank() == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report),flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main(args):
    if args.capacity_global256 and args.distributed_steps != 2:
        raise ValueError("Capacity mode requires the bounded two-update DDP test.")
    if args.distributed_steps:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group("nccl", device_id=torch.device("cuda",torch.cuda.current_device()))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(1000)
    started = time.monotonic()
    wrapper = RobotWinZevaPolicy.from_handoff(
        args.handoff, foundation_checkpoint=Path(args.handoff)/"checkpoint/pretrained_model-best-v1",
        install_injection_hooks=False, device="cuda",
    )
    foundation = wrapper.foundation
    source = TorchCodecRoboTwinDataset(args.dataset, "train")
    index = next(i for i,r in enumerate(source.dataset._records) if r["key"][1] == "beat_block_hammer")
    record = source.dataset._records[index]
    sample = dict(source.dataset[int(source.dataset._cumulative[index])])
    images = source.read_images(record, [0])
    for key in ROBOTWIN_CAMERA_KEYS:
        sample[key] = prepare_robotwin_pi_image(images[key][0], name=key)
    processed = wrapper.preprocessor(sample)
    if processed["action"].ndim == 2:
        processed["action"] = processed["action"].unsqueeze(0)
    if args.distributed_steps:
        distributed_smoke(foundation,processed,output,args.distributed_steps,started,args.capacity_global256)
        return
    foundation.eval()
    # Inactive hooks must be exactly identity for baseline samples.
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    with torch.no_grad():
        baseline = foundation.predict_action_chunk(processed)
    pbd = ZevaPBD().cuda().eval()
    pbd.install(foundation.model)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    with torch.no_grad():
        inactive = foundation.predict_action_chunk(processed)
    torch.testing.assert_close(inactive, baseline, rtol=0, atol=0)
    assert baseline.shape == (1,50,16), baseline.shape
    features = [torch.randn(1,256,device="cuda") for _ in range(3)]
    pbd.activate(*features)
    try:
        with torch.no_grad():
            conditioned = foundation.predict_action_chunk(processed)
    finally:
        pbd.clear()
    assert conditioned.shape == (1,50,16) and torch.isfinite(conditioned).all()
    print("INFERENCE_PASS: inactive identity, conditioned H50/EEF16", flush=True)
    # Exercise a real full-PI backward, not just a frozen-backbone adapter graph.
    foundation.requires_grad_(True).train()
    foundation.model.gradient_checkpointing_enable()
    pbd.train()
    prior = pbd.activate(*features)
    try:
        result = foundation(processed)
        flow = result[0] if isinstance(result, tuple) else result
        total = flow.mean() + pbd.prior_loss(prior, processed["action"])
    finally:
        pbd.clear()
    total.backward()
    gradients = {}
    for name, parameter in pbd.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            gradients[name] = float(parameter.grad.float().norm())
    for name in ("global_projector.weight", "effect_projector.weight", "action_prior.dist_head.2.weight"):
        assert gradients.get(name, 0) > 0, (name, gradients.get(name))
    pi_gradients = {name:float(p.grad.float().norm()) for name,p in foundation.named_parameters()
                    if p.grad is not None and p.requires_grad and name.endswith("action_out_proj.weight")}
    assert pi_gradients and all(value > 0 for value in pi_gradients.values())
    assert all(torch.isfinite(p.grad).all() for p in foundation.parameters() if p.grad is not None)
    report = {"status":"PASS", "scope":"real PI/data single-rank integration; no trained policy or success-rate evidence",
              "inactive_hooks_bit_exact":True, "conditioned_actions_shape":list(conditioned.shape),
              "full_pi_backward":True, "loss":float(total.detach()), "pbd_gradients":gradients,
              "pi_gradients":pi_gradients, "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
              "elapsed_seconds":time.monotonic()-started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
