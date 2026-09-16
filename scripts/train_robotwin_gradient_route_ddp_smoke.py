"""Two-rank smoke for the opt-in Stage 2 gradient-routing DDP contract."""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from scripts.train_robotwin_stage2 import _ActionExpertGradientRouter
from scripts.train_robotwin_stage2 import _zero_parameter_gradient_link


class _TinyRoutePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action = nn.Parameter(torch.tensor(2.0))
        self.zeva = nn.Parameter(torch.tensor(5.0))

    def forward(self, value: torch.Tensor, *, foundation_only: bool) -> torch.Tensor:
        result = self.action * value
        if not foundation_only:
            result = result + self.zeva * value
        return result


def main() -> None:
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    policy = _TinyRoutePolicy().to(device)
    router = _ActionExpertGradientRouter([policy.action])
    router.begin_first_step_audit()
    wrapped = DistributedDataParallel(
        policy,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Run twice so the reducer must finish one routed iteration and start the next.
    # Each iteration uses two accumulated micro-steps, matching the production
    # global-batch path.  The residual-on backward is always no_sync; only the
    # final residual-off backward synchronizes accumulated gradients.
    for iteration in range(2):
        wrapped.zero_grad(set_to_none=True)
        accumulation_steps = 2
        for micro_step in range(accumulation_steps):
            value = torch.tensor(float(rank + 1 + micro_step), device=device)
            router.set_phase("residual_on")
            with wrapped.no_sync():
                on_loss = wrapped(value, foundation_only=False)
                on_loss.div(accumulation_steps).backward()
            if iteration == 0 and micro_step == 0:
                assert policy.action.grad is not None and policy.action.grad.item() == 0
                assert policy.zeva.grad is not None
            router.set_phase("residual_off")
            off_sync_context = (
                wrapped.no_sync() if micro_step + 1 < accumulation_steps else nullcontext()
            )
            with off_sync_context:
                off_loss = wrapped(value, foundation_only=True)
                off_loss = off_loss + _zero_parameter_gradient_link([policy.zeva], off_loss)
                off_loss.div(accumulation_steps).backward()

        # DDP averages the two rank-local micro-step means:
        # rank 0=(1+2)/2, rank 1=(2+3)/2, hence both final gradients are 2.
        torch.testing.assert_close(policy.action.grad, torch.tensor(2.0, device=device))
        torch.testing.assert_close(policy.zeva.grad, torch.tensor(2.0, device=device))
        if iteration == 0:
            router.assert_first_step_audit()
            router.end_first_step_audit()

    router.close()
    torch.distributed.barrier()
    if rank == 0:
        print("GRADIENT_ROUTE_DDP_SMOKE_OK", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
