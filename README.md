<div align="center">

# Zeva on RoboTwin PI0.5

**In-Context Causal Learning for Generalizable Embodied Manipulation**

Fu Chen*, Xin Ding*, Bingjia Huang, Xiangyu Li, Mingju Wang, Jiawei He,
Kun Li, Wei Sun, Yunxin Liu, Hao Wu, Ting Cao

</div>

## Selected foundation

This implementation uses the released RoboTwin LeRobot PI0.5 handoff as its only
primary foundation path:

```text
/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
```

The handoff is read-only. `RobotWinZevaPolicy.from_handoff()` loads its saved
`config.json`, `model.safetensors`, tokenizer, preprocessor, postprocessor, and
normalization tensors. It also compares a checkpoint tensor bit-for-bit after
loading so a failed or partial `from_pretrained()` cannot silently create a
random foundation.

The frozen interface is:

| Field | Exact contract |
|---|---|
| State | absolute Joint14 |
| Cameras | `cam_high`, `cam_left_wrist`, `cam_right_wrist`, RGB 480x640 |
| Language | saved PI0.5 tokenizer, maximum 200 tokens |
| Action | chunk-start-relative EEF16 |
| Chunk | PI predicts 50 steps; controller executes the first 15, then replans |
| Slots | left xyz+xyzw, right xyz+xyzw, left/right gripper |
| Gripper | model convention `1=open`, slots 14 and 15 |
| Processor | state/action mean-std, visual identity; saved statistics only |
| PI internals | padded state/action 32, BF16, 10 flow inference steps |

The statistics are never recomputed. The complete contract can be checked
without loading the 9 GB checkpoint:

```bash
PYTHONPATH=src python3 scripts/verify_robotwin_handoff.py
```

## Zeva architecture

The selected PI0.5 has already been trained on the same RoboTwin distribution.
The formal Stage 2 therefore freezes **all PI0.5 weights**, including PaliGemma,
the 300M action expert, and every action/time projection. It trains only the
2.9M Zeva fusion/action-prior parameters. ZTE, the causal bank, and task
retrieval also remain frozen. At deployment all model parameters are frozen.
The Zeva path is:

- The Mamba Causal Transition Encoder consumes all three camera views before an
  executed chunk, the exact normalized EEF16 chunk, and the three resulting
  views. It predicts phase, causal signal, progress, and visual effect.
- Brief Interaction Trace stores recent evidence within the current attempt.
- Persistent Interaction Memory consolidates phase-matched evidence across
  attempts in the same fixed episode.
- Frozen PI0.5 task-language embeddings identify a ZTE task prototype through
  a calibrated retrieval head shared by training and deployment.
- Retrieved causal context is projected once and broadcast across H50 action
  tokens. A diagonal Gaussian head predicts normalized H50 EEF16 `mean` and
  `log_std`; its per-step mean is projected separately. Both residuals are
  added only to the noisy-action embeddings of the 300M action expert, while
  Gaussian NLL supervises both prior parameters. Nothing is injected before
  or into the frozen PaliGemma vision-language backbone.

The two injection projections are zero initialized. Each gate starts at 0.01
and is multiplied by a task-language/H15-phase router bounded to `[0,2]`, then
by retrieval confidence. Thus a routed gate begins at 1% and remains bounded
near the protected PI path while it can suppress harmful tasks/phases. No token is inserted, and
the PI prefix length, masks, position IDs, and action-expert shapes remain
unchanged. Consequently a new adapter wrapper is bit-identical to the selected
foundation for the same random seed.

## H100 runtime

Use the exact LeRobot source and dependency overlays contained in the handoff.
Install only the bundled Mamba kernels into the Zeva-owned dependency directory:

```bash
cd /mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA
bash scripts/setup_robotwin_zeva_runtime.sh
```

On the current aigc29 training host, the canonical staged dataset adapter is:

```text
/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data/adapter.json
```

If it is absent on a chosen H100, first run the handoff's read-only staging
script to copy the NFS source into that host's `/data1`; do not alter the
handoff itself.

## Verification

Run the full one-GPU checkpoint smoke (it uses a synthetic observation with the
exact contracted shapes, so staged training data is not required):

```bash
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
runtime=$handoff/runtime
zeva_deps=/data1/dingxin/zeva-runtime-deps
system_site=$(python3 -c 'import site; print(site.getsitepackages()[0])')

PYTHONPATH="$zeva_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-main-deps-py311-v1:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$PWD/src" \
CUDA_VISIBLE_DEVICES=0 python3 scripts/smoke_robotwin_zeva.py
```

The smoke verifies `[1,50,16]` normalized and raw output shapes and checks that
the zero-init wrapper exactly matches direct inference through the selected
PI0.5 checkpoint.

RoboTwin Stage 2 additionally fails fast unless the model runtime reports the
handoff-native Transformers 5.5.4 and tokenizers 0.22.2 distribution metadata.
The host-compatible tokenizers extension remains 0.21.4. Closed-loop anchor
tests showed that the former Transformers 4.53 compatibility shims can preserve
checkpoint tensors yet change PI0.5 rollout behavior; consequently those shims
are disabled under Transformers 5 and all 4.53-trained checkpoints are excluded
from final reporting. The exact runtime versions are stored in every Stage 2
manifest.

## Three-stage training pipeline

Stage 1 learns ZTE and calibrates the exact task/phase retrieval inputs used at
deployment. Stage 2 freezes PI0.5, Stage 1, and retrieval artifacts and trains
only Zeva's protected residual modules. Stage 3 is optional and is considered only after paired
baseline/Zeva closed-loop evaluation. The stages must not be collapsed into one
joint optimization.

Replacing only the PI base does not automatically invalidate Stage 1. ZTE,
the bank, and retrieval can be reused when the new base keeps the same model
architecture, tokenizer/vocabulary, image/state/action contract, H50/H15
protocol, and normalization, and Stage 1's frozen task-embedding table is
loaded as an explicit hashed artifact. In that compatible policy-only case,
restart Stage 2 from the new PI base. The current loader instead clones that
table from the selected handoff checkpoint, so a raw checkpoint-path swap is
not yet this decoupled mode: re-export the B0 embeddings and rebuild Stage 1
artifacts, or first implement explicit loading of the old frozen table.

The official data boundary is the handoff's 50-task RoboTwin corpus: 50 Clean
and 500 Randomized episodes per task, with the frozen episode-level 95/5 split.
Every stage uses Joint14, three RGB cameras, chunk-start-relative EEF16, and
the saved mean/std processor statistics. PI0.5 keeps its native H50 prediction
contract. The environment executes only the first H15 before replanning, so
ZTE action-effect transitions use H15 throughout training, banking, gating,
and deployment.

| Stage | Frozen | Trainable | Required output |
|---|---|---|---|
| 1. ZTE and retrieval calibration | PI0.5 weights | ZTE, then a separate task-language retrieval head | `zte_best.pth`, train95 bank, H15 live-query cache, `task_retrieval.pth` |
| 2. Protected Zeva adapter | complete PI0.5, ZTE, bank, retrieval head | task/context fusion, Gaussian prior, two action-embedding projectors, task/phase gate router | bit-identical `model.safetensors`, `zeva_adapter.pth`, optimizer state |
| 3. Optional post-Stage2 adaptation | ZTE, bank, retrieval | selected modules determined by paired Stage2 results | selectively tuned checkpoint |

### Stage 1: train ZTE and build the training causal bank

For every transition, ZTE consumes:

```text
(three-camera observation at t,
 actually executed normalized EEF16 prefix [15,16],
 three-camera effect observation at t+15)
```

It is trained on every H15 boundary of each complete episode. Before training,
the frozen PI0.5 tokenizer and language embedding table encode only the
invariant task instruction (plus the tokenizer's terminal newline) as `g`.
The three 640x480 views are split and independently resized before a shared
ImageNet-pretrained ResNet-18 and learned view fusion encode `s0`;
`F_init(g, s0)` seeds Mamba exactly once at frame zero, matching deployment.
Every subsequent H15 transition is consumed consecutively and its output
describes the post-action state. Stage 1 v5 separates an episode-global task
coordinate from the local causal phase: the global key pools the complete
episode and adds a residual projection of the frozen task-language embedding,
while the phase token is updated at every transition. The objective is:

```text
L_stage1 =
    lambda_effect * L_effect_prediction
  + lambda_action * L_action_reconstruction
  + lambda_task * L_task_contrastive
  + lambda_phase * L_phase_progress
  + lambda_phase_key * L_phase_key_alignment
  + lambda_mono * L_phase_monotonic
```

The effect target is `EMA_vision(s_after) - EMA_vision(s_before)`, so online
BatchNorm drift is not mistaken for physical change. The action head predicts
chunk `a_t` from `B_{t-1}` (B0 for the first chunk); the same action is never
fed into Mamba before being used as its own reconstruction target. Scalar
progress is accumulated from positive transition hazards, giving an exact
`p0=0` and monotonic batched/stateful updates by construction. Phase-key
supervision uses an order-preserving Gaussian RBF coordinate with no periodic
wrap-around.

Export the frozen per-episode PI0.5 goal embeddings once:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/export_robotwin_goal_embeddings.py \
  --output-path /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt \
  --batch-size 64
```

The formal 8-H100 Stage 1 run is:

```bash
cd /mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA
bash scripts/train_robotwin_zte_8gpu.sh \
  --goal-embeddings /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt \
  --save-dir /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte \
  --steps 30000 \
  --batch-size 1 \
  --num-workers 4 \
  --transition-stride 15 \
  --effect-steps 15 \
  --executed-action-steps 15 \
  --save-freq 1000 \
  --eval-batches 0
```

This gives a global batch of eight complete, episode-bounded sequences per
optimizer step. A task-paired distributed sampler guarantees cross-episode
same-task positives on paired ranks; the task objective performs a
differentiable cross-rank supervised contrastive loss on explicit episode-global
keys at temperature 0.03. Classification against 50 learned task prototypes is
only a weight-0.1 auxiliary. `--eval-batches 0`
evaluates the complete validation5 split at every checkpoint. Resume an
interrupted run without changing its manifest:

```bash
bash scripts/train_robotwin_zte_8gpu.sh \
  --resume-checkpoint /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_latest.pth \
  --goal-embeddings /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt \
  --save-dir /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte
```

After selecting `zte_best.pth`, export the train95-only causal bank. The
exporter replays every training episode consecutively from its true `B0` and
collects four stratified, episode-index-shifted outputs; sparse collection never
truncates Mamba history. It stores episode-pooled global task prototypes
separately from the 32-bin local phase keys and causal values. Every episode's
true `B0` is inserted into phase bin zero with zero causal evidence:

```bash
bash scripts/export_robotwin_causal_bank_8gpu.sh \
  --goal-embeddings /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt \
  --zte-checkpoint /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth \
  --output-path /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt \
  --transition-horizon 15 \
  --phase-bins 32 \
  --samples-per-episode 4 \
  --batch-size 1 \
  --num-workers 4
```

Select the best ZTE checkpoint on held-out episodes, freeze it, and use it to
export normalized global task prototypes, local phase keys, and causal values
from the training split. This bank is supervision for Stages 2 and 3; it is not
the deployment PIM.

Expected artifacts:

```text
/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt
/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth
/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt
/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/manifest.json
```

Before Stage 2, export the frozen ZTE's exact stateful phase at every H15
decision boundary. This cache is generated by replaying each complete episode
from B0; it is not computed from frame number or normalized progress:

```bash
bash scripts/export_robotwin_live_queries_8gpu.sh
```

Then calibrate the frozen task-only PI0.5 language embedding to the train95 ZTE
task prototypes:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  python3 scripts/train_robotwin_task_retrieval.py
```

The held-out task retrieval accuracy must be at least 95%. Both artifacts are
hashed by Stage 2, which refuses stale or mismatched inputs.

The manifest must record the handoff path, stats hash, camera order, EEF16 slot
order, horizon, dataset split, seed, and ZTE configuration. A later stage must
refuse checkpoints whose manifest differs from the frozen contract.

Before Stage 2, run the capability gate on all validation5 episodes:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_robotwin_stage1.py \
  --goal-embeddings /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt \
  --zte-checkpoint /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth \
  --causal-bank /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt \
  --training-horizon 15 \
  --deployment-horizon 15 \
  --output-path /data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/stage1_gate.json
```

This evaluates episode-global task retrieval, phase/progress estimation, effect and
action prediction against trivial baselines, initial-phase bank lookup, and
the exact complete stateful H15 deployment path in addition to the batched
complete-episode path. Both begin at the true first frame and share one `B0`.
The gate also requires batched/online causal latents to agree within `5e-3`
max absolute error and auxiliary vision/action outputs within `2e-2` (the
latter allows CUDA convolution batch-shape rounding), preventing future
training/deployment recurrence drift.
Stage 1 may advance only when `gate.passed` is true. In particular, the H15
gate requires global-task macro R@1 at least 95%, phase-end MAE at most 0.10,
phase-end Spearman at least 0.80, monotonic violation rate at most 5%, bank
phase-end MAE at most 0.10, bank monotonic violation rate at most 5%, initial
phase bank MAE at most 0.05, and at least 20% action/effect improvement over
their zero baselines. Validation5 contains one episode per task, so the former
binary minimum-per-task recall check is intentionally removed. A low weighted
validation loss alone is not a pass.

### Stage 2: frozen PI0.5, protected Zeva residual training

Stage 2 freezes the complete PI0.5, including PaliGemma, the 300M Gemma action
expert, and PI0.5 action input/output/time projections. It trains only the Zeva task/context projection, BIT/PIM fusion,
diagonal-Gaussian H50 EEF16 action prior, two zero-initialized action-embedding
projectors, and bounded task/phase-routed confidence gates. Stage 1 ZTE/Mamba, the train95 causal bank, and the
task-language retrieval head remain frozen. Training uses H15 decision frames
and the same recurrent-phase `bank.retrieve()` path used in deployment.
Ground-truth task IDs only measure retrieval accuracy and never select a bank
entry.

The new fusion/action-prior path uses `5e-5`; no PI learning-rate group exists.
This preserves the already aligned RoboTwin policy while retaining Zeva's
non-oracle task-language retrieval and real H15 recurrent phase path.

```text
L_stage2 = L_PI_flow
          + lambda_prior * L_Gaussian_NLL
          + lambda_preserve * mean_i max(0, L_PI_flow_i - L_frozen_baseline_i)
          + lambda_gate * L_gate
```

Gaussian NLL is summed over EEF16 and averaged over batch/H50, with `log_std`
clamped to `[-5,2]`. The causal context and prior mean are injected only into
the noisy-action embedding: context is broadcast to H50, while the prior is
per-step H50. PaliGemma receives neither residual.
During training, an independent Bernoulli mask drops the complete prior
residual with probability 0.4 (keep probability 0.6), while the separate
whole-memory dropout remains 0.1. The frozen-baseline loss uses the same flow
noise as the enhanced forward pass. The hinge is evaluated per example, so a
gain on one task cannot cancel a regression on another. To control cost, the
matched frozen baseline runs every two optimizer steps and the sampled term is
multiplied by two. Validation
continues to run the matched baseline on every batch, preserving the expected
training objective and exact validation gate.

```bash
bash scripts/train_robotwin_advantage10_safe_router.sh
```

The released joint PaliGemma/action-expert attention forward and both noisy-
action residuals are retained unchanged, but every PI0.5 parameter has
`requires_grad=False` and receives no gradient. The trainer installs the residual hook
before compiling the foundation training forward. It uses per-GPU micro-batch
16 with two-step gradient accumulation, giving `16 x 8 GPUs x 2 = 256`. This must pass the formal
eight-H100 one-step memory preflight before launch. The protected ten-task run is 1,500
optimizer steps. Every checkpoint
still stores the complete `model.safetensors` for self-contained deployment,
plus `zeva_adapter.pth`, Zeva-only optimizer and scheduler state,
manifest, and validation metrics.

#### Ten-task ZeVA specialization

The fixed method-aligned subset is declared in
`configs/robotwin_zeva_advantage10.json`; the rationale and the complete
50-task classification are recorded in
`docs/ROBOTWIN_50_TASK_SELECTION_CN.md`. Task filtering is applied only to
Stage 2 decision samples. Stage 1's full 50-task vocabulary, ZTE, causal bank,
and language retrieval IDs remain unchanged and frozen.

```bash
bash scripts/train_robotwin_stage2_8gpu.sh \
  --task-subset configs/robotwin_zeva_advantage10.json \
  --save-dir /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage2-advantage10-v1 \
  --steps 1000 \
  --warmup-steps 250 \
  --save-freq 250 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --num-workers 4 \
  --video-backend torchcodec \
  --baseline-preserve-interval 4 \
  --compile-model \
  --compile-mode default
```

This is a clean specialization run initialized from the original trained
RoboTwin PI0.5 handoff, not a continuation of the 50-task ZeVA checkpoint.
At global batch 256, 1,000 subset steps give approximately the same mean
per-task sample exposure as 5,000 steps over all 50 tasks. Its result must be
reported as a 10-task specialized model and evaluated with a frozen paired
seed manifest; it is not a replacement for the 50-task benchmark score. The
1,000-step value is an initial budget, not an assertion of convergence:
checkpoints are saved at 250/500/750/1000. Held-out flow, retrieval >=95%, and
enhanced-flow non-regression determine the checkpoint before any test rollout.
The 200-episode native-runtime closed loop is then run exactly once as the final
measurement; it is not used to choose among checkpoints.

The earlier controlled action-expert fine-tuning experiment used the sequential
pair launcher below. It is retained only for reproducibility and is not the
current method:

```bash
bash scripts/train_robotwin_advantage10_bestv1_pair.sh
```

It trained the matched action-expert-only PI0.5 baseline and then the Zeva
variant on the same eight GPUs. Both use best-v1 as the actual PI foundation,
freeze PaliGemma, and share the ten-task samples, seed, global batch 256,
1,000-step schedule with 250-step warmup, and action-expert LR `1e-6`. The Zeva branch alone adds
the dual residual/action-prior modules at LR `5e-5`. Because best-v1 changed
its language embedding weights, Zeva explicitly reads the old frozen Stage 1
language table via `--goal-embedding-checkpoint`; the trainer verifies tokenizer
identity and records both foundation and auxiliary-language lineages. This
prevents silent B0/bank coordinate drift without retraining Stage 1.

The corrected native-runtime pair runs sequentially on aigc29's eight H100s;
aigc24 is reserved for RoboTwin rendering. Code, task manifest, Stage 1
artifacts, and outputs live on the shared filesystem; best-v1 and its minimal
frozen Stage 1 language table are hash-verified local NVMe caches on aigc29.

That superseded experiment used the monitored evaluation chain:

```bash
bash scripts/robotwin_eval/wait_and_launch_advantage10_bestv1_eval.sh
```

It audits three conditions on the exact same 20 expert-valid seeds per task:
the untouched `pretrained_model-best-v1` anchor, the matched ten-task baseline,
and Zeva. Evaluation loads each Stage 2 directory's complete
`model.safetensors`; Zeva additionally loads its adapter, frozen Stage 1 goal
embedding table, ZTE, retrieval head, and causal bank. The acceptance gate is
`baseline >= untouched anchor` and `Zeva > baseline`. All conditions use
Large_D435 640x480 RGB, absolute Joint14, EEF16 H50 prediction, H15 execution,
seen instructions, a continuous diffusion RNG stream matching the frozen
handoff evaluator, episode causal-state reset, and video/result cross-checking.
For reproducibility, all three conditions set `model_rng_seed=20260907` only
after the complete checkpoint is loaded, then consume that stream continuously;
the environment seed is never reused to reset PI flow noise per episode.
RoboTwin GPU physics is not bit deterministic, so a seed that passed the Base
expert filter can occasionally raise `UnStableError` when the same scene is
initialized in Anchor or Zeva. In a fixed-manifest condition the evaluator
retries that exact `(seed, instruction)` before any policy query, up to 20
times; it must never advance to a replacement seed. These pre-inference setup
retries do not consume the model's continuous diffusion RNG stream.

The first closed-loop shortlist was baseline step 1000 and Zeva step 250.
Its diagnostic 10-task x 20-episode evaluation runs under
`eval/advantage10-best-v1-paired-v2-native-tf5/formal-b1000-z250-seeded-v1`.
The repeated three-task gate is only a loading/alignment smoke test: GPU physics
made the same anchor vary from 10/15 to 13/15 even with fixed model RNG. Final
acceptance therefore comes only from the complete 200-episode reports and still
requires `baseline >= anchor` and `Zeva > baseline`.

The first formal Base condition completed with 99/200 successes (49.5%),
200/200 final-labelled videos, ten zero return codes, and an exact 10 x 20
frozen seed/instruction manifest. Its untouched-anchor replay reached 114/200
(57.0%), so this baseline is rejected as an abnormal degradation and cannot be
used for the final Base-versus-Zeva claim, irrespective of that run's eventual
Zeva score.

Because the foundation is already RoboTwin-trained, the replacement candidate
uses a deliberately conservative matched adaptation instead of relabeling the
untouched anchor as the baseline:

```bash
bash scripts/train_robotwin_advantage10_conservative_variant.sh baseline
bash scripts/train_robotwin_advantage10_conservative_variant.sh zeva
```

Both branches keep PaliGemma frozen and train the action expert with LR
`1e-7`, global batch 256 (`16 x 8 x 2`), 250 optimizer steps, 250 warmup steps,
and checkpoints at steps 125 and 250. Zeva alone trains the dual residual and
Gaussian-NLL action-prior modules at LR `5e-5`, with prior residual dropout
0.4; Stage 1 ZTE/Mamba, the causal bank, live queries, task-language table,
and retrieval module remain frozen. Offline selection retains baseline step
125 (best held-out flow) and Zeva step 250 (mature prior while enhanced flow
remains better than its injection-off self-baseline).

The first conservative three-condition formal run was
`eval/advantage10-best-v1-paired-v2-native-tf5/formal-conservative-b125-z250-seeded-v1`.
It replays the exact frozen manifest from the first run and is subject to the
same hard gate: complete reports and videos for all 600 episodes, exact
seed/instruction equality, all task return codes zero, `baseline >= anchor`,
and `Zeva > baseline`. At 194/200 Base episodes it had 107 successes;
even six additional successes could reach only 113/200, below the existing
same-protocol untouched anchor's 114/200. The run was therefore stopped before
starting its Anchor and Zeva conditions and marked `rejected_early`; its 194
episodes remain diagnostic evidence.

The next matched candidate halves only the action-expert LR to `5e-8`; all
other training and evaluation settings remain unchanged. Offline selection
again retains Base step 125 (validation flow `0.020620564`) and Zeva step 250
(enhanced flow `0.021710902`, injection-off flow `0.021731101`, retrieval
`99.7803%`). Its authoritative run is
`eval/advantage10-best-v1-paired-v2-native-tf5/formal-conservative-ae5e8-b125-z250-seeded-v1`.
The selected complete checkpoints are cached on aigc32, and the three
conditions replay the same frozen manifest under the same hard acceptance
gate.

That Base was also rejected early: at 187/200 episodes it had 99 successes,
so even a perfect remainder could reach only 112/200, below the established
normal-PI result of 114/200. The run did not proceed to Anchor or Zeva. This
shows that simply reducing the action-expert LR does not reliably preserve the
already-trained RoboTwin policy in closed loop.

The next candidate therefore applies an anchor-preserving post-training merge
to both matched branches. Before its closed-loop run, `alpha=0.25` was fixed:
each trained action-path tensor is `0.75 * original_PI + 0.25 * fine_tuned`,
while the Zeva adapter remains unmerged. The merge script rejects changes to
any frozen PaliGemma tensor; the produced checkpoints verified 209 action-path
tensors and 604 bit-identical frozen tensors. This is not a replacement of the
trained Base by the untouched Anchor: both Base and Zeva retain the same
pretrained-to-finetuned action-expert displacement at the same fixed scale.

The authoritative run is
`eval/advantage10-best-v1-paired-v2-native-tf5/formal-wise025-b125-z250-seeded-v1`.
In addition to comparing against the Anchor replay, its manifest and audit
encode the previously established normal-PI floor `0.57`; acceptance requires
`Base >= max(current Anchor, 0.57)` and `Zeva > Base`. GPU-physics variation in
a new Anchor replay therefore cannot silently weaken the retention gate.

Closed loop rejected this merge as well: at 199/200 Base episodes it had 101
successes, hence a maximum of only 102/200. Across the three nontrivial
action-expert adaptations, reducing the LR and shrinking the final weight
displacement did not recover the hard-task capability. The protocol therefore
returns to the actual experimental question: the existing RoboTwin-trained PI
is the normal Base, and Zeva must improve it without changing any PI weight.

The first frozen-PI scalar-gate adapter completed its preregistered formal run
but did not pass: the untouched Base stayed `114/200 = 57.0%`, while Zeva was
`110/200 = 55.0%` (paired delta -2 points; 31 Zeva-only versus 35 Base-only;
exact McNemar p=0.712). It improved hammer, dual-bottle and scan, but regressed
RGB/size ranking, handover, dustbin and bowl stacking. All 400 condition videos
and the structural audit passed, so this is a model failure rather than a
protocol failure. The run is preserved as diagnostic evidence and is not a
final result. The current non-regression repair is:

```bash
bash scripts/train_robotwin_advantage10_safe_router.sh
```

It runs 1,500 optimizer steps with global batch 256 (`16 x 2 x 8`), 250-step
warmup, Zeva LR `5e-5`, Gaussian-NLL weight `0.01`, action-prior residual
dropout `0.4`, a matched frozen-PI teacher every two steps, TorchCodec, and
the compiled H50/H15 forward. PaliGemma, the complete PI action expert and
input/output/time projections, ZTE/Mamba, the causal bank, and task retrieval
are frozen. The optimizer contains only the Zeva task projector, memory
encoder, Gaussian prior, two action residual projections, scalar gates, and the
bounded task-language/H15-phase router. The preservation hinge is per example,
not a batch mean. The checkpoint is selected only from validation5 reports at
250-step boundaries, never from test rollouts. Since absolute flow at different
checkpoints consumes different training RNG, ranking uses the within-checkpoint
matched-noise metrics among checkpoints with retrieval >=95%, nonnegative mean
improvement on every one of the ten tasks under full validation coverage, positive aggregate improvement, and
per-example win fraction >=50%: maximize win fraction, maximize the worst-task
improvement, then minimize mean degradation, maximize aggregate improvement,
minimize prior NLL, and choose the earlier step.

Safe-router v3 completed this validation-only selection at step 1,250, but its
strict 200-episode closed loop still did not pass: the untouched Base remained
`114/200 = 57.0%`, while Zeva reached `112/200 = 56.0%` (33 Zeva-only versus
35 Base-only; paired bootstrap 95% CI `[-0.09, 0.07]`; exact McNemar
`p=0.904`). The independent structural audit passed with 200 videos per
condition, exact seed/instruction pairing, and no protocol drift. This proves
that per-example offline flow protection is necessary but is not a sufficient
proxy for closed-loop success.

The v4 repair therefore adds a deployment-only **closed-loop residual trust
calibration** after Stage 2. It does not retrain or alter PI0.5, ZTE/Mamba, the
causal bank, retrieval, either learned residual, or the H15 recurrent state.
For each task, task-language retrieval selects one shared multiplier for both
the context and Gaussian-prior residual. The candidates are preregistered as
`0.25`, `0.5`, and `1.0`; they are evaluated on eight expert-valid episodes per
task beginning at seed 2000, disjoint from all formal seeds. A task enables the
smallest scale tied for the best validation result only if it improves over the
paired Base by at least two successes; otherwise its scale is zero and the
action path exactly falls back to the untouched PI0.5. Formal-test metrics are
rejected by the selector, and both seed-manifest hashes are stored in the
calibrated adapter.

```bash
bash scripts/robotwin_eval/launch_closed_loop_residual_calibration_v4.sh
```

This launcher evaluates Base once on the calibration split, reuses the exact
same seeds and instructions for all three Zeva scales, writes the task-language
trust table, and only then starts the fixed 10-task x 20-episode formal run.

The final comparison has two semantic conditions. `Base` is the untouched,
already RoboTwin-trained `pretrained_model-best-v1`; its preregistered run was
established before adapter training at 114/200. The evaluator imports that
immutable evidence with its source path and SHA256 instead of rerunning a
non-bit-deterministic physics simulation until it gets a convenient number.
`Zeva` replays the exact same 200 `(task, seed, instruction)` entries. The gate
is therefore `Base == 114/200` and `Zeva > Base`, with 200 videos per condition
and an independent two-condition audit. The rejected action-expert branches
remain diagnostics and can never replace this normal Base.

#### Decoder and throughput configuration

Formal v8 uses the same TorchCodec backend as the released RoboTwin PI0.5
baseline. Each persistent worker keeps a bounded 32-entry decoder LRU instead
of launching three FFmpeg processes per sample. On a real episode, TorchCodec
and the former FFmpeg path were verified bit-exact for all three cameras. The
FFmpeg backend remains available only for diagnostics.

The 1.1--2 second v8 figures were short kernel-only smoke measurements and are
not a valid end-to-end throughput estimate. The current frozen-PI adapter run
processes global batch 256 and three video streams per sample at about 9--11
seconds per optimizer step on eight H100s; CPU TorchCodec random access plus
the frozen PI forward/backward path to the residual injection points dominate.
Formal Stage 2 retains H15 recurrent samples, H50 action output, 1,500
optimizer steps, and the float32 `[0,1]` image contract.

Stage 2 passes its offline gate only when task retrieval remains at least 95%
and held-out enhanced flow does not regress against the matched unconditioned
branch. Final selection additionally requires paired baseline/Zeva closed-loop
evaluation with identical seeds, instructions, and diffusion RNG.

The former `stage2a-adapter/005000` checkpoint is invalid for final reporting.
Its FFmpeg images reached the visual-identity PI0.5 processor as uint8
`[0,255]`, so both its baseline and enhanced offline losses were measured in
the wrong model domain. Stage 2 now converts every view at the dataset boundary
to contiguous CHW float32 `[0,1]`, uses the accelerated action-only dual-residual v8
manifest/training state, and rejects full-v5, adapter-only, or old image-
contract checkpoints. Retrain from the original handoff into the separate
`stage2-action-expert-v8-accelerated` directory. The stopped v7 eager run is an
audit artifact. The v6 run injected
causal context into the PaliGemma prefix and is retained only as an audit
artifact; it is structurally incompatible with v8. Before the full run, verify
one real decoded frame on the target host:

```bash
python scripts/verify_robotwin_stage2_image_contract.py
```

The verifier requires bit-exact equality between the Stage 2 CHW path and the
formal evaluator's HWC path after conversion. Only the best checkpoint from
this corrected run may be used by the paired closed-loop evaluation.

### Stage 3: optional selective action adaptation

Stage 3 is entered only after action-expert Stage 2 passes the paired
non-regression gate. If needed, restrict it to the final action blocks or a
smaller action-expert learning rate; never unfreeze the PI0.5 VLM first. Retain
the baseline-preservation loss and early-stop on paired RoboTwin rollouts. A
full-PI0.5 fine-tune is kept only as an ablation; the formal Stage 2 keeps
PaliGemma frozen. The previous
`train_robotwin_stage3.py` VLM-retrieval recipe is legacy and is not part of
this protected-baseline protocol because task retrieval is now calibrated
before Stage 2 from the actual task-language input.

### Leakage and deployment rules

- Training/validation banks may contain only their declared split.
- RoboTwin test trajectories, evaluation attempts, and success labels must
  never be written into the offline Stage 1 bank.
- At deployment, PIM starts empty unless an experiment explicitly declares a
  training-only or human-provided warm start.
- BIT resets at every attempt. PIM persists across attempts only within the same
  fixed episode and resets at the episode boundary.
- Online interaction never updates model parameters; only Mamba state, BIT, and
  PIM change.

### Current implementation status

`scripts/train_robotwin_zte.py` and `scripts/train_robotwin_zte_8gpu.sh` are the
formal Stage 1 trainer and launcher. `scripts/train_robotwin_zeva_adapter.py` is
retained only as an end-to-end joint training/smoke path: it combines Stage 1
representation losses with Stage 2 flow loss and is not the formal three-stage
recipe. Formal experiments must use the Stage 1 live-query/task-retrieval
artifacts and the formal action-expert Stage 2 checkpoint, and must not report results from
the joint compatibility path as the final Zeva model.

## Deployment API

```python
from openpi.zeva.robotwin_policy import RobotWinZevaPolicy

policy = RobotWinZevaPolicy.from_handoff(
    "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1",
    zte_checkpoint="/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth",
    stage2_checkpoint="/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage2-action-expert-v8-accelerated/005000",
    adapter_checkpoint="/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage2-action-expert-v8-accelerated/005000/zeva_adapter.pth",
    retrieval_checkpoint="/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1.5-task-retrieval/task_retrieval.pth",
    causal_bank="/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt",
)

# raw_observation contains Joint14, the exact three RGB keys, and task text.
eef16_chunk = policy.infer_chunk(raw_observation, reset_scope="episode")

# After executing the chunk, pass the actually executed raw EEF16 commands with
# the next observation. They are normalized using the frozen checkpoint stats.
eef16_chunk = policy.infer_chunk(next_observation, executed_actions=executed_eef16)

# A failed retry keeps PIM but clears attempt-local Mamba/BIT state.
policy.reset(scope="attempt")
```

`reset(scope="attempt")` clears Mamba recurrence and BIT while preserving PIM.
`reset(scope="episode")` clears all online causal state. `observe_final()` commits
the terminal action-effect transition without generating another chunk.

On the first observation of each episode, the frozen task-language head projects the
saved PI0.5 task-token embedding into the 50-task ZTE key table. The selected task and live ZTE
phase retrieve read-only train95 causal values; these are fused with online
BIT/PIM values by the Stage 2 context encoder. `retrieval_diagnostics()` exposes
the selected task and cosine score for evaluation logs.

## Formal RoboTwin evaluation

The reportable experiment is a paired comparison on the fixed ten-task subset
in `configs/robotwin_zeva_advantage10.json`. Both conditions use
`zeva_randomized`, seen instructions, `Large_D435` 640x480 three-view RGB,
absolute Joint14, chunk-start-relative EEF16, H50 prediction, and H15 execution
before replanning. The untouched `pretrained_model-best-v1` PI0.5 is the Base;
its established normal result is 114/200 (57.0%) and must not be replaced by a
lower checkpoint or a changed action expert.

For each task, the formal manifest freezes the first 20 expert-valid seeds
selected from absolute seed 1000. Base and Zeva replay the exact same seed and
instruction pairs. The policy's recurrent Zeva state resets once per episode,
while the model flow RNG starts at 20260907 after model load and is consumed
continuously within each condition, matching the released PI0.5 evaluator.
GPU-PhysX initialization retries keep the same seed and happen before the first
model call; a failed seed is never silently substituted. Every episode must
produce a video whose success label agrees with the atomic progress record.

Before the single formal comparison, residual trust is calibrated only on a
disjoint closed-loop validation manifest beginning at seed 2000. Base runs once
for eight expert-valid episodes per task; Zeva evaluates residual scales 0.25,
0.5, and 1.0 on those exact 80 seed/instruction pairs. A task enables the best
scale only when it gains at least 2/8 successes over Base; ties choose the
smaller scale, and every other task uses scale 0 (an exact untouched-PI
fallback). The selector records both manifest hashes and
`test_metrics_used=false`; formal-test outcomes are never read by calibration.

```bash
cd /mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA
nohup bash scripts/robotwin_eval/launch_closed_loop_residual_calibration_v4.sh \
  > /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v4-closed-loop/launcher.log \
  2>&1 < /dev/null &
```

The validated deployment uses eight model servers on `aigc29` and eight SAPIEN
clients on the renderer-compatible `aigc24`. Do not combine results from nodes
with different NVIDIA/Vulkan stacks. The launcher is resumable from per-task
atomic progress files and emits condition reports, a paired report, videos,
calibration evidence, and an independent final audit under:

```text
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v4-closed-loop
```

## Code map

```text
src/openpi/zeva/robotwin_contract.py       frozen handoff and mean-std validation
src/openpi/zeva/robotwin_policy.py         exact PI0.5 wrapper and causal injection
src/openpi/zeva/transition_encoder.py      Mamba action-effect CTE
src/openpi/zeva/causal_bank.py             frozen train95 bank loading and lookup
src/openpi/zeva/memory.py                  BIT/PIM state, merge, retrieval, reset
src/openpi/zeva/context.py                 causal context construction
scripts/verify_robotwin_handoff.py         read-only artifact validation
scripts/smoke_robotwin_zeva.py             full checkpoint/interface smoke
scripts/train_robotwin_zte.py              formal Stage 1 ZTE training
scripts/train_robotwin_zte_8gpu.sh         formal eight-H100 Stage 1 launcher
scripts/export_robotwin_causal_bank.py      train95 causal bank export
scripts/export_robotwin_causal_bank_8gpu.sh eight-H100 bank exporter
scripts/train_robotwin_stage2.py            full/action-expert/adapter Stage 2 training modes
scripts/train_robotwin_stage2_8gpu.sh       formal eight-H100 Stage 2 launcher
scripts/export_robotwin_live_queries.py      deployment-recurrent H15 phase cache
scripts/train_robotwin_task_retrieval.py     task-language retrieval calibration
scripts/train_robotwin_stage3.py             legacy VLM retrieval experiment
scripts/train_robotwin_stage3_8gpu.sh        legacy VLM retrieval launcher
scripts/robotwin_eval/zeva_policy.py         official simulator/model-server adapter
scripts/calibrate_robotwin_residual_trust.py validation-only per-task residual selector
scripts/make_robotwin_residual_scale_candidate.py immutable residual-scale checkpoint builder
scripts/robotwin_eval/launch_paired_formal_eval.sh strict paired resumable evaluator
scripts/robotwin_eval/launch_closed_loop_residual_calibration_v4.sh validation-to-formal orchestrator
scripts/robotwin_eval/audit_advantage10_formal.py independent protocol/result audit
scripts/train_robotwin_zeva_adapter.py      frozen-PI adapter training
scripts/train_robotwin_zeva_8gpu.sh         eight-H100 launcher
```

## LIBERO pipeline (PI0.5 baseline aligned)

LIBERO is a separate experiment family from RoboTwin: it has independent data,
normalization, checkpoints, causal banks, logs, and evaluation outputs under
`/data1/dingxin/zeva-runs/libero-v3-h5-lang`. The foundation is the frozen
handoff contract `egoscale-libero-memory-baseline-handoff-v1`; its selected
`model.safetensors` SHA-256 is
`d6eabd264bb4b7b4fbde795068a1615ba6a2ce5c18fefcbefb0c5498b5582d92`.

The physical contract is fixed throughout all stages:

| Item | LIBERO setting |
| --- | --- |
| Suites | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` (40 tasks) |
| Data split | official train 1,614 episodes / validation 79 episodes |
| Cameras | `agentview_rgb`, `eye_in_hand_rgb`; source 256x256, PI0.5 input 224x224 |
| State | absolute EEF16 at decision time |
| Action | chunk-start-relative EEF16, frozen q01/q99 normalization |
| PI0.5 output | H10, 32 model slots with the first 16 physical slots |
| Closed loop | execute the first H5 and replan every H5 |
| ZTE transition | exactly the executed H5 action-effect interval |
| Initial context `B0` | task-only PI0.5 language embedding (task text plus newline) and `s0` |
| Recurrence | Mamba, matching the Zeva architecture used for RoboTwin |

The three stages are intentionally gated. A later stage must use hashes from
the exact selected artifacts of the preceding stage; the scripts reject split,
horizon, normalization, or checkpoint drift.

`libero-v3-h5-lang` is the current admissible Stage 1 lineage. The archived v1
lineage is audit-only: it resized the concatenated cameras as one panorama,
mean-pooled the H5 action chunk, reconstructed the current action from a state
that had already consumed that action, and did not guarantee cross-episode task
positives for per-GPU batch size one. V2 fixed those defects by splitting and
fusing the two camera views, retaining ordered action tokens, predicting each
action from the preceding causal state, using one frozen EMA space for both
sides of the visual effect target, and using paired cross-rank task sampling
plus 40 learned task prototypes. V3 additionally makes task identity an
explicit episode-global key and replaces pointwise progress regression with an
accumulated positive-hazard state. Consequently offline and recurrent inference
share the same exact `p0=0`, strictly increasing progress construction. V1 and
V2 checkpoints are intentionally rejected by the V3 loaders.

### LIBERO Stage 1: train ZTE

Stage 1 trains only the causal transition encoder. Its visual encoders start
from ImageNet-pretrained ResNet-18 weights; the target encoder is an EMA copy.
The goal input is a frozen, task-only PI0.5 language embedding rather than an
episode observation embedding.

```bash
python3 scripts/export_libero_goal_embeddings.py \
  --handoff-root /data1/dingxin/libero-memory-baseline-v1 \
  --dataset-root /data1/dingxin/libero-memory-baseline-v1/data \
  --output-path /data1/dingxin/zeva-runs/libero-v1-h5-lang/pi05_goal_embeddings.pt

bash scripts/train_libero_zte_8gpu.sh \
  --goal-embeddings /data1/dingxin/zeva-runs/libero-v1-h5-lang/pi05_goal_embeddings.pt \
  --save-dir /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte \
  --steps 30000 --batch-size 1 --num-workers 4 \
  --transition-stride 5 --effect-steps 5 --executed-action-steps 5 \
  --save-freq 1000 --eval-batches 0
```

Checkpoint selection uses the complete held-out validation pass written by the
trainer, not training loss. After selection, export a read-only train-only
causal bank from complete consecutive H5 episode histories. The bank stores an
explicit B0 phase entry and episode-global task prototypes; it never constructs
causal states from temporally disconnected samples. Then run the online H5
capability gate:

```bash
bash scripts/export_libero_causal_bank_8gpu.sh \
  --zte-checkpoint /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/zte_best.pth \
  --output-path /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/train_causal_bank.pt

python3 scripts/eval_libero_stage1.py \
  --zte-checkpoint /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/zte_best.pth \
  --causal-bank /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/train_causal_bank.pt
```

Stage 1 advances only when `stage1_gate.json` has `gate.passed=true`. The gate
checks held-out task retrieval, phase progress and monotonicity, bank phase
lookup, and action/effect reconstruction relative to constant baselines. Both
offline and recurrent measurements use a complete consecutive H5 rollout from
the real episode `s0`; resetting a zero-progress cache at a mid-episode frame is
not a valid evaluation protocol.

### LIBERO Stage 1.5: deployment retrieval preparation

Before Stage 2, export the frozen ZTE state at every real H5 replanning boundary
and train a small task-language retrieval head against the frozen train-bank task
prototypes. This removes both oracle task IDs and oracle progress from Stage 2.
The cache replays each episode consecutively from its real `B0`; the retrieval
head uses the same task-only PI0.5 language embedding used at deployment.

```bash
bash scripts/export_libero_live_queries_8gpu.sh \
  --zte-checkpoint /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/zte_best.pth \
  --output /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/live_queries_h5.pt

python3 scripts/train_libero_task_retrieval.py \
  --causal-bank /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/train_causal_bank.pt \
  --output /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1.5-task-retrieval/task_retrieval.pth
```

Stage 1.5 passes only when held-out task retrieval accuracy is at least 95%.
Stage 2 refuses a stale cache, a bank trained from another ZTE, or a retrieval
head trained against another bank.

### LIBERO Stage 2A: protected residual on frozen PI0.5

The current Stage 2 protocol matches RoboTwin. It freezes the complete LIBERO
PI0.5 baseline, ZTE, causal bank, and task-language retrieval head. Only the
task projector, memory context encoder, action prior, prefix/action projectors,
and two scalar residual gates are trainable. Each batch infers its task from
language and its phase from the recurrent H5 ZTE cache, then combines frozen
train-bank memory with episode-local BIT/PIM memory. Retrieval confidence gates
the injection; training adds phase noise 0.02 and memory dropout 0.1.

The matched frozen baseline is evaluated with the identical RNG state. The loss
is `L_flow + 0.5 L_prior + relu(L_flow - L_frozen_PI) + 1e-3 L_gate`. Thus Stage
2 cannot hide a PI regression behind a lower auxiliary loss.

```bash
bash scripts/train_libero_stage2_8gpu.sh \
  --zte-checkpoint /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/zte_best.pth \
  --causal-bank /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/train_causal_bank.pt \
  --live-queries /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/live_queries_h5.pt \
  --task-retrieval /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1.5-task-retrieval/task_retrieval.pth \
  --save-dir /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage2a-adapter \
  --steps 5000 --batch-size 16 --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 --warmup-steps 500 --save-freq 500
```

The effective global batch is 256 on eight H100s. Step directories contain
only the compact adapter and optimizer/scheduler state; the frozen 7.5GB PI0.5
checkpoint is referenced by hash instead of copied. Gradient invariants require
zero gradients on PI0.5/ZTE and nonzero gradients on all residual modules and
both gates. Select the checkpoint from `best.json` using held-out total loss.

```bash
python3 scripts/eval_libero_stage2.py \
  --stage2-dir /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage2a-adapter
```

Stage 2A passes only when task retrieval remains at least 95%, validation flow
does not regress relative to the matched frozen PI0.5 baseline, the enhanced
total loss is finite, and every artifact/hash/gradient invariant is complete.

### LIBERO Stage 3: optional selective action adaptation

Stage 3 is no longer the task-retrieval stage; retrieval is frozen before Stage
2. Enter Stage 3 only if Stage 2A passes non-regression but closed-loop success
still needs adaptation. Keep the VLM, ZTE, bank, retrieval head, and Stage 2A
memory modules frozen, and selectively unfreeze only the PI0.5 action expert or
its final action blocks. A Stage 3 checkpoint is admissible only if it improves
closed-loop validation without violating the frozen-baseline non-regression
gate. Otherwise the Stage 2A checkpoint is the final model.

The recommended LIBERO recipe unfreezes action-expert blocks 16–17 plus
`action_out_proj`, uses a 10x smaller learning rate than Stage 2A, and compares
every enhanced forward with an immutable Stage 2A functional anchor under the
same RNG. It trains for 2,000 steps and retains only the selective action delta:

```bash
bash scripts/train_libero_stage3_8gpu.sh \
  --stage2-checkpoint /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage2a-adapter/005000 \
  --save-dir /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage3-selective-action \
  --steps 2000 --final-action-layers 2 \
  --batch-size 16 --gradient-accumulation-steps 2 \
  --learning-rate 5e-6 --warmup-steps 200 --save-freq 250

python3 scripts/eval_libero_stage3.py \
  --stage3-dir /data1/dingxin/zeva-runs/libero-v3-h5-lang/stage3-selective-action
```

The offline Stage 3 gate checks strict matched-Stage-2A non-regression,
retrieval accuracy, hashes, selective parameter scope, and gradient invariants.
Passing it means the checkpoint is ready for paired closed-loop evaluation;
the paired Stage 2A/Stage 3 success result remains the final selection gate.

### Formal LIBERO and LIBERO-Pro evaluation

Train every ZeVA stage only on the official LIBERO training split. LIBERO-Pro is
an OOD evaluation benchmark; none of its perturbed images, states, instructions,
or trajectories may enter ZTE, retrieval, causal-bank, Stage 2, or Stage 3
training.

For in-distribution LIBERO validation, use the matched H10-predict/H5-execute
protocol. For frozen LIBERO-Pro evaluation, retain its official R10 controller:
PI0.5 predicts H10, all ten targets are executed, and the policy replans only
after target 10. ZeVA's causal clock remains aligned to its H5 training by
recording the real observation after target 5 and committing two consecutive
transitions per R10 block:

```text
PI inference at B0 -> execute targets 1..5 -> CTE/Mamba commit (B0,a1:5,B5)
                   -> execute targets 6..10 -> CTE/Mamba commit (B5,a6:10,B10)
                   -> next PI inference at B10
```

The midpoint commit updates only ZeVA causal state and memory; it must not call
PI0.5 again or replace targets 6..10. Both baseline and ZeVA therefore retain
identical R10 physical control, replan IDs, sampled noise, seeds, and success
predicates. Because all ten PI targets are originally chunk-start-relative to
B0, targets 6..10 are re-expressed against the actually observed B5 position
and rotation before they enter the second causal transition; gripper commands
remain absolute. This matches the Stage 1 H5 dataset transform instead of
silently feeding B0-relative actions to an ostensibly H5 transition. The
derived evaluator overlay is generated from the frozen handoff
by `scripts/libero_eval/libero_pro_h5_overlay.patch`; the original handoff is
never edited. `scripts/libero_eval/run_libero_pro_full20_4090.sh` materializes
that immutable overlay and runs all four suites, five OOD conditions, ten tasks,
and twenty episodes per task-condition (4,000 episodes total, seed 7), with an
independent memory reset and video/result audit for every episode.

A model is ready for comparison only after all three stage gates pass and the
runtime assertions confirm image, state, action, normalization, camera order,
R10 physical execution, and H5 causal-transition contracts.

LIBERO code map:

```text
src/openpi/zeva/libero_contract.py          frozen handoff, H10/H5 and quantile contract
src/openpi/zeva/libero_data.py              official split and relative-EEF16 loaders
src/openpi/zeva/libero_bank.py              frozen official-train causal bank
src/openpi/zeva/libero_policy.py            PI0.5 wrapper and Stage 2 injection
scripts/export_libero_goal_embeddings.py    task-only B0 language embeddings
scripts/train_libero_zte.py                 Stage 1 Mamba ZTE trainer
scripts/export_libero_causal_bank.py         official-train causal bank exporter
scripts/eval_libero_stage1.py                held-out H5 Stage 1 gate
scripts/export_libero_live_queries.py         deployment-recurrent H5 phase cache
scripts/train_libero_task_retrieval.py        Stage 1.5 task-language retrieval
scripts/train_libero_stage2.py                protected frozen-PI0.5 Stage 2A trainer
scripts/eval_libero_stage2.py                Stage 2 artifact/loss gate
scripts/train_libero_stage3.py                selective-action Stage 3 trainer
scripts/eval_libero_stage3.py                 Stage 3 offline non-regression gate
scripts/libero_eval/libero_pro_provider.py    R10 rollout with two explicit H5 causal commits
scripts/libero_eval/libero_pro_h5_overlay.patch frozen-evaluator midpoint transport overlay
scripts/libero_eval/run_libero_pro_full20_4090.sh formal 4,000-episode LIBERO-Pro runner
```
