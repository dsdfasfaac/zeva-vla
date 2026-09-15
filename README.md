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

Development update (2026-09-15 19:19 CST): the preregistered **100-step
gate-initialization comparison** (0.01 versus 0.10) completed both arms and all
5,874 validation decisions. The stronger gate increased context/prior residual
norms 7.62x/8.45x but did not improve H15: error rose 0.10341% versus the 0.01
arm, and both residual-on paths were slightly worse than their own off paths.
This does not support extending the stronger-gate run; it is not a significance
claim or evidence that ZTE cannot work. No new full training or formal rollout
has started. The read-only gradient check completed on 32 fixed training
decisions: weighted NLL gradients in the shared memory encoder were 50–1,014x
the flow gradients, with negative alignment in three of four batches in each
arm. These are local raw-gradient observations, not Adam-update ratios or
causal proof of rollout failures. An opt-in NLL-only input-stop-gradient route
passed direct Torch and real-weight routing checks: flow/NLL values and flow
gradient norms were unchanged, while NLL-to-shared-feature gradients became
zero. A preregistered 100-step candidate using this routing has now been
dispatched; optimizer progress and validation benefit are not yet confirmed.
See the [gradient evidence and routing scope](docs/ZTEV2_OBJECTIVE_GRADIENTS_20260915.md).
This is not a new matched-Base
performance result or a test-label gate sweep. See the
[fixed design and execution record](docs/ZTEV2_GATE_MECHANISM_20260915.md).

Current status (2026-09-14): both fixed-teacher-pair branches completed their
predeclared 1,000 fresh-optimizer steps from trained Base004500. Full matched
validation on 5,874 decisions also completed. H15 flow is 0.01014354 for ZeVA,
0.01014233 for its residual-off path, 0.01018073 for the fixed starting Base,
and 0.01003096 for the newly trained Base. Thus ZeVA improves over the starting
Base by 0.3653%, but is 1.1224% worse than the matched new Base; its residuals
also slightly worsen H15 (+0.01195%). The fixed-teacher change has not shown
incremental ZTE utility. These are point estimates, not significance or
closed-loop success-rate claims. No test-label checkpoint reselection or
inference gate sweep was used. The later short initialization experiment is
separate from these completed results. The fixed-seed/instruction evaluation
controller started on aigc28 at 2026-09-15 01:00 CST after placement checks;
The completed paired evaluation (2026-09-15) is Base 110/200 (55.0%),
ZeVA 110/200 (55.0%), and untouched Anchor 104/200 (52.0%). ZeVA's measured
delta is zero (paired 95% CI −7.5 to +8.0 percentage points); Base is below
the declared 57% reference floor. Acceptance is false. See the
[complete ten-task results](docs/ROBOTWIN_FIXED_ANCHOR_PAIRED_RESULTS_20260915.md).
Stage1's four auxiliary
metrics still clear their configured reference lines and are not official
BehaviorVLA gates. The preservation hinge is expert-loss reweighting, not
teacher-action distillation or a guarantee of preserved success rate. See the
[evidence and next-step boundaries](docs/ZTEV2_POST_EVAL_DIAGNOSIS_20260912.md).

The selected PI0.5 has already been trained on the same RoboTwin distribution.
The active work is a practical ZTE v2 → PI action-expert integration, followed
by matched ten-task Base/ZeVA training and closed-loop evaluation. Auxiliary
representation metrics diagnose problems; they are not an all-pass prerequisite
for policy training. The former v19 task-gated PI
consensus run was stopped and marked `superseded`; it must not be resumed as the
main method. Its predecessor v18 achieved Base `43/80` and ZeVA `43/80` in a
paired closed-loop split despite improving offline expert-action MSE. A grouped
audit also showed that the former Stage 1 representation did not add candidate
ranking information. These results reject that implementation and proxy, not
the goal of learning an action-effect representation.

ZTE v2 follows the three-stream temporal factorization of BehaviorVLA while
retaining ZeVA's action-effect semantics:

- Separate causal Mamba streams encode visual state, the ordered H15 EEF16
  chunk, and the observed visual effect. The action stream may not mean-pool the
  raw chunk.
- A pre-effect state predicts EMA visual change without reading the target
  after-image. A controlled `--action-prediction-context phase` variant predicts
  the next H15 directly from the exported post-transition phase: the current
  after-image is already available at that replanning boundary, but future
  images/actions are not. Legacy checkpoints and the `pre` control retain the
  old next-action head. This new supervision path is a hypothesis under test,
  not a demonstrated improvement or a verbatim BehaviorVLA implementation.
- The representation is factorized into an episode-level task prototype,
  recurrent local phase, and action-effect token. Language is permitted in
  `B0`, but every sensorimotor representation claim must also pass a
  language-masked probe.
- Brief Interaction Trace stores recent evidence within the current attempt.
- Persistent Interaction Memory consolidates phase-matched evidence across
  attempts in the same fixed episode.
- The current integration target retains the existing action-expert-side dual
  residual: memory/context conditioning and a phase-conditioned Gaussian prior
  modify action-expert inputs, not final EEF16 outputs. The before-VLM prefix
  engineering smoke is a separate candidate, not the selected route or proof
  of BehaviorVLA-equivalent injection.
- PI0.5 still returns H50 and RoboTwin still executes H15 before replanning.

Stage 1 diagnostics guide representation improvements; the delivery criterion
is normal matched Base capability and measured ZeVA closed-loop improvement.
Checkpoint/bank lineage, no future/test leakage, H15 recurrence and numerical
loading/injection correctness remain required. The scientific-design document
records diagnostic hypotheses, not mandatory all-pass Stage2 gates. Older
v11--v19 sections below are failure-analysis history, not current launch plans.

Implementation evidence and outstanding gates are tracked separately in
`docs/ZTE_V2_IMPLEMENTATION_STATUS_CN.md`. The legacy-path encoder passes eight
information-flow tests using real Mamba on H100, including a real ResNet
training-mode test. These are correctness tests, not a Stage 1 capability pass.
The v2 smoke entry point is `scripts/run_robotwin_zte_v2.sh`; the historical
Stage 1/2 commands below do not launch the redesigned method.

The active loss audit found another non-equivalence with BehaviorVLA: its
prediction losses sum coordinates before averaging valid times, while earlier
v2 pilots averaged coordinate-wise SmoothL1. The explicit `vector_mse` control
now sums feature/EEF coordinates and averages valid transitions (and H15 for
actions), retaining the same external weights. It is not a capability pass.
The vector-MSE run targets 4096 steps (about five epochs). The planned parallel
legacy-Huber control and expanded old/v2 frozen-probe run were cancelled to
prioritize policy integration. The loss-control launcher remains for
reproducibility; its existence does not mean both experiments are running.

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

Latest completed paired evaluation (2026-09-12): **Base 54.5% (109/200),
ZeVA 55.0% (110/200), Anchor 50.5% (101/200)**. All 600 videos and exact
seed/instruction pairs were audited. The +0.5 pp ZeVA delta is not evidence of
a stable advantage (paired 95% CI -7.5 to +8.5 pp); the predeclared 57% Base
floor also failed, so delivery acceptance is false. Training and evaluation
are complete; evidence-based diagnosis continues, with no new training started.
See the [full ten-task result and limitations](docs/ROBOTWIN_ZTEV2_PAIRED_RESULTS_20260912.md).
The timestamped progress notes below are historical.

Stage 1 trains ZTE v2; a selected checkpoint exports a train-only bank and
matching recurrent live queries. After real checkpoint loading, zero-init and
H15-state smoke tests, Stage 2 freezes ZTE/bank and the vision-language backbone,
and trains the PI action expert (LR `5e-6`) plus ZeVA modules (LR `5e-5`) with
global batch 256. Representation and shuffled-input probes are diagnostics,
not mandatory all-pass prerequisites. Stage 2 must be followed by paired
closed-loop evaluation. Stage 3 is optional and addresses an observed failure,
not an automatic extra stage. As of 2026-09-12 14:04 (China time), both matched
Stage2 runs have completed 5,000 steps with full checkpoints. ZeVA's interrupted
last 500 steps were recovered on aigc24 with the original four-rank/global-256
settings and verified identical dataset contents. This recovery is not bit-exact
(RNG was not saved and auxiliary library versions differ); provenance is retained.
Independent held-out flow selection chose Base step 4,500 (`0.01872689`) and
ZeVA step 5,000 (`0.01828257`). These are offline losses, not success rates.
The selected pair passed explicit evaluation preflight. At 14:13 the formal
Base -> Anchor -> ZeVA pipeline started on aigc24 (8 slots); model servers are
loading, and no completed rollout results are available yet. The dedicated
serving-script release is `eval-release-ztev2-20260912`; all conditions share
H50/H15, Large_D435, seen randomized scenes, and the same 20 expert-valid
seeds/instructions per task starting from seed 1000.

At 16:27, Base is complete: **109/200 (54.5%)**, with all 200 videos and
seed/instruction entries audited. This is below the predeclared 57% reference
floor, so normality has not passed. Anchor is running on the identical cases;
ZeVA follows automatically. Do not treat this partial comparison as a final
result or replace the selected checkpoints/seeds in response to test outcomes.

At 19:02, Anchor is also complete: **101/200 (50.5%)**, with video and exact
seed/instruction checks passed. Base is 4 percentage points higher on these
paired cases, but remains below the historical 57% floor. ZeVA evaluation is
running. Historical-reference comparability is being audited without changing
the gate or using test outcomes to reselect weights.

Historical-reference audit (19:43): the 57% Anchor used the same protocol but
a different frozen seed/instruction manifest. Of 200 task-seeds, 181 overlap;
only 2 overlapping cases also have identical instructions. Model hosting also
changed from aigc29 to aigc24. Thus 57% versus 54.5% is not a paired capability
comparison; only the current Base/Anchor/ZeVA share exact cases. The predeclared
reference floor remains unchanged, and no test-driven checkpoint reselection
is performed.
The fresh-run entry point is
`scripts/train_robotwin_advantage10_ztev2.sh`; historical commands below are
not this experiment. Training is complete; closed-loop delivery is not.

The final requested comparison trains both ordinary Base and ZeVA from the
specified best-v1 on the same selected ten tasks with matched PI optimization
and data budgets. Untouched best-v1 remains a separate capability anchor, not
a replacement for that trained Base; a degraded Base cannot establish success.

For this matched run, finish both 5,000-step budgets and select each side's
checkpoint by minimum held-out **flow loss** (earlier step breaks a tie).
ZeVA's total loss includes Gaussian NLL and is not the common selection metric.
Freeze the selected paths before closed-loop evaluation; neither test successes
nor the ZeVA model's internal residual-off proxy selects the final pair.

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
| 1. ZTE v2 representation | PI0.5 weights and EMA visual target | three-stream ZTE v2 | gated `zte_v2_best.pth`, probe report, train95 causal bank |
| 2A. Injection attribution | PI0.5, ZTE v2, bank | zero-init causal-prompt and Gaussian-prior adapters | Base/global-only/prior-only/both plus correct/zero/shuffled report |
| 2B. Joint policy tuning | ZTE v2 and bank | PI0.5 plus prompt/PBD adapters at separate LRs | full model, adapters, optimizer/scheduler, paired validation |
| 3. Optional diagnosed repair | selected modules depend on the observed failure | only the module implicated by Stage2 evidence | two independent positive paired splits before final evaluation |

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

### Stage 2: frozen-PI H15 direct output residual (active v14)

The normal Base is the already RoboTwin-trained, untouched best-v1 checkpoint,
whose audited ten-task result is `114/200 = 57%`. Active v14 freezes all 813
PI0.5 tensors, ZTE/Mamba, the train95 bank and task retrieval. It trains only
the task/context/prior representation and direct residual corrector at `5e-5`.

This change follows two falsification experiments. v11 action-token injection
lost closed-loop (`43/80` Base versus `41/80` ZeVA). v13 then froze PI and
interpolated Base toward the Gaussian prior, but the stronger PI made the
optimal gate collapse from about 1% to `8.5e-8`. An independent Stage1 probe
still found useful information: held-out action MSE fell from `0.280884`
(task only) to `0.130684` (+phase) and `0.060073` (+causal), with 10/10 tasks
improving. v14 therefore uses the Stage1 information to predict the smaller
conditional target `expert - Base`, instead of replacing Base with the weaker
prior mean. No `episode_index` or progress oracle is used.

```text
L_stage2 = MSE(Base[:15] + Delta[:15], expert[:15])
          + lambda_residual * MSE(Delta, clip(expert - Base))
          + lambda_preserve * mean_i max(0, MSE_ZeVA_i - MSE_Base_i)
          + lambda_trust * max(0, |effective Delta| - 0.10)^2
          + lambda_prior * L_Gaussian_NLL
          + lambda_gate * L_centered_gate
```

Gaussian NLL is an auxiliary representation loss and never directly writes an
action. The residual is hard bounded to `0.25` in normalized EEF16 coordinates,
and only positions 0--14 are changed. The hinge is evaluated per example, so a
gain on one task cannot cancel a regression on another. Frozen PI actions are
exported once into a hashed cache and reused for training and validation.

```bash
# Export the immutable Base action cache once, then train active v14.
bash scripts/cache_robotwin_base_actions_v13_8gpu.sh
bash scripts/train_robotwin_advantage10_output_residual_v14.sh
```

The released joint PaliGemma/action-expert inference remains intact and runs
outside gradient computation. The residual trainer uses per-GPU micro-batch 16
with two-step accumulation, giving
`16 x 8 GPUs x 2 = 256`. A real eight-H100 one-step smoke must pass module-
freeze and gradient assertions before launch. The active run is 1,500 optimizer
steps and saves every 250 steps. Every checkpoint stores the complete adapted
`model.safetensors`, `zeva_adapter.pth`, optimizer/scheduler state, manifest and
validation metrics; the complete PI model is tensor-identical to untouched
best-v1.

#### Ten-task ZeVA specialization

The fixed method-aligned subset is declared in
`configs/robotwin_zeva_advantage10.json`; the rationale and the complete
50-task classification are recorded in
`docs/ROBOTWIN_50_TASK_SELECTION_CN.md`. Task filtering is applied only to
Stage 2 decision samples. Stage 1's full 50-task vocabulary, ZTE, causal bank,
and language retrieval IDs remain unchanged and frozen.

The active run is under `advantage10-output-residual-v14/zeva`. Held-out
validation5 metrics are only a screening gate: retrieval must remain at least
95%, all ten tasks must be covered, and no task may have a negative mean paired
flow improvement. Because earlier versions passed this offline screen and still
lost closed-loop success, offline loss is never sufficient to select the final
checkpoint.

The selected step1500 checkpoint has Base MSE `0.01201938`, ZeVA MSE
`0.01196872`, paired improvement `5.06596e-5`, sample win fraction `68.19%`,
and positive improvement on all ten tasks. Candidate checkpoints must first
pass frozen-foundation tensor identity, token-injection-off, H15-only and H35
exact-Base audits. They are then
compared against the same untouched Base in four pre-registered seed x model-
RNG cells: seeds 15000 and 16000 crossed with RNGs 20260907 and 20260908. Each
cell contains 10 tasks x 8 paired episodes. Every cell and every task aggregated
over cells must be non-regressive, and the total gain must be at least 12/320.
Only then may the checkpoint enter the seed-1000 confirmatory test. That test
imports the immutable, video-audited Base result
`114/200 = 57%` and evaluates Zeva on the same 10x20 expert-valid
`(seed, instruction)` manifest. The final gate is `Base >= 57%` and
`Zeva > Base`, so Zeva must achieve at least 115/200.

The uncalibrated v14 development split-j result is Base `43/80` and ZeVA
`38/80`; this rejects unconditional residual use despite its positive offline
MSE. Deployment therefore uses a pre-registered binary safety route selected
only on split-j: a task receives scale 1 iff it gains at least one success and
has more ZeVA-only wins than Base-only losses; otherwise scale 0 gives exact
Base. This enables only `hanging_mug` and `scan_object`, for a deterministic
development composition of `46/80`. The route is frozen before independent
split-i confirmation and may not be revised from confirmation or final data.

```bash
# v14 evaluation reuses a frozen Base seed manifest and identical model RNG.
bash scripts/robotwin_eval/launch_paired_formal_eval.sh
```

If the four-cell gate fails, the only next attribution experiment is an
action-expert-only run from untouched best-v1 with the same `5e-7` LR, step
budget, paired teacher and validation cells, while prior/context and Gaussian
NLL are fully disabled. Gate, dropout and residual scale are not tuned first.

For audit history, frozen anchored-v9 passed its seed9000/12000 development
splits by `+7/160`, but its action-expert Base reached only 95 successes after
193/200 seed-10000 episodes and could finish at no more than 102/200. It was
therefore rejected before Anchor/Zeva could consume further formal resources;
its development result is not a deliverable comparison.

The earlier 1,000-step action-expert experiment used the sequential launcher
below. It is retained only for historical reproduction; its evaluated Base
regressed below the untouched 57% anchor and it is not the active v9 run:

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

That single-split v4 calibration did **not** pass the formal gate.  It selected
scale `0.5` for `put_bottles_dustbin`, scale `1.0` for
`stack_bowls_three`, and scale `0` for the other eight tasks.  On the fixed
formal manifest, Base remained `114/200 = 57.0%`, while calibrated Zeva fell to
`100/200 = 50.0%`.  The protocol and structural audit passed, including 200
videos per condition and exact seed/instruction pairing; the failure was the
performance criterion itself.  A subsequent tensor-by-tensor audit also found
all 813 foundation tensors bit-identical to the untouched PI0.5, ruling out a
damaged or accidentally fine-tuned backbone.  The evidence instead shows that
one eight-episode closed-loop split is too noisy for residual trust selection.

The current v5 repair requires the gain to replicate on **two disjoint
closed-loop validation splits**, beginning at seeds 2000 and 3000.  Each split
contains eight expert-valid episodes per task and evaluates the preregistered
scales `0.25`, `0.5`, and `1.0` on the exact paired Base seed/instruction
manifest.  A task/scale is eligible only when its paired success gain is
nonnegative on each split and at least `+3/16` in aggregate.  Selection
maximizes aggregate gain and breaks ties toward the smaller scale; tasks with
no eligible candidate use scale zero.  Both validation manifests must be
pairwise disjoint from the formal seed-1000 manifest, and
`test_metrics_used=false` is recorded.  The v5 launcher intentionally stops at
`calibration_complete_pending_audit`; it never starts a formal run without an
explicit evidence review.

```bash
bash scripts/robotwin_eval/launch_closed_loop_residual_calibration_v5.sh
```

The two-split v5 calibration completed on 2026-09-08.  Split B used the
frozen manifest SHA256
`8a088aa85bee0e757a0fd74b5808a9c67c3e0a20f75e4c614307be68bcf6cff2`;
paired Base scored `43/80`, while residual scales `0.25`, `0.5`, and `1.0`
scored `47/80`, `47/80`, and `45/80`.  Every condition has 80 videos, ten
successful worker exits, and exact seed/instruction pairing.  Only
`put_bottles_dustbin` at scale `0.5` replicated: its paired gains were `+3/8`
on split A and `+2/8` on split B, or `+5/16` in aggregate.  The calibrated
adapter therefore sets that task to `0.5`, the other nine tasks and the
default to `0`, records `seed_sets_pairwise_disjoint=true` and
`test_metrics_used=false`, and has SHA256
`8a12b0dbed2da7741aeff407e62d6eeaa676e958dc1eef1f772aafd8f95810cc`.
After this evidence review, the fixed seed-1000 formal evaluation ran all 200
episodes under `formal-calibrated-v5`.  It failed the performance gate:
Base was `114/200 = 57.0%`, while ZeVA was `109/200 = 54.5%` (absolute delta
`-2.5` points; Base-only/ZeVA-only discordant pairs `35/30`; exact McNemar
`p=0.620`).  The independent audit passed all protocol checks, including 200
ZeVA videos, ten successful worker exits, and exact seed/instruction pairing.
The v5 result is therefore a model-selection failure, not an evaluation
misalignment, and is retained only as negative evidence.

A lightweight branch audit can be run without loading PI0.5.  For the v3
adapter, the mean-language/all-phase-bin diagnostic found injected context RMS
`0.004881` versus prior RMS `0.0000826`, a `59.1x` ratio, while the learned
context gate was nearly constant across tasks.  This is a diagnostic over
prototypes and bank phases, not a closed-loop success metric; it is used to
decide whether a failed v5 run needs branch isolation or prior-strength
retraining instead of another blind shared-scale sweep.

v6 followed that preregistered diagnosis.  It exactly zeroed the context
projector and restores the Gaussian action prior to an absolute `0.5` gate,
matching BehaviorVLA's inference guidance magnitude while retaining the
trained task/phase router, task-language retrieval, and recurrent H15 phase.
PI0.5, ZTE/Mamba, the causal bank, and retrieval remain frozen.  On two
disjoint closed-loop validation splits the global candidate scored `50/80`
against Base `50/80` on split A and `45/80` against `43/80` on split B.  The
same per-task replication gate as v5 (non-negative on both splits and at least
three aggregate wins) enables only `beat_block_hammer` (`+1,+3`) and
`blocks_ranking_rgb` (`0,+3`); all other tasks and the default fall back
exactly to PI0.5.  The calibrated adapter SHA256 is
`28d99c1fd9092310acb6fd7a9040df619de45b773e6caa5feaa1d1caafc84709`.
Its metadata records pairwise-disjoint validation/formal seeds and
`test_metrics_used=false`.  A third frozen holdout split scored Base `41/80`
and ZeVA `46/80`, but the two actually enabled tasks contributed only `+1`;
the other `+4` came from tasks whose residual was disabled and is therefore
physics variance, not model gain.  The fixed 10x20 formal run then completed
under `advantage10-prior-guidance-v6/formal-calibrated-prior05-v6`: Base was
`114/200 = 57.0%`, while ZeVA was `102/200 = 51.0%` (absolute delta `-6.0`
points; Base-only/ZeVA-only discordant pairs `44/32`; exact McNemar
`p=0.207`, paired bootstrap 95% CI `[-14.5,+2.5]` points).  The independent
audit passed all structural checks: 200 episodes and videos per condition,
ten tasks with 20 episodes each, and exact seed/instruction pairing.  v6 is
therefore rejected as a model failure and cannot be delivered as an
improvement.

```bash
bash scripts/robotwin_eval/launch_prior_guidance_validation_v6.sh
bash scripts/robotwin_eval/launch_prior_guidance_formal_v6.sh
```

The rejected v7 candidate attempted to remove a train/deploy mismatch rather
than changing a trained 1% gate to 50% after optimization.  The v7
`prior_adapter` variant keeps PI0.5 (including the action expert), ZTE/Mamba,
the causal bank, retrieval, both scalar gates, and the complete context branch
frozen.  The context projector is exactly zero.  The prior gate is fixed at
`0.5` from the first training step, while the zero-initialized prior projector
keeps initialization exactly equal to PI0.5.  Only the task projector, memory
encoder, Gaussian prior, prior action projector, and bounded task/phase router
are optimized.  It retains prior residual dropout `0.4`, per-example matched
PI preservation, H50 prediction/H15 recurrence, and global batch 256.  This
candidate is trained and selected only with the existing train95/validation5
data; no formal-test metric is an optimizer or checkpoint-selection input.

```bash
bash scripts/train_robotwin_advantage10_prior_only_v7.sh
```

The first v7 attempt was stopped after its step-250 audit exposed an
Accelerate scheduler error: the saved scheduler had `last_epoch=2000`, so the
intended 250-global-step warmup had advanced eight times per step and ended at
about step 32.  Its superficially positive offline metric is invalid for
selection.  The corrected trainer sets
`step_scheduler_with_optimizer=False`, records the scheduler contract in the
manifest, and fails immediately unless `scheduler.last_epoch == completed
global optimizer steps`.  The clean v7 run restarts from PI0.5 at step zero in
`advantage10-prior-only-v7-corrected`; no checkpoint from the stopped directory
may be used for evaluation.

The corrected v7 selected step 1750 from train95/validation5 only, then ran two
fresh, disjoint 10x8 closed-loop validation splits. Split A was Base `47/80`
versus Zeva `44/80`; split B was Base `44/80` versus Zeva `41/80`. The combined
delta was therefore `-6/160`, so the preregistered gate rejected v7 and no
seed-10000 final run was launched. This is model evidence, not a protocol
failure: both manifests replayed the same expert-valid seed/instruction pairs.

After all eight checkpoints are present, derive the deployment choice from
validation5 only, then run two fresh closed-loop validation streams.  A fresh
seed-10000 final run is allowed only if each validation split is nonnegative
and their combined gain is at least six successes out of 160 episodes per
condition:

```bash
python3 scripts/select_robotwin_prior_adapter_checkpoint.py \
  /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-prior-only-v7-corrected/zeva
bash scripts/robotwin_eval/launch_prior_adapter_validation_v7.sh
bash scripts/robotwin_eval/launch_prior_adapter_fresh_final_v7.sh
```

The final launcher reruns untouched PI0.5 and ZeVA contemporaneously on the
same 10x20 expert-valid `(seed,instruction)` pairs beginning at seed 10000.  It
requires Base at or above the historical 57% floor, ZeVA strictly above Base,
200 videos per condition, and a fresh independent audit.  The final manifest
is pairwise checked against both validation streams and cannot be used to
revise the selected checkpoint.

```bash
PYTHONPATH=src python scripts/audit_robotwin_residual_branches.py \
  --adapter /path/to/zeva_adapter.pth \
  --goal-embeddings /path/to/pi05_task_embeddings.pt \
  --causal-bank /path/to/train_causal_bank.pt \
  --task-manifest configs/robotwin_zeva_advantage10.json \
  --output /path/to/residual_branch_audit.json
```

#### Historical v7/v9 audit (superseded by v14)

The historical v9 final comparison was designed with two trained conditions plus a same-seed
normality anchor. `Base` is the validation-selected action-expert-only step 3000;
`Zeva` starts from that exact Base and learns the dual residual with an immutable
Base teacher. The earlier untouched seed-1000 result was
114/200 and establishes the 57% floor; a trained Base below that floor cannot
be delivered. Fresh validation/final launchers must run Base and Zeva
contemporaneously on exactly the same pairs and verify the best-v1 parent SHA256
`7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe`
before rollout. The final gate is `Base >= max(same-seed Anchor,114/200)` and
`Zeva > Base`, with 200 videos per condition and an independent three-condition
audit. The
v7/v9 results and older action-expert branches remain diagnostics and can never
be substituted for the active v14 same-seed comparison.

#### Decoder and throughput configuration

Historical v9 used the same TorchCodec backend as the released RoboTwin PI0.5
baseline. Each persistent worker keeps a bounded 32-entry decoder LRU instead
of launching three FFmpeg processes per sample. On a real episode, TorchCodec
and the former FFmpeg path were verified bit-exact for all three cameras. The
FFmpeg backend remains available only for diagnostics.

The historical v9 run processed global batch 256 and three video streams per sample.
CPU TorchCodec random access plus backward-through-frozen-action-expert are the
main costs. The same-model residual-off teacher is sampled every two steps. Formal
Stage 2 retains H15 recurrent samples, H50 action output, 2,000 optimizer steps,
and the float32 `[0,1]` image contract. Validation is
forced to `num_workers=0`; this prevents forked validation workers from
retaining a CUDA context after every 500-step checkpoint.

Stage 2 passes its offline gate only when task retrieval remains at least 95%
and held-out enhanced flow does not regress against the matched unconditioned
branch. Final selection additionally requires paired baseline/Zeva closed-loop
evaluation with identical seeds, instructions, and diffusion RNG.

The former `stage2a-adapter/005000` checkpoint is invalid for final reporting.
Its FFmpeg images reached the visual-identity PI0.5 processor as uint8
`[0,255]`, so both its baseline and enhanced offline losses were measured in
the wrong model domain. Stage 2 now converts every view at the dataset boundary
to contiguous CHW float32 `[0,1]`; this historical fix was first recorded in
the anchored-v9 manifest/training state. The active v14 rejects full-v5,
adapter-only, or old image-contract checkpoints. The superseded candidate is
isolated under `advantage10-anchored-v9/adapter`; the joint
sibling, rejected v8 Zeva, and stopped v7 eager
runs remain audit artifacts. The v6 run injected
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
    foundation_checkpoint="/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1",
    goal_embedding_checkpoint="/mnt/100T/users/dingxin/VLA/runtime/pretrained_model-stage1-language-v1",
    zte_checkpoint="/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/zte_best.pth",
    stage2_checkpoint="/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-anchored-v9/adapter/001250",
    adapter_checkpoint="/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-anchored-v9/adapter/001250/zeva_adapter.pth",
    retrieval_checkpoint="/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1.5-task-retrieval/task_retrieval.pth",
    causal_bank="/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/train_causal_bank.pt",
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
before replanning. The v8 `Base` is the matched action-expert-only branch; the
untouched `pretrained_model-best-v1` is kept as an explicit same-seed Anchor.
Its established 114/200 (57.0%) result is also a historical floor, so a trained
Base below either the same-seed Anchor or 57.0% is rejected.

Checkpoint selection uses validation5 only. Two closed-loop validation streams
freeze eight expert-valid seeds per task beginning at 7000 and 8000; the final
is unreachable unless both are non-negative and their combined gain is at
least +6/160. For each task, the fresh final freezes the first 20 expert-valid
seeds selected from absolute seed 10000. Anchor, Base, and Zeva replay the exact
same seed and instruction pairs. The policy's recurrent Zeva state resets once per episode,
while the model flow RNG starts at 20260907 after model load and is consumed
continuously within each condition, matching the released PI0.5 evaluator.
GPU-PhysX initialization retries keep the same seed and happen before the first
model call; a failed seed is never silently substituted. Every episode must
produce a video whose success label agrees with the atomic progress record.

Before the formal comparison, residual trust is calibrated only on two
disjoint closed-loop validation manifests beginning at seeds 2000 and 3000.
Base runs once per split for eight expert-valid episodes per task; Zeva
evaluates residual scales 0.25, 0.5, and 1.0 on the exact paired
seed/instruction entries. A task/scale must have nonnegative paired gain on
both splits and aggregate gain of at least 3/16. The selector maximizes the
aggregate gain, breaks ties toward the smaller scale, and otherwise uses scale
0. It records all validation/formal manifest hashes,
`seed_sets_pairwise_disjoint=true`, and `test_metrics_used=false`; formal-test
outcomes are never an input. Calibration stops for audit before any formal run.
The completed v5 audit selected only `put_bottles_dustbin=0.5`; the other nine
tasks use scale zero.  The subsequent formal run is stored in
`formal-calibrated-v5` beneath the root below and retains the immutable
`114/200` Base evidence.

```bash
cd /mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA
nohup bash scripts/robotwin_eval/launch_closed_loop_residual_calibration_v5.sh \
  > /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v5-multisplit/launcher.log \
  2>&1 < /dev/null &
```

The validated deployment uses eight model servers on `aigc29` and eight SAPIEN
clients on the renderer-compatible `aigc24`. Do not combine results from nodes
with different NVIDIA/Vulkan stacks. The launcher is resumable from per-task
atomic progress files and emits condition reports, a paired report, videos,
calibration evidence, and an independent final audit under:

```text
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v5-multisplit
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
scripts/calibrate_robotwin_residual_trust_multisplit.py two-split non-regression selector
scripts/make_robotwin_residual_scale_candidate.py immutable residual-scale checkpoint builder
scripts/audit_robotwin_residual_branches.py foundation-free branch-strength diagnostic
scripts/robotwin_eval/launch_paired_formal_eval.sh strict paired resumable evaluator
scripts/robotwin_eval/launch_closed_loop_residual_calibration_v4.sh validation-to-formal orchestrator
scripts/robotwin_eval/launch_closed_loop_residual_calibration_v5.sh two-split calibration-only orchestrator
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
