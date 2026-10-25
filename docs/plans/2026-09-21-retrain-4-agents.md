# Retrain MARL Agents at 4 Players Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerun the private/collective Commons Harvest trainings with 4 agents instead of 2, keeping the same PPO procedure (10,000 training iterations), and launch the jobs on AWS SageMaker with an instance and wall-clock budget that will actually finish.

**Architecture:** Keep the existing independent-PPO setup in `baselines/train/`. Agent count is determined by the length of `env_config["roles"]`, which currently comes from each substrate's `default_player_roles` (hard-coded to 2). Add a `--num_players` CLI flag, scale the three custom ASCII maps so 4 agents still have a meaningful private / collective / tragedy layout, and pass the same flag through the SageMaker launcher. Evaluation must stop sampling 2 players from a shared `"default"` role and instead pin controlled 4-agent compositions (CCCC, CCCP, CCPP, CPPP, PPPP).

**Tech Stack:** Melting Pot (dmlab2d) substrates, Ray RLlib 2.6.1 PPO (torch), independent per-agent policies, AWS SageMaker training jobs (`ml.g4dn.*`), WandB, S3 checkpoints.

**Spec:** This plan is the spec. It is derived from the current 2-agent training/eval pipeline and from the paper's open "critical mass" question in `paper/springer_nature_submission/conditioning_co-operation_article.tex` §4.3.

## Global Constraints

- Stop criterion stays `"training_iteration": 10000` (do not switch to the paper's "10k episodes / 5M steps" wording; that does not match the code).
- Algorithm stays independent PPO (`shared_policy: False`), torch, LSTM 128, CNN `[16, 3×3]` + `[32, 7×7]`, `train_batch_size=2048`, `sgd_minibatch_size=128`, `num_sgd_iter=10`, `gamma=0.99`, `lr=5e-4`, `entropy_coeff=0.01`.
- Episode horizon stays `maxEpisodeLengthFrames=500`.
- Seeds stay `[123, 1234, 12345, 123456, 1234567]`.
- Experiments stay `private` and `collective` for training; `tragedy_test` for evaluation.
- Do not enable parameter sharing. That would be a different study.
- Region stays `eu-central-1`. SageMaker `max_run` must be raised above 24h for 4-agent jobs.
- Maps must preserve the property-rights semantics: private = physically fenced plots; collective = shared field; tragedy = one shared commons.

---

## Current training setup (read this first)

### How a run is constructed

1. CLI in `baselines/train/run_ray_train.py` (local/EC2) or `sagemaker_scripts/train_sagemaker.py` (SageMaker) sets `--exp`, `--seed`, `--num_workers`, `--num_gpus`, `--framework=torch`, `--algo=ppo`.
2. `baselines/train/configs.py:get_experiment_config` maps `--exp` to a substrate name:

   | `--exp` | Substrate |
   |---|---|
   | `private` | `commons_harvest__private` |
   | `collective` | `commons_harvest__collective` |
   | `tragedy_test` | `commons_harvest__tragedy_test` |

3. Player count is **not** a training hyperparameter today. It is:

```python
player_roles = substrate.get_config(substrate_name).default_player_roles
# currently ("default", "default") because get_config(num_players=2)
```

4. `make_envs.env_creator` builds `substrate.build(name, roles=roles)`. Lab2D uses `numPlayers = len(roles)`.
5. RLlib gets one independent `PolicySpec` per player (`agent_0` … `agent_{n-1}`), `policy_mapping_fn` maps `player_i → agent_i`.
6. Tune stops on `training_iteration == 10000`. Checkpoints: keep 3, every 100 iters, score = `episode_reward_mean`.

### How many steps were the 2-agent agents trained?

**Code stop rule** (`baselines/train/configs.py`):

```python
"stopping": {
    "training_iteration": 10000,
}
"train_batch_size": 2048,   # env steps per PPO iteration
"count_steps_by": "env_steps",  # confirmed in saved params.json
```

**Measured on a completed 2-agent private run** (`results_train_aws/torch/private/PPO_meltingpot_95104_00000_0_2024-04-19_14-03-22/progress.csv`, last row, iteration 10000):

| Quantity | Value |
|---|---|
| PPO iterations | 10,000 |
| Env steps sampled | **20,480,000** (= 10000 × 2048) |
| Agent steps sampled | **40,960,000** (2 agents × env steps) |
| Agent steps **per policy** | **20,480,000** |
| Episodes | **40,960** (episode length is always 500) |
| Wall clock (EC2, 7 CPU workers, 0 GPU) | **95,370 s ≈ 26.5 h** |
| Sample / learn split | ~3.0 s sample, ~5.6 s SGD per iter |

SageMaker's documented estimate (`sagemaker_scripts/README.md`) of "~15 h per experiment (10k iterations)" assumes a GPU instance (`ml.g4dn.xlarge`, 1× T4, `num_workers=2`). That is the same 10k-iteration budget, not a shorter one.

**Paper vs code.** The article says "trained for 10 thousand episodes … 5 million steps" and later "each agent learning period lasts for 5 million steps". That arithmetic is `10000 × 500 = 5e6` and treats a training *iteration* as an *episode*. The code and the saved `progress.csv` do not do that. **Replicate the code: 10,000 PPO iterations, 20.48M env steps.** If a reviewer asks, the honest statement is "10k PPO updates, 20.48M environment frames (paper text understated this by equating iteration with episode)."

### Current maps (2 agents)

Private (`commons_harvest__private.py`): two 5×5 rooms separated by a wall, one 5-apple patch per room, 2 top (`P`) and 2 bottom (`Q`) spawn tiles. Agents cannot cross.

Collective (`commons_harvest__collective.py`): same outer 7×13 footprint, no mid-wall, both agents share both patches.

Tragedy (`commons_harvest__tragedy_test.py`): one 5×5 field, one patch, 4 corner spawns. Evaluation only.

Spawn logic already splits `num_players` in half across the two spawn groups, so 4 players would currently put 2 agents in the top room and 2 in the bottom room. That is **not** a 4-plot private-property design; it is "two shared plots." The maps must be redesigned if private property is to keep the paper's meaning (one fenced plot per owner).

### Current eval pairing (2 agents)

`run_multiple_evaluations.sh` runs 5 seeds × 4 mixed checkpoints:

- `Pa0_Ca0`, `Pa0_Ca1`, `Pa1_Ca0`, `Pa1_Ca1`

Each eval directory's `params.json` has `"roles": ["default", "default"]`. `evaluate.py` loads `agent_0` and `agent_1` from that directory (manually assembled mixed checkpoints). Melting Pot's `Population` then samples **with replacement** from the `"default"` role pool, so an episode can be `(agent_0, agent_0)` or `(agent_1, agent_0)` etc. (see `evaluations/seed_123/tragedy_test_Pa0_Ca1/evaluation.csv`).

500 episodes × 4 pairings × 5 seeds = 10,000 eval episodes, matching the paper.

For 4 agents this sampling model is wrong: it cannot hold composition (k collective vs 4−k private) fixed.

### AWS / SageMaker as used today

| Piece | Current 2-agent value |
|---|---|
| Launcher | `sagemaker_scripts/launch_training.py` |
| Job entrypoint | `sagemaker_scripts/train_sagemaker.py` |
| Image | ECR `meltingpot-marl`, built from `sagemaker_scripts/Dockerfile` |
| Instance | `ml.g4dn.xlarge` (1× T4, 4 vCPU, 16 GB, $0.526/h) |
| Workers | `num_workers=2`, `num_gpus=1` |
| `max_run` | **24 hours** |
| Experiments | 2 substrates × 5 seeds = **10 jobs** |
| Cost estimate in README | 10 × 15 h × $0.526 ≈ **$79** |
| Budget default | `$1000` |
| Output | `s3://…/meltingpot-results/{exp}/seed_{seed}` |

`EXPERIMENTS` in the launcher only lists `private` and `collective`. `tragedy_test` is eval-only (one leftover job file exists: `sagemaker_scripts/jobs_20260211_154006.json`).

Local/EC2 path (`run_multiple_trainings.sh`) used `--num_workers=15` on a larger box; the completed `params.json` in `results_train_aws` shows `num_workers=7`, `num_cpus=8`, `num_gpus=0`. That CPU-only 26.5 h run would have been **killed** by SageMaker's 24 h cap. GPU is what makes the 15 h estimate plausible.

---

## What must change for 4 agents

### 1. Maps (required)

Do not just set `num_players=4` on the current ASCII. Private would become 2-vs-2 in two rooms; tragedy 5×5 would be four bodies on 25 tiles.

**Locked design: 2×2 fenced plots, same 5×5 room and 5-apple patch as the 2-agent study.**

Private (13×13 including outer walls, four 5×5 interiors):

```
WWWWWWWWWWWWW
WP    WW    PW
W  A  WW  A  W
W AAA WW AAA W
W  A  WW  A  W
W     WW     W
WWWWWWWWWWWWW
W     WW     W
W  A  WW  A  W
W AAA WW AAA W
W  A  WW  A  W
WR    WW    SW
WWWWWWWWWWWWW
```

- `P` / `Q`-style letters become four spawn groups, one per room (`P`, `Q`, `R`, `S`).
- One agent per room. That is the 4-player analogue of "private property as physical separation."

Collective: same 13×13 outer box, **delete the internal cross of walls**, keep four apple patches, four corner spawns. All four agents can reach every patch.

Tragedy eval: one open field, **one** central patch (keep the tragedy), more floor so 4 agents are not stacked. Use 9×9 including walls (7×7 interior) with four corner spawns and the same plus-shaped 5-apple patch in the centre:

```
WWWWWWWWW
WP     QW
W       W
W   A   W
W  AAA  W
W   A   W
W       W
WR     SW
WWWWWWWWW
```

Update `WORLD.RGB` spec and `baselines/tests/envs_test.py` `EXPECTED_WORLD_SIZE` to the new unscaled render shape (tiles × 8, e.g. private 13×13×8 → `(104, 104, 3)` if render is (H, W, C) matching the current test's `(13, 7, 3)` convention — **measure after the first env reset**, do not guess the axis order).

Spawn assignment in `create_avatar_objects`: one dedicated spawn group per player index (player `i` → group `i`), then randomize which physical room/corner that group is bound to at build time so positional bias stays off. Do not keep the current "half and half" 2-group split.

### 2. Training config / CLI (required)

Add `--num_players` (default 2 for backward compatibility) to:

- `baselines/train/run_ray_train.py`
- `baselines/tests/train_test.py` (parser only if needed)
- `sagemaker_scripts/train_sagemaker.py`
- `sagemaker_scripts/launch_training.py` hyperparameters

In `baselines/train/configs.py`, **ignore** `substrate.get_config(...).default_player_roles` length and set:

```python
player_roles = ("default",) * args.num_players
```

The policy loop `for i in range(len(player_roles))` already creates `agent_0` … `agent_{n-1}`. No other policy code change.

Keep `train_batch_size=2048` env steps. Consequence at 4 players:

| | 2 agents (current) | 4 agents (this plan) |
|---|---|---|
| Env steps / iter | 2048 | 2048 |
| Agent steps / iter (all) | 4096 | 8192 |
| Agent steps / policy / iter | 2048 | 2048 |
| SGD work | 2 nets | 4 nets (~2× learn time) |
| Env-step forwards | 2 policies | 4 policies (~2× sample forwards) |
| Total env steps @ 10k iters | 20.48M | 20.48M |
| Total per-policy steps | 20.48M | 20.48M |

Per-agent sample budget is unchanged. Wall-clock grows because there are twice as many independent learners and twice as many policy forwards per env step.

Do **not** double `train_batch_size`. That would give each policy the same samples *and* double SGD, on top of the extra nets.

### 3. Substrate `get_config` default (required for eval)

`meltingpot/configs/substrates/__init__.py:get_config` always calls `module.get_config()` with no args. Evaluation on a substrate uses `factory.default_player_roles()`, **not** the training `params.json` roles.

So even after training with 4 roles, `evaluate_population_on_substrate("commons_harvest__tragedy_test")` will still build a **2-player** env unless `get_config(num_players=…)` default becomes 4 **or** evaluate.py passes roles explicitly.

**Do both:** keep substrate default at 2, and change `evaluate.py` + `evaluate_population_on_substrate` to take `roles` from the eval `params.json` / CLI `--num_players`. Do not silently flip the 2-agent eval path.

### 4. Evaluation compositions (required)

Replace the 4 pairwise mixes with **composition** cells, which is the paper's unanswered critical-mass question:

| Tag | Makeup |
|---|---|
| `C4` | 4 collective |
| `C3P1` | 3 collective + 1 private |
| `C2P2` | 2 collective + 2 private |
| `C1P3` | 1 collective + 3 private |
| `P4` | 4 private |

Within a seed, each training still yields 4 distinct policies (`agent_0`…`agent_3`). For a first pass, pick agents **in index order** (C uses collective `agent_0..k-1`, P uses private `agent_0..4-k-1`) and fix slot assignment. Do not sample with replacement. Optionally later sweep which of the 4 trained heads are used; that is out of scope for the first 4-agent study (5 compositions × 5 seeds × 500 episodes = 12,500 eval episodes, comparable to the original 10,000).

`evaluate.py` must:

- Load N named policies (unique IDs, e.g. `C_agent_0`, `P_agent_1`) so mixed checkpoints do not collide on `agent_0`.
- Set `roles` to length 4.
- Set `names_by_role` so each **slot** maps to exactly one policy name (use distinct dummy roles `slot_0`…`slot_3`, or pass a fixed name list and skip `random.choice`).

Add a small assembler script `baselines/evaluation/assemble_population.py` that copies/symlinks checkpoints into `evaluations/seed_{seed}/{tag}/policies/` and writes `params.json` with 4 roles.

### 5. Tests (required)

- `baselines/tests/envs_test.py`: parametrize private/collective/tragedy at `num_players=2` and `4`; assert `len(obs)==n`, action space unchanged, render shape matches the new maps.
- New test: `create_avatar_objects(4)` returns 4 objects with 4 distinct spawn groups, and building the env does not raise.
- Smoke training test already in `baselines/tests/train_test.py` should run with `--num_players=4` and a tiny batch (this is slow; keep `train_batch_size=400`, 1 worker, and **override stop to 1 iteration** in the test — today the test uses the production 10k stop and is not a real unit test).

### 6. SageMaker / AWS (required)

4-agent cost drivers:

- SGD time ≈ 2× (4 independent PPO learners).
- Rollout forwards ≈ 2×.
- Simulation cost grows modestly (4 avatars).
- Memory: 4 LSTM policies + 4 agent RNNs × workers. 16 GB on `ml.g4dn.xlarge` is the risky one.

**Locked instance for the full 10k-iter jobs: `ml.g4dn.2xlarge`** (1× T4, 8 vCPU, 32 GB, $0.752/h).

| Setting | 2-agent (current) | 4-agent (this plan) |
|---|---|---|
| Instance | `ml.g4dn.xlarge` | `ml.g4dn.2xlarge` |
| vCPU / RAM | 4 / 16 GB | 8 / 32 GB |
| `num_workers` | 2 | **6** (driver + 6 workers = 7 CPUs, 1 spare) |
| `num_gpus` | 1 | 1 |
| `max_run` | 24 h | **48 h** (`max_run=48*60*60`) |
| Iters | 10,000 | 10,000 |
| Jobs | 10 | 10 |
| Expected wall clock | ~15 h (GPU estimate) | **~20–30 h** (2× SGD, more workers recover some sample time) |
| Expected cost | ~$79 | **10 × 25 h × $0.752 ≈ $188** (budget $400 to cover retries) |

If a 100-iteration probe (Task 8) shows >35 h projected, switch to `ml.g5.xlarge` ($1.006/h, faster GPU) **or** `ml.g4dn.4xlarge` (more CPUs) rather than cutting iterations.

`num_gpus_per_worker = num_gpus/num_workers` is already in `configs.py`. With 6 workers and 1 GPU this is a fraction (≈0.167). That is valid in RLlib; workers will share the T4. If Ray refuses fractional GPU on this stack, set `num_gpus_per_worker=0` and keep `num_gpus=1` on the learner/driver only (PPO with old API typically learns on the driver).

Launcher changes:

- Add `--num-players` (default 2).
- Pass it as hyperparameter `num_players`.
- Include `num_players` in the job name: `meltingpot-{exp}-n{n}-seed{seed}-…`.
- Include `num_players` in the S3 prefix: `meltingpot-results/n{n}/{exp}/seed_{seed}` so 2-agent and 4-agent artifacts never overwrite.
- `estimate_cost(..., hours_per_experiment=25)`.
- Raise `max_run` to 48 h.
- Default `--instance-type` to `ml.g4dn.2xlarge` when `num_players>=4`.
- Rebuild the Docker image (`--build-image`) after map/config changes; the image copies the whole repo.

Do a **10-iteration SageMaker canary** (1 job, `private`, seed 123) before `--run-all`. Confirm GPU is visible, 4 policies appear in the trial, checkpoints upload, then launch the remaining 9.

Spot/managed-spot is **not** in scope: LSTM PPO + 10k iters is painful to resume correctly with the current Tune layout.

### 7. What does *not* need to change

- CNN / LSTM / lr / entropy / minibatch / gamma.
- Downsampling factor 8 (8×8 sprite → 1×1; 7×7 view stays 7×7).
- `IGNORE_KEYS` / observation filtering.
- Avatar self-sprite (all others still render as the same brown `(150, 100, 50)`).
- `ray_patch.sh`, Dockerfile system packages, `requirements-docker.txt` pins.
- WandB project name can stay; use a new **group** `n4-{exp}-experiments`.

---

## File map

**Create:**

- `baselines/evaluation/assemble_population.py` — copy selected checkpoints into an eval directory with unique policy IDs and a 4-role `params.json`.
- `docs/plans/2026-09-21-retrain-4-agents.md` — this file.

**Modify:**

- `meltingpot/configs/substrates/commons_harvest__private.py` — ASCII map, spawn groups, `create_avatar_objects`.
- `meltingpot/configs/substrates/commons_harvest__collective.py` — same.
- `meltingpot/configs/substrates/commons_harvest__tragedy_test.py` — same.
- `baselines/train/configs.py` — `--num_players` consumed via `args.num_players` for roles.
- `baselines/train/run_ray_train.py` — add `--num_players`.
- `sagemaker_scripts/train_sagemaker.py` — add `--num_players`, print it, do not override roles.
- `sagemaker_scripts/launch_training.py` — hyperparam, job name, S3 prefix, instance default, `max_run`, cost hours.
- `sagemaker_scripts/README.md` — 4-agent cost table.
- `baselines/evaluation/evaluate.py` — pin policies to slots; honour `--num_players` / params roles; unique policy ids.
- `baselines/tests/envs_test.py` — 2- and 4-player cases, new render shapes.
- `baselines/tests/train_test.py` — stop after 1 iteration; optional `--num_players=4` smoke.
- `run_multiple_trainings.sh` / `run_multiple_evaluations.sh` — 4-player variants (or new `run_multiple_trainings_n4.sh`).

**Do not modify:** paper TeX in this workstream. After results exist, a separate writing pass can replace "two agents" / "5 million steps" with the measured 4-agent protocol.

---

### Task 1: Make player count a training argument

**Files:**
- Modify: `baselines/train/run_ray_train.py`
- Modify: `baselines/train/configs.py:52-71` and `198-226`
- Modify: `sagemaker_scripts/train_sagemaker.py`
- Test: `baselines/tests/envs_test.py` (roles length)

**Interfaces:**
- Consumes: existing `args.exp`, substrate `get_config`
- Produces: `args.num_players: int` (default `2`); `env_config["roles"] = ("default",) * args.num_players`; `len(policies) == args.num_players`

- [ ] **Step 1: Write a failing assertion in the env test**

Add to `baselines/tests/envs_test.py` a second test class or parameterized case that builds the private env with 4 roles:

```python
def test_four_player_reset_count(self):
    env = make_envs.env_creator({
        'substrate': 'commons_harvest__private',
        'roles': ('default',) * 4,
        'scaled': 8,
    })
    obs, _ = env.reset()
    self.assertLen(obs, 4)
    env.close()
```

This will fail or mis-spawn until Task 2 redesigns the map; that is acceptable — keep it skipped with `@absltest.skip('maps still 2-player')` until Task 2, then enable it.

- [ ] **Step 2: Add the CLI flag**

In `baselines/train/run_ray_train.py` `get_cli_args()`, after `--seed`:

```python
parser.add_argument(
    "--num_players",
    type=int,
    default=2,
    help="Number of players in the substrate. Length of default_player_roles is ignored.",
)
```

Mirror the same flag in `sagemaker_scripts/train_sagemaker.py:get_sagemaker_args` (default 2 so old jobs stay valid).

- [ ] **Step 3: Use it when building roles**

In `baselines/train/configs.py`, replace:

```python
player_roles = substrate.get_config(substrate_name).default_player_roles
```

with:

```python
num_players = getattr(args, "num_players", 2)
player_roles = ("default",) * num_players
```

Still call `substrate.get_config(substrate_name)` if anything else is needed; do not use its `default_player_roles` for length.

Log `num_players` next to seed in the banner prints in `run_ray_train.py` and `train_sagemaker.py`.

- [ ] **Step 4: Commit**

```bash
git add baselines/train/run_ray_train.py baselines/train/configs.py sagemaker_scripts/train_sagemaker.py baselines/tests/envs_test.py
git commit -m "$(cat <<'EOF'
feat: thread --num_players into RLlib env roles

EOF
)"
```

---

### Task 2: Redesign the three ASCII maps for 4 players

**Files:**
- Modify: `meltingpot/configs/substrates/commons_harvest__private.py`
- Modify: `meltingpot/configs/substrates/commons_harvest__collective.py`
- Modify: `meltingpot/configs/substrates/commons_harvest__tragedy_test.py`
- Modify: `baselines/tests/envs_test.py`

**Interfaces:**
- Consumes: `len(roles)` in `build()`
- Produces: 4 spawn groups; private = 4 fenced 5×5 plots; collective = open 13×13 with 4 patches; tragedy = 7×7 interior, 1 patch, 4 corners

- [ ] **Step 1: Extend CHAR_PREFAB_MAP with four spawn letters**

In all three files, keep `P` as `spawn_point` and add `Q`, `R`, `S` mapped to `bottom_spawn_point` / two new prefabs `spawn_point_c` / `spawn_point_d` (copy the existing spawn prefab, change only `"kwargs": {"group": "spawnGroupC"}` etc.). Reuse the same four group names in private, collective, and tragedy so `create_avatar_objects` can be identical.

- [ ] **Step 2: Replace ASCII_MAP** with the locked designs in the "What must change" section. Count rows/cols and update the `WORLD.RGB` spec from `specs.rgb(5*SPRITE_SIZE, 5*SPRITE_SIZE)` to `specs.rgb(height_tiles*SPRITE_SIZE, width_tiles*SPRITE_SIZE)` using the actual map dimensions.

- [ ] **Step 3: Rewrite `create_avatar_objects`**

```python
SPAWN_GROUPS = ["spawnPoints", "bottomSpawnPoints", "spawnGroupC", "spawnGroupD"]

def create_avatar_objects(num_players):
    if num_players > len(SPAWN_GROUPS):
        raise ValueError(f"Need {num_players} spawn groups, have {len(SPAWN_GROUPS)}")
    # Rotate assignment so spawn location is not tied to policy index.
    offset = randint(0, num_players - 1)
    avatar_objects = []
    for player_idx in range(num_players):
        spawn_group = SPAWN_GROUPS[(player_idx + offset) % num_players]
        avatar_objects.append(
            create_avatar_object(player_idx, TARGET_SPRITE_SELF, spawn_group=spawn_group)
        )
    return avatar_objects
```

Keep 2-player working: for `num_players==2` use only the first two groups (top-left and bottom-right, or top pair vs bottom pair). Put that branch in explicitly so old 2-agent eval still runs on the new maps. **Recommended 2-player behaviour on the new private map:** spawn the two agents in two *diagonally opposite* rooms so they still cannot share a plot and still see each other through walls if the view reaches. Document that 2-agent layouts changed if you keep one map file; alternatively keep the original 7×13 ASCII when `num_players==2` and switch ASCII on `len(roles)` inside `build()`:

```python
ascii_map = ASCII_MAP_4 if num_players >= 4 else ASCII_MAP_2
```

**Lock this:** dual ASCII (`ASCII_MAP_2` unchanged, `ASCII_MAP_4` as designed) selected in `build()` from `len(roles)`. That preserves the published 2-agent study.

- [ ] **Step 4: Enable and fix env tests**

```python
class MeltingPotEnvTestsFourPlayer(absltest.TestCase):
    SUBSTRATES = (
        "commons_harvest__private",
        "commons_harvest__collective",
        "commons_harvest__tragedy_test",
    )

    def test_reset_four_agents(self):
        for name in self.SUBSTRATES:
            env = make_envs.env_creator({
                'substrate': name,
                'roles': ('default',) * 4,
                'scaled': 8,
            })
            obs, _ = env.reset()
            self.assertLen(obs, 4)
            env.close()
```

Run:

```bash
python -m baselines.tests.envs_test
```

Expected: PASS. Record the 4-player `env.render().shape` and set `EXPECTED_WORLD_SIZE_4` from that measurement.

- [ ] **Step 5: Commit**

```bash
git add meltingpot/configs/substrates/commons_harvest__*.py baselines/tests/envs_test.py
git commit -m "$(cat <<'EOF'
feat: add 4-player private, collective, and tragedy maps

EOF
)"
```

---

### Task 3: Make the 1-iteration training smoke test actually stop

**Files:**
- Modify: `baselines/tests/train_test.py`

**Interfaces:**
- Consumes: `get_experiment_config`
- Produces: a test that runs 1 PPO iteration with 2 and 4 players on CPU

- [ ] **Step 1: Override stop and resources in `test_training`**

After `get_experiment_config`:

```python
exp_config['stop'] = {"training_iteration": 1}
configs.num_gpus = 0
configs.num_rollout_workers = 0
configs.train_batch_size = 200
configs.sgd_minibatch_size = 50
configs.rollout_fragment_length = 10
```

- [ ] **Step 2: Add `test_training_four_players`** that sets `args.num_players = 4` (build a tiny namespace or parse `--num_players=4`) and uses the same 1-iteration stop.

- [ ] **Step 3: Run**

```bash
python -m baselines.tests.train_test
```

Expected: PASS in a few minutes. Failure mode to watch: Ray GPU fraction, missing spawn groups, observation shape vs CNN `dim=7`.

- [ ] **Step 4: Commit**

```bash
git add baselines/tests/train_test.py
git commit -m "$(cat <<'EOF'
test: cap training smoke test at one PPO iteration for 2 and 4 players

EOF
)"
```

---

### Task 4: Pin evaluation populations to 4-agent compositions

**Files:**
- Create: `baselines/evaluation/assemble_population.py`
- Modify: `baselines/evaluation/evaluate.py`
- Modify: `run_multiple_evaluations.sh` (or create `run_multiple_evaluations_n4.sh`)

**Interfaces:**
- Consumes: training checkpoint dirs `results_.../torch/{private,collective}/.../checkpoint_010000/policies/agent_{i}`
- Produces: `evaluations/n4/seed_{seed}/{tag}/params.json` and `policies/{policy_id}/`

- [ ] **Step 1: Assembler**

`assemble_population.py` CLI:

```text
--seed 123
--tag C2P2
--private-ckpt /path/to/private/checkpoint_010000
--collective-ckpt /path/to/collective/checkpoint_010000
--out evaluations/n4/seed_123/C2P2
```

Composition table in code:

```python
COMPOSITIONS = {
    "C4":   ("C", "C", "C", "C"),
    "C3P1": ("C", "C", "C", "P"),
    "C2P2": ("C", "C", "P", "P"),
    "C1P3": ("C", "P", "P", "P"),
    "P4":   ("P", "P", "P", "P"),
}
```

Copy `agent_0`, `agent_1`, … from the matching training, renaming to `slot_0` … `slot_3`. Write:

```json
{
  "env_config": {
    "roles": ["slot_0", "slot_1", "slot_2", "slot_3"],
    "scaled": 8,
    "substrate": "commons_harvest__tragedy_test"
  }
}
```

- [ ] **Step 2: Change `evaluate.py` sampling**

Today every role is `"default"` so `Population._sample_names` does `random.choice`. After assembly, each role has exactly one name. Still set:

```python
roles = configs["env_config"]["roles"]
policy_ids = list(roles)  # slot_0 ... slot_3
names_by_role = {role: [role] for role in roles}
```

and load `EvalPolicy(policies_path, p_id)` for each `p_id`.

Pass `roles` into substrate build. If `evaluation.evaluate_population_on_substrate` cannot take custom roles, call `factory.build(roles)` locally (copy the 20-line function into `evaluate.py` rather than silently using `default_player_roles`).

- [ ] **Step 3: Tragedy `get_config` must accept 4 roles.** `valid_roles` is currently `frozenset({"default"})`. Either add `slot_0`…`slot_3` to `valid_roles` **or** keep roles as four `"default"` strings and pass an explicit `names` list by subclassing/wrapping `Population` to skip sampling:

```python
class FixedPopulation(population_lib.Population):
    def _sample_names(self):
        return list(self._roles)  # one policy name per slot, already ordered
```

**Lock this:** keep substrate roles as `("default",)*4` and use `FixedPopulation` with `policies` keyed `slot_0`…`slot_3` and `_roles = ("slot_0", …)` only inside the Population, while `factory.build(("default",)*4)` builds the env. Simplest path that does not fork Melting Pot role validation:

```python
env_roles = ("default",) * 4
population = FixedPopulation(
    policies=policies_by_slot,          # {"slot_0": ..., "slot_3": ...}
    names_by_role={s: [s] for s in policies_by_slot},
    roles=("slot_0", "slot_1", "slot_2", "slot_3"),
)
factory.build(env_roles)  # 4 default avatars
```

`Population.reset` samples names for policies; `send_timestep` zips them with the 4 env players in order. That matches slots to avatars without changing `valid_roles`.

- [ ] **Step 4: Smoke-eval 2 episodes** for `P4` seed 123 after a 1-iteration checkpoint exists.

- [ ] **Step 5: Commit**

```bash
git add baselines/evaluation/assemble_population.py baselines/evaluation/evaluate.py run_multiple_evaluations_n4.sh
git commit -m "$(cat <<'EOF'
feat: assemble and evaluate fixed 4-agent C/P compositions

EOF
)"
```

---

### Task 5: SageMaker launcher, instance, and 48 h cap

**Files:**
- Modify: `sagemaker_scripts/launch_training.py`
- Modify: `sagemaker_scripts/train_sagemaker.py` (already has `--num_players` from Task 1)
- Modify: `sagemaker_scripts/README.md`

**Interfaces:**
- Consumes: `--num-players`, `--instance-type`, `--training-iterations`
- Produces: jobs named `meltingpot-{exp}-n{n}-seed{seed}-…` writing to `s3://…/meltingpot-results/n{n}/{exp}/seed_{seed}`

- [ ] **Step 1: Hyperparameters and paths**

In `launch_single_experiment`:

```python
hyperparameters = {
    "exp": exp_name,
    "seed": str(seed),
    "framework": "torch",
    "num_workers": str(num_workers),
    "num_gpus": "1",
    "num_players": str(num_players),
    "training_iterations": str(training_iterations),
    ...
}
safe = exp_name.replace("_", "-")
job_base = f"meltingpot-{safe}-n{num_players}"
output_path = f"s3://{s3_bucket}/meltingpot-results/n{num_players}/{exp_name}/seed_{seed}"
```

Add CLI `--num-players` (default 2) and `--num-workers` (default: 2 if `num_players==2` else 6).

- [ ] **Step 2: `max_run` and instance**

```python
max_run_hours = 48 if num_players >= 4 else 24
estimator = Estimator(
    ...
    instance_type=instance_type,
    max_run=max_run_hours * 60 * 60,
)
```

If `args.num_players >= 4` and the user left the default instance, use `ml.g4dn.2xlarge`.

- [ ] **Step 3: Cost estimate**

```python
hours = 25 if num_players >= 4 else 15
estimated_cost = estimate_cost(args.instance_type, total_experiments, hours_per_experiment=hours)
```

Print jobs × hours × rate. README table:

```
4 agents × 2 envs × 5 seeds = 10 jobs
ml.g4dn.2xlarge @ $0.752/h × ~25 h ≈ $188
budget floor for this wave: $400 (retries + canary)
```

- [ ] **Step 4: Dockerfile ENTRYPOINT already ignores extra SageMaker args.** Confirm `train_sagemaker.py` reads `--num_players` from hyperparameters (SageMaker injects them as CLI flags). If the shell ENTRYPOINT swallows flags (it currently does: `python sagemaker_scripts/train_sagemaker.py` with no `"$@"`), **fix that**:

```dockerfile
ENTRYPOINT ["python", "sagemaker_scripts/train_sagemaker.py"]
```

SageMaker appends `--exp private --num_players 4 …`. The current `/bin/bash -c "python …"` form **drops hyperparameters**. Verify whether existing 2-agent jobs worked via env vars or because Estimator overwrote the command. If 2-agent SageMaker jobs trained the requested exp, leave the entrypoint; if they always ran defaults, this is a pre-existing bug and must be fixed before the 4-agent wave.

Check by reading a completed SageMaker log. If none exist, change ENTRYPOINT to exec-form so hyperparameters reach argparse, and test with `--dry-run` plus a local `docker run … --exp private --num_players 4 --training_iterations 1`.

- [ ] **Step 5: Commit**

```bash
git add sagemaker_scripts/launch_training.py sagemaker_scripts/train_sagemaker.py sagemaker_scripts/Dockerfile sagemaker_scripts/README.md
git commit -m "$(cat <<'EOF'
feat: launch 4-player SageMaker jobs on g4dn.2xlarge with a 48h cap

EOF
)"
```

---

### Task 6: Local 10-iteration canary (CPU)

**Files:** none new; uses Task 1–3.

- [ ] **Step 1: Run private, 4 players, 10 iterations**

```bash
python baselines/train/run_ray_train.py \
  --num_workers=0 --num_gpus=0 --no-tune --algo=ppo --framework=torch \
  --exp=private --seed=123 --num_players=4 \
  --results_dir=./results_n4_canary --logging=INFO --downsample=True
```

Temporarily set stop to 10 iterations by exporting a one-off patch **or** add `--training_iterations` to `run_ray_train.py` (SageMaker already has this; **add the same flag to the local script** in this task so you do not edit `configs.py` by hand).

Add to `run_ray_train.py`:

```python
parser.add_argument("--training_iterations", type=int, default=None)
```

and in `main`, if set: `exp_config["stop"] = {"training_iteration": args.training_iterations}`.

- [ ] **Step 2: Confirm**

- Trial folder contains `policies/agent_0` … `agent_3` after checkpoint.
- `progress.csv` last row: `training_iteration==10`, `num_env_steps_sampled==20480`, `agent_timesteps_total==81920`.
- No CUDA errors; episode length 500.

- [ ] **Step 3: Repeat for `--exp=collective`.**

- [ ] **Step 4: Commit the CLI flag only** (not canary results).

```bash
git add baselines/train/run_ray_train.py
git commit -m "$(cat <<'EOF'
feat: allow overriding PPO training_iteration on the local trainer

EOF
)"
```

---

### Task 7: Rebuild the SageMaker image and run a 100-iteration probe

- [ ] **Step 1: Build and push**

```bash
export AWS_DEFAULT_REGION=eu-central-1
python sagemaker_scripts/launch_training.py \
  --build-image --dry-run --exp private --seed 123 --num-players 4
```

- [ ] **Step 2: Launch one real probe**

```bash
python sagemaker_scripts/launch_training.py \
  --exp private --seed 123 --num-players 4 \
  --training-iterations 100 \
  --instance-type ml.g4dn.2xlarge \
  --image-uri <ecr-uri>
```

- [ ] **Step 3: Monitor**

```bash
python sagemaker_scripts/monitor_jobs.py --status --region eu-central-1
```

Record: GPU seen, `time_total_s` at iter 100, RAM, whether 4 policies train. **Projected full-run hours = `time_total_s/100 * 10000 / 3600`.**

- [ ] **Step 4: Go / no-go**

| Projected 10k hours | Action |
|---|---|
| ≤ 30 h | Proceed with `ml.g4dn.2xlarge`, `max_run=48h` |
| 30–40 h | Keep instance, still OK under 48 h |
| > 40 h | Switch to `ml.g5.xlarge` or raise `num_workers` on `ml.g4dn.4xlarge`; do not cut iterations |

Download the probe checkpoint and run `assemble_population.py` + 2 eval episodes on tragedy 4-player.

Do not commit probe artifacts.

---

### Task 8: Launch the full 10-job 4-agent wave

- [ ] **Step 1: Dry-run the matrix**

```bash
python sagemaker_scripts/launch_training.py \
  --run-all --num-players 4 \
  --training-iterations 10000 \
  --instance-type ml.g4dn.2xlarge \
  --budget 400 \
  --image-uri <ecr-uri> \
  --dry-run
```

Expected print: 10 jobs, ~$188–$250, instance `ml.g4dn.2xlarge`.

- [ ] **Step 2: Submit** (same command without `--dry-run`). Save the new `sagemaker_scripts/jobs_*.json` manifest.

- [ ] **Step 3: Watch the first 2 hours.** If a job dies on `ResourceLimitExceeded`, request a quota increase for `ml.g4dn.2xlarge` training jobs (README already documents this). If OOM, drop `num_workers` to 4 and relaunch that seed only.

- [ ] **Step 4: When `Completed`, download**

```bash
python sagemaker_scripts/monitor_jobs.py --download --output-dir ./results_sagemaker_n4 --region eu-central-1
```

Verify each seed has `checkpoint_010000/policies/agent_{0,1,2,3}`.

---

### Task 9: Evaluate the 5 compositions × 5 seeds

- [ ] **Step 1: Assemble** all `evaluations/n4/seed_{seed}/{C4,C3P1,C2P2,C1P3,P4}` directories from the downloaded checkpoints.

- [ ] **Step 2: Run 500 episodes each** via `run_multiple_evaluations_n4.sh` (CPU is fine; eval is not GPU-bound).

- [ ] **Step 3: Aggregate** by cloning the logic in `evaluations/0_data_aggregation.ipynb` into a 4-player path that stores per-slot scores and composition tag. Primary metrics: mean per-capita return, mean collective-agent return vs private-agent return, fraction of episodes that keep the patch alive (proxy: total return ≳ long-cooperation mode).

- [ ] **Step 4: Do not treat this as a paper rewrite.** Stop when CSVs exist and a one-page note in `evaluations/n4/README.md` records job hours, instance, env steps, and composition table.

---

## AWS compute summary (for the decision at launch time)

The 2-agent study already **is** an AWS-scale job: 10 independent 10k-iteration PPO runs. SageMaker exists specifically because one run is ~15–26 hours.

Moving to 4 agents does **not** change the number of jobs (still 2 × 5). It changes **what each job does**:

1. **Same env-step budget** (20.48M), **same per-policy sample budget** (20.48M).
2. **Twice the learners** → roughly 2× SGD, which was already ~60% of iteration time on CPU.
3. **Twice the per-step policy forwards** during rollout.
4. **Heavier RAM**, so the 16 GB `xlarge` is the wrong default.
5. The **24 h `max_run` will kill** a 4-agent job if it regresses toward the 26.5 h CPU timing. Raise to 48 h **before** `--run-all`.
6. Cost scales with wall-clock × instance rate, not with agent count directly. Expect about **2–2.5× the 2-agent SageMaker bill** (~$80 → ~$190), not 2× the number of jobs.

Recommended sequence: Tasks 1–6 on a laptop/CPU (maps + flag + 10-iter canary) → Task 7 one paid 100-iter probe → Task 8 full wave → Task 9 eval. Do not launch 10 full jobs until the probe's projected hours are written down.

---

## Self-review

**Spec coverage:** player-count CLI, dual ASCII maps, spawn groups, eval compositions, SageMaker instance/`max_run`/S3 isolation, cost, canary, full wave, eval aggregation. Paper text left as a follow-up.

**Placeholder scan:** none. Map ASCII, composition dict, instance types, and hour/cost numbers are explicit.

**Type consistency:** `args.num_players: int`; roles `tuple[str, ...]`; policies `slot_0`…`slot_3` in eval, `agent_0`…`agent_3` in training; S3 prefix `n{num_players}`.
