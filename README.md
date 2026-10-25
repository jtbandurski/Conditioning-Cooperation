# Conditioning Cooperation

Multi-agent reinforcement learning on Commons Harvest substrates. Two agents learn with independent PPO, then mixed pairs are evaluated on a shared apple field.

The training and evaluation code is based on the [Melting Pot Challenge](https://www.aicrowd.com/challenges/melting-pot-challenge-2023) (NeurIPS 2023). Substrates live in `meltingpot/` and follow the [Melting Pot](https://github.com/google-deepmind/meltingpot) Commons Harvest design: an apple is worth 1, and a patch regrows only while a neighbouring apple remains (`regrowth radius 2`, probability `0.025`). Eating the last apple in a patch stops that patch from growing back. Episodes last 500 frames.

## Experiments

Three substrates, selected with `--exp`:

| `--exp` | Substrate | Layout | Role |
| --- | --- | --- | --- |
| `private` | `commons_harvest__private` | Two rooms split by a wall. Each agent has its own apple patch and cannot cross. | Training |
| `collective` | `commons_harvest__collective` | Same outer map with an open passage between the two patches. Both agents can reach both patches. | Training |
| `tragedy_test` | `commons_harvest__tragedy_test` | One open field and one apple patch. | Evaluation |

Training uses two players and one policy per player (`shared_policy` is off). The learner is PPO in PyTorch: LSTM of size 128, CNN filters `[16, 3×3]` then `[32, 7×7]`, `train_batch_size` 2048, minibatch 128, 10 SGD passes, `gamma` 0.99, learning rate `5e-4`, entropy coefficient `0.01`. Observations are downsampled by 8. A run stops after 10,000 training iterations (20,480,000 environment steps). Checkpoints are written every 100 iterations; the three best by `episode_reward_mean` are kept.

The study repeats both training substrates on seeds `123`, `1234`, `12345`, `123456`, and `1234567`.

Evaluation places one privately trained policy and one collectively trained policy together on `commons_harvest__tragedy_test`. Each seed has four pairings, named `Pa{i}_Ca{j}`: private-trained agent `i` with collective-trained agent `j`. Each pairing is 500 episodes. Per-episode returns are written to `evaluation.csv` in the config directory (`focal_per_capita_return`, `focal_player_returns`). Finished pairings are under `evaluations/seed_<seed>/tragedy_test_Pa*_Ca*/`. Videos from those runs are in `evaluations/videos/`.

Saved training runs are under `results_train_aws/torch/{private,collective}/`.

## Installation

Tested with Python 3.10. Use a virtual environment. `dmlab2d` is installed before the rest of the package, and `ray_patch.sh` patches Ray 2.6.1 so the LSTM policy can train.

```bash
conda create -n mpc_main python=3.10
conda activate mpc_main
SYSTEM_VERSION_COMPAT=0 pip install dmlab2d
pip install -e ".[dev]"
sh ray_patch.sh
```

Weights & Biases logging needs an API key:

```bash
export WANDB_API_KEY=<your-key>
```

Local runs log to a project named `<exp>_<framework>` in the `meltingpot` group.

## Run training

One substrate and one seed:

```bash
python baselines/train/run_ray_train.py \
  --num_workers=15 \
  --num_gpus=0 \
  --no-tune \
  --algo=ppo \
  --framework=torch \
  --exp=private \
  --seed=123 \
  --results_dir=./results_train_aws \
  --logging=INFO \
  --wandb=True \
  --downsample=True
```

Set `--exp` to `private` or `collective`. Checkpoints and `params.json` are written under `--results_dir`. With a GPU, set `--num_gpus` and, if several devices are present, `CUDA_VISIBLE_DEVICES`.

`run_multiple_trainings.sh` launches the five seeds for `collective` in separate `screen` sessions (15 workers, results in `./results_train_aws`). Create `screenlogs/` before running it. Repeat with `--exp=private` for the other substrate.

## Evaluate

An evaluation directory needs `params.json` and `policies/agent_0` plus `policies/agent_1`. `params.json` records the roles, the downsample scale, and the substrate. The checked-in directories already contain those mixed checkpoints.

One pairing:

```bash
python baselines/evaluation/evaluate.py \
  --tragedy_test=True \
  --num_episodes=500 \
  --config_dir=evaluations/seed_123/tragedy_test_Pa0_Ca0
```

`--tragedy_test=True` forces `commons_harvest__tragedy_test`. The script prints the episode table and writes `evaluation.csv` into `--config_dir`.

`run_multiple_evaluations.sh` starts all 5 seeds × 4 pairings (500 episodes each) in `screen`. It expects the `evaluations/seed_*/tragedy_test_*` directories to exist.

To record videos during evaluation, add `--create_videos=True`. Files go to `<config_dir>/videos/`.

To watch a checkpoint in a window on the tragedy map:

```bash
python baselines/train/render_models.py \
  --config_dir=evaluations/seed_123/tragedy_test_Pa0_Ca0 \
  --policies_dir=evaluations/seed_123/tragedy_test_Pa0_Ca0/policies \
  --horizon=200
```

## Validate

Environment checks for `commons_harvest__private` (action space, reset, step, render shape `(13, 7, 3)`):

```bash
pytest baselines/tests/envs_test.py
```

The full suite configured in `pyproject.toml` covers the Melting Pot substrates and the baseline tests:

```bash
pytest
```

A training run is doing the right thing when Ray reports PPO iterations climbing toward 10,000 and checkpoints appear under `results_train_aws/torch/<exp>/`. An evaluation is doing the right thing when `evaluation.csv` has one row per episode and `focal_player_returns` lists both agents.

## AWS SageMaker

`sagemaker_scripts/` launches the same training entry point on SageMaker. Jobs use `ml.g4dn.xlarge` (1× T4, 4 vCPU, 16 GB) in `eu-central-1`, with `num_workers=2`, `num_gpus=1`, and a 24 hour limit. Outputs go to `s3://<bucket>/meltingpot-results/<exp>/seed_<seed>`.

Install the launcher dependencies and configure AWS:

```bash
pip install -r sagemaker_scripts/requirements-sagemaker.txt
aws configure
export AWS_DEFAULT_REGION=eu-central-1
export WANDB_API_KEY=<your-key>
```

Build the image once and push it to ECR (`meltingpot-marl`). Docker must be running. The image is built for `linux/amd64`.

```bash
python sagemaker_scripts/launch_training.py \
  --build-image \
  --dry-run \
  --exp private
```

`--dry-run` builds the image and prints the job plan. Later launches can pass the printed URI with `--image-uri`.

From your own machine, pass the SageMaker execution role ARN:

```bash
python sagemaker_scripts/launch_training.py \
  --exp private \
  --seed 123 \
  --role arn:aws:iam::<account-id>:role/<SageMakerRole> \
  --image-uri <account-id>.dkr.ecr.eu-central-1.amazonaws.com/meltingpot-marl:latest
```

`--exp` accepts `private`, `collective`, or `tragedy_test`. Training still stops at 10,000 iterations unless `--training-iterations` is set. WandB logs to the project `meltingpot-sagemaker`.

All ten training jobs (2 substrates × 5 seeds):

```bash
python sagemaker_scripts/launch_training.py \
  --run-all \
  --role arn:aws:iam::<account-id>:role/<SageMakerRole> \
  --image-uri <account-id>.dkr.ecr.eu-central-1.amazonaws.com/meltingpot-marl:latest \
  --budget 1000
```

The launcher estimates cost at about 15 hours per job and asks for confirmation. It refuses to submit when the estimate is over `--budget` unless `--force` is set. Job names are written to `sagemaker_scripts/jobs_<timestamp>.json`.

Other instance types: `ml.g4dn.2xlarge` ($0.752/h), `ml.g4dn.4xlarge` ($1.204/h), `ml.g5.xlarge` ($1.006/h), `ml.p3.2xlarge` ($3.06/h). The default `ml.g4dn.xlarge` is $0.526/h, so ten jobs at ~15 hours are about $79.

Monitor, download, or stop jobs:

```bash
python sagemaker_scripts/monitor_jobs.py --status --region eu-central-1
python sagemaker_scripts/monitor_jobs.py --download --output-dir ./results_sagemaker --region eu-central-1
python sagemaker_scripts/monitor_jobs.py --stop-all --region eu-central-1
```

Downloaded artifacts are extracted under `--output-dir`. Use those checkpoints with the evaluation commands above. If a job fails with `ResourceLimitExceeded`, raise the SageMaker training quota for the instance type in Service Quotas.
