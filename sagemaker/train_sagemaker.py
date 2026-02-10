#!/usr/bin/env python3
"""
SageMaker Training Script for MeltingPot MARL
Adapted from baselines/train/run_ray_train.py for SageMaker environment
"""

import argparse
import os
import datetime
import json
import ray

from typing import *
from ray import air
from ray import tune
from ray.rllib.algorithms import ppo, a3c
from ray.tune import registry
from ray.air.integrations.wandb import WandbLoggerCallback

# Add parent directory to path for imports
import sys
sys.path.insert(0, '/opt/ml/code')

from baselines.train.configs import get_experiment_config
from baselines.train import make_envs


def get_sagemaker_args():
    """Parse arguments from SageMaker hyperparameters"""
    
    parser = argparse.ArgumentParser(description="SageMaker Training Script for MARL")
    
    # SageMaker specific paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))
    
    # Training parameters (same as original)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_gpus", type=float, default=1)
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--no-tune", action="store_true", default=True)
    parser.add_argument("--algo", choices=["ppo", "a3c"], default="ppo")
    parser.add_argument("--framework", choices=["tf", "torch"], default="torch")
    parser.add_argument("--exp", type=str, 
                       choices=['private', 'collective', 'tragedy_test', 'pd_arena', 'al_harvest', 'clean_up', 'territory_rooms'],
                       default="private")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--results_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--logging", choices=["DEBUG", "INFO", "WARN", "ERROR"], default="INFO")
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="meltingpot-sagemaker")
    parser.add_argument("--wandb_group", type=str, default="marl-experiments")
    parser.add_argument("--downsample", type=bool, default=True)
    parser.add_argument("--training_iterations", type=int, default=10000)
    parser.add_argument("--as-test", action="store_true", default=False)

    args = parser.parse_args()
    return args


def setup_wandb(args):
    """Setup WandB logging for SageMaker"""
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    
    if wandb_api_key and args.wandb:
        return [
            WandbLoggerCallback(
                project=args.wandb_project,
                group=args.wandb_group,
                api_key=wandb_api_key,
                log_config=True,
                upload_checkpoints=True,
            )
        ]
    else:
        print("WARNING: No WANDB_API_KEY found, running without WandB logging!")
        return []


def main():
    args = get_sagemaker_args()
    
    print("#" * 120)
    print(f"SageMaker Training Started: {datetime.datetime.now()}")
    print(f"Parameters: exp={args.exp}, workers={args.num_workers}, gpus={args.num_gpus}, seed={args.seed}")
    print(f"Model output dir: {args.model_dir}")
    print("#" * 120)

    # Initialize algorithm config
    if args.algo == "ppo":
        trainer = "PPO"
        default_config = ppo.PPOConfig()
    elif args.algo == "a3c":
        trainer = "A3C"
        default_config = a3c.A3CConfig()
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    # Get experiment configurations
    configs, exp_config, tune_config = get_experiment_config(args, default_config)
    
    # Override results directory for SageMaker
    exp_config['dir'] = args.model_dir
    
    # Override stopping criteria if specified
    exp_config['stop'] = {"training_iteration": args.training_iterations}

    # Initialize Ray
    ray.init(
        num_cpus=configs.num_cpus,
        num_gpus=configs.num_gpus,
        local_mode=args.local,
        ignore_reinit_error=True
    )

    # Register environment
    registry.register_env("meltingpot", make_envs.env_creator)

    # Check GPU availability
    if configs.num_gpus > 0:
        import torch
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, falling back to CPU")
            configs.num_gpus = 0

    # Setup WandB callbacks
    wdb_callbacks = setup_wandb(args)

    # Setup tune config
    tune_config = tune.TuneConfig(reuse_actors=False)

    # Setup checkpointing
    ckpt_config = air.CheckpointConfig(
        num_to_keep=exp_config['keep'],
        checkpoint_frequency=exp_config['freq'],
        checkpoint_at_end=True,  # Always save at end for SageMaker
        checkpoint_score_attribute=exp_config['checkpoint_score_attribute'],
        checkpoint_score_order=exp_config['checkpoint_score_order'],
    )

    # Run training
    results = tune.Tuner(
        trainer,
        param_space=configs.to_dict(),
        run_config=air.RunConfig(
            name=exp_config['name'],
            callbacks=wdb_callbacks,
            local_dir=exp_config['dir'],
            stop=exp_config['stop'],
            checkpoint_config=ckpt_config,
            verbose=1
        ),
    ).fit()

    # Get best result
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Best result: {best_result}")

    # Save training summary
    summary = {
        "best_reward": float(best_result.metrics.get("episode_reward_mean", 0)),
        "experiment": args.exp,
        "seed": args.seed,
        "training_iterations": args.training_iterations,
        "framework": args.framework,
        "timestamp": str(datetime.datetime.now())
    }
    
    summary_path = os.path.join(args.model_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    ray.shutdown()

    print("#" * 120)
    print(f"Training completed: {datetime.datetime.now()}")
    print(f"Results saved to: {args.model_dir}")
    print("#" * 120)


if __name__ == "__main__":
    main()
