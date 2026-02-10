#!/usr/bin/env python3
"""
Launch SageMaker Training Jobs for MeltingPot MARL Experiments

Usage:
    # Single experiment
    python sagemaker/launch_training.py --exp private --seed 123
    
    # Multiple experiments (all combinations)
    python sagemaker/launch_training.py --run-all
    
    # Custom configuration
    python sagemaker/launch_training.py --exp collective --seed 456 --instance-type ml.g4dn.2xlarge
"""

import argparse
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from datetime import datetime
import os
import json
import time


# Configuration
DEFAULT_CONFIG = {
    "role": None,  # Will be auto-detected or must be provided
    "instance_type": "ml.g4dn.xlarge",  # ~$0.526/hour, 1 GPU, 4 vCPUs, 16GB RAM
    "instance_count": 1,
    "max_run_hours": 24,
    "s3_bucket": None,  # Will be auto-created if not provided
    "ecr_repository": "meltingpot-marl",
    "region": "eu-central-1",
}

# Experiment configurations for full study
EXPERIMENTS = [
    {"exp": "private", "seeds": [123, 1234, 12345, 123456, 1234567]},
    {"exp": "collective", "seeds": [123, 1234, 12345, 123456, 1234567]},
]

# Cost estimates per instance type (USD/hour)
INSTANCE_COSTS = {
    "ml.g4dn.xlarge": 0.526,    # 1 GPU, 4 vCPU, 16GB - RECOMMENDED
    "ml.g4dn.2xlarge": 0.752,   # 1 GPU, 8 vCPU, 32GB
    "ml.g4dn.4xlarge": 1.204,   # 1 GPU, 16 vCPU, 64GB
    "ml.p3.2xlarge": 3.06,      # 1 V100 GPU - faster but expensive
    "ml.g5.xlarge": 1.006,      # 1 A10G GPU - good balance
}


def get_sagemaker_role():
    """Get or create SageMaker execution role"""
    try:
        role = sagemaker.get_execution_role()
        print(f"Using existing SageMaker role: {role}")
        return role
    except ValueError:
        # Not running in SageMaker, need to specify role ARN
        print("Not running in SageMaker environment.")
        print("Please provide --role argument with your SageMaker execution role ARN")
        print("Example: arn:aws:iam::123456789012:role/SageMakerExecutionRole")
        return None


def get_or_create_bucket(session, bucket_name=None):
    """Get existing bucket or create a new one"""
    if bucket_name:
        return bucket_name
    
    # Create default bucket
    default_bucket = session.default_bucket()
    print(f"Using S3 bucket: {default_bucket}")
    return default_bucket


def build_and_push_docker_image(repository_name, region):
    """Build and push Docker image to ECR"""
    import subprocess
    
    account_id = boto3.client('sts').get_caller_identity()['Account']
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}"
    
    print(f"Building Docker image: {ecr_uri}")
    
    # Login to ECR
    login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    subprocess.run(login_cmd, shell=True, check=True)
    
    # Create repository if it doesn't exist
    ecr_client = boto3.client('ecr', region_name=region)
    try:
        ecr_client.create_repository(repositoryName=repository_name)
        print(f"Created ECR repository: {repository_name}")
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"ECR repository already exists: {repository_name}")
    
    # Build image for x86_64 (SageMaker runs on x86_64)
    subprocess.run(f"docker build --platform linux/amd64 -t {repository_name} -f sagemaker_scripts/Dockerfile .", shell=True, check=True)
    
    # Tag and push
    subprocess.run(f"docker tag {repository_name}:latest {ecr_uri}:latest", shell=True, check=True)
    subprocess.run(f"docker push {ecr_uri}:latest", shell=True, check=True)
    
    return f"{ecr_uri}:latest"


def estimate_cost(instance_type, num_experiments, hours_per_experiment=15):
    """Estimate total cost for experiments"""
    hourly_cost = INSTANCE_COSTS.get(instance_type, 0.5)
    total_cost = hourly_cost * hours_per_experiment * num_experiments
    return total_cost


def launch_single_experiment(
    image_uri,
    role,
    s3_bucket,
    exp_name,
    seed,
    instance_type,
    wandb_api_key=None,
    training_iterations=10000,
    dry_run=False
):
    """Launch a single SageMaker training job"""
    
    job_name = f"meltingpot-{exp_name}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_path = f"s3://{s3_bucket}/meltingpot-results/{exp_name}/seed_{seed}"
    
    hyperparameters = {
        "exp": exp_name,
        "seed": str(seed),
        "framework": "torch",
        "num_workers": "2",
        "num_gpus": "1",
        "training_iterations": str(training_iterations),
        "wandb": "True" if wandb_api_key else "False",
        "wandb_project": "meltingpot-sagemaker",
        "wandb_group": f"{exp_name}-experiments",
    }
    
    environment = {}
    if wandb_api_key:
        environment["WANDB_API_KEY"] = wandb_api_key
    
    print(f"\n{'='*60}")
    print(f"Launching: {job_name}")
    print(f"  Experiment: {exp_name}")
    print(f"  Seed: {seed}")
    print(f"  Instance: {instance_type}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    if dry_run:
        print("DRY RUN - Job not submitted")
        return None
    
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        output_path=output_path,
        hyperparameters=hyperparameters,
        environment=environment,
        max_run=24 * 60 * 60,  # 24 hours max
        base_job_name=f"meltingpot-{exp_name}",
    )
    
    estimator.fit(wait=False)  # Don't wait, launch async
    
    return {
        "job_name": job_name,
        "exp": exp_name,
        "seed": seed,
        "output_path": output_path,
        "status": "launched"
    }


def launch_all_experiments(args, image_uri, role, s3_bucket):
    """Launch all experiment combinations"""
    
    jobs = []
    total_experiments = sum(len(exp["seeds"]) for exp in EXPERIMENTS)
    
    # Cost estimate
    estimated_cost = estimate_cost(args.instance_type, total_experiments)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT PLAN")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Instance type: {args.instance_type}")
    print(f"Estimated cost: ${estimated_cost:.2f} (assuming ~15h per experiment)")
    print(f"{'='*60}")
    
    if estimated_cost > args.budget:
        print(f"\nWARNING: Estimated cost (${estimated_cost:.2f}) exceeds budget (${args.budget:.2f})")
        if not args.force:
            print("Use --force to proceed anyway, or reduce experiments")
            return []
    
    if args.dry_run:
        print("\nDRY RUN MODE - No jobs will be submitted")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    for exp_config in EXPERIMENTS:
        exp_name = exp_config["exp"]
        for seed in exp_config["seeds"]:
            job = launch_single_experiment(
                image_uri=image_uri,
                role=role,
                s3_bucket=s3_bucket,
                exp_name=exp_name,
                seed=seed,
                instance_type=args.instance_type,
                wandb_api_key=args.wandb_api_key,
                training_iterations=args.training_iterations,
                dry_run=args.dry_run
            )
            if job:
                jobs.append(job)
            
            # Small delay between job submissions
            if not args.dry_run:
                time.sleep(5)
    
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker Training Jobs")
    
    # Experiment selection
    parser.add_argument("--exp", type=str, choices=['private', 'collective', 'tragedy_test'],
                       help="Single experiment to run")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--run-all", action="store_true", help="Run all experiment combinations")
    
    # SageMaker configuration
    parser.add_argument("--role", type=str, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", type=str, default="ml.g4dn.xlarge",
                       choices=list(INSTANCE_COSTS.keys()))
    parser.add_argument("--s3-bucket", type=str, help="S3 bucket for outputs")
    parser.add_argument("--region", type=str, default="eu-central-1")
    
    # Training configuration
    parser.add_argument("--training-iterations", type=int, default=10000)
    parser.add_argument("--wandb-api-key", type=str, default=os.environ.get("WANDB_API_KEY"))
    
    # Budget and safety
    parser.add_argument("--budget", type=float, default=1000.0, help="Budget limit in USD")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without launching")
    parser.add_argument("--force", action="store_true", help="Proceed even if over budget")
    
    # Docker
    parser.add_argument("--image-uri", type=str, help="Pre-built Docker image URI")
    parser.add_argument("--build-image", action="store_true", help="Build and push Docker image")
    
    args = parser.parse_args()
    
    # Setup SageMaker session
    session = sagemaker.Session()
    
    # Get role
    role = args.role or get_sagemaker_role()
    if not role:
        return
    
    # Get S3 bucket
    s3_bucket = get_or_create_bucket(session, args.s3_bucket)
    
    # Get or build Docker image
    if args.image_uri:
        image_uri = args.image_uri
    elif args.build_image:
        image_uri = build_and_push_docker_image(DEFAULT_CONFIG["ecr_repository"], args.region)
    else:
        print("\nERROR: Must provide --image-uri or --build-image")
        print("First time setup: python sagemaker/launch_training.py --build-image --dry-run")
        return
    
    print(f"Using Docker image: {image_uri}")
    
    # Launch experiments
    if args.run_all:
        jobs = launch_all_experiments(args, image_uri, role, s3_bucket)
    elif args.exp:
        job = launch_single_experiment(
            image_uri=image_uri,
            role=role,
            s3_bucket=s3_bucket,
            exp_name=args.exp,
            seed=args.seed,
            instance_type=args.instance_type,
            wandb_api_key=args.wandb_api_key,
            training_iterations=args.training_iterations,
            dry_run=args.dry_run
        )
        jobs = [job] if job else []
    else:
        print("ERROR: Must specify --exp or --run-all")
        return
    
    # Save job manifest
    if jobs and not args.dry_run:
        manifest_path = f"sagemaker_scripts/jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(manifest_path, "w") as f:
            json.dump(jobs, f, indent=2)
        print(f"\nJob manifest saved to: {manifest_path}")
    
    print(f"\nLaunched {len(jobs)} training jobs")


if __name__ == "__main__":
    main()
