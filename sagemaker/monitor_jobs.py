#!/usr/bin/env python3
"""
Monitor SageMaker Training Jobs and Download Results

Usage:
    # Monitor all running jobs
    python sagemaker/monitor_jobs.py --status
    
    # Download results from completed jobs
    python sagemaker/monitor_jobs.py --download --output-dir ./results_sagemaker
    
    # Stop all running jobs (emergency)
    python sagemaker/monitor_jobs.py --stop-all
"""

import argparse
import boto3
import json
import os
from datetime import datetime, timedelta
from tabulate import tabulate


def get_training_jobs(sagemaker_client, name_contains="meltingpot", max_results=50):
    """Get list of training jobs"""
    response = sagemaker_client.list_training_jobs(
        NameContains=name_contains,
        MaxResults=max_results,
        SortBy='CreationTime',
        SortOrder='Descending'
    )
    return response.get('TrainingJobSummaries', [])


def get_job_details(sagemaker_client, job_name):
    """Get detailed info about a training job"""
    response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    return response


def format_duration(seconds):
    """Format duration in human readable format"""
    if seconds is None:
        return "N/A"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours}h {minutes}m"


def calculate_cost(instance_type, duration_seconds):
    """Calculate estimated cost"""
    hourly_rates = {
        "ml.g4dn.xlarge": 0.526,
        "ml.g4dn.2xlarge": 0.752,
        "ml.g4dn.4xlarge": 1.204,
        "ml.p3.2xlarge": 3.06,
        "ml.g5.xlarge": 1.006,
    }
    rate = hourly_rates.get(instance_type, 0.5)
    hours = duration_seconds / 3600 if duration_seconds else 0
    return rate * hours


def show_status(sagemaker_client, name_contains="meltingpot"):
    """Show status of all training jobs"""
    jobs = get_training_jobs(sagemaker_client, name_contains)
    
    if not jobs:
        print("No training jobs found")
        return
    
    table_data = []
    total_cost = 0
    
    for job_summary in jobs:
        job_name = job_summary['TrainingJobName']
        status = job_summary['TrainingJobStatus']
        
        # Get detailed info for cost calculation
        try:
            details = get_job_details(sagemaker_client, job_name)
            instance_type = details.get('ResourceConfig', {}).get('InstanceType', 'unknown')
            
            # Calculate duration
            start_time = details.get('TrainingStartTime')
            end_time = details.get('TrainingEndTime')
            
            if start_time and end_time:
                duration = (end_time - start_time).total_seconds()
            elif start_time:
                duration = (datetime.now(start_time.tzinfo) - start_time).total_seconds()
            else:
                duration = None
            
            cost = calculate_cost(instance_type, duration)
            total_cost += cost
            
            # Extract experiment info from job name
            parts = job_name.split('-')
            exp_name = parts[1] if len(parts) > 1 else "unknown"
            
            table_data.append([
                job_name[:40] + "..." if len(job_name) > 40 else job_name,
                exp_name,
                status,
                instance_type,
                format_duration(duration),
                f"${cost:.2f}"
            ])
        except Exception as e:
            table_data.append([job_name[:40], "error", str(e)[:20], "", "", ""])
    
    headers = ["Job Name", "Experiment", "Status", "Instance", "Duration", "Cost"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nTotal estimated cost: ${total_cost:.2f}")
    
    # Summary by status
    status_counts = {}
    for job in jobs:
        status = job['TrainingJobStatus']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nStatus Summary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")


def download_results(sagemaker_client, s3_client, output_dir, name_contains="meltingpot"):
    """Download results from completed training jobs"""
    jobs = get_training_jobs(sagemaker_client, name_contains)
    
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = 0
    for job_summary in jobs:
        if job_summary['TrainingJobStatus'] != 'Completed':
            continue
        
        job_name = job_summary['TrainingJobName']
        details = get_job_details(sagemaker_client, job_name)
        
        # Get S3 output path
        output_path = details.get('ModelArtifacts', {}).get('S3ModelArtifacts')
        if not output_path:
            print(f"No output found for {job_name}")
            continue
        
        # Parse S3 path
        # s3://bucket/path/model.tar.gz
        parts = output_path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        # Download
        local_path = os.path.join(output_dir, job_name)
        os.makedirs(local_path, exist_ok=True)
        
        local_file = os.path.join(local_path, "model.tar.gz")
        
        print(f"Downloading {job_name}...")
        try:
            s3_client.download_file(bucket, key, local_file)
            downloaded += 1
            
            # Extract tar.gz
            import tarfile
            with tarfile.open(local_file, "r:gz") as tar:
                tar.extractall(local_path)
            
            print(f"  Extracted to {local_path}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nDownloaded {downloaded} completed jobs to {output_dir}")


def stop_all_jobs(sagemaker_client, name_contains="meltingpot"):
    """Stop all running training jobs"""
    jobs = get_training_jobs(sagemaker_client, name_contains)
    
    stopped = 0
    for job_summary in jobs:
        if job_summary['TrainingJobStatus'] == 'InProgress':
            job_name = job_summary['TrainingJobName']
            print(f"Stopping {job_name}...")
            try:
                sagemaker_client.stop_training_job(TrainingJobName=job_name)
                stopped += 1
            except Exception as e:
                print(f"  Error: {e}")
    
    print(f"\nStopped {stopped} jobs")


def main():
    parser = argparse.ArgumentParser(description="Monitor SageMaker Training Jobs")
    
    parser.add_argument("--status", action="store_true", help="Show status of all jobs")
    parser.add_argument("--download", action="store_true", help="Download completed results")
    parser.add_argument("--stop-all", action="store_true", help="Stop all running jobs")
    parser.add_argument("--output-dir", type=str, default="./results_sagemaker")
    parser.add_argument("--name-contains", type=str, default="meltingpot")
    parser.add_argument("--region", type=str, default="us-east-1")
    
    args = parser.parse_args()
    
    # Initialize clients
    sagemaker_client = boto3.client('sagemaker', region_name=args.region)
    s3_client = boto3.client('s3', region_name=args.region)
    
    if args.status:
        show_status(sagemaker_client, args.name_contains)
    elif args.download:
        download_results(sagemaker_client, s3_client, args.output_dir, args.name_contains)
    elif args.stop_all:
        confirm = input("Are you sure you want to stop ALL running jobs? (yes/no): ")
        if confirm.lower() == "yes":
            stop_all_jobs(sagemaker_client, args.name_contains)
        else:
            print("Cancelled")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
