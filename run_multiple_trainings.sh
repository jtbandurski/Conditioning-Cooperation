#!/bin/bash

# Define an array-like structure with pairs of parameters
declare -a params_array=("123" "1234" "12345" "123456" "1234567")

# Loop over the array
for param in "${params_array[@]}"
do

   # Run the Python script with the seed parameter
   screen -L -Logfile screenlogs/train_collective_15cpus_5000_iter_${param}_seed.txt python baselines/train/run_ray_train.py \
   --num_workers=15 --num_gpus=0 --no-tune --algo=ppo --framework=torch --exp=collective\ 
   --seed=$param --results_dir=./results_train_aws --logging=INFO --wandb=True --downsample=True

   # Wait for 60 seconds
   sleep 60
done