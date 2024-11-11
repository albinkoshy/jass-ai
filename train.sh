#!/bin/bash

TAU_VALUES=(0.0001 0.0005 0.001 0.005)
LR_VALUES=(0.00001 0.00005 0.0001 0.0005 0.001 0.005)

# Counter to track the number of jobs
job_count=0

# Loop over TAU and LR values
for tau in "${TAU_VALUES[@]}"; do
    for lr in "${LR_VALUES[@]}"; do
        if (( job_count >= 48 )); then
            break  # Stop if we've reached the max number of jobs
        fi

        # Define the log directory for this run
        LOG_DIR="./logs/run_tau_${tau}_lr_${lr}"
        OUTPUT_FILE="output_logs/dqn_run_tau_${tau}_lr_${lr}.out"
        # Submit the job with sbatch
        sbatch -n 1 --mem-per-cpu=4096 --time=11:00:00 --output="${OUTPUT_FILE}" --wrap="python src/train.py --n_episodes=1000000 --tau=${tau} --lr=${lr} --log_dir=${LOG_DIR}"

        # Increment the job counter
        ((job_count++))
    done

    # Break out of the outer loop if we've reached max number of jobs
    if (( job_count >= 48 )); then
        break
    fi
done

echo "Submitted $job_count jobs."
