#!/bin/bash

N_EPISODES = 500000
HIDDEN_SIZE=(128 256 512)
HIDDEN_LAYERS=(1 2 3 4)
TAU_VALUES=(0.0005 0.001)
LR_VALUES=(0.00005 0.0001)

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
        sbatch -n 1 --mem-per-cpu=4096 --time=11:00:00 --output="${OUTPUT_FILE}" --wrap="python src/train.py \
        --n_episodes=${N_EPISODES} \
        --hidden_size=${HIDDEN_SIZE} \
        --hidden_layers=${HIDDEN_LAYERS} \
        --gamma=0.99 \
        --tau=${tau} \
        --lr=${lr} \
        --log_dir=${LOG_DIR}"

        # Increment the job counter
        ((job_count++))
    done

    # Break out of the outer loop if we've reached max number of jobs
    if (( job_count >= 48 )); then
        break
    fi
done

JOB_INFO="job_info.txt"

# Write information to the output file
{
    echo "N_EPISODES = $N_EPISODES"
    echo "HIDDEN_SIZE = ${HIDDEN_SIZE[*]}"
    echo "HIDDEN_LAYERS = ${HIDDEN_LAYERS[*]}"
    echo "TAU_VALUES = ${TAU_VALUES[*]}"
    echo "LR_VALUES = ${LR_VALUES[*]}"
} > "$JOB_INFO"

echo "Submitted $job_count jobs."