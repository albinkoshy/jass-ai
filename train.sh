#!/bin/bash

N_EPISODES=500000
HIDDEN_SIZES=("128,128,128" "256,256,256" "512,512,512")
GAMMA=0.99
TAU_VALUES=(0.0005 0.001)
LR_VALUES=(0.00005 0.0001)

# Ensure directories exist
mkdir -p ./logs ./output_logs

job_count=0

for hidden_sizes in "${HIDDEN_SIZES[@]}"; do
    for tau in "${TAU_VALUES[@]}"; do
        for lr in "${LR_VALUES[@]}"; do
            if (( job_count >= 48 )); then
                break 3
            fi

            log_dir="./logs/run_${N_EPISODES}_${hidden_sizes}_${GAMMA}_${tau}_${lr}"
            output_file="./output_logs/run_${N_EPISODES}_${hidden_sizes}_${GAMMA}_${tau}_${lr}.out"

            sbatch -n 1 --mem-per-cpu=4096 --time=11:00:00 --output="${output_file}" --wrap="python src/train.py \
            --n_episodes=${N_EPISODES} \
            --hidden_sizes=${hidden_sizes} \
            --gamma=${GAMMA} \
            --tau=${tau} \
            --lr=${lr} \
            --log_dir=${log_dir}"

            ((job_count++))
        done
    done
done

JOB_INFO="./job_info.txt"

# Write information to the output file
{
    echo "N_EPISODES = ${N_EPISODES[*]}"
    echo "HIDDEN_SIZES = ${HIDDEN_SIZES[*]}"
    echo "GAMMA = ${GAMMA[*]}"
    echo "TAU_VALUES = ${TAU_VALUES[*]}"
    echo "LR_VALUES = ${LR_VALUES[*]}"
} > "${JOB_INFO}"

echo "Submitted $job_count jobs."