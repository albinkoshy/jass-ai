#!/bin/bash

N_EPISODES=500000
AGENT=("dqn", "double_dqn")
HIDDEN_SIZES=("128,64" "128,128" "256,256")
ACTIVATION=("relu" "tanh" "sigmoid")
BATCH_SIZE=(512)
GAMMA=(1)
TAU=(0.005)
LR=(0.00005 0.0001)
BUFFER_SIZE=(50000)
LOSS=("smooth_l1" "mse")

# Ensure directories exist
mkdir -p ./logs ./output_logs

job_count=0

for agent in "${AGENT[@]}"; do
    for hidden_sizes in "${HIDDEN_SIZES[@]}"; do
        for activation in "${ACTIVATION[@]}"; do
            for batch_size in "${BATCH_SIZE[@]}"; do
                for gamma in "${GAMMA[@]}"; do
                    for tau in "${TAU[@]}"; do
                        for lr in "${LR[@]}"; do
                            for buffer_size in "${BUFFER_SIZE[@]}"; do
                                for loss in "${LOSS[@]}"; do
                                    if (( job_count >= 48 )); then
                                        break 9
                                    fi

                                    hidden_sizes_hyphen=$(echo "${hidden_sizes}" | tr ',' '-')
                                    run_string="run_${N_EPISODES}_${hidden_sizes_hyphen}_${activation}_${batch_size}_${gamma}_${tau}_${lr}_${buffer_size}_${loss}"
                                    log_dir="./logs/${run_string}"
                                    output_file="./output_logs/${run_string}.out"

                                    sbatch -n 1 --mem-per-cpu=2048 --time=11:00:00 --output="${output_file}" --wrap="python src/train.py \
                                    --agent=${agent} \
                                    --n_episodes=${N_EPISODES} \
                                    --hide_opponents_hands \
                                    --hidden_sizes=${hidden_sizes} \
                                    --activation=${activation} \
                                    --batch_size=${batch_size} \
                                    --gamma=${gamma} \
                                    --tau=${tau} \
                                    --lr=${lr} \
                                    --replay_buffer_size=${buffer_size} \
                                    --loss=${loss} \
                                    --log_dir=${log_dir}"

                                    ((job_count++))
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

JOB_INFO="./job_info.txt"

# Write information to the output file
{
    echo "N_EPISODES = ${N_EPISODES[*]}"
    echo "AGENT = ${AGENT[*]}"
    echo "HIDE_OPPONENTS_HANDS = True"
    echo "HIDDEN_SIZES = ${HIDDEN_SIZES[*]}"
    echo "ACTIVATION = ${ACTIVATION[*]}"
    echo "BATCH_SIZE = ${BATCH_SIZE[*]}"
    echo "GAMMA = ${GAMMA[*]}"
    echo "TAU = ${TAU[*]}"
    echo "LR = ${LR[*]}"
    echo "BUFFER_SIZE = ${BUFFER_SIZE[*]}"
    echo "LOSS = ${LOSS[*]}"
} > "${JOB_INFO}"

echo "Submitted $job_count jobs."