#!/bin/bash

source ./env_config.sh

################################################################################
# 1. Global Configurations
################################################################################
CKPT_PATH="nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
STATS_PATH="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"
EMB_PATH="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl"

# Hyperparameters
CHUNK_SIZE=16
OPEN_LOOP_STEPS=16
STEPS_ACT=5
STEPS_FUTURE=1
STEPS_VALUE=1
DETERMINISTIC="True"
SAVE_VIDEO="False"

################################################################################
# 2. Execution Function
################################################################################
run_task() {
    local gpu=$1
    local task=$2
    local seed=$3
    local note="eval_${task}_s${seed}_step${STEPS_ACT}_DETERMINISTIC-${DETERMINISTIC}"
    local local_dir="logs/libero/${task}_${seed}"
    
    mkdir -p "$local_dir"

    echo "[GPU $gpu] Launching: $task | Seed: $seed | ID: $note"

    CUDA_VISIBLE_DEVICES=$gpu uv run --extra cu128 --group libero --python 3.10 \
        python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
            --config cosmos_predict2_2b_480p_libero__inference_only \
            --ckpt_path "$CKPT_PATH" \
            --config_file cosmos_policy/config/config.py \
            --use_wrist_image True \
            --use_proprio True \
            --normalize_proprio True \
            --unnormalize_actions True \
            --dataset_stats_path "$STATS_PATH" \
            --t5_text_embeddings_path "$EMB_PATH" \
            --trained_with_image_aug True \
            --chunk_size "$CHUNK_SIZE" \
            --num_open_loop_steps "$OPEN_LOOP_STEPS" \
            --task_suite_name "$task" \
            --local_log_dir "$local_dir" \
            --randomize_seed False \
            --data_collection False \
            --available_gpus "0,1,2,3,4,5,6,7" \
            --seed "$seed" \
            --use_variance_scale False \
            --deterministic "$DETERMINISTIC" \
            --run_id_note "$note" \
            --ar_future_prediction False \
            --ar_value_prediction False \
            --use_jpeg_compression True \
            --flip_images True \
            --num_denoising_steps_action "$STEPS_ACT" \
            --num_denoising_steps_future_state "$STEPS_FUTURE" \
            --num_denoising_steps_value "$STEPS_VALUE" \
            --save_videos "$SAVE_VIDEO"
}

################################################################################
# 3. Task Distribution
################################################################################

( run_task 0 libero_spatial 195; run_task 0 libero_spatial 196 ) &
( run_task 1 libero_spatial 197; run_task 1 libero_object 195 ) &
( run_task 2 libero_object  196; run_task 2 libero_object  197 ) &
( run_task 3 libero_goal    195; run_task 3 libero_goal    196 ) &
run_task 4 libero_goal 197 &
run_task 5 libero_10    195 &
run_task 6 libero_10    196 &
run_task 7 libero_10    197 &

wait
echo "All evaluation tasks are finished."