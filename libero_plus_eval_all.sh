#!/bin/bash

source ./env_config.sh

################################################################################
# 1. Global Configurations
################################################################################

CKPT_PATH="nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
STATS_PATH="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"

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
    local range_start=$4
    local range_end=$5
    
    local emb_path="./logs/libero_plus_t5_embedding_cache/${task}.pkl"
    local note="libero_plus_${task}_s${seed}_step${STEPS_ACT}_det${DETERMINISTIC}"
    local local_dir="./logs/libero_plus_test/${task}_seed${seed}"
    
    mkdir -p "$local_dir"

    echo "[GPU $gpu] Launching: $task | Seed: $seed | ID: $note | Range: [$range_start,$range_end]"

    CUDA_VISIBLE_DEVICES=$gpu uv run --extra cu128 --group libero_plus --python 3.10 \
        python -m cosmos_policy.experiments.robot.libero_plus.run_libero_plus_eval \
            --config cosmos_predict2_2b_480p_libero__inference_only \
            --ckpt_path "$CKPT_PATH" \
            --config_file cosmos_policy/config/config.py \
            --use_wrist_image True \
            --use_proprio True \
            --normalize_proprio True \
            --unnormalize_actions True \
            --dataset_stats_path "$STATS_PATH" \
            --t5_text_embeddings_path "$emb_path" \
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
            --save_videos "$SAVE_VIDEO" \
            --num_trials_per_task 1 \
            --task_id_range "[$range_start, $range_end]"
}

################################################################################
# 3. Task Distribution (Parallel Execution)
################################################################################


( run_task 0 libero_spatial 195 0 300;    run_task 0 libero_object 195 0 300;    run_task 0 libero_goal 195 0 300;    run_task 0 libero_10 195 0 300;    ) &
( run_task 1 libero_spatial 195 300 600;  run_task 1 libero_object 195 300 600;  run_task 1 libero_goal 195 300 600;  run_task 1 libero_10 195 300 600;  ) &
( run_task 2 libero_spatial 195 600 900;  run_task 2 libero_object 195 600 900;  run_task 2 libero_goal 195 600 900;  run_task 2 libero_10 195 600 900;  ) &
( run_task 3 libero_spatial 195 900 1200; run_task 3 libero_object 195 900 1200; run_task 3 libero_goal 195 900 1200; run_task 3 libero_10 195 900 1200; ) &
( run_task 4 libero_spatial 195 1200 1500; run_task 4 libero_object 195 1200 1500; run_task 4 libero_goal 195 1200 1500; run_task 4 libero_10 195 1200 1500; ) &
( run_task 5 libero_spatial 195 1500 1800; run_task 5 libero_object 195 1500 1800; run_task 5 libero_goal 195 1500 1800; run_task 5 libero_10 195 1500 1800; ) &
( run_task 6 libero_spatial 195 1800 2100; run_task 6 libero_object 195 1800 2100; run_task 6 libero_goal 195 1800 2100; run_task 6 libero_10 195 1800 2100; ) &
( run_task 7 libero_spatial 195 2100 -1;   run_task 7 libero_object 195 2100 -1;   run_task 7 libero_goal 195 2100 -1;   run_task 7 libero_10 195 2100 -1;   ) &


# ( run_task 0 libero_10 195 0 1; ) &
# ( run_task 1 libero_10 195 100 101; ) &
# ( run_task 2 libero_goal 195 0 1; ) &
# ( run_task 3 libero_goal 195 100 101; ) &
# ( run_task 4 libero_spatial 195 0 1; ) &
# ( run_task 5 libero_spatial 195 100 101; ) &
# ( run_task 6 libero_object 195 0 1; ) &
# ( run_task 7 libero_object 195 100 101; ) &



# ( run_task 2 libero_10 195 0 1200; ) &
# ( run_task 3 libero_10 195 1200 -1; ) &
# ( run_task 2 libero_goal 195 0 1200; ) &
# ( run_task 3 libero_goal 195 1200 -1; ) &
# ( run_task 4 libero_spatial 195 0 1200; ) & 
# ( run_task 5 libero_spatial 195 1200 -1; ) &
# ( run_task 6 libero_object 195 0 1200; ) &
# ( run_task 7 libero_object 195 1200 -1; ) &


# ( run_task 0 libero_spatial 195; run_task 0 libero_spatial 196 ) &
# ( run_task 1 libero_spatial 197; run_task 1 libero_object 195 ) &
# ( run_task 2 libero_object  196; run_task 2 libero_object  197 ) &
# ( run_task 3 libero_goal    195; run_task 3 libero_goal    196 ) &
# ( run_task 4 libero_goal    197 ) &
# ( run_task 5 libero_10      195 ) &
# ( run_task 6 libero_10      196 ) &
# ( run_task 7 libero_10      197 ) &

wait
echo "All LIBERO-Plus evaluation tasks have completed."