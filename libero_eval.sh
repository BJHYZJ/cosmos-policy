#!/bin/bash
source ./env_config.sh

# *** Main checkpoint: 98.5% success rate ***
#   Replace `task_suite_name` with one of {libero_spatial, libero_object, libero_goal, libero_10}
#   Replace `seed` with one of {195, 196, 197}
#   Replace `run_id_note` with a unique identifier for the run

# export EGL_DEVICE_ID=0  
# export MUJOCO_GL=osmesa  
# export PYOPENGL_PLATFORM=osmesa  
CUDA_VISIBLE_DEVICES=0 uv run --extra cu128 --group libero --python 3.10 \
    python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name libero_10 \
        --local_log_dir ./logs/libero/test \
        --randomize_seed False \
        --data_collection False \
        --available_gpus "0,1,2,3,4,5,6,7" \
        --seed 195 \
        --use_variance_scale False \
        --deterministic True \
        --run_id_note chkpt45000--5stepAct--seed195--deterministic \
        --ar_future_prediction False \
        --ar_value_prediction False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --save_videos False \
        --task_id_range "[2400, -1]"