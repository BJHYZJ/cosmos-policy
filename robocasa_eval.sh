#!/bin/bash
source ./env_config.sh
# *** Main checkpoint: 67.1% avg success rate ***
#   Replace `task_suite_name` with one of {PnPCounterToCab, PnPCabToCounter, PnPCounterToSink, \
#   PnPSinkToCounter, PnPCounterToMicrowave, PnPMicrowaveToCounter, PnPCounterToStove, PnPStoveToCounter, \
#   OpenSingleDoor, CloseSingleDoor, OpenDoubleDoor, CloseDoubleDoor, OpenDrawer, CloseDrawer, TurnOnStove, \
#   TurnOffStove, TurnOnSinkFaucet, TurnOffSinkFaucet, TurnSinkSpout, CoffeeSetupMug, CoffeeServeMug, CoffeePressButton, \
#   TurnOnMicrowave, TurnOffMicrowave}
#   Replace `seed` with one of {195, 196, 197}
#   Replace `run_id_note` with a unique identifier for the run

CUDA_VISIBLE_DEVICES=0 uv run --extra cu128 --group robocasa --python 3.10 \
  python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --num_wrist_images 1 \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 32 \
    --num_open_loop_steps 16 \
    --task_name TurnOffMicrowave \
    --num_trials_per_task 50 \
    --run_id_note chkpt45000--5stepAct--seed195--deterministic \
    --local_log_dir ./logs/robocasa/test/ \
    --seed 195 \
    --randomize_seed False \
    --deterministic True \
    --use_variance_scale False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --data_collection False \
    --save_videos True