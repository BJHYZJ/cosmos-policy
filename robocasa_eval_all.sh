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

################################################################################
# 1. Global Configurations
################################################################################
CKPT_PATH="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
STATS_PATH="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json"
EMB_PATH="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl"

CHUNK_SIZE=32
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
    local local_dir="logs/robocasa/${task}_${seed}"
    
    mkdir -p "$local_dir"

    echo "[GPU $gpu] Launching: $task | Seed: $seed | ID: $note"

    CUDA_VISIBLE_DEVICES=$gpu uv run --extra cu128 --group robocasa --python 3.10 \
        python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
            --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
            --ckpt_path "$CKPT_PATH" \
            --config_file cosmos_policy/config/config.py \
            --use_wrist_image True \
            --num_wrist_images 1 \
            --use_proprio True \
            --normalize_proprio True \
            --unnormalize_actions True \
            --dataset_stats_path "$STATS_PATH" \
            --t5_text_embeddings_path "$EMB_PATH" \
            --trained_with_image_aug True \
            --chunk_size "$CHUNK_SIZE" \
            --num_open_loop_steps "$OPEN_LOOP_STEPS" \
            --task_name "$task" \
            --num_trials_per_task 50 \
            --run_id_note "$note" \
            --local_log_dir "$local_dir" \
            --seed "$seed" \
            --randomize_seed False \
            --deterministic "$DETERMINISTIC" \
            --use_variance_scale False \
            --use_jpeg_compression True \
            --flip_images True \
            --num_denoising_steps_action "$STEPS_ACT" \
            --num_denoising_steps_future_state "$STEPS_FUTURE" \
            --num_denoising_steps_value "$STEPS_VALUE" \
            --data_collection False \
            --save_videos "$SAVE_VIDEO"
}


################################################################################
# 3. Task Distribution
################################################################################

( run_task 0 PnPCounterToCab 195; run_task 0 PnPCounterToCab 196; run_task 0 PnPCounterToCab 197; \
  run_task 0 PnPCabToCounter 195; run_task 0 PnPCabToCounter 196; run_task 0 PnPCabToCounter 197; \
  run_task 0 PnPCounterToSink 195; run_task 0 PnPCounterToSink 196; run_task 0 PnPCounterToSink 197 ) &

( run_task 1 PnPSinkToCounter 195; run_task 1 PnPSinkToCounter 196; run_task 1 PnPSinkToCounter 197; \
  run_task 1 PnPCounterToMicrowave 195; run_task 1 PnPCounterToMicrowave 196; run_task 1 PnPCounterToMicrowave 197; \
  run_task 1 PnPMicrowaveToCounter 195; run_task 1 PnPMicrowaveToCounter 196; run_task 1 PnPMicrowaveToCounter 197 ) &

( run_task 2 PnPCounterToStove 195; run_task 2 PnPCounterToStove 196; run_task 2 PnPCounterToStove 197; \
  run_task 2 PnPStoveToCounter 195; run_task 2 PnPStoveToCounter 196; run_task 2 PnPStoveToCounter 197; \
  run_task 2 OpenSingleDoor 195; run_task 2 OpenSingleDoor 196; run_task 2 OpenSingleDoor 197 ) &

( run_task 3 CloseSingleDoor 195; run_task 3 CloseSingleDoor 196; run_task 3 CloseSingleDoor 197; \
  run_task 3 OpenDoubleDoor 195; run_task 3 OpenDoubleDoor 196; run_task 3 OpenDoubleDoor 197; \
  run_task 3 CloseDoubleDoor 195; run_task 3 CloseDoubleDoor 196; run_task 3 CloseDoubleDoor 197 ) &

( run_task 4 OpenDrawer 195; run_task 4 OpenDrawer 196; run_task 4 OpenDrawer 197; \
  run_task 4 CloseDrawer 195; run_task 4 CloseDrawer 196; run_task 4 CloseDrawer 197; \
  run_task 4 TurnOnStove 195; run_task 4 TurnOnStove 196; run_task 4 TurnOnStove 197 ) &

( run_task 5 TurnOffStove 195; run_task 5 TurnOffStove 196; run_task 5 TurnOffStove 197; \
  run_task 5 TurnOnSinkFaucet 195; run_task 5 TurnOnSinkFaucet 196; run_task 5 TurnOnSinkFaucet 197; \
  run_task 5 TurnOffSinkFaucet 195; run_task 5 TurnOffSinkFaucet 196; run_task 5 TurnOffSinkFaucet 197 ) &

( run_task 6 TurnSinkSpout 195; run_task 6 TurnSinkSpout 196; run_task 6 TurnSinkSpout 197; \
  run_task 6 CoffeeSetupMug 195; run_task 6 CoffeeSetupMug 196; run_task 6 CoffeeSetupMug 197; \
  run_task 6 CoffeeServeMug 195; run_task 6 CoffeeServeMug 196; run_task 6 CoffeeServeMug 197 ) &

( run_task 7 CoffeePressButton 195; run_task 7 CoffeePressButton 196; run_task 7 CoffeePressButton 197; \
  run_task 7 TurnOnMicrowave 195; run_task 7 TurnOnMicrowave 196; run_task 7 TurnOnMicrowave 197; \
  run_task 7 TurnOffMicrowave 195; run_task 7 TurnOffMicrowave 196; run_task 7 TurnOffMicrowave 197 ) &

wait
echo "All evaluation tasks are finished."


# kill
# pkill -9 -f "robocasa_eval_all.sh"
# pkill -9 -f "uv run"
# ps -ef | grep "run_robocasa_eval" | grep -v grep | awk '{print $2}' | xargs -r kill -9
