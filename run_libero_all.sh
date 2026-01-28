#!/bin/bash

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# Define the arrays for task suites and seeds
task_suites=("libero_spatial" "libero_object" "libero_goal" "libero_10")
seeds=(195 196 197)

# Loop through each task suite
for suite in "${task_suites[@]}"; do
    # Loop through each seed
    for seed in "${seeds[@]}"; do
        
        echo "================================================================"
        echo "STARTING: Task Suite = $suite | Seed = $seed"
        echo "================================================================"

        # Execute the evaluation command
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
            --task_suite_name "$suite" \
            --local_log_dir "logs/libero/${suite}_seed${seed}" \
            --randomize_seed False \
            --data_collection False \
            --use_parallel_inference True \
            --available_gpus "0,1,2,3,4,5,6,7" \
            --num_queries_best_of_n 8 \
            --seed "$seed" \
            --use_variance_scale False \
            --deterministic True \
            --run_id_note "chkpt45000--5stepAct--seed${seed}--deterministic--${suite}" \
            --ar_future_prediction False \
            --ar_value_prediction False \
            --use_jpeg_compression True \
            --flip_images True \
            --num_denoising_steps_action 5 \
            --num_denoising_steps_future_state 1 \
            --num_denoising_steps_value 1

        echo "FINISHED: Task Suite = $suite | Seed = $seed"
        echo "================================================================"
        echo ""
    done
done

echo "All evaluation tasks have been completed successfully."