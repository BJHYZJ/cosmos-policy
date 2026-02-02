import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import torch
import tqdm
import pickle
import shutil
from pathlib import Path
from filelock import FileLock, Timeout as FileLockTimeout
from libero.libero import benchmark
from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder

def generate_and_save_official_cache(suite_name, output_dir, encoder, batch_size=32):
    """
    Generate and save embedding cache with batch processing for faster inference.
    """
    t5_text_embeddings_path = os.path.join(output_dir, f"{suite_name}.pkl")
    lock_path = t5_text_embeddings_path + ".lock"
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    num_tasks = task_suite.n_tasks
    
    # 1. Collect all instructions for the suite
    instructions = [task_suite.get_task(i).language for i in range(num_tasks)]
    t5_text_embeddings_cache = {}

    print(f"\n>>> Processing suite: {suite_name} ({num_tasks} tasks) with batch_size {batch_size}")
    
    # 2. Batch inference
    for i in tqdm.tqdm(range(0, num_tasks, batch_size), desc=f"Batching {suite_name}"):
        batch_instructions = instructions[i : i + batch_size]
        
        with torch.no_grad():
            # encoder.encode_prompts can handle a list of strings
            # Returns shape: (batch_size, seq_len, embed_dim)
            batch_embs = encoder.encode_prompts(
                batch_instructions,
                max_length=512,
                return_mask=False,
            )
        
        # Unpack batch and store in dictionary
        for j, instr in enumerate(batch_instructions):
            # Each embedding is stored as (1, seq_len, embed_dim) to match official format
            t5_text_embeddings_cache[instr] = batch_embs[j:j+1].cpu()

    # 3. Official Save Logic
    lock = FileLock(lock_path, timeout=30)
    try:
        with lock:
            if os.path.exists(t5_text_embeddings_path):
                shutil.copy2(t5_text_embeddings_path, t5_text_embeddings_path + ".backup")
            
            with open(t5_text_embeddings_path, "wb") as f:
                pickle.dump(t5_text_embeddings_cache, f)
            print(f"Successfully saved {suite_name} cache.")
    except Exception as e:
        print(f"Error saving {suite_name}: {e}")

def main():
    suites = ['libero_object', 'libero_goal', 'libero_10', 'libero_90']
    output_dir = "./logs/libero_plus_t5_embedding_cache"
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = CosmosT5TextEncoder(device=device, cache_dir="./checkpoints/hf_cache")
    encoder.text_encoder.to(device)

    for suite_name in suites:
        # You can adjust batch_size based on your GPU VRAM
        generate_and_save_official_cache(suite_name, output_dir, encoder, batch_size=16)

if __name__ == "__main__":
    main()