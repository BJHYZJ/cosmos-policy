import os
import re
import json

def process_class(path):
    with open(path, "r", encoding="utf-8") as f:
        task_classification = json.load(f)
    
    for key, value in task_classification

def parse_libero_logs(root_dir="."):
    # Regex for filename range: e.g., range_[0, 300]
    range_pattern = re.compile(r"range_\[(-?\d+),\s*(-?\d+)\]")
    
    # Regex for metrics in content
    episodes_pattern = re.compile(r"Total episodes: (\d+)")
    successes_pattern = re.compile(r"Total successes: (\d+)")
    
    # Regex for latency: matches "Action query time = 1.871 sec"
    latency_pattern = re.compile(r"Action query time = ([\d\.]+) sec")

    # Get subdirectories, excluding 'test'
    subdirs = [d for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d)) and d != "test"]

    for subdir in sorted(subdirs):
        dir_path = os.path.join(root_dir, subdir)
        files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
        
        extracted_data = []
        folder_total_eps = 0
        folder_total_sucs = 0
        folder_all_latencies = []
        
        for filename in files:
            range_match = range_pattern.search(filename)
            if not range_match: continue
                
            range_label = range_match.group(0).replace("range_", "")
            start_idx = int(range_match.group(1))
            
            file_path = os.path.join(dir_path, filename)
            
            # Local metrics for the current file
            file_eps = 0
            file_sucs = 0
            file_latencies = []

            # Read line by line to handle large logs efficiently
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Search for latency in every line
                    lat_match = latency_pattern.search(line)
                    if lat_match:
                        file_latencies.append(float(lat_match.group(1)))
                    
                    # Search for final results (usually at the end)
                    ep_match = episodes_pattern.search(line)
                    if ep_match: file_eps = int(ep_match.group(1))
                    
                    suc_match = successes_pattern.search(line)
                    if suc_match: file_sucs = int(suc_match.group(1))

            if file_eps > 0:
                success_rate = (file_sucs / file_eps * 100)
                avg_lat = (sum(file_latencies) / len(file_latencies)) if file_latencies else 0
                
                extracted_data.append({
                    "start": start_idx,
                    # "output": f"{range_label}: success rate: {success_rate:.1f}%, avg latency: {avg_lat:.3f}s",
                    "output": f"{range_label}: success rate: {success_rate:.1f}%",
                    "eps": file_eps,
                    "sucs": file_sucs,
                    "latencies": file_latencies
                })
                
                folder_total_eps += file_eps
                folder_total_sucs += file_sucs
                folder_all_latencies.extend(file_latencies)

        # Sort by range start index
        extracted_data.sort(key=lambda x: x["start"])

        # Final Folder Output
        if extracted_data:
            print(f"### {subdir} ###")
            for data in extracted_data:
                print(data["output"])
            
            overall_rate = (folder_total_sucs / folder_total_eps * 100) if folder_total_eps > 0 else 0
            overall_lat = (sum(folder_all_latencies) / len(folder_all_latencies)) if folder_all_latencies else 0
            
            print(f"\nSummary for {subdir}:")
            print(f"  > Total Success Rate: {overall_rate:.1f}%")
            print(f"  > Average Action Query Time: {overall_lat:.4f} sec")
            print("-" * 60 + "\n")

if __name__ == "__main__":
    # Ensure the script runs in the directory where your log folders are located
    root_dir = "./logs/libero_plus"
    task_classification_path = "LIBERO-plus/libero/libero/benchmark/task_classification.json"
    process_class(path=task_classification_path)
    parse_libero_logs(root_dir=root_dir)