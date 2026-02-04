import os
import re

def parse_libero_tasks(root_dir="."):
    """
    Parses robot logs to extract Task descriptions and their corresponding Success status.
    """
    # Regex to capture the content after 'Task:' and 'Success:'
    task_pattern = re.compile(r"Task:\s*(.*)")
    success_pattern = re.compile(r"Success:\s*(.*)")

    # Identify subdirectories, excluding the 'test' folder
    subdirs = [d for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d)) and d != "test"]

    for subdir in sorted(subdirs):
        dir_path = os.path.join(root_dir, subdir)
        # Filter for log files ending in .txt
        files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
        
        folder_pairs = []
        
        for filename in files:
            file_path = os.path.join(dir_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Find all occurrences in the current file
                tasks = task_pattern.findall(content)
                results = success_pattern.findall(content)
                
                # Ensure 1:1 mapping before adding to the list
                if len(tasks) == len(results):
                    folder_pairs.extend(zip(tasks, results))
                else:
                    continue

        # Print the results for the current folder
        if folder_pairs:
            print(f"### Folder: {subdir} ###")
            print(f"{'No.':<4} | {'Status':<8} | {'Task Description'}")
            print("-" * 70)
            for i, (task, success) in enumerate(folder_pairs, 1):
                # Normalize the status for display
                print(f"{i:<4} | {success:<8} | {task.strip()}")
            print("-" * 70 + "\n")

if __name__ == "__main__":
    # Set the path to your log directory
    target_directory = "./logs/libero_plus"
    parse_libero_tasks(root_dir=target_directory)