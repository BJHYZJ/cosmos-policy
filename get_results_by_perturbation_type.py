import os
import re
import json
# from libero.libero import benchmark

# benchmark_dict = benchmark.get_benchmark_dict()

# def process_class(path):
#     with open(path, "r", encoding="utf-8") as f:
#         task_classification = json.load(f)
    
#     # task_id_to_language = {}
#     # task_id_to_category = {}
#     language_to_category = {
#         "libero_spatial": {},
#         "libero_object": {},
#         "libero_goal": {},
#         "libero_10": {},
#     }
#     for key, value in task_classification.items():
#         task_suite = benchmark_dict[key]()
#         num_tasks = task_suite.n_tasks
#         print(num_tasks)
        # task_id_to_language[key] = {task_id: task_suite.get_task(task_id).language for task_id in range(num_tasks)}
        # task_id_to_category[key] = {task_id: value[task_id]['category'] for task_id in range(num_tasks)}
        # for i in range(num_tasks):
        #     assert value[i]['id'] == i + 1
        #     language = task_suite.get_task(i).language
        #     if language in language_to_category[key].keys():
        #         assert 1 == 1
        #     language_to_category[key][language] = value[i]['category']
            # language_to_category[key] = {task_suite.get_task(task_id).language: value[task_id]['category'] for task_id in range(num_tasks)}

    # for key, value in task_classification.items():

    #     for info in value:
    #         raw_name = info['name']
            
    #         # Use regex to find everything after SCENEx_
    #         # \d+ handles cases like SCENE4 or SCENE10
    #         match = re.search(r"SCENE\d+_(.*)", raw_name)
            
    #         if match:
    #             # Take only the part after SCENEx_
    #             extracted_name = match.group(1)
    #         else:
    #             # Fallback to full name if pattern is not found
    #             extracted_name = raw_name
            
    #         # Now clean it up: lower case and replace underscores with spaces
    #         task_description = extracted_name.lower().replace("_", " ")
    #         description_to_category[task_description] = info['category']
            
    # return language_to_category

def parse_libero_tasks(root_dir, task_classification_path):
    """
    Parses robot logs to extract Task descriptions and their corresponding Success status.
    """
    with open(task_classification_path, "r", encoding="utf-8") as f:
        task_classification = json.load(f)

    # Regex to capture the content after 'Task:' and 'Success:'
    task_pattern = re.compile(r"Task:\s*(.*)")
    success_pattern = re.compile(r"Success:\s*(.*)")

    # Identify subdirectories, excluding the 'test' folder
    subdirs = [d for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d)) and d != "test"]

    category_to_task_success = {}
    for subdir in sorted(subdirs):
        sub_task_classification = task_classification[subdir.split("_seed")[0]]
        categories = [item["category"] for item in sub_task_classification]
        dir_path = os.path.join(root_dir, subdir)
        # Filter for log files ending in .txt
        files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
        files = sorted(files, key=lambda x: int(x.split("[")[-1].split(",")[0]))
        
        results_all = []

        for filename in files:
            file_path = os.path.join(dir_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Find all occurrences in the current file
                # tasks = task_pattern.findall(content)
                results = success_pattern.findall(content)
                results_all.extend(results)
        
        if len(results_all) != len(sub_task_classification):
            print(f"Fail in process: {subdir}")
            continue
                
        for idx in range(len(categories)):
            if categories[idx] not in category_to_task_success.keys():
                category_to_task_success[categories[idx]] = {
                    "success": 0,
                    "total": 0
                }
            category_to_task_success[categories[idx]]['total'] += 1
            category_to_task_success[categories[idx]]['success'] += int(eval(results_all[idx]))
    
    total_success = 0
    total_length = 0
    for key, value in category_to_task_success.items():
        category_to_task_success[key]['success_rate'] = value['success'] / value['total']
        total_success += value['success']
        total_length += value['total']
    
    print("=" * 60)
    print("=" * 60)
    print("Total length:", total_length)
    print("Total success:", total_success / total_length)
    print(category_to_task_success)

    
    
        

if __name__ == "__main__":
    # Ensure the script runs in the directory where your log folders are located
    root_dir = "./logs/libero_plus"
    task_classification_path = "LIBERO-plus/libero/libero/benchmark/task_classification.json"
    # description_to_category = process_class(path=task_classification_path)
    parse_libero_tasks(root_dir=root_dir, task_classification_path=task_classification_path)


