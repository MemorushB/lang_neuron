import json
import random
import os
from typing import List

def load_json(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Data loaded from the JSON file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    """
    Save data to a JSON file.

    Args:
        data (dict): Data to be saved.
        file_path (str): Path to the JSON file.
    """
    with open(file_path, 'w') as output_file:
        json.dump(data, output_file, indent=4)

def process_dataset(file_path, limit=3000):
    """
    Load, modify, and save the dataset.

    Args:
        file_path (str): Path to the JSON file.
        limit (int): Maximum number of positive sentences to keep.

    Returns:
        str: Path to the modified JSON file.
    """
    data = load_json(file_path)

    if 'positive' in data['sentences']:
        if len(data['sentences']['positive']) > limit:
            data['sentences']['positive'] = random.sample(data['sentences']['positive'], limit)

    save_json(data, file_path)
    return file_path

def update_negative_key(main_file_key, main_file_data, other_files_data):
    """
    Update the 'negative' key with positive examples from other files.

    Args:
        main_file_key (str): Key of the main file.
        main_file_data (dict): Data of the main file.
        other_files_data (dict): Data of other files.

    Returns:
        dict: Updated main file data.
    """
    main_file_negatives = main_file_data["sentences"].get("negative", [])
    for key, data in other_files_data.items():
        if key != main_file_key:
            main_file_negatives.extend(data["sentences"]["positive"])
    
    main_file_data["sentences"]["negative"] = main_file_negatives
    return main_file_data

def json_process(file_paths, limit=3000):
    """
    Main function to process and update datasets.

    Args:
        file_paths (dict): Dictionary of file keys and their paths.

    Returns:
        dict: Dictionary of file keys and their updated paths.
    """
    # Process all files and get their output paths
    modified_files = {key: process_dataset(path, limit=limit) for key, path in file_paths.items()}

    # Load all the modified files
    data_files = {key: load_json(path) for key, path in modified_files.items()}

    # Update each file's "negative" key
    updated_files = {}
    for key, data in data_files.items():
        updated_data = update_negative_key(key, data, data_files)
        updated_files[key] = updated_data

    # Save the updated data back to the respective files
    for key, data in updated_files.items():
        output_path = modified_files[key]
        save_json(data, output_path)

    output_paths = {key: modified_files[key] for key in modified_files}
    return output_paths

def build_json(prompts_path, answers_path):
    """
    Build the antonym JSON structure from prompts and answers.

    Returns:
        str: Path to the final corrected antonym JSON file.
    """
    # Load the necessary data
    prompts = load_json(prompts_path)
    answers = load_json(answers_path)

    # Rebuild the antonym mapping from subject to object
    mapping = {item['subject']: item['object'] for item in answers}

    # The final JSON structure to be built
    json_final = []

    # Loop through each word in antonym_prompts.json
    for item in prompts:
        word = item['word']
        prompts = item['prompts']
        
        # Randomly select 3 prompts
        selected_prompts = random.sample(prompts, 3)
        
        # Find the corresponding antonym (keyword) from antonym_answers.json
        keyword = mapping.get(word, "")
        
        # Create the structure, updating only the positive key with selected prompts and the corresponding antonym as the keyword
        entry = {
            "concept": word,
            "group": "sense",
            "source": "language",
            "sentences": {
                "positive": [{"sentence": prompt, "keyword": keyword} for prompt in selected_prompts],
                "negative": []
            }
        }
        
        json_final.append(entry)

    # Save the final antonym.json as requested
    output_path= prompts_path.replace("_prompts.json", ".json")
    save_json(json_final, output_path)
    return output_path


if __name__ == "__main__":
    data_files = "datasets"
    files: List[List[str]] = []
    for file in os.listdir(data_files):
        if file.endswith("prompts.json"):
            files.append([file, file.replace("_prompts.json", "_answers.json")])
            
    print(f"Files found: {files}\n")
            
    for f in files:
        prompts_path, answers_path = f
        # Build the JSON
        print(f"Building JSON for {prompts_path.replace('_prompts.json', '')}")
        file_path = build_json(prompts_path==prompts_path, answers_path==answers_path)

        output_paths = json_process(file_path, limit=1000)
        
        print(f"JSON saved to: {output_paths}")
