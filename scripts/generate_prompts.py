import os
import json
import glob
import random

def load_json_files(folder_path):
    """Load all JSON files from the specified folder."""
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    data = []
    filenames = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
                data.append(json_data)
                filenames.append(os.path.basename(file))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file}: {e}")
    return data, filenames

def generate_combined_prompts(json_data, few_shot_count=3):
    """
    Generate combined prompts using QA form and few-shot prompting with random sampling,
    using all templates in 'prompt_templates'.

    Args:
        json_data (dict): The JSON data for a single relation.
        few_shot_count (int): Number of few-shot examples to include.

    Returns:
        list: A list of dictionaries containing prompts and answers.
    """
    results = []
    samples = json_data.get('samples', [])
    prompt_templates = json_data.get('prompt_templates', [])

    # Ensure there are enough samples
    total_required_samples = few_shot_count + 1  # Few-shot examples + test sample
    if len(samples) < total_required_samples:
        print(f"Warning: Not enough samples for random sampling in relation '{json_data.get('name', 'relation')}'.")
        return results

    for template in prompt_templates:

        random.seed(42)

        for sample in samples:
            # Split into few-shot examples and test sample
            selection = samples.copy()
            selection.remove(sample)
            few_shot_samples = random.sample(selection, total_required_samples)

            if not template.endswith("?"):
                template = template.strip() + "?"

            combined_prompt = "Prompt:"
            # Generate few-shot examples
            for f in few_shot_samples:
                subject = f.get('subject', '').strip()
                answer = f.get('object', '').strip()
                prompt = template.format(subject)
                combined_prompt += f" {prompt} Answer: {answer}."

            # Add the actual sample's prompt without the answer
            actual_subject = sample.get('subject', '').strip()
            actual_answer = sample.get('object', '').strip()
            actual_prompt = template.format(actual_subject)
            combined_prompt += f" {actual_prompt} Answer: __"

            # Append the result
            results.append({
                "prompt": combined_prompt,
                "answer": actual_answer
            })

    return results

def process_json_file(json_data, few_shot_count=3):
    """
    Process a single JSON data and generate combined prompts.

    Args:
        json_data (dict): JSON data dictionary.
        few_shot_count (int): Number of few-shot examples to include.

    Returns:
        list: A list of dictionaries containing prompts and answers.
    """
    prompts = generate_combined_prompts(json_data, few_shot_count)
    return prompts

def save_results(output_file, results):
    """
    Save the results to a JSON file.

    Args:
        output_file (str): Path to the output JSON file.
        results (list): List of dictionaries containing prompts and answers.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")

def main(input_folder, output_folder, few_shot_count=3):
    """Main function to process JSON files and generate combined prompts."""
    json_data_list, filenames = load_json_files(input_folder)
    if not json_data_list:
        print("No JSON data found.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for json_data, filename in zip(json_data_list, filenames):
        results = process_json_file(json_data, few_shot_count)
        if results:
            # Use the input filename to create the output filename
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_folder, f"{base_name}_prompts.json")
            save_results(output_file, results)
        else:
            print(f"No prompts were generated for file {filename}.")

if __name__ == "__main__":
        # Define the input folder containing JSON files and the output folder path
        input_folder = 'datasets/LRE/'  # Replace with your JSON folder path
        output_folder = 'datasets/LRE/lre_prompts/'  # Replace with your desired output folder path

        # Optional: Adjust the number of few-shot examples per relation
        few_shot_count = 3  # You can modify this value as needed

        main(input_folder, output_folder, few_shot_count)