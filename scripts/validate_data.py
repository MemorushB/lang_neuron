import os
import json
import random
import argparse
import re
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_json_file(file_path: str) -> List[Dict]:
    """Load a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_samples_from_file(data: List[Dict]) -> List[str]:
    """Extract samples from the JSON data."""
    samples = []
    for item in data:
        prompt = item.get('prompt')
        answer = item.get('answer')
        if prompt and answer:
            # Remove the 'Prompt: ' prefix if present
            prompt = re.sub(r'^Prompt:\s*', '', prompt)
            # Replace '__' with the answer
            prompt_filled = prompt.replace('__', answer)
            samples.append(prompt_filled.strip())
    return samples

def convert_json_to_format(
    input_file: str,
    folder_path: str,
    model_name: str,
    max_samples: Optional[int] = None
) -> Dict:
    """Convert the JSON file to the specified format."""
    # Load the input JSON file
    data = load_json_file(input_file)
    # Extract positive samples
    positive_samples = extract_samples_from_file(data)
    # Validate positive samples using the LLM
    positive_samples = validate_samples_with_llm(positive_samples, model_name)
    # Number of positive samples
    num_positive = len(positive_samples)
    
    if num_positive == 0:
        raise ValueError("No positive samples were validated successfully. Please check your data or model.")

    # Determine the number of negative samples (4 times the number of positive samples)
    num_negative = num_positive * 4
    
    # Collect negative samples from other JSON files
    negative_samples = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and filename != os.path.basename(input_file):
            file_path = os.path.join(folder_path, filename)
            other_data = load_json_file(file_path)
            negative_samples.extend(extract_samples_from_file(other_data))
    
    # Check if there are enough negative samples
    if len(negative_samples) < num_negative:
        print(f"Warning: Not enough negative samples available. Requested {num_negative}, but only {len(negative_samples)} available.")
        num_negative = len(negative_samples)
    
    # Use downsampling to get the required number of negative samples
    negative_samples = random.sample(negative_samples, num_negative)
    
    # Create the final dictionary
    concept_name = os.path.basename(input_file).replace('_prompts.json', '')
    result = {
        "concept": concept_name,
        "group": "sense",
        "source": "relation",
        "sentences": {
            "positive": positive_samples,
            "negative": negative_samples
        }
    }
    return result

def validate_samples_with_llm(samples: List[str], model_name: str) -> List[str]:
    """Validate samples using an LLM and return only the successfully answered samples."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.to(device)
    model.eval()
    
    valid_samples = []
    for sample in samples:
        # Extract the question and the expected answer
        match = re.match(r'(.+?) Answer: (.+)', sample)
        if match:
            question, expected_answer = match.groups()
            # Generate the model's response
            prompt = f"{question.strip()} Answer:"
            response = generate_response(prompt, model, tokenizer, device)
            # Check if the response contains the expected answer
            if is_correct_answer(response, expected_answer):
                valid_samples.append(sample)
    return valid_samples

def generate_response(prompt: str, model, tokenizer, device, max_new_tokens=50) -> str:
    """Generate a response from the model."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    # Remove any leading non-alphanumeric characters
    response = re.sub(r'^[^\w]*', '', response)
    return response

def is_correct_answer(model_response: str, expected_answer: str) -> bool:
    """Check if the model's response contains the expected answer."""
    return expected_answer.lower() in model_response.lower()

def dataset_generator(input_file: str, output_file: str, model_name: str, folder_path: str):
    """Generate the dataset with the specified format."""
    # Convert the JSON file to the specified format
    result = convert_json_to_format(
        input_file=input_file,
        folder_path=folder_path,
        model_name=model_name
    )
    # Save the result to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON file to specified format with LLM validation.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output JSON file.')
    parser.add_argument('--model_name', type=str, required=True, help='Name or path of the language model.')
    parser.add_argument('--folder_path', type=str, default='.', help='Path to the folder containing other JSON files.')
    args = parser.parse_args()
    
    dataset_generator(args.input_file, args.output_file, args.model_name, args.folder_path)