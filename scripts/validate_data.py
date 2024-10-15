"""
This script is used to filter the data from the raw data.
We first feed the prompts to the model and only the prompts can be answered correctly are saved.
"""

import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = "Please take a deep breathe, and answer the following question correctly.\n"

def load_json_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_model(model_name):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if device.type == "cuda":
        model.half()
    
    return tokenizer, model, device

def generate_response(tokenizer, model, device, prompt, max_length=200):
    # Optionally, adjust the prompt formatting
    prompt = SYSTEM_PROMPT + prompt.strip() + "\nAnswer:"
    
    # Ensure the prompt does not contain the eos token
    if tokenizer.eos_token in prompt:
        prompt = prompt.replace(tokenizer.eos_token, '')
    
    # Encode the prompt without adding special tokens
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    
    # Generate a response
    try:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_length,
            min_length=5,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=1
            # Removed eos_token_id
        )
    except Exception as e:
        print(f"An error occurred during generation: {e}")
        return ""
    
    # Extract the generated tokens
    generated_ids = output_ids[0][input_ids.shape[1]:]
    if len(generated_ids) == 0:
        print("Warning: No new tokens generated.")
        return ""
    
    # Decode the response
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if not response.strip():
        print("Warning: Decoded response is empty.")
        return ""
    
    return response.strip()

def is_correct_answer(model_response, expected_answer):
    # Basic comparison (case-insensitive)
    return expected_answer.lower() in model_response.lower()

def filter_questions(data, tokenizer, model, device):
    filtered_questions = []
    for entry in data:
        word = entry.get('word', '')
        expected_answer = entry.get('answer', '')
        prompts = entry.get('prompts', [])
        
        for prompt in prompts:
            response = generate_response(tokenizer, model, device, prompt)
            if is_correct_answer(response, expected_answer):
                filtered_questions.append({
                    'prompt': prompt,
                    'response': response,
                    'expected_answer': expected_answer
                })
                print(f"Prompt: {prompt}")
                print(f"Model Response: {response}")
                print(f"Expected Answer: {expected_answer}")
                print("Status: Correct\n")
            else:
                print(f"Prompt: {prompt}")
                print(f"Model Response: {response}")
                print(f"Expected Answer: {expected_answer}")
                print("Status: Incorrect\n")
    return filtered_questions

def main(args):
    # Load the dataset
    data = load_json_dataset(args.dataset)
    
    # Choose the model
    model_sizes = {
        '564M': "facebook/xglm-564M",
        '1.7B': "facebook/xglm-1.7B",
        '2.9B': "facebook/xglm-2.9B"
    }
    
    model_name = model_sizes.get(args.model_size)
    if not model_name:
        print("Invalid model size selected.")
        return
    
    # Load the model and tokenizer
    tokenizer, model, device = get_model(model_name)
    
    # Filter the questions
    filtered_questions = filter_questions(data, tokenizer, model, device)
    
    # Save the filtered questions to a JSON file
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(filtered_questions, f, ensure_ascii=False, indent=4)
        print(f"Filtered questions saved to {args.output}")
    else:
        print("Filtered questions:")
        print(json.dumps(filtered_questions, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter questions that the XGLM model can answer correctly.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the JSON dataset file.')
    parser.add_argument('--model_size', type=str, choices=['564M', '1.7B', '2.9B'], default='564M', help='Size of the XGLM model to use.')
    parser.add_argument('--output', type=str, default="/datasets/filtered_data", help='Path to save the filtered questions as a JSON file.')
    args = parser.parse_args()
    
    main(args)