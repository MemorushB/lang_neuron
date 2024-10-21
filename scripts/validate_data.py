import json
import torch
import argparse
import time
from vllm import LLM, SamplingParams

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def reformat_prompts(prompt_templates):
    # Simply strip the templates
    qa_templates = [template.strip() for template in prompt_templates]
    return qa_templates

def generate_prompt(subject, expected_answer, qa_templates):
    prompts = []
    for template in qa_templates:
        # Build the question using the template and subject
        # Count the number of '{}' placeholders in the template
        num_placeholders = template.count('{}')

        # Format the question with the appropriate number of arguments
        if num_placeholders == 0:
            # Template doesn't have placeholders, so assume it is a question itself
            question = template
        elif num_placeholders == 1:
            # Template expects one placeholder, fill with subject
            question = template.format(subject)
        elif num_placeholders == 2:
            # Template expects two placeholders, fill with subject and expected_answer
            question = template.format(subject, expected_answer)
        else:
            raise ValueError(f"Unexpected number of placeholders in template: {template}")

        # Ensure question ends with '?'
        if not question.endswith('?'):
            question += '?'

        # Construct the prompt in the desired format
        prompt = f"Question: {question} Answer:"

        prompts.append(prompt)
    return prompts

def generate_few_shot_prompt(subject, expected_answer, few_shot_examples, template):
    # Construct a few-shot prompt
    prompt = ""
    for example in few_shot_examples:
        example_prompt = template.format(example['subject'], example['object'])
        prompt += f"{example_prompt}\n\n"
    # Add the actual question
    actual_prompt = template.format(subject, expected_answer)
    prompt += f"{actual_prompt}"
    return prompt

def generate_response(prompt, llm_engine, max_new_tokens=50):
    # Prepare the prompt for Llama 2
    system_prompt = "You are a helpful assistant."
    bos_token = '<s>'
    eos_token = '</s>'
    full_prompt = f"{bos_token}[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt.strip()} [/INST]{eos_token}"

    # Set sampling parameters
    sampling_params = SamplingParams(
        n=1,
        best_of=1,
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_new_tokens,
        stop=["</s>"],
    )

    # Generate the response using vLLM
    outputs = llm_engine.generate([full_prompt], sampling_params)

    # Extract the response
    response = outputs[0].outputs[0].text.strip().split('\n')[0]
    return response

def is_correct_answer(model_response, expected_answer):
    # Basic comparison (case-insensitive)
    return expected_answer.lower() in model_response.lower()

def validate_samples(samples, qa_templates, llm_engine, prompt_method, few_shot_examples):
    filtered_samples = []
    total_samples = len(samples)
    total_time = 0.0

    start_time = time.time()
    for idx, sample in enumerate(samples):
        sample_start_time = time.time()

        subject = sample['subject']
        expected_answer = sample['object']

        if prompt_method == 'qa':
            prompts = generate_prompt(subject, expected_answer, qa_templates)
        elif prompt_method == 'few_shot':
            prompts = []
            # Use the first template for few-shot prompts
            template = qa_templates[0]
            # Exclude the current sample from few_shot_examples
            current_few_shot_examples = [ex for ex in few_shot_examples if ex != sample]
            prompt = generate_few_shot_prompt(subject, expected_answer, current_few_shot_examples, template)
            prompts.append(prompt)
        else:
            raise ValueError("Invalid prompt method. Choose 'qa' or 'few_shot'.")

        for prompt in prompts:
            response = generate_response(prompt, llm_engine)
            if is_correct_answer(response, expected_answer):
                # Save the sample if the answer is correct
                filtered_samples.append({
                    'subject': subject,
                    'object': expected_answer,
                    'prompt': prompt,
                    'response': response,
                })
                print(f"Subject: {subject}")
                print(f"Prompt:\n{prompt}")
                print(f"Expected Answer: {expected_answer}")
                print(f"Model Response: {response}")
                print("Status: Correct\n")
                break  # Stop after the first correct answer
            else:
                print(f"Subject: {subject}")
                print(f"Prompt:\n{prompt}")
                print(f"Expected Answer: {expected_answer}")
                print(f"Model Response: {response}")
                print("Status: Incorrect\n")

        sample_end_time = time.time()
        sample_time = sample_end_time - sample_start_time
        total_time += sample_time
        print(f"Sample {idx + 1}/{total_samples} processed in {sample_time:.2f} seconds.")

    end_time = time.time()
    average_time = total_time / total_samples if total_samples > 0 else 0
    print(f"Validation completed in {end_time - start_time:.2f} seconds.")
    print(f"Average time per sample: {average_time:.2f} seconds.")
    return filtered_samples

def main(args):
    total_start_time = time.time()

    # Load the data
    data = load_json_data(args.data_file)
    samples = data.get('samples', [])
    prompt_templates = data.get('prompt_templates', []) + data.get('prompt_templates_zs', [])

    # Reformat the prompts into QA form
    qa_templates = reformat_prompts(prompt_templates)

    # Prepare few-shot examples
    few_shot_examples = []
    if args.prompt_method == 'few_shot':
        few_shot_examples = samples[:args.num_few_shot_examples]

    # Initialize the vLLM engine
    start_time = time.time()
    llm_engine = LLM(
        model=args.model_name, 
        tokenizer=args.model_name, 
        trust_remote_code=True,
        #quantization="aqlm"
        max_model_len = 1024,
        gpu_memory_utilization=0.9,
        )
    end_time = time.time()
    print(f"vLLM engine initialized in {end_time - start_time:.2f} seconds.")

    # Validate samples
    filtered_samples = validate_samples(
        samples,
        qa_templates,
        llm_engine,
        prompt_method=args.prompt_method,
        few_shot_examples=few_shot_examples
    )

    # Save the filtered samples
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_samples, f, ensure_ascii=False, indent=4)
        print(f"Filtered samples saved to {args.output_file}")
    else:
        print("Filtered Samples:")
        print(json.dumps(filtered_samples, ensure_ascii=False, indent=4))

    total_end_time = time.time()
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate data using Llama2 model with vLLM.")
    parser.add_argument('--data_file', type=str, required=True, help='Path to the JSON data file.')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='Name or path of the Llama2 model.')
    parser.add_argument('--output_file', type=str, help='Path to save the filtered data.')
    parser.add_argument('--prompt_method', type=str, choices=['qa', 'few_shot'], default='qa', help="Prompt method to use: 'qa' or 'few_shot'.")
    parser.add_argument('--num_few_shot_examples', type=int, default=3, help='Number of few-shot examples to include (only used if prompt_method is "few_shot").')
    args = parser.parse_args()

    main(args)