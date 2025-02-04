import numpy as np
import json
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import concurrent.futures
from module.tools import get_response
from tqdm import tqdm  # Added tqdm import

# Prompt template
prompt_template = (
    "Please think step by step to solve the question"
    "Question: {question}\n\nNow give you answer following the above requirements\n"
)


def evaluate_simple_cot(task, shots, model, temperature=1.0):
    output_file = f'results/simple_cot/{model}.json'
    if not os.path.exists('results/simple_cot'):
        os.makedirs('results/simple_cot')
    # Process a single question
    def process_question(question_data):
        prompt = prompt_template.format(question=question_data['question'])
        # Call get_response with assumed parameters (adjust as needed)
        response = get_response(model=model, prompt=prompt, temperature=temperature)
        if not response:
            # API did not return anything, skip with answer as None
            predicted_reasoning = None
        else:
            predicted_reasoning = response.split("\n")
        correct_option = question_data['answer']
        result = {
            "question": question_data['question'],
            "correct_option": correct_option,
            "predicted_reasoning": predicted_reasoning,
        }
        return result

    results = []
    test_questions = task[shots:]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for res in tqdm(executor.map(process_question, test_questions), total=len(test_questions), desc="Processing With Simple CoT"):  # Wrapped with tqdm
            results.append(res)
            
    # Write all results to output_file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    return 0

with open('data/Combine_of_5.json', 'r', encoding='utf-8') as f:
    all_task = json.load(f)
    
accuracy = evaluate_simple_cot(
    task=all_task,
    shots=1200,
    model="gpt-4o-mini",
    temperature=1.0
)
