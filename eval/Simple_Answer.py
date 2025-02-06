import numpy as np
import json
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from module.tools import get_response
import concurrent.futures
from tqdm import tqdm

# Load the dataset

prompt_template = (
    "Please think short and respond quick in a divergent way."
    "Question: {question}\n\nNow give you answer following the above requirements\n"
)

def evaluate_simple_answer(task, shots, model, temperature=1.0):
    test_questions = task[shots:]
    
    def process_question(question_data):
        prompt = prompt_template.format(question=question_data['question'])
        response = get_response(model=model, prompt=prompt, temperature=temperature)
        predicted_reasoning = response.split("\n") if response else None
        result = {
            "question": question_data['question'],
            "correct_option": question_data['answer'],
            "predicted_reasoning": predicted_reasoning,
        }
        return result

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for res in tqdm(executor.map(process_question, test_questions), total=len(test_questions), desc="Processing With Simple Answer"):
            results.append(res)
            
    output_file = f"results/simple_answer/{model}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    return 0

with open('data/Combine_of_5.json', 'r', encoding='utf-8') as f:
    all_task = json.load(f)

accuracy = evaluate_simple_answer(
    task=all_task,
    shots=100,
    model="gpt-4o",
    temperature=1.0
)
