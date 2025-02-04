import sys
import os
import json

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(PROJECT_ROOT)
sys.path.append('./baselines/auto-cot-main')
sys.argv=['']
from api import cot

import concurrent.futures
from tqdm import tqdm


def evaluate_auto_cot(task, shots, output_file):
    # Ensure output directory exists
    dir_name = os.path.dirname(output_file)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Process a single question
    def process_question(question_data):
        response = cot(method="zero_shot", question=question_data['question'])
        print(response)
        reasoning_steps = response.split("\n")
        correct_option = question_data['answer']
        result = {
            "question": question_data['question'],
            "correct_option": correct_option,
            "predicted_reasoning": reasoning_steps,
        }
        return result

    test_questions = task[shots:]
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for res in tqdm(
            executor.map(process_question, test_questions),
            total=len(test_questions),
            desc="Processing With Auto CoT"
        ):
            results.append(res)
            
    # Write all results to output_file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    return 0


with open('data/Combine_of_5.json', 'r', encoding='utf-8') as f:
    all_task = json.load(f)

accuracy = evaluate_auto_cot(
    task=all_task,
    shots=1200,
    output_file="results/auto_cot.json"
)
