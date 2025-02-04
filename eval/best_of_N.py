import numpy as np
import json
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import concurrent.futures
from module.tools import get_response
from tqdm import tqdm  

# Prompt Template
prompt_template = (
    "Please think step by step to solve the question. "
    "Question: {question}\n\nNow give you answer following the above requirements\n"
)

def select_best_response(responses):
    """
    Simple heuristic: choose the answer with the most characters.
    If all answers are None, return None.
    """
    valid_responses = [r for r in responses if r is not None]
    if not valid_responses:
        return None
    # Calculate string length for each answer (joining list elements)
    best = max(valid_responses, key=lambda r: len("".join(r)))
    return best

def evaluate_responses(responses):
    for r in responses:
        if r is not None:
            return r
    return None

def evaluate_best_of_N(task, shots, model, temperature=1.0, N=5, use_eval=True):
    """
    For the questions in the task, skip the first 'shots' examples,
    then call get_response N times for each question.
    If use_eval is True, select the best answer using the evaluation function,
    otherwise record all N answers directly.
    """
    output_dir = 'results/best_of_N'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'{model}.json')
    
    cached_dict = {}
    if use_eval and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            cached_results = json.load(f)
        for entry in cached_results:
            cached_dict[entry["question"]] = entry.get("all_responses", [])
    
    def process_question(question_data):
        q_text = question_data['question']
        if use_eval and q_text in cached_dict and cached_dict[q_text]:
            responses = cached_dict[q_text]
        else:
            prompt = prompt_template.format(question=q_text)
            responses = []
            for i in range(N):
                response = get_response(model=model, prompt=prompt, temperature=temperature)
                responses.append(None if not response else response.split("\n"))
        if use_eval:
            best_response = evaluate_responses(responses)
            result = {
                "question": q_text,
                "correct_option": question_data['answer'],
                "predicted_reasoning": best_response,
                "all_responses": responses,
            }
        else:
            result = {
                "question": q_text,
                "correct_option": question_data['answer'],
                "all_responses": responses,
            }
        return result

    results = []
    test_questions = task[shots:]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for res in tqdm(executor.map(process_question, test_questions), total=len(test_questions), desc="Processing With Best-of-N"):
            results.append(res)
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    return 0

# Read data
with open('data/Combine_of_5.json', 'r', encoding='utf-8') as f:
    all_task = json.load(f)

# Call best_of_N baseline, setting shots and N (here N=5)
accuracy = evaluate_best_of_N(
    task=all_task,
    shots=1200,
    model="gpt-4o-mini",
    temperature=1.0,
    N=5,
    use_eval=False
)
