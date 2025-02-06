import json
import os
import re
import sys
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(PROJECT_ROOT)
import concurrent.futures
from tqdm import tqdm
from module.tools import get_response

def load_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def append_result(result, file_path):
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
    
    results.append(result)
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

def robust_parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pattern = r'\{.*\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                print("Regex extraction JSON parse error:", e)
        return None
    

def reward(question, correct_answer, reasoning_process, model, temperature):
    prompt = f"""Strictly determine if ANY reasoning step CONTAINS or LOGICALLY LEADS TO the correct answer. Follow these criteria:

Judgment Rules (MUST FOLLOW):
Content Match: Accept different numerical formats (0.5=50%=1/2) or unit variations.
Logical Derivation: Verify if steps mathematically/ logically imply the answer.
Option Substance: For MCQs, match answer CONTENT not just labels (e.g. "Option B" vs actual answer text).
Partial Evidence: Check if key components appear across multiple steps.
Semantic Equivalence: Recognize paraphrased answers with identical meaning.
Mathematical Simplification: For answers that are mathematical expressions, attempt to simplify step-by-step to check if they match the correct answer in its simplest form. Consider answers correct even if they are not fully simplified to the final correct form.

All your thinking process must be entirely provided in the "details" key.
Provide your final verdict in the "result" key, with the value either "Yes" or "No". 
Only output a single JSON object with no additional text.

Question:
{question}

Required Answer:
{correct_answer}

Candidate Reasoning:
{reasoning_process}
"""
    response = get_response(model=model, prompt=prompt, temperature=temperature)
    print(response)
    result = robust_parse_json(response)
    if result is None:
        stripped = response.strip().lower()
        if stripped in ["yes", "no"]:
            result = {"result": stripped, "details": ""}
        else:
            result = {"result": None, "details": "Response could not be parsed as JSON."}
    return result

# Helper function to process a single item
def process_item(item, judge_model, temperature):
    if not item.get("predicted_reasoning"):
        return None
    judge_result = reward(
        question=item["question"],
        correct_answer=item["correct_option"],
        reasoning_process=item["predicted_reasoning"],
        model=judge_model,
        temperature=temperature
    )
    result_entry = {
        "question": item["question"],
        "correct_option": item["correct_option"],
        "predicted_reasoning": item["predicted_reasoning"],
        "evaluation": judge_result
    }
    return result_entry

def evaluate_judge(task_type='simple_cot', source_model='gpt-4o-mini', judge_model='gpt-4o', temperature=0.7):
    input_file = f'results/{task_type}/{source_model}.json'
    output_dir = f'results/{task_type}'
    output_file = f'{output_dir}/{source_model}_judge.json'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = load_questions(input_file)
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed = executor.map(lambda item: process_item(item, judge_model, temperature), data)
        for result in tqdm(processed, total=len(data), desc="Concurrent Judge Processing"):
            results.append(result)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results appended to {output_file}")

if __name__ == "__main__":
    task_type = 'best_of_N'
    source_model = 'gpt-4o-mini'
    judge_model = 'chatgpt-4o-latest'
    temperature = 0.1
    evaluate_judge(task_type=task_type, source_model=source_model, judge_model=judge_model, temperature=temperature)
