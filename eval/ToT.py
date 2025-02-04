import json
import numpy as np
import os
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Added import for progress bar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from module.tools import get_response  # Updated import

# ======= Adjustable Parameters =======
MAX_STEPS = 3             # Maximum number of expansion steps in Tree of Thoughts
BRANCHING_FACTOR = 3      # Number of thought branches generated at each step
BEAM_SIZE = 2             # Number of top paths to retain at each step
NUM_WORKERS = 15          # Number of concurrent workers

prompt_template = (
    "Please think step by step to solve the following question.\n"
    "Question: {question}\n\n"
    "Please continue your reasoning at step {current_step}.\n"
)

def generate_candidate_thoughts(model, question, partial_solution, current_step, temperature=1.0):
    """
    Generate possible next thoughts (extensions) for the current reasoning path.
    """
    prompt = (
        f"{partial_solution}\n\n" if partial_solution else ""
    ) + prompt_template.format(question=question, current_step=current_step)
    
    candidates = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(get_response, model=model, prompt=prompt, temperature=temperature)
                   for _ in range(BRANCHING_FACTOR)]
        for future in as_completed(futures):
            try:
                response = future.result().strip()
                candidates.append(response)
            except Exception as e:
                print("Error during candidate generation:", e)
    return candidates

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

def improved_dense_reward(model, question, reasoning_process, options=None, temperature=1.0):
    """
    Evaluate the reasoning process without relying on ground truth answers.
    """
    prompt = (
        f"Question: {question}\n\n"
        "Your task is to evaluate the following reasoning process. "
        "Please output a valid JSON with the following keys and values (all values should be floats between 0 and 1):\n"
        "  - LogicalFlaw: Evaluate the logical consistency and absence of errors.\n"
        "  - Coverage: Evaluate how well the reasoning covers all relevant aspects of the question.\n"
        "  - Confidence: Rate your confidence in the correctness of the reasoning.\n"
        "  - Rationale: Assess the quality of the explanation provided.\n\n"
        "Output Requirements:\n"
        "  - Output must be a valid JSON string, for example:\n"
        '    {"LogicalFlaw": 0.9, "Coverage": 0.85, "Confidence": 0.9, "Rationale": 0.95}\n\n'
        f"Reasoning Process: {reasoning_process}\n"
        f"Options (if any): {options}\n"
    )
    
    response = get_response(model=model, prompt=prompt, temperature=temperature)
    print("LLM Evaluation Result:", response)
    
    evaluation = robust_parse_json(response)
    if evaluation is None:
        print("Error: Unable to parse evaluation response. Defaulting reward to -1.0")
        return -1.0
    try:
        keys = ["LogicalFlaw", "Coverage", "Confidence", "Rationale"]
        reward = sum(evaluation[key] for key in keys) / len(keys)
        return reward
    except KeyError as e:
        print("Missing key in evaluation response:", e)
        return -1.0

def evaluate_path(model, question, reasoning_process, options=None, temperature=1.0):
    """
    Evaluate the reasoning process using the improved_dense_reward function.
    """
    score = improved_dense_reward(model, question=question, reasoning_process=reasoning_process, options=options, temperature=temperature)
    return score

def tree_of_thoughts_search(model, question, options=None, temperature=1.0):
    """
    Execute the Tree of Thoughts search process and return the best reasoning path and its score.
    """
    initial_path = ""
    paths = [(initial_path, 0.0)]  # (path_text, score)

    for step in range(1, MAX_STEPS + 1):
        new_paths = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as gen_executor:
            future_to_path = {}
            for (path_text, _) in paths:
                futures = [gen_executor.submit(generate_candidate_thoughts, model, question, path_text, step, temperature)
                           for _ in range(1)] 
                for future in futures:
                    future_to_path[future] = path_text

            for future in as_completed(future_to_path):
                base_path = future_to_path[future]
                try:
                    candidates = future.result()
                    for candidate in candidates:
                        new_path_text = f"{base_path}\n{candidate}" if base_path else candidate
                        new_paths.append(new_path_text)
                except Exception as e:
                    print("Error during candidate generation in tree search:", e)
        
        scored_paths = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as eval_executor:
            eval_futures = {eval_executor.submit(evaluate_path, model, question, path_text, options, temperature): path_text
                            for path_text in new_paths}
            for future in as_completed(eval_futures):
                path_text = eval_futures[future]
                try:
                    score = future.result()
                    scored_paths.append((path_text, score))
                except Exception as e:
                    print("Error during evaluation of a path:", e)
                    scored_paths.append((path_text, -1.0))
        
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        paths = scored_paths[:BEAM_SIZE]
        print(f"Step {step} completed, retained paths: {len(paths)}")
    
    best_path, best_score = paths[0]
    return best_path, best_score

def evaluate_tot(model, task, shots, temperature=1.0):
    """
    Evaluate each question using Tree of Thoughts and save the results to a file.
    """
    out_dir = "results/tot"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file = f"{out_dir}/{model}.json"
    
    results = []
    # Ensure task length is sufficient
    task = task[shots:] if shots < len(task) else task

    # Helper function to process each question
    def process_question(idx, question_data):
        question = question_data["question"]
        options = question_data.get("options", None)
        best_path, best_score = tree_of_thoughts_search(model, question, options=options, temperature=temperature)
        predicted_reasoning = best_path.split("\n")
        print(f"Processed question {idx + 1}/{len(task)}")
        return {
            "question": question,
            "predicted_reasoning": predicted_reasoning,
            "score": best_score
        }

    # Concurrency: process questions concurrently in main function
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as main_executor:
        futures = {main_executor.submit(process_question, idx, q_data): idx for idx, q_data in enumerate(task)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing questions", unit="question"):
            try:
                results.append(future.result())
            except Exception as e:
                print("Error processing question:", e)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    return 0

# ======= Execute and Save Results =======
if __name__ == "__main__":
    with open('data/Combine_of_5.json', 'r', encoding='utf-8') as f:
        all_task = json.load(f)
    shots = 1200
    model = "gpt-4o-mini"
    temperature = 1.0
    evaluate_tot(
        model=model,
        task=all_task,
        shots=shots,
        temperature=temperature
    )
    print("Completed. Results saved.")
