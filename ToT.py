import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ======= Import Existing Utility Functions/Modules =======
from module.feature_extraction import extract_features
from module.utils import load_questions, send_openai_prompt

# ======= Adjustable Parameters =======
MAX_STEPS = 3             # Maximum number of expansion steps in Tree of Thoughts
BRANCHING_FACTOR = 3      # Number of thought branches generated at each step
BEAM_SIZE = 2             # Number of top paths to retain at each step

# ======= Load Data and Models =======
all_task = load_questions("./data/shuffled_combined_HARDMATH.json")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
hf_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# ======= Prompt Template =======
prompt_template = (
    "Please think step by step to solve the following question.\n"
    "Question: {question}\n\n"
    "Please continue your reasoning at step {current_step}.\n"
)

def generate_candidate_thoughts(question, partial_solution, current_step, temperature=1.0, token_limit=1024):
    """
    Generate possible next thoughts (extensions) for the current reasoning path.
    """
    prompt = (
        f"{partial_solution}\n\n"
        + prompt_template.format(question=question, current_step=current_step)
    )
    
    candidates = []
    for _ in range(BRANCHING_FACTOR):
        response = send_openai_prompt(prompt, 1.0, temperature, token_limit)
        candidates.append(response.strip())
    return candidates

def improved_dense_reward(question, reasoning_process, options=None):
    """
    Evaluate the reasoning process without relying on ground truth answers.
    """
    prompt = (
        f"Question: {question}\n\n"
        "Your task is to evaluate the reasoning process provided below based on specific dimensions and output the results in a JSON-like plain text format (not actual JSON).\n\n"

        "Evaluation Dimensions:\n"
        "1. **Logical Flaw**: Assess the logical consistency and absence of errors in the reasoning process (0 to 1).\n"
        "2. **Coverage**: Evaluate how well the reasoning process addresses all relevant aspects of the question (0 to 1).\n"
        "3. **Confidence**: Rate the confidence level in the correctness of the reasoning process (0 to 1).\n"
        "4. **Rationale**: Judge the quality of the reasoning and explanation provided (0 to 1).\n\n"

        "Output Requirements:\n"
        "1. Provide a single set of scores for the reasoning process based on the evaluation dimensions.\n"
        "2. Format the output as plain text that resembles JSON but is not actual JSON code.\n"
        "3. Use this specific format for the output:\n"
        "{\n"
        "  \"LogicalFlaw\": 0.9,\n"
        "  \"Coverage\": 0.85,\n"
        "  \"Confidence\": 0.9,\n"
        "  \"Rationale\": 0.95\n"
        "}\n\n"

        "Prohibited Actions:\n"
        "1. Do not provide real JSON code.\n"
        "2. Ensure the output remains in plain text resembling JSON.\n"
        "3. Do not deviate from the defined evaluation dimensions or format.\n\n"

        f"Reasoning Process: {reasoning_process}\n"
        f"Options (if any): {options}\n"
    )
    
    response = send_openai_prompt(prompt)
    
    print("LLM Evaluation Result:", response)
    
    # Parse the JSON-like response
    try:
        # Replace single quotes with double quotes and ensure proper formatting
        response_corrected = response.replace("'", "\"")
        evaluation = json.loads(response_corrected)
        # Aggregate the scores for a dense reward
        reward = sum(evaluation[key] for key in ["LogicalFlaw", "Coverage", "Confidence", "Rationale"]) / 4.0
    except (json.JSONDecodeError, KeyError) as e:
        print("Error in parsing the LLM response:", e)
        reward = -1.0  # Default to penalty for invalid or unexpected response
    
    return reward

def evaluate_path(reasoning_process, options=None):
    """
    Evaluate the reasoning process using the improved_dense_reward function.
    """
    # Since ground truth is not used, we don't pass correct_answer
    score = improved_dense_reward(question="", reasoning_process=reasoning_process, options=options)
    return score

def tree_of_thoughts_search(question, options=None):
    """
    Execute the Tree of Thoughts search process and return the best reasoning path and its score.
    """
    initial_path = ""
    paths = [(initial_path, 0.0)]  # (path_text, score)

    for step in range(1, MAX_STEPS + 1):
        new_paths = []
        for (path_text, _) in paths:
            # Generate candidate thoughts based on the current path
            candidates = generate_candidate_thoughts(question, path_text, step)
            for c in candidates:
                new_path_text = path_text + "\n" + c if path_text else c
                new_paths.append(new_path_text)
        
        # Evaluate all new paths
        scored_paths = []
        for path_text in new_paths:
            score = evaluate_path(reasoning_process=path_text, options=options)
            scored_paths.append((path_text, score))
        
        # Sort paths by score in descending order and retain top BEAM_SIZE paths
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        paths = scored_paths[:BEAM_SIZE]
        print(f"Step {step} completed, retained paths: {len(paths)}")
    
    # Select the path with the highest score as the final result
    best_path, best_score = paths[0]
    return best_path, best_score

def evaluate_tree_of_thoughts_save(task, shots, output_file):
    """
    Evaluate each question using Tree of Thoughts and save the results to a file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")
        
        for idx, question_data in enumerate(task):
            # Skip few-shot examples if necessary
            if idx < shots:
                continue

            question = question_data["question"]
            options = question_data.get("options", None)  # Assuming some questions have options

            # Extract features if needed
            context = extract_features(question)

            # Perform Tree of Thoughts search
            best_path, best_score = tree_of_thoughts_search(question, options)

            # Split the reasoning path into a list
            predicted_reasoning = best_path.split("\n")

            # Organize the result
            result = {
                "question": question,
                "predicted_reasoning": predicted_reasoning,
                "score": best_score
            }

            # Write to file
            f.write(json.dumps(result, ensure_ascii=False, indent=4))
            if idx < len(task) - 1:
                f.write(",\n")
            print(f"Processed question {idx + 1}/{len(task)}")
        
        f.write("\n]")
    return 0

# ======= Execute and Save Results =======
if __name__ == "__main__":
    output_path = "results/tot_results.json"
    shots = 20  # Number of few-shot examples
    evaluate_tree_of_thoughts_save(
        task=all_task,
        shots=shots,
        output_file=output_path
    )
    print(f"Completed. Results saved to {output_path}")
