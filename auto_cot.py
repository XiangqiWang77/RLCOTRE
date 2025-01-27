import sys
sys.path.append('./baselines/auto-cot-main')
sys.argv=['']


import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
from module.feature_extraction import extract_features
from module.Reasoning_reward import simple_reward
from module.utils import load_questions, send_openai_prompt

# Load the dataset
all_task = load_questions("./data/shuffled_combined_HARDMATH.json")

embedding_dim = 768

from api import cot

def evaluate_few_shot_single_save(task,shots, output_file):

    with open(output_file, "w") as f:
        f.write("[\n")

        """
        for idx, question_data in enumerate(task[:shots]):
            context = extract_features(question_data['question'])
            
            temperature = 1.0
            token_limit = 1000000

            prompt = prompt_template.format(
                question=question_data['question']
            )
            response = send_openai_prompt(prompt, 1.0, temperature, token_limit)
            print(response)

            correct_option = question_data['answer']
            reward = simple_reward(question_data['question'], correct_option, response)
        """

        for idx, question_data in enumerate(task[shots:]):
            response=cot(method="zero_shot", question=question_data['question'])
            print(response)

            reasoning_steps = response.split("\n")
            correct_option = question_data['answer']

            result = {
                "question": question_data['question'],
                "correct_option": correct_option,
                "predicted_reasoning": reasoning_steps,
            }
            f.write(json.dumps(result, indent=4))
            if idx < len(task[shots:]) - 1:
                f.write(",\n")

        f.write("\n]")
    return 0

accuracy = evaluate_few_shot_single_save(
    task=all_task,
    shots=20,
    output_file="results/auto_cot.json"
)
