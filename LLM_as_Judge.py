import json
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 获取 API Key
api_key = os.getenv('API_KEY')

# 加载问题

def load_questions(file_path):
    """从 JSON 文件加载问题"""
    with open(file_path, 'r') as f:
        return json.load(f)

# 追加保存结果到 JSON 文件

def append_result(result, file_path):
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
    
    results.append(result)
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

# 调用 OpenAI API 进行评估

def send_openai_prompt(prompt_text, model_name="gpt-4o", temperature=0.7, top_p=0.7, token_limit=1000):
    """调用 OpenAI API"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            model=model_name,
            temperature=temperature,
            max_tokens=token_limit
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content 
    except Exception as e:
        return f"Request failed: {e}"

# 简单奖励函数，检查推理过程是否包括正确答案

def simple_reward(question, correct_answer, reasoning_process, options=None):
    prompt = (
        f"Please check whether these reasoning results successfully or partially successfully output the ground truth answer of the given question.\n"
        f"Please carefully examine the reasoning steps and output and identify whether the correct answer is covered within the given process.\n"
        f"Note that if the reasoning process and answer covers the ground truth count as yes.\n"
        f"Question: {question}\n"
        f"Correct Answer or Option: {correct_answer}\n"
        f"Reasoning Process (all of them): {reasoning_process}\n"
        f"Options if any: {options}\n"
        f"Answer with 'yes' or 'no'. Just 'yes' and 'no'. No explanation or other things."
    )
    response = send_openai_prompt(prompt)
    return response

# 测试代码
if __name__ == "__main__":
    input_file = './results/RLCOTfinal.json'
    output_file = './LLM_as_Judge/RLCOT_Judge.json'
    
    data = load_questions(input_file)
    
    for item in data:
        if item["best_candidate"]:
            result = simple_reward(
                question=item["question"],
                correct_answer=item["ground_truth"],
                reasoning_process=item["best_candidate"]
            )
            result_entry = {
                "question": item["question"],
                "correct_option": item["ground_truth"],
                "predicted_reasoning": item["best_candidate"],
                "evaluation": result
            }
            append_result(result_entry, output_file)
            print(f"Result appended to {output_file}")
