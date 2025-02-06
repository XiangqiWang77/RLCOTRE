import json
api_key = OPENAI_API

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

def robust_reward_v2(question, correct_answer, reasoning_process, options=None):
    prompt = f"""Strictly determine if ANY reasoning step CONTAINS or LOGICALLY LEADS TO the correct answer. Follow these criteria:

# Judgment Rules (MUST FOLLOW)
1. Content Match: Accept different numerical formats (0.5=50%=1/2) or unit variations
2. Logical Derivation: Verify if steps mathematically/ logically imply the answer
3. Option Substance: For MCQs, match answer CONTENT not just labels (e.g. "Option B" vs actual answer text)
4. Partial Evidence: Check if key components appear across multiple steps
5. Semantic Equivalence: Recognize paraphrased answers with identical meaning

# Question
{question}

# Required Answer
{correct_answer}{f" (Options: {options})" if options else ""}

# Candidate Reasoning
{reasoning_process}

Just output yes or no and don't output anything else beside that.

Final verdict (only 'yes'/'no'):"""

    response = send_openai_prompt(prompt).strip().lower()
    return 'yes' if response[:1] == 'y' else 'no'

# 测试代码
if __name__ == "__main__":
    input_file = './new_results/0206filter.json'
    output_file = '0206new_Judge.json'
    
    data = load_questions(input_file)
    
    for item in data:
        if item["best_candidate"]:
            result = robust_reward_v2(
                question=item["question"],
                correct_answer=item["ground_truth"],
                reasoning_process=item["best_candidate"]
            )
            result_entry = {
                "question": item["question"],
                "correct_option": item["ground_truth"],
                "predicted_reasoning": item["best_candidate"],
                "current_action": item["current_action"],
                "evaluation": result
            }
            append_result(result_entry, output_file)
            print(f"Result appended to {output_file}")
