import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 初始化模型
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')  # Semantic similarity model
entailment_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')  # Entailment model tokenizer
entailment_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')  # Entailment model

def calculate_reward_v4(question, correct_answer, reasoning_process, alpha=0.2, beta=0.8, num_samples=2):
    """
    使用 Shapley Score 计算推理步骤的奖励，返回每一步的贡献度列表。
    
    Args:
        question (str): 当前问题。
        correct_answer (str): Ground Truth。
        reasoning_process (list): 推理步骤的列表 [r1, r2, ..., rn]。
        alpha (float): 语义相似性的权重。
        beta (float): 逻辑一致性的权重。
        num_samples (int): 用于近似 Shapley Score 的采样次数。
        
    Returns:
        list: 每个推理步骤的 Shapley 分数列表，与 reasoning_process 的长度一致。
    """
    
    def semantic_similarity(steps, correct_answer):
        """
        计算子集推理输出与 Ground Truth 的语义相似性。
        """
        combined_response = " ".join(steps)  # 将步骤合并为一个完整的推理路径
        response_emb = semantic_model.encode(combined_response, convert_to_tensor=True)
        gt_emb = semantic_model.encode(correct_answer, convert_to_tensor=True)
        # 使用余弦相似度计算语义相似性
        similarity = torch.nn.functional.cosine_similarity(response_emb, gt_emb, dim=0)
        return similarity.item()

    def detect_negation(steps, correct_answer, threshold=0.6):
        """
        检测子集推理输出是否否决了正确答案。

        Args:
            steps (list): 推理步骤的列表。
            correct_answer (str): 正确答案。
            entailment_tokenizer: 用于编码输入的 tokenizer。
            entailment_model: 用于计算蕴含关系的模型。
            threshold (float): 用于判断否决的概率阈值。

        Returns:
            bool: 如果否决正确答案，返回 True；否则返回 False。
        """
        combined_response = " ".join(steps)
        
        # 将推理步骤和正确答案编码为模型输入
        inputs = entailment_tokenizer(
            combined_response,
            correct_answer,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            # 使用文本蕴含模型进行推断
            outputs = entailment_model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        entailment_prob = probabilities[0][0].item()  # Entailment 类别概率
        contradiction_prob = probabilities[0][2].item()  # Contradiction 类别概率

        # 如果检测到否决，则返回负的否决概率值作为惩罚
        if contradiction_prob > threshold:
            return -1 * contradiction_prob  # 否决时的负分
        # 否则返回支持概率值
        return entailment_prob

    def evaluate_subset(steps, correct_answer):
        """
        评估子集的综合得分。
        """
        Temp_res="The final and correct answer is"+correct_answer
        semantic_score = semantic_similarity(steps, Temp_res)
        logical_score = detect_negation(steps, Temp_res)
        return alpha * semantic_score + beta * logical_score

    # 初始化 Shapley 分数
    n = len(reasoning_process)
    shapley_values = np.zeros(n)

    # 使用蒙特卡罗采样近似 Shapley Score
    for _ in range(num_samples):
        permutation = np.random.permutation(n)  # 随机排列步骤
        current_subset = []  # 当前构造的步骤子集
        v_current = 0.0  # 当前子集得分

        for i in permutation:
            # 加入新步骤后的得分
            v_with_step = evaluate_subset(current_subset + [reasoning_process[i]], correct_answer)
            # 计算边际贡献
            marginal_contribution = v_with_step - v_current
            shapley_values[i] += marginal_contribution
            # 更新当前子集和得分
            current_subset.append(reasoning_process[i])
            v_current = v_with_step

    # 归一化 Shapley 分数
    shapley_values /= num_samples
    return shapley_values.tolist()

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 获取 API Key
api_key = os.getenv('API_KEY')


def send_openai_prompt(prompt_text, model_name="gpt-4o", temperature=0.7, top_p=0.7, token_limit=250):
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
            model="gpt-4o",
            temperature=temperature,
            #top_p=top_p,
            max_tokens=token_limit
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content 
    except Exception as e:
        return f"Request failed: {e}"
    

def simple_reward(question, correct_answer, reasoning_process, options=None):
    prompt = (
        f"Please check whether this reasoning process successfully or partially successfully output the ground truth answer of the given question.\n"
        f"Please carefully examine the reasoning steps and output and identify whether the correct answer is covered within the given process.\n"
        f"Question: {question}\n"
        f"Correct Answer or Option: {correct_answer}\n"
        f"Reasoning Process: {reasoning_process}\n"
        f"Options if any: {options}\n"
        f"Answer with 'yes' or 'no'. Just 'yes' and 'no'. No explanation or other things."
    )
    
    response = send_openai_prompt(prompt)

    print("The LLM as Judge result",response)
    
    if 'yes' in response.lower():
        return 1.0
    else:
        return -1.0

import json

def improved_dense_reward(question, correct_answer, reasoning_process, options=None):
    prompt = (
    # [Objective]
        f"Question: {question}\n\n"
        "Your task is to evaluate the reasoning process provided below based on specific dimensions and output the results in a JSON-like plain text format (not actual JSON).\n\n"

        # [Evaluation Dimensions]
        "Evaluation Dimensions:"
        "1. **Logical Flaw**: Assess the logical consistency and absence of errors in the reasoning process (0 to 1)."
        "2. **Coverage**: Evaluate how well the reasoning process addresses all relevant aspects of the question (0 to 1)."
        "3. **Confidence**: Rate the confidence level in the correctness of the reasoning process (0 to 1)."
        "4. **Rationale**: Judge the quality of the reasoning and explanation provided (0 to 1).\n\n"

        # [Output Requirements]
        "Output Requirements:"
        "1. Provide a single set of scores for the reasoning process based on the evaluation dimensions."
        "2. Format the output as plain text that resembles JSON but is not actual JSON code."
        "3. Use this specific format for the output:"
        "{"
        "  \"LogicalFlaw\": 0.9,"
        "  \"Coverage\": 0.85,"
        "  \"Confidence\": 0.9,"
        "  \"Rationale\": 0.95"
        "}\n\n"

        # [Prohibited Actions]
        "Prohibited Actions:"
        "1. Do not provide real JSON code."
        "2. Ensure the output remains in plain text resembling JSON."
        "3. Do not deviate from the defined evaluation dimensions or format.\n\n"

        # [Reasoning and Context]
        "Reasoning Process: {reasoning_process}\n"
        "Correct Answer: {correct_answer}\n"
        "Options (if any): {options}\n"
    )

    
    response = send_openai_prompt(prompt)
    
    print("LLM Evaluation Result:", response)
    
    # Parse the JSON-like response
    try:
        evaluation = json.loads(response)
        # Aggregate the scores for a dense reward
        reward = sum(evaluation[key] for key in ["LogicalFlaw", "Coverage", "Confidence", "Rationale"]) / 4.0
    except (json.JSONDecodeError, KeyError) as e:
        print("Error in parsing the LLM response:", e)
        reward = -1.0  # Default to penalty for invalid or unexpected response
    
    return reward


def majority_voting_reward(question, correct_answer, best_responses):
    """
    Evaluates a set of responses using a scoring mechanism (0 to 1) by leveraging LLM.

    Parameters:
    - question (str): The original question being evaluated.
    - correct_answer (str): The correct answer to the question.
    - best_responses (list): A list of generated responses to be evaluated.

    Returns:
    - float: The average score (between 0 and 1) based on the LLM's evaluation.
    """
    
    # Define the prompt template
    prompt_template = (
        "Question: {question}\n\n"
        "Evaluate the correctness of the response based on the correct answer.\n"
        "Provide a score from 0 to 1:\n"
        "- 1.0: Fully correct\n"
        "- 0.5: Mostly correct but with minor errors\n"
        "- 0.0: Incorrect\n\n"
        "Output only the score (0, 0.5, or 1.0) with no extra text.\n\n"
        "Correct Answer: {correct_answer}\n"
        "Response: {response}\n"
    )

    # Ensure we don't divide by zero
    if not best_responses:
        return 0.0

    total_score = 0
    valid_responses = 0
    
    for response in best_responses:
        # Format the prompt for the current response
        formatted_prompt = prompt_template.format(
            question=question, correct_answer=correct_answer, response=response
        )
        
        # Get the evaluation result from LLM
        eval_response = send_openai_prompt(formatted_prompt).strip()

        # Try to parse the response as a float
        try:
            score = float(eval_response)
            if 0.0 <= score <= 1.0:  # Ensure the score is within the expected range
                total_score += score
                valid_responses += 1
        except ValueError:
            pass  # Ignore invalid responses

    # Return the average score
    return total_score / valid_responses if valid_responses > 0 else 0.0


def majority_voting_reward_v2(question, correct_answer, best_responses, options=None):
    """
    利用 LLM 对一组候选推理进行评估，判断其中是否有任一步骤包含或逻辑推出了正确答案。
    这里允许 LLM 输出 "yes"、"no" 或 "0.5"（或 "half"）来表示部分正确。
    最终得分为所有有效候选推理的平均得分。

    参数：
    - question (str): 要评估的问题文本。
    - correct_answer (str): 问题的正确答案。
    - best_responses (list): 候选推理列表，每个元素代表一次生成的推理过程。
    - options (可选): 对于选择题，可传入候选项（例如列表或字符串），默认为 None。

    返回：
    - float: 平均得分（介于 0 和 1 之间）。
    """
    
    if not best_responses:
        return 0.0

    # 如果提供了选项，则构造选项字符串
    options_str = f" (Options: {options})" if options else ""

    # 使用新的 prompt 模板（注意：这里提示 LLM 输出 "yes"/"no"/"0.5"）
    prompt_template = (
        "Strictly determine if ANY reasoning step CONTAINS or LOGICALLY LEADS TO the correct answer. Follow these criteria:\n\n"
        "# Judgment Rules (MUST FOLLOW)\n"
        "1. Content Match: Accept different numerical formats (0.5=50%=1/2) or unit variations\n"
        "2. Logical Derivation: Verify if steps mathematically/ logically imply the answer\n"
        "3. Option Substance: For MCQs, match answer CONTENT not just labels (e.g. \"Option B\" vs actual answer text)\n"
        "4. Partial Evidence: Check if key components appear across multiple steps\n"
        "5. Semantic Equivalence: Recognize paraphrased answers with identical meaning\n\n"
        "# Question\n{question}\n\n"
        "# Required Answer\n{correct_answer}{options_str}\n\n"
        "# Candidate Reasoning\n{reasoning_process}\n\n"
        "Just output yes, no, or 0.5 (or 'half') and don't output anything else beside that.\n\n"
        "Final verdict (only 'yes'/'no'/'0.5'):"
        "Output nothing else despite yes, no or 0.5. "
    )

    total_score = 0
    valid_responses = 0
    
    for reasoning in best_responses:
        formatted_prompt = prompt_template.format(
            question=question,
            correct_answer=correct_answer,
            options_str=options_str,
            reasoning_process=reasoning
        )
        
        eval_response = send_openai_prompt(formatted_prompt).strip().lower()
        
        # 判断返回值，并给予相应分数：
        if eval_response in {"yes", "1", "1.0"}:
            score = 1.0
        elif eval_response in {"no", "0", "0.0"}:
            score = 0.0
        elif eval_response in {"0.5", "half"}:
            score = 0.5
        else:
            # 对于不符合预期的输出，忽略该次评分
            continue
        
        total_score += score
        valid_responses += 1

    return total_score / valid_responses if valid_responses > 0 else 0.0


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加载模型和分词器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "OpenAssistant/reward-model-deberta-v3-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def compute_reward_score(question: str, candidate: str) -> float:
    """
    使用 OpenAssistant/reward-model-deberta-v3-large-v2 模型计算奖励分数。
    """
    # 将输入格式化为对话
    inputs = tokenizer(question, candidate, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # 从模型的输出中提取奖励分数
    reward_score = outputs.logits.squeeze().item()
    return reward_score

def evaluate_responses(question: str, best_responses: list) -> float:
    """
    使用奖励模型评估候选回答。
    """
    if not best_responses:
        return 0.0

    total_score = 0.0
    valid_count = 0

    for candidate in best_responses:
        # 计算奖励分数
        score = compute_reward_score(question, candidate)
        total_score += score
        valid_count += 1

    #print(total_score / valid_count if valid_count > 0 else 0.0)
    return total_score / valid_count if valid_count > 0 else 0.0

def compute_reward_score1(question: str, candidate: str, ground_truth: str) -> float:
    """
    使用 OpenAssistant/reward-model-deberta-v3-large 模型计算候选回答与参考答案的匹配度得分。
    """
    # 将输入格式化为对话
    inputs = tokenizer(question, candidate, ground_truth, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # 从模型的输出中提取奖励分数
    reward_score = outputs.logits.squeeze().item()
    return reward_score

def evaluate_responses1(question: str, ground_truth: str, best_responses: list) -> float:
    """
    使用奖励模型评估候选回答与参考答案的匹配度。
    """
    if not best_responses:
        return 0.0

    total_score = 0.0
    valid_count = 0

    for candidate in best_responses:
        # 计算奖励分数
        score = compute_reward_score1(question, candidate, ground_truth)
        total_score += score
        valid_count += 1

    average_score = total_score / valid_count if valid_count > 0 else 0.0
    print(average_score)
    return average_score