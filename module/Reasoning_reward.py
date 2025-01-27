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
