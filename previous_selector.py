import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
from module.feature_extraction import extract_features
from module.Reasoning_reward import improved_dense_reward
from module.utils import load_questions, send_openai_prompt
from module.new_toolbox import AdaptiveContextualMLPAgent
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

# Load the dataset
all_task = load_questions("./long_dataset/raw_files/preprocessing/Finalcombined.json")

embedding_dim = 768
step_lengths = list(range(0, 10))  # 0 to 9

# Prompt pool replaced with a dynamic generator
base_templates = [
    "Try reasoning step by step and explain your thought process clearly: {variation}",
    "Focus on a structured reasoning approach and explain each step in detail: {variation}",
    "Think deeply about the connections between ideas and articulate them: {variation}",
    "Approach the problem broadly and consider multiple perspectives: {variation}",
    "Use creative reasoning to explore unconventional solutions: {variation}",
    "Adopt a meticulous approach, considering both details and broader implications: {variation}",
    "Engage in reflective reasoning by questioning your assumptions and conclusions: {variation}",
    "Use an exploratory approach, considering new angles or possibilities: {variation}"
]

variations = [
    "Be meticulous in considering all possible interpretations of the information.",
    "Break down the problem into smaller, manageable components to reason effectively.",
    "Evaluate your reasoning by comparing it with similar examples or past cases.",
    "Be open to challenging your assumptions and refine your argument accordingly.",
    "Try reasoning more creatively by thinking outside the conventional framework.",
    "Consider how different premises might lead to alternative conclusions.",
    "Explore how subtle details might influence the overall reasoning process.",
    "Test your reasoning by explaining it as if to someone unfamiliar with the topic."
]



def prompt_generator(base_templates, variations):
    """
    Generate diverse prompts using templates and variations.
    """
    generated_prompts = []
    for template in base_templates:
        for variation in variations:
            generated_prompts.append(template.format(variation=variation))
    return generated_prompts

dynamic_prompts = prompt_generator(base_templates, variations)

few_shot_ts = AdaptiveContextualMLPAgent(
    step_lengths,
    dynamic_prompts,
    embedding_dim
)

# Initialize BERT-based model for reasoning evaluation
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
hf_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Prompt tem
prompt_template = (
    # [Objective]
    "Generate a single distinct and diverse answer to the following question. The reasoning must strictly follow {optimal_steps} reasoning steps—no more, no less.\n\n"

    # [Reasoning Instructions]
    "You must strictly follow this reasoning process: {instruction_prompt}.\n\n"

    # [Question]
    "Question: {question}\n\n"

    # [Output Requirements]
    "Output Requirements:"
    "1. Provide a single answer that fully complies with the question's requirements."
    "2. The reasoning must align with the specified steps and fulfill all necessary details of the question."
    "3. Ensure the answer is clear, accurate, and aligns with the question requirements."
    "4. Avoid repetitive or ambiguous content; ensure the response is distinct and well-reasoned."

    "Diversity Requirements:"
    "Provided below are some previous generated content on this question, you must reason divergently and try your best not to generate duplicate answer and process. Make the final answer as diverse and different. "
    "You need to try your best to avoid concluding same result wih the previous generated content and provide new content."
    "Preious contents: {previousones}"
)




def extract_embeddings_and_params(model):
    embeddings = []
    step_lengths = []
    prompts = []
    temperatures = []
    token_limits = []

    for (step_length, prompt), data in model.models.items():
        embeddings.append(data['mu'])
        step_lengths.append(step_length)
        prompts.append(hash(prompt) % 100)  # Map prompt to unique ID
        temperatures.append(model.temperature)
        token_limits.append(model.token_limit)

    return np.array(embeddings), step_lengths, prompts, temperatures, token_limits

def plot_kde_heatmaps(embeddings, param, title):
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
    else:
        reduced_embeddings = embeddings

    x, y = reduced_embeddings[:, 0], reduced_embeddings[:, 1]

    # Kernel Density Estimation
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, weights=param)  # Weighted estimation
    z = kde(xy)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c=z, cmap='viridis', s=100)
    plt.colorbar(label=f'{title}')
    plt.title(f'{title} Heatmap (KDE)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()




from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Initialize semantic model for embedding-based evaluations
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize HuggingFace model for logical evaluation
logic_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
logic_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")




import torch
import torch.nn as nn
import torch.optim as optim
import json
from transformers import AutoModel, AutoTokenizer
from typing import List

class AdaptivePreferenceModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_weights=4, learning_rate=1e-4):
        """
        使用深度学习模型，根据 question 和 candidate_answers 动态选择权重层
        - model_name: 预训练 Transformer 模型 (BERT, RoBERTa)
        - num_weights: 评分维度的数量 (LogicalFlaw, Coverage, Confidence, Rationale)
        """
        super(AdaptivePreferenceModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 线性层用于预测最佳 weight vector
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_weights)  # 输出 4 个权重
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # 采用均方误差损失

    def forward(self, question: str, candidate_answers: List[str]) -> torch.Tensor:
        """
        输入：
        - question: 问题文本 (str)
        - candidate_answers: 候选答案列表 (List[str])
        
        输出：
        - 预测的最佳权重向量 (Tensor)
        """
        # 编码问题文本
        question_inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True)
        question_embedding = self.encoder(**question_inputs).last_hidden_state[:, 0, :]  # 取 CLS token 作为表示

        # 编码所有候选答案
        response_embeddings = []
        for response in candidate_answers:
            response_inputs = self.tokenizer(response, return_tensors="pt", padding=True, truncation=True)
            response_embedding = self.encoder(**response_inputs).last_hidden_state[:, 0, :]
            response_embeddings.append(response_embedding)
        
        # 计算 candidate_answers 的平均 embedding
        response_tensor = torch.stack(response_embeddings).mean(dim=0)

        # 合并 question + response 作为最终特征
        combined_features = torch.cat((question_embedding, response_tensor), dim=-1)

        # 通过 MLP 预测最佳权重向量
        predicted_weights = self.fc(combined_features)
        predicted_weights = torch.softmax(predicted_weights, dim=-1)  # 归一化成概率分布

        return predicted_weights
    
    def update(self, reward: float, question: str, candidate_answers: List[str]):
        """
        使用给定的 reward 更新模型参数。
        - reward: 反馈分数 (float)
        - question: 问题文本 (str)
        - candidate_answers: 候选答案列表 (List[str])
        """
        self.train()
        self.optimizer.zero_grad()
        
        # 预测权重
        predicted_weights = self.forward(question, candidate_answers)
        
        # 创建目标值 (理想情况下，我们希望模型的预测权重与 reward 成正比)
        target_weights = torch.full_like(predicted_weights, reward)
        
        # 计算损失并更新模型
        loss = self.criterion(predicted_weights, target_weights)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# 初始化模型
preference_model = AdaptivePreferenceModel()

def select_best_answer(candidate_answers, question, temperature, token_limit):
    """
    选择最佳答案：
    1. 让模型自动推导问题类型，并选择最优的权重层。
    2. 结合 response 评分优化权重。
    3. 计算加权得分，返回最优答案。
    """

    # 让模型自动预测最佳权重
    # 让模型生成 (num_candidates, 4) 形状的权重
    weights = preference_model(question, candidate_answers).detach().numpy()

    # 确保 weights 的行数和 candidate_answers 对齐
    if weights.shape[0] != len(candidate_answers):
        weights = np.tile(weights, (len(candidate_answers), 1))  # 复制到正确形状

    print("Selected weight",weights)

    # 构造 LLM 评分提示
    #scoring_prompt = f"Question: {question}\n\n"
    scoring_prompt = (
        # [Objective]
        f"Question: {question}\n\n"
        "Your task is to evaluate the given answers based on specific scoring dimensions and provide a plain-text output in a JSON-like format (not actual JSON).\n\n"

        # [Scoring Dimensions]
        "Scoring Dimensions:"
        "1. **Logical Flaw**: Rate the logical consistency and absence of errors in the answer (0 to 1)."
        "2. **Coverage**: Assess how well the answer addresses all relevant aspects of the question (0 to 1)."
        "3. **Confidence**: Evaluate the confidence level of the correctness of the answer (0 to 1)."
        "4. **Rationale**: Judge the quality of reasoning and explanation provided in the answer (0 to 1).\n\n"

        # [Output Requirements]
        "Output Requirements:"
        "1. Provide scores for each dimension for all answers."
        "2. Combine the scores to identify the best answer."
        "3. Output must be plain text formatted like JSON but not actual JSON code."
        "4. Don't generate anything other than like the example output"
        "4. Use this format for the output:"
        "["
        "  {\"Answer\": 1, \"LogicalFlaw\": 0.9, \"Coverage\": 0.85, \"Confidence\": 0.9, \"Rationale\": 0.95},"
        "  {\"Answer\": 2, \"LogicalFlaw\": 0.7, \"Coverage\": 0.8, \"Confidence\": 0.75, \"Rationale\": 0.8}"
        "]\n\n"

        # [Prohibited Actions]
        "Prohibited Actions:"
        "1. Do not provide real JSON code."
        "2. Do not deviate from the required scoring dimensions or format."
        "3. Ensure the output is plain text that looks like JSON, but never in actual JSON syntax.\n\n"
        "4. Don't generate anything else other than the json like results."

        # [Answers to Evaluate]
        "Answers:"
    )

    for idx, answer in enumerate(candidate_answers):
        scoring_prompt += f"Answer {idx + 1}: {answer}\n"

    # 3. **调用 LLM 进行评分**
    try:
        response = send_openai_prompt(
            scoring_prompt,
            temperature=temperature,
            token_limit=token_limit
        )
        print(f"LLM Response: {response}")
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return candidate_answers[0]  # 失败时返回默认答案

    try:
        # 解析 LLM JSON 响应
        scores = json.loads(response)

        # 构建有效评分字典
        valid_scores = {}
        for entry in scores:
            if "Answer" in entry and isinstance(entry["Answer"], int):
                answer_idx = entry["Answer"] - 1  # 转换为 0-based 索引
                if 0 <= answer_idx < len(candidate_answers):  # 确保索引合法
                    valid_scores[answer_idx] = entry

        if not valid_scores:
            raise ValueError("No valid scores extracted.")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing scores: {e}")
        return candidate_answers[0]  # 解析失败返回默认答案

    # 定义 key 映射函数
    def map_key_to_response_field(key_name: str) -> str:
        return "".join(word.capitalize() for word in key_name.split("_"))

    # 计算最佳答案索引
    best_answer_idx = max(
        valid_scores.keys(),
        key=lambda idx: sum(
            weights[idx][i] * valid_scores[idx].get(map_key_to_response_field(k), 0.0)
            for i, k in enumerate(["LogicalFlaw", "Coverage", "Confidence", "Rationale"])
        ),
    )

    # 返回最佳答案
    return candidate_answers[best_answer_idx]






class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle non-serializable types such as numpy types.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def evaluate_few_shot_with_multiple_responses(ts_model, task, shots, tokenizer, hf_model, max_attempts=10, output_file="results.json"):
    """
    Evaluate using Thompson Sampling with iterative response generation, selection, and hyperparameter tuning.
    After few-shot evaluation, apply the trained model to the full dataset.
    """
    correct_counts = []

    with open(output_file, "w") as f:
        f.write("[\n")

        # Few-shot training and evaluation
        for idx, question_data in enumerate(task[:shots]):
            context = extract_features(question_data['question'])
            ground_truth = question_data['answer']
            best_reward = -999  # Initialize reward as -1
            attempts = 0
            best_responses = []  # Store generated responses iteratively
            current_action = None

            while len(best_responses) < 5:
                attempts += 1

                # Select action
                current_action = ts_model.select_action(context)
                step_length, prompt, temperature, token_limit = current_action

                print("Current action:", current_action)

                # Generate the prompt using TS-selected instruction_prompt and optimal_steps
                #best_responses = []
                # Generate the prompt using TS-selected instruction_prompt and optimal_steps
                formatted_prompt = prompt_template.format(
                    question=question_data['question'],
                    instruction_prompt=prompt,
                    optimal_steps=step_length,
                    previousones=best_responses
                )

                try:
                    # Generate the next answer iteratively
                    new_answer = send_openai_prompt(
                        formatted_prompt,
                        temperature=temperature,
                        token_limit=10000
                    )
                    if new_answer not in best_responses:
                        best_responses.append(new_answer)

                    print(f"Generated {len(best_responses)} answers so far.")
                except ValueError as e:
                    print(f"Error in generating answers: {e}")

            # Evaluate and update model with the best response
            if best_responses:
                best_candidate = select_best_answer(
                    best_responses,
                    question_data['question'],
                    temperature,
                    10000
                )
                reward = improved_dense_reward(question_data['question'], ground_truth, best_candidate)
                ts_model.update(context, reward, current_action)
                preference_model.update(reward, question_data['question'], best_responses)

                print(f"Few-shot Question {idx + 1}: Best Reward: {reward}")
                print(f"Best Responses: {best_responses}")

                # Save few-shot results
                result = {
                    "question": question_data['question'],
                    "ground_truth": question_data['answer'],
                    "best_responses": best_responses,
                    "best_candidate": best_candidate,
                    "best_reward": float(reward),
                    "attempts": int(attempts),
                    "current_action": [
                        int(current_action[0]) if isinstance(current_action[0], np.integer) else current_action[0],
                        str(current_action[1]),
                        float(current_action[2]),
                        int(current_action[3])
                    ]
                }
                f.write(json.dumps(result, indent=4, cls=NumpyEncoder))
                if idx < len(task[:shots]) - 1 or shots < len(task):
                    f.write(",\n")

        print("Few-shot training completed.")

        ts_model.save_parameters('RL_parameters.pkl')

        # Apply to the full dataset
        print("Applying to the full dataset...")
        for idx, question_data in enumerate(task[shots:], start=shots):
            context = extract_features(question_data['question'])

            # Use the trained TS model to select action
            current_action = ts_model.select_action(context)
            step_length, prompt, temperature, token_limit = current_action

            best_responses = []
            # Generate the prompt using TS-selected instruction_prompt and optimal_steps
            formatted_prompt = prompt_template.format(
                question=question_data['question'],
                instruction_prompt=prompt,
                optimal_steps=step_length,
                previousones=best_responses
            )

            # Generate multiple candidate answers
            
            while len(best_responses) < 5:
                try:
                    new_answer = send_openai_prompt(
                        formatted_prompt,
                        temperature=temperature,
                        token_limit=10000
                    )
                    if new_answer not in best_responses:
                        best_responses.append(new_answer)

                    print("best response",best_responses)
                except ValueError as e:
                    print(f"Error in generating full dataset answers: {e}")
                    break

            if best_responses:
                # Use the best answer selection logic
                best_candidate = select_best_answer(
                    best_responses,
                    question_data['question'],
                    temperature,
                    10000
                )


                result = {
                    "question": question_data['question'],
                    "ground_truth": question_data['answer'],
                    "best_responses": best_responses,
                    "best_candidate": best_candidate,
                    "current_action": [
                        int(current_action[0]) if isinstance(current_action[0], np.integer) else current_action[0],
                        str(current_action[1]),
                        float(current_action[2]),
                        int(current_action[3])
                    ]
                }

                print("Result:", result)
                f.write(json.dumps(result, indent=4, cls=NumpyEncoder))
                if idx < len(task) - 1:
                    f.write(",\n")

        f.write("\n]")  # Close the JSON array

    print("Full dataset application completed.")





# Run evaluation
accuracy = evaluate_few_shot_with_multiple_responses(
    ts_model=few_shot_ts,
    task=all_task,
    shots=100,
    tokenizer=tokenizer,
    hf_model=hf_model,
    max_attempts=3,
    output_file="./new_results/RLCOT.json"
)
