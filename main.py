import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
from module.feature_extraction import extract_features
from module.Reasoning_reward import improved_dense_reward
from module.utils import load_questions, send_openai_prompt
from module.mlp_toolbox import AdaptiveContextualMLPAgent
from module.ts_toolbox import AdaptiveContextualThompsonSampling
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

# Load the dataset
all_task = load_questions("./data/shuffled_combined_HARDMATH.json")

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




class AdaptivePreferenceModel:
    def __init__(self, initial_weights=None):
        self.weights = initial_weights or {
            "LogicalFlaw": 0.35,
            "Coverage": 0.35,
            "Confidence": 0.1,
            "Rationale": 0.1,
        }

    def adjust_weights(self, few_shot_examples):
        if not few_shot_examples:
            return
        avg_scores = {key: np.mean([ex[key] for ex in few_shot_examples]) for key in self.weights}
        total = sum(avg_scores.values())
        for key in self.weights:
            self.weights[key] = avg_scores[key] / total

    def get_weights(self):
        return self.weights


preference_model = AdaptivePreferenceModel()

def select_best_answer(candidate_answers, question, temperature, token_limit, few_shot_examples=None):
    """
    从一组候选答案中挑选最佳答案。评分维度包括：
    - Logical Flaw (逻辑是否严谨、是否存在错误)
    - Coverage (对问题的覆盖度)
    - Confidence (对答案正确性的信心程度)
    - Rationale (解释/推理的质量)

    如果传入了 few_shot_examples，将先用它来调整 preference_model 的权重。
    随后会生成一个评分用的提示（prompt），调用 LLM 为每个答案打分，
    最后根据加权结果，返回分数最高的那个答案。
    """

    # 若提供了 few-shot 示例，用它来调整模型权重
    if few_shot_examples:
        preference_model.adjust_weights(few_shot_examples)
    weights = preference_model.get_weights()
    # 可能返回形如：
    # weights = {
    #     "logical_flaw": 0.25,
    #     "coverage": 0.25,
    #     "confidence": 0.25,
    #     "rationale": 0.25,
    # }

    # 构造给 LLM 的提示信息
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

        # [Answers to Evaluate]
        "Answers:"
    )

    # 将每个候选答案加入提示中
    for idx, answer in enumerate(candidate_answers):
        scoring_prompt += f"Answer {idx + 1}: {answer}\n"

    # 调用 LLM 进行评分
    try:
        response = send_openai_prompt(
            scoring_prompt,
            temperature=temperature,
            token_limit=10000
        )
        print(f"LLM Response: {response}")
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return None

    # 解析 LLM 返回的 JSON
    try:
        scores = json.loads(response)  # 预期得到一个列表，每个元素包含各项评分
        print("scores",scores)
        # 只保留合法的评分项，并构建 answer_idx -> score_dict 的字典
        valid_scores = {}
        for score_entry in scores:
            if ("Answer" in score_entry 
                and isinstance(score_entry["Answer"], int) 
                and 1 <= score_entry["Answer"] <= len(candidate_answers)):
                answer_index = score_entry["Answer"] - 1
                valid_scores[answer_index] = score_entry

        if not valid_scores:
            raise ValueError("No valid scores were extracted from the LLM response.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing scores: {e}")
        print(f"Raw response: {response}")
        return None

    # 针对每个候选答案，根据预先设定的权重加权求和，选出最佳答案
    # 注意：LLM 返回的 JSON 中的键是 "LogicalFlaw", "Coverage", "Confidence", "Rationale"，
    # 而 weights 中的键可能是 "logical_flaw", "coverage", "confidence", "rationale"，
    # 因此需要进行映射以避免 KeyError。
    def map_key_to_response_field(key_name: str) -> str:
        """
        将 weight 中的 key（如 logical_flaw）映射到 LLM JSON 中对应的字段名（如 LogicalFlaw）。
        比如：logical_flaw -> LogicalFlaw, coverage -> Coverage, 等。
        """
        # 先把下划线拆分，再分别首字母大写，最后拼回去
        parts = key_name.split("_")
        return "".join(word.capitalize() for word in parts)
    
    best_answer_idx = max(
        valid_scores.keys(),
        key=lambda idx: sum(
            weights[key] * valid_scores[idx].get(map_key_to_response_field(key), 0.0)
            for key in weights
        ),
    )

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
    shots=40,
    tokenizer=tokenizer,
    hf_model=hf_model,
    max_attempts=3,
    output_file="./results/RLCOTiter.json"
)
