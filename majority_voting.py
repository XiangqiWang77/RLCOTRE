import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
from module.feature_extraction import extract_features
from module.Reasoning_reward import improved_dense_reward, majority_voting_reward
from module.utils import load_questions, send_openai_prompt
from module.new_toolbox import AdaptiveContextualMLPAgent
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

# Load the dataset
all_task = load_questions("./Combine_of_5.json")

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




import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer, util

# 预加载 SBERT 模型
#sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_answer_similarity(candidate_answers):
    """
    计算候选答案之间的语义相似性矩阵
    """
    embeddings = semantic_model.encode(candidate_answers, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()
    return similarity_matrix

def cluster_answers(candidate_answers, threshold=0.8):
    """
    语义聚类：将相似答案归为同一类
    :param candidate_answers: List[str] - 候选答案
    :param threshold: float - 语义相似度阈值
    :return: List[List[str]] - 语义上相似的答案分组
    """
    if len(candidate_answers) == 1:
        return [candidate_answers]
    
    similarity_matrix = compute_answer_similarity(candidate_answers)
    
    # 使用层次聚类，注意这里的距离为 1 - 相似度
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        affinity="precomputed", 
        linkage="average",
        distance_threshold=1 - threshold
    )
    labels = clustering.fit_predict(1 - similarity_matrix)
    
    # 按簇分类答案
    clustered_answers = defaultdict(list)
    for i, label in enumerate(labels):
        clustered_answers[label].append(candidate_answers[i])
    
    return list(clustered_answers.values())

def select_cluster_representative(cluster, question, reward_function=None):
    """
    选择簇内代表答案
    1. 若提供 reward_function，则直接返回簇内得分最高的答案
    2. 否则，采用“medoid”方法，即计算每个答案与其它答案的平均相似度，
       选择中心性最高的答案作为代表
    """
    if reward_function:
        # 计算每个答案的质量评分，选择得分最高的答案作为代表
        scores = [reward_function(ans, question) for ans in cluster]
        best_idx = np.argmax(scores)
        return cluster[best_idx]
    else:
        if len(cluster) == 1:
            return cluster[0]
        # 计算簇内答案的相似度矩阵
        embeddings = semantic_model.encode(cluster, convert_to_tensor=True)
        sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()
        # 计算每个答案与其它答案的平均相似度
        avg_sim = sim_matrix.mean(axis=1)
        best_idx = np.argmax(avg_sim)
        return cluster[best_idx]

def select_best_answer(candidate_answers, question, temperature=0.7, token_limit=512, reward_function=None):
    """
    使用语义聚类 + 加权 Majority Voting 选择最佳答案

    :param candidate_answers: List[str] - 候选答案
    :param question: str - 提出的问题
    :param temperature: float - 控制生成的随机性（目前未用到）
    :param token_limit: int - 生成答案的 token 限制（目前未用到）
    :param reward_function: function - 计算答案质量评分的函数（可选）
    :return: str - 选择的最佳答案
    """
    # 1. 进行语义聚类
    clustered_answers = cluster_answers(candidate_answers, threshold=0.8)
    
    # 2. 对每个簇计算投票权重，并选取簇内代表答案
    cluster_votes = {}
    for cluster in clustered_answers:
        # 选取该簇内最具代表性的答案
        representative_answer = select_cluster_representative(cluster, question, reward_function)
        
        # 计算该簇的权重：如果提供了 reward_function，则使用所有答案的打分和；否则使用答案数量
        if reward_function:
            weight = sum([reward_function(ans, question) for ans in cluster])
        else:
            weight = len(cluster)
        
        cluster_votes[representative_answer] = weight

    # 3. 选择加权得分最高的代表答案
    best_answer = max(cluster_votes, key=cluster_votes.get)

    return best_answer

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
    print("Let's do it.")
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
                reward = majority_voting_reward(question_data['question'], correct_answer=ground_truth, best_responses=best_responses)
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

        ts_model.save_parameters('RL_parameters_whole_MJ.pkl')

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
    output_file="./new_results/0203.json"
)
