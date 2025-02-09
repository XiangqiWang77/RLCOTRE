import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
from module.feature_extraction import extract_features
from module.Reasoning_reward import evaluate_responses1
from module.utils import load_questions, send_openai_prompt
from module.mmlp_toolbox import FactorizedAdaptiveContextualMLPAgent
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

# Load the dataset
all_task = load_questions("./long_dataset/raw_files/preprocessing/Newdata.json")

embedding_dim = 768
step_lengths = list(range(3, 10))  # 0 to 9

# Prompt pool replaced with a dynamic generator
# Ultimate Enhanced Base Templates for Diverse and Robust Reasoning:
base_templates = [
    "Break down your reasoning into clear, sequential steps: {variation}",
    "Systematically structure your analysis, elaborating on each step with thorough detail: {variation}",
    "Examine the logical connections between concepts and articulate each step in depth: {variation}",
    "Consider multiple perspectives and explore alternative viewpoints comprehensively: {variation}",
    "Apply creative reasoning to unearth unconventional insights and challenge standard assumptions: {variation}",
    "Adopt a detailed and rigorous approach, balancing specific details with overarching themes: {variation}",
    "Reflect on your assumptions and refine your argument through critical self-questioning and validation: {variation}",
    "Explain your reasoning step-by-step in a clear, accessible manner for all audiences: {variation}",
    "Include a systematic self-check and verification of your reasoning process to ensure consistency: {variation}",
    "Conclude by summarizing your key points and re-evaluating your final answer for completeness: {variation}"
]

# Ultimate Enhanced Variations for Maximum Reasoning Diversity:
variations = [
    "Thoroughly analyze all possible interpretations to guarantee a comprehensive understanding.",
    "Decompose the problem into smaller, logical components to enhance clarity and precision.",
    "Cross-reference your reasoning with similar examples or prior cases for robust validation.",
    "Review and double-check each reasoning step to ensure no key detail is overlooked.",
    "Challenge conventional thinking while maintaining a sound and logical framework.",
    "Ensure every premise is clearly understood and meticulously applied.",
    "Pay close attention to minor details that might otherwise be neglected, ensuring depth in your analysis.",
    "Explain your reasoning in simple, straightforward language to guarantee clarity and accessibility.",
    "Perform a detailed self-audit of your reasoning to detect and correct any inconsistencies.",
    "Validate your conclusions by aligning them with established principles or empirical data."
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

few_shot_ts = FactorizedAdaptiveContextualMLPAgent(
    step_lengths,
    dynamic_prompts,
    embedding_dim
)

# Initialize BERT-based model for reasoning evaluation
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
hf_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Prompt tem
prompt_template = (
    "TASK OBJECTIVE:\n"
    "Generate a comprehensive, accurate, and direct answer to the following question. "
    "Your reasoning must be structured in exactly {optimal_steps} clear and sequential steps.\n\n"

    "REASONING GUIDELINES:\n"
    "Utilize the following instructions to construct your reasoning process:\n"
    "{instruction_prompt}\n\n"

    "ANSWER REQUIREMENTS:\n"
    "After developing your reasoning, provide a definitive and explicit answer at the beginning or end of your response. "
    "Under no circumstances should you indicate that you cannot answer the question.\n\n"

    "QUESTION:\n"
    "{question}\n\n"

    "OUTPUT REQUIREMENTS:\n"
    "1. Deliver a single, complete answer that directly meets the question's requirements.\n"
    "2. Follow exactly {optimal_steps} reasoning steps, ensuring all relevant details are covered.\n"
    "3. Ensure that the final answer is clear, accurate, and logically consistent with the question.\n"
    "4. Avoid repetition, ambiguity, or unnecessary verbosity—the response should be unique, concise, and thoroughly reasoned.\n\n"

    "DIVERSITY REQUIREMENTS:\n"
    "Below are some previously generated responses for this question. You must apply divergent reasoning to ensure your answer "
    "is distinct and does not duplicate the following:\n"
    "{previousones}\n"
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


import concurrent.futures
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
                current_action = ts_model.select_action(context,idx,few_shot=True)
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

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    tasks = [executor.submit(send_openai_prompt, formatted_prompt, temperature) for _ in range(5)]
                    best_responses = [future.result() for future in concurrent.futures.as_completed(tasks)]

            # Evaluate and update model with the best response
            if best_responses:
                best_candidate = select_best_answer(
                    best_responses,
                    question_data['question'],
                    temperature,
                    10000
                )
                reward = evaluate_responses1(question_data['question'], ground_truth, best_responses=best_responses)
                #tem_action=step_length, prompt, temperature
                ts_model.update(context, reward, current_action,idx)

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
            current_action = ts_model.select_action(context,idx+shots,few_shot=True)
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
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                tasks = [executor.submit(send_openai_prompt, formatted_prompt, temperature) for _ in range(5)]
                best_responses = [future.result() for future in concurrent.futures.as_completed(tasks)]


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
    output_file="./new_results/0209few.json"
)
