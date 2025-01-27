import numpy as np
import json
from collections import OrderedDict
from sentence_transformers import SentenceTransformer

# Initialize semantic model for embedding-based evaluations
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

from sentence_transformers import util
class AdaptiveContextualThompsonSampling:
    def __init__(self, step_lengths, prompts, embedding_dim, beta=1.0, learning_rate=0.5, cache_size=100, exploration_weight=0.1, epsilon=0.2):
        self.step_lengths = step_lengths
        self.prompts = prompts  # Dynamic prompt pool
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.learning_rate = learning_rate
        self.cache_size = cache_size
        self.exploration_weight = exploration_weight
        self.epsilon = epsilon  # ε-greedy exploration

        self.models = OrderedDict()  # Store models for each (step_length, prompt) pair
        self.temperature = 0.7
        self.token_limit = 250

        # Reward tracking for weight selection
        self.rewards = {"correctness": [], "logical_flow": [], "language_quality": []}

    def select_action(self, task_embedding):
        """
        Select the best action based on current exploration vs exploitation.
        Enhanced to ensure more exploration after receiving negative rewards.
        """
        # 强制探索条件：检查最近的奖励记录
        if len(self.rewards["correctness"]) > 5:
            recent_rewards = self.rewards["correctness"][-5:]  # 最近的 5 个 reward
            if all(r == -1 for r in recent_rewards):  # 如果最近 5 次奖励都为 -1
                # 强制探索
                random_step = np.random.choice(self.step_lengths)
                random_prompt = np.random.choice(self.prompts)
                return random_step, random_prompt, self.temperature, self.token_limit

        # epsilon-greedy 探索策略
        if not self.models or np.random.rand() < self.epsilon:
            # 随机探索
            random_step = np.random.choice(self.step_lengths)
            random_prompt = np.random.choice(self.prompts)
            return random_step, random_prompt, self.temperature, self.token_limit

        # 利用（exploitation）：使用 Thompson Sampling
        sampled_rewards = {}
        for key in self.models:
            mu = self.models[key]['mu']
            Sigma = self.models[key]['Sigma']
            # 从分布中采样
            w_sample = np.random.multivariate_normal(mu, Sigma)
            exploration_bonus = self.exploration_weight * np.sqrt(np.diag(Sigma)).mean()
            reward_estimate = np.dot(w_sample, task_embedding) + exploration_bonus
            sampled_rewards[key] = reward_estimate

        # 找到最优 action
        best_action = max(sampled_rewards, key=sampled_rewards.get)
        reward_value = sampled_rewards[best_action]

        # 动态调整温度和 token 限制
        self.temperature = np.clip(0.1 + 0.05 * reward_value, 0.1, 1.0)
        self.token_limit = int(np.clip(200 + 10 * reward_value, 150, 500))

        return best_action[0], best_action[1], self.temperature, self.token_limit


    def select_weights(self, context):
        """
        Dynamically select weights for correctness, logical flow, and language quality.
        """
        selected_weights = {}
        for key, values in self.rewards.items():
            mean_reward = np.mean(values) if values else 0.3  # Default weight if no data
            selected_weights[key] = max(mean_reward, 0.1)  # Ensure minimum weight
        # Normalize weights to sum to 1
        total = sum(selected_weights.values())
        return {k: v / total for k, v in selected_weights.items()}

    def select_best_response(self, candidate_embeddings, task_embedding, step_length, prompt):
        """
        Use TS model to select the most reasonable response among candidates.
        """
        best_index = -1
        best_reward_estimate = -float("inf")

        for idx, candidate_embedding in enumerate(candidate_embeddings):
            # Estimate reward for each candidate
            key = (step_length, prompt)
            if key not in self.models:
                continue
            mu = self.models[key]['mu']
            reward_estimate = np.dot(mu, candidate_embedding)

            if reward_estimate > best_reward_estimate:
                best_reward_estimate = reward_estimate
                best_index = idx

        return best_index, best_reward_estimate

    def update(self, step_length, prompt, temperature, token_limit, task_embedding, reward):
        """
        Update the TS model for the given (step_length, prompt) action.
        """
        key = (step_length, prompt)

        if key not in self.models:
            if len(self.models) >= self.cache_size:
                self.models.popitem(last=False)  # 移除最旧的模型
            self.models[key] = {
                'mu': np.zeros(self.embedding_dim, dtype=np.float32),
                'Sigma': np.eye(self.embedding_dim, dtype=np.float32)
            }

        model = self.models[key]
        mu = model['mu']
        Sigma = model['Sigma']

        # 缩放 reward 到 [-1, 1] 范围
        scaled_reward = (reward - 0.5) * 2 if reward > 0 else reward * 2  # 增强负 reward 的影响
        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_new_inv = Sigma_inv + self.beta * np.outer(task_embedding, task_embedding)
        Sigma_new = np.linalg.inv(Sigma_new_inv)
        mu_new = Sigma_new @ (Sigma_inv @ mu + self.beta * scaled_reward * task_embedding)

        # 使用学习率更新参数
        model['mu'] = (1 - self.learning_rate) * mu + self.learning_rate * mu_new
        model['Sigma'] = Sigma_new

        # 更新奖励记录
        self.rewards["correctness"].append(reward * 0.4)
        self.rewards["logical_flow"].append(reward * 0.3)
        self.rewards["language_quality"].append(reward * 0.3)

        # 减少探索概率，但保持合理下限
        self.epsilon = max(self.epsilon * 0.99, 0.05)
        self.models.move_to_end(key)

    def add_dynamic_prompt(self, new_prompt):
        """
        Dynamically add new prompts to the pool.
        """
        if new_prompt not in self.prompts:
            self.prompts.append(new_prompt)

    def estimate_parameters(self, best_action):
        step_length, prompt = best_action
        return {
            'step_length': step_length,
            'prompt': prompt,
            'temperature': self.temperature,
            'token_limit': self.token_limit
        }

    def probe_correctness_score(self, question, answer):
        """
        Estimate correctness in unsupervised scenarios using semantic similarity.
        """
        embeddings = semantic_model.encode([question, answer])  # semantic_model 已在全局初始化
        similarity = util.cos_sim(embeddings[0], embeddings[1])  # 计算语义相似度
        return similarity.item()
    
    def save_to_json(self, filepath):
        """
        Save model parameters to JSON.
        """
        parameters = {
            key: {
                'mu': self.models[key]['mu'].tolist(),
                'Sigma': self.models[key]['Sigma'].tolist()
            }
            for key in self.models
        }
        with open(filepath, 'w') as f:
            json.dump(parameters, f)

    def load_from_json(self, filepath):
        """
        Load model parameters from JSON.
        """
        with open(filepath, 'r') as f:
            parameters = json.load(f)
        for key in parameters:
            self.models[key] = {
                'mu': np.array(parameters[key]['mu'], dtype=np.float32),
                'Sigma': np.array(parameters[key]['Sigma'], dtype=np.float32)
            }

    def fit_on_datasets(self, datasets):
        """
        Train the model on multiple datasets.
        """
        for dataset in datasets:
            for question_embedding, reward, action in dataset:
                self.update(*action, question_embedding, reward)
