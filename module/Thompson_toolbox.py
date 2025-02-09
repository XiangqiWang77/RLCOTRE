import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

class FactorizedTSAgent:
    def __init__(self, step_lengths, prompts, embedding_dim, lambda_reg=1.0, sigma=1.0):
        """
        参数说明：
            step_lengths: 步长候选列表
            prompts: 提示语候选列表，会通过 TF-IDF+PCA 进行排序
            embedding_dim: 上下文（任务 embedding）的维度
            lambda_reg: 贝叶斯线性回归中 A 矩阵的正则化系数
            sigma: 噪声标准差，用于后验采样
        """
        self.step_lengths = step_lengths
        self.prompts = self.rank_prompts_by_tfidf_pca(prompts)
        self.embedding_dim = embedding_dim
        self.lambda_reg = lambda_reg
        self.sigma = sigma

        # 温度候选集合（较小）
        self.temperature_options = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # 为每个维度分别初始化 TS 模型参数：
        # --- Step_lengths TS 参数
        self.num_steps = len(step_lengths)
        self.A_step = [np.eye(self.embedding_dim) * self.lambda_reg for _ in range(self.num_steps)]
        self.b_step = [np.zeros(self.embedding_dim) for _ in range(self.num_steps)]
        self.counts_step = np.zeros(self.num_steps)

        # --- Prompts TS 参数
        self.num_prompts = len(self.prompts)
        self.A_prompt = [np.eye(self.embedding_dim) * self.lambda_reg for _ in range(self.num_prompts)]
        self.b_prompt = [np.zeros(self.embedding_dim) for _ in range(self.num_prompts)]
        self.counts_prompt = np.zeros(self.num_prompts)

        # --- Temperature TS 参数
        self.num_temps = len(self.temperature_options)
        self.A_temp = [np.eye(self.embedding_dim) * self.lambda_reg for _ in range(self.num_temps)]
        self.b_temp = [np.zeros(self.embedding_dim) for _ in range(self.num_temps)]
        self.counts_temp = np.zeros(self.num_temps)

        self.token_limit = 250
        self.embedding_memory = []
        
        # 用于奖励归一化（可选）
        self.min_reward = float('inf')
        self.max_reward = float('-inf')
        
    def rank_prompts_by_tfidf_pca(self, prompts):
        """
        使用 TF-IDF 计算 prompts 的文本特征，并通过 PCA 降维排序，
        排序顺序从“发散 (divergent)” 到“严谨 (serious)”。
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(prompts)
        pca = PCA(n_components=1)
        pca_embeddings = pca.fit_transform(tfidf_matrix.toarray()).flatten()
        sorted_prompts = [p for _, p in sorted(zip(pca_embeddings, prompts))]
        return sorted_prompts

    def _select_candidate(self, A_list, b_list, counts, x):
        """
        对一个维度的 TS 模型，根据上下文向量 x 选择候选索引：
          - 如果某候选从未被选中过，则赋值 np.inf（促使随机探索）
          - 否则，根据 TS 后验采样计算得分
        """
        scores = []
        for i in range(len(A_list)):
            if counts[i] == 0:
                scores.append(np.inf)
            else:
                A_inv = np.linalg.inv(A_list[i])
                mu = A_inv.dot(b_list[i])
                cov = self.sigma**2 * A_inv
                theta_sample = np.random.multivariate_normal(mu, cov)
                scores.append(np.dot(x, theta_sample))
        # 如果全部候选都是未尝试过的，则随机选一个
        if np.all(np.isinf(scores)):
            return np.random.randint(len(A_list))
        else:
            return np.argmax(scores)

    def select_action(self, task_embedding):
        """
        给定当前任务的上下文 embedding，从三个 TS 模型中分别采样，
        得到 step_length、prompt 和 temperature，再构成最终动作。
        """
        # 归一化上下文向量
        x = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)

        # 分别对三个维度进行 TS 采样
        step_index = self._select_candidate(self.A_step, self.b_step, self.counts_step, x)
        prompt_index = self._select_candidate(self.A_prompt, self.b_prompt, self.counts_prompt, x)
        temp_index = self._select_candidate(self.A_temp, self.b_temp, self.counts_temp, x)

        chosen_step = self.step_lengths[step_index]
        chosen_prompt = self.prompts[prompt_index]
        chosen_temp = self.temperature_options[temp_index]

        # 动态调整 token_limit（例如：随着 step_length 增大 token_limit 增大）
        self.token_limit = min(250 + chosen_step * 20, 500)

        return chosen_step, chosen_prompt, chosen_temp, self.token_limit

    def update(self, task_embedding, reward, chosen_action):
        """
        根据获得的 reward 更新对应维度的 TS 模型参数。
        参数：
          task_embedding: 当前任务的上下文 embedding
          reward: 获得的反馈奖励
          chosen_action: (step_length, prompt, temperature)
        """
        # 更新最小、最大奖励（用于归一化奖励，可选）
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        if self.max_reward > self.min_reward:
            norm_reward = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        else:
            norm_reward = reward

        # 归一化上下文向量
        x = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)

        # 解析 chosen_action 得到各个维度的索引
        chosen_step, chosen_prompt, chosen_temp = chosen_action[:3]
        step_index = self.step_lengths.index(chosen_step)
        prompt_index = self.prompts.index(chosen_prompt)
        temp_index = int(np.where(self.temperature_options == chosen_temp)[0][0])

        # 更新 TS 参数：A <- A + x*x^T, b <- b + norm_reward * x
        self.A_step[step_index] += np.outer(x, x)
        self.b_step[step_index] += norm_reward * x
        self.counts_step[step_index] += 1

        self.A_prompt[prompt_index] += np.outer(x, x)
        self.b_prompt[prompt_index] += norm_reward * x
        self.counts_prompt[prompt_index] += 1

        self.A_temp[temp_index] += np.outer(x, x)
        self.b_temp[temp_index] += norm_reward * x
        self.counts_temp[temp_index] += 1

    def save_parameters(self, file_path):
        params = {
            'step_lengths': self.step_lengths,
            'prompts': self.prompts,
            'temperature_options': self.temperature_options,
            'embedding_dim': self.embedding_dim,
            'lambda_reg': self.lambda_reg,
            'sigma': self.sigma,
            'A_step': self.A_step,
            'b_step': self.b_step,
            'counts_step': self.counts_step,
            'A_prompt': self.A_prompt,
            'b_prompt': self.b_prompt,
            'counts_prompt': self.counts_prompt,
            'A_temp': self.A_temp,
            'b_temp': self.b_temp,
            'counts_temp': self.counts_temp,
            'token_limit': self.token_limit,
            'embedding_memory': self.embedding_memory,
            'min_reward': self.min_reward,
            'max_reward': self.max_reward,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)

    def load_parameters(self, file_path):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        self.step_lengths = params['step_lengths']
        self.prompts = params['prompts']
        self.temperature_options = params['temperature_options']
        self.embedding_dim = params['embedding_dim']
        self.lambda_reg = params['lambda_reg']
        self.sigma = params['sigma']
        self.A_step = params['A_step']
        self.b_step = params['b_step']
        self.counts_step = params['counts_step']
        self.A_prompt = params['A_prompt']
        self.b_prompt = params['b_prompt']
        self.counts_prompt = params['counts_prompt']
        self.A_temp = params['A_temp']
        self.b_temp = params['b_temp']
        self.counts_temp = params['counts_temp']
        self.token_limit = params['token_limit']
        self.embedding_memory = params['embedding_memory']
        self.min_reward = params['min_reward']
        self.max_reward = params['max_reward']
