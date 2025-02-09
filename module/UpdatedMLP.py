import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# 用于语义评价的句子嵌入模型
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

class FactorizedAdaptiveContextualMLPAgent:
    def __init__(self, step_lengths, prompts, embedding_dim, hidden_dim=64, learning_rate=0.001):
        """
        参数：
          - step_lengths: 候选步长列表
          - prompts: 候选提示语列表，内部会先按 TF-IDF+PCA 排序
          - embedding_dim: 上下文（任务）嵌入的维度
          - hidden_dim: 隐层维度
          - learning_rate: 学习率
        """
        self.step_lengths = step_lengths
        self.prompts = self.rank_prompts_by_tfidf_pca(prompts)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # 温度候选值（相对较小）
        self.temperature_options = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # 各因子候选数
        self.num_steps = len(self.step_lengths)
        self.num_prompts = len(self.prompts)
        self.num_temps = len(self.temperature_options)
        
        # 为 step_length 建立一个两层 MLP
        self.W1_step = np.random.randn(embedding_dim, hidden_dim) * 0.01
        self.b1_step = np.random.randn(hidden_dim) * 0.01
        self.W2_step = np.random.randn(hidden_dim, self.num_steps) * 0.01
        self.b2_step = np.random.randn(self.num_steps) * 0.01

        # 为 prompt 建立一个两层 MLP
        self.W1_prompt = np.random.randn(embedding_dim, hidden_dim) * 0.01
        self.b1_prompt = np.random.randn(hidden_dim) * 0.01
        self.W2_prompt = np.random.randn(hidden_dim, self.num_prompts) * 0.01
        self.b2_prompt = np.random.randn(self.num_prompts) * 0.01

        # 为 temperature 建立一个两层 MLP
        self.W1_temp = np.random.randn(embedding_dim, hidden_dim) * 0.01
        self.b1_temp = np.random.randn(hidden_dim) * 0.01
        self.W2_temp = np.random.randn(hidden_dim, self.num_temps) * 0.01
        self.b2_temp = np.random.randn(self.num_temps) * 0.01

        self.token_limit = 250
        self.embedding_memory = []  # 用于记录历史嵌入，防止过度重复

    def rank_prompts_by_tfidf_pca(self, prompts):
        """
        先用 TF-IDF 提取文本特征，再利用 PCA 降维排序（从发散到严谨）。
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(prompts)
        pca = PCA(n_components=1)
        pca_embeddings = pca.fit_transform(tfidf_matrix.toarray()).flatten()
        sorted_prompts = [p for _, p in sorted(zip(pca_embeddings, prompts))]
        return sorted_prompts

    # -------------------------------
    # 每个因子单独的前向传播
    # -------------------------------
    def forward_step(self, embedding):
        z = np.dot(embedding, self.W1_step) + self.b1_step
        a = np.tanh(z)
        logits = np.dot(a, self.W2_step) + self.b2_step
        return logits

    def forward_prompt(self, embedding):
        z = np.dot(embedding, self.W1_prompt) + self.b1_prompt
        a = np.tanh(z)
        logits = np.dot(a, self.W2_prompt) + self.b2_prompt
        return logits

    def forward_temp(self, embedding):
        z = np.dot(embedding, self.W1_temp) + self.b1_temp
        a = np.tanh(z)
        logits = np.dot(a, self.W2_temp) + self.b2_temp
        return logits

    # -------------------------------
    # 动作选择
    # -------------------------------
    def select_action(self, task_embedding, training_step=0,few_shot=True):
        """
        根据当前任务 embedding 分别预测各因子：
          - step_length：通过 MLP 得到 logits 后用 Boltzmann Softmax(τ) 采样；
          - prompt：根据 step_length 决定使用前半（发散）或后半（严谨）的候选，再采样；
          - temperature：独立采样。
        """
        # 归一化上下文向量
        x = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)
        decay_rate = 0.0001
        temperature_tau = max(1.0 - decay_rate * training_step, 0.1)

        # --- Step 选择 ---
        logits_step = self.forward_step(x)
        probs_step = np.exp((logits_step - np.max(logits_step)) / temperature_tau)
        probs_step /= np.sum(probs_step)
        if few_shot:
            step_index = np.random.choice(self.num_steps, p=probs_step)
        else:
            step_index = np.argmax(probs_step)
        chosen_step = self.step_lengths[step_index]

        # --- Prompt 选择 ---
        # 根据 step 的大小决定使用哪一部分的 prompt（耦合逻辑）
        if chosen_step > 6:
            valid_indices = list(range(self.num_prompts // 2, self.num_prompts))
        else:
            valid_indices = list(range(0, self.num_prompts // 2))
        logits_prompt = self.forward_prompt(x)
        valid_logits = logits_prompt[valid_indices]
        probs_prompt = np.exp((valid_logits - np.max(valid_logits)) / temperature_tau)
        probs_prompt /= np.sum(probs_prompt)
        if few_shot:
            rel_prompt_index = np.random.choice(len(valid_indices), p=probs_prompt)
        else:
            rel_prompt_index = np.argmax(probs_prompt)
        prompt_index = valid_indices[rel_prompt_index]
        chosen_prompt = self.prompts[prompt_index]

        # --- Temperature 选择 ---
        logits_temp = self.forward_temp(x)
        probs_temp = np.exp((logits_temp - np.max(logits_temp)) / temperature_tau)
        probs_temp /= np.sum(probs_temp)
        if few_shot:
            temp_index = np.random.choice(self.num_temps, p=probs_temp)
        else:
            temp_index = np.argmax(probs_temp)
        chosen_temp = self.temperature_options[temp_index]

        # 更新嵌入记忆（可用于多样性惩罚）
        self.embedding_memory.append(x.copy())
        if len(self.embedding_memory) > 100:
            self.embedding_memory.pop(0)

        # 根据 step_length 动态调整 token_limit
        self.token_limit = min(250 + chosen_step * 20, 500)

        return chosen_step, chosen_prompt, chosen_temp, self.token_limit

    # -------------------------------
    # 策略更新（基于简单的策略梯度）
    # -------------------------------
    def update(self, task_embedding, reward, chosen_action, training_step):
        """
        分别对三个因子的 MLP 进行基于策略梯度的更新。
        这里采用对数概率的梯度（归一化奖励已计算）进行反向传播更新。
        """
        # 解析动作：注意 chosen_action 包含 (step_length, prompt, temperature, token_limit)
        chosen_step, chosen_prompt, chosen_temp, _ = chosen_action
        step_index = self.step_lengths.index(chosen_step)
        prompt_index = self.prompts.index(chosen_prompt)
        temp_index = int(np.where(self.temperature_options == chosen_temp)[0][0])

        x = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)
        decay_rate = 0.0001
        temperature_tau = max(1.0 - decay_rate * training_step, 0.1)

        # 计算各因子的预测概率
        # Step
        logits_step = self.forward_step(x)
        probs_step = np.exp((logits_step - np.max(logits_step)) / temperature_tau)
        probs_step /= np.sum(probs_step)
        prob_step = probs_step[step_index]

        # Prompt：根据 step 决定 valid 范围
        if chosen_step > 6:
            valid_indices = list(range(self.num_prompts // 2, self.num_prompts))
        else:
            valid_indices = list(range(0, self.num_prompts // 2))
        logits_prompt = self.forward_prompt(x)
        valid_logits = logits_prompt[valid_indices]
        probs_prompt = np.exp((valid_logits - np.max(valid_logits)) / temperature_tau)
        probs_prompt /= np.sum(probs_prompt)
        rel_prompt_index = valid_indices.index(prompt_index)
        prob_prompt = probs_prompt[rel_prompt_index]

        # Temperature
        logits_temp = self.forward_temp(x)
        probs_temp = np.exp((logits_temp - np.max(logits_temp)) / temperature_tau)
        probs_temp /= np.sum(probs_temp)
        prob_temp = probs_temp[temp_index]

        # 组合各部分的概率（假设独立）
        chosen_prob = prob_step * prob_prompt * prob_temp

        # 更新奖励归一化（记录最近 100 个 reward）
        if not hasattr(self, "reward_history"):
            self.reward_history = []
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        reward_mean = np.mean(self.reward_history)
        reward_std = np.std(self.reward_history) + 1e-8
        norm_reward = (reward - reward_mean) / reward_std

        # 策略梯度：对于选中的输出，梯度大致为 -norm_reward/概率
        # --- 更新 step MLP ---
        dL_dlogit_step = np.zeros_like(logits_step)
        dL_dlogit_step[step_index] = - norm_reward / (prob_step + 1e-8)
        z1_step = np.dot(x, self.W1_step) + self.b1_step
        a_step = np.tanh(z1_step)
        dW2_step = np.outer(a_step, dL_dlogit_step)
        db2_step = dL_dlogit_step
        da_step = np.dot(self.W2_step, dL_dlogit_step)
        dz1_step = da_step * (1 - a_step**2)
        dW1_step = np.outer(x, dz1_step)
        db1_step = dz1_step
        self.W1_step += self.learning_rate * dW1_step
        self.b1_step += self.learning_rate * db1_step
        self.W2_step += self.learning_rate * dW2_step
        self.b2_step += self.learning_rate * db2_step

        # --- 更新 prompt MLP ---
        dL_dlogit_prompt = np.zeros(self.num_prompts)
        dL_dlogit_prompt[prompt_index] = - norm_reward / (prob_prompt + 1e-8)
        z1_prompt = np.dot(x, self.W1_prompt) + self.b1_prompt
        a_prompt = np.tanh(z1_prompt)
        dW2_prompt = np.outer(a_prompt, dL_dlogit_prompt)
        db2_prompt = dL_dlogit_prompt
        da_prompt = np.dot(self.W2_prompt, dL_dlogit_prompt)
        dz1_prompt = da_prompt * (1 - a_prompt**2)
        dW1_prompt = np.outer(x, dz1_prompt)
        db1_prompt = dz1_prompt
        self.W1_prompt += self.learning_rate * dW1_prompt
        self.b1_prompt += self.learning_rate * db1_prompt
        self.W2_prompt += self.learning_rate * dW2_prompt
        self.b2_prompt += self.learning_rate * db2_prompt

        # --- 更新 temperature MLP ---
        dL_dlogit_temp = np.zeros_like(logits_temp)
        dL_dlogit_temp[temp_index] = - norm_reward / (prob_temp + 1e-8)
        z1_temp = np.dot(x, self.W1_temp) + self.b1_temp
        a_temp = np.tanh(z1_temp)
        dW2_temp = np.outer(a_temp, dL_dlogit_temp)
        db2_temp = dL_dlogit_temp
        da_temp = np.dot(self.W2_temp, dL_dlogit_temp)
        dz1_temp = da_temp * (1 - a_temp**2)
        dW1_temp = np.outer(x, dz1_temp)
        db1_temp = dz1_temp
        self.W1_temp += self.learning_rate * dW1_temp
        self.b1_temp += self.learning_rate * db1_temp
        self.W2_temp += self.learning_rate * dW2_temp
        self.b2_temp += self.learning_rate * db2_temp

    # -------------------------------
    # 参数存储与加载
    # -------------------------------
    def save_parameters(self, file_path):
        params = {
            'W1_step': self.W1_step,
            'b1_step': self.b1_step,
            'W2_step': self.W2_step,
            'b2_step': self.b2_step,
            'W1_prompt': self.W1_prompt,
            'b1_prompt': self.b1_prompt,
            'W2_prompt': self.W2_prompt,
            'b2_prompt': self.b2_prompt,
            'W1_temp': self.W1_temp,
            'b1_temp': self.b1_temp,
            'W2_temp': self.W2_temp,
            'b2_temp': self.b2_temp,
            'step_lengths': self.step_lengths,
            'prompts': self.prompts,
            'temperature_options': self.temperature_options,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'token_limit': self.token_limit,
            'embedding_memory': self.embedding_memory
        }
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)

    def load_parameters(self, file_path):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        self.W1_step = params['W1_step']
        self.b1_step = params['b1_step']
        self.W2_step = params['W2_step']
        self.b2_step = params['b2_step']
        self.W1_prompt = params['W1_prompt']
        self.b1_prompt = params['b1_prompt']
        self.W2_prompt = params['W2_prompt']
        self.b2_prompt = params['b2_prompt']
        self.W1_temp = params['W1_temp']
        self.b1_temp = params['b1_temp']
        self.W2_temp = params['W2_temp']
        self.b2_temp = params['b2_temp']
        self.step_lengths = params['step_lengths']
        self.prompts = params['prompts']
        self.temperature_options = params['temperature_options']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.learning_rate = params['learning_rate']
        self.token_limit = params['token_limit']
        self.embedding_memory = params['embedding_memory']
