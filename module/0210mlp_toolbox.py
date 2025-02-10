import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 用于语义评价的句子嵌入模型
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

class FactorizedAdaptiveContextualMLPAgent:
    def __init__(self, step_lengths, prompts, embedding_dim, hidden_dim=64, learning_rate=0.001):
        """
        参数：
          - step_lengths: 候选步长列表
          - prompts: 候选提示语列表，内部会先按 TF-IDF+PCA 排序
          - embedding_dim: 上下文（任务）嵌入的维度
          - hidden_dim: 隐层维度（这里为共享层的维度）
          - learning_rate: 学习率
        """
        self.step_lengths = step_lengths
        self.prompts = self.rank_prompts_by_tfidf_pca(prompts)  # 预排序，但后续选择不再分段
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # 温度候选值（相对较小）
        self.temperature_options = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # 各因子候选数
        self.num_steps = len(self.step_lengths)
        self.num_prompts = len(self.prompts)
        self.num_temps = len(self.temperature_options)
        
        # --- 新架构：共享层 + 三个头 ---
        # 共享层：将任务 embedding 映射到一个隐藏表示
        self.W_shared = np.random.randn(embedding_dim, hidden_dim) * 0.01
        self.b_shared = np.random.randn(hidden_dim) * 0.01

        # Step Head
        self.W_step = np.random.randn(hidden_dim, self.num_steps) * 0.01
        self.b_step = np.random.randn(self.num_steps) * 0.01

        # Prompt Head
        self.W_prompt = np.random.randn(hidden_dim, self.num_prompts) * 0.01
        self.b_prompt = np.random.randn(self.num_prompts) * 0.01

        # Temperature Head
        self.W_temp = np.random.randn(hidden_dim, self.num_temps) * 0.01
        self.b_temp = np.random.randn(self.num_temps) * 0.01

        self.token_limit = 250
        self.embedding_memory = []  # 用于记录历史嵌入，防止过度重复

        # 用于记录训练日志（例如梯度范数），便于后续画图观察收敛性
        self.training_logs = {
            "training_step": [],
            "reward": [],
            "loss_step": [],
            "loss_prompt": [],
            "loss_temp": []
        }
        self.reward_history = []

    def rank_prompts_by_tfidf_pca(self, prompts):
        """
        用 TF-IDF 提取文本特征，再利用 PCA 降维排序（从发散到严谨）。
        排序后的顺序可以用于后续提示语选择的参考（尽管现在不做分段限制）。
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(prompts)
        pca = PCA(n_components=1)
        pca_embeddings = pca.fit_transform(tfidf_matrix.toarray()).flatten()
        sorted_prompts = [p for _, p in sorted(zip(pca_embeddings, prompts))]
        return sorted_prompts

    def forward(self, x):
        """
        前向传播：先经过共享层，再由各 head 分别输出 logits
        """
        # 共享层
        h = np.tanh(np.dot(x, self.W_shared) + self.b_shared)
        # 各 head 输出
        logits_step = np.dot(h, self.W_step) + self.b_step
        logits_prompt = np.dot(h, self.W_prompt) + self.b_prompt
        logits_temp = np.dot(h, self.W_temp) + self.b_temp
        return h, logits_step, logits_prompt, logits_temp

    def select_action(self, task_embedding, training_step=0, few_shot=True):
        """
        根据当前任务 embedding 分别预测各因子：
          - step_length、prompt 和 temperature 均基于共享层+head输出的 logits，
            使用 Boltzmann Softmax(τ) 采样进行选择（few_shot=True 下主动探索）。
          - 移除了 prompt 的分段限制，所有提示语均可被采样。
        """
        # 归一化任务 embedding
        x = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)
        decay_rate = 0.0001
        temperature_tau = max(1.0 - decay_rate * training_step, 0.1)

        h, logits_step, logits_prompt, logits_temp = self.forward(x)
        
        # Step 选择
        exp_logits_step = np.exp((logits_step - np.max(logits_step)) / temperature_tau)
        probs_step = exp_logits_step / np.sum(exp_logits_step)
        if few_shot:
            step_index = np.random.choice(self.num_steps, p=probs_step)
        else:
            step_index = np.argmax(probs_step)
        chosen_step = self.step_lengths[step_index]

        # Prompt 选择（不再受 step 大小限制，全候选均可）
        exp_logits_prompt = np.exp((logits_prompt - np.max(logits_prompt)) / temperature_tau)
        probs_prompt = exp_logits_prompt / np.sum(exp_logits_prompt)
        if few_shot:
            prompt_index = np.random.choice(self.num_prompts, p=probs_prompt)
        else:
            prompt_index = np.argmax(probs_prompt)
        chosen_prompt = self.prompts[prompt_index]

        # Temperature 选择
        exp_logits_temp = np.exp((logits_temp - np.max(logits_temp)) / temperature_tau)
        probs_temp = exp_logits_temp / np.sum(exp_logits_temp)
        if few_shot:
            temp_index = np.random.choice(self.num_temps, p=probs_temp)
        else:
            temp_index = np.argmax(probs_temp)
        chosen_temp = self.temperature_options[temp_index]

        # 更新 embedding 记忆与 token limit（与 step_length 相关）
        self.embedding_memory.append(x.copy())
        if len(self.embedding_memory) > 100:
            self.embedding_memory.pop(0)
        self.token_limit = min(250 + chosen_step * 20, 500)

        return (chosen_step, chosen_prompt, chosen_temp, self.token_limit)

    def update(self, task_embedding, reward, chosen_action, training_step):
        """
        使用策略梯度更新网络参数：
          - 基于共享层+head的联合结构，计算每个 head 的梯度，
            并将各 head 对共享层的梯度求和后反向传播更新共享参数。
          - 此处对数概率梯度采用形式：dL/dlogits = - norm_reward/ (概率)
          - 同时记录每个 head 梯度范数作为训练日志，便于后续观察收敛性。
        """
        chosen_step, chosen_prompt, chosen_temp, _ = chosen_action
        # 找出各候选在列表中的索引
        step_index = self.step_lengths.index(chosen_step)
        prompt_index = self.prompts.index(chosen_prompt)
        temp_index = int(np.where(self.temperature_options == chosen_temp)[0][0])

        x = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)
        decay_rate = 0.0001
        temperature_tau = max(1.0 - decay_rate * training_step, 0.1)
        
        # 前向传播
        h, logits_step, logits_prompt, logits_temp = self.forward(x)
        # 计算 softmax 概率
        exp_logits_step = np.exp((logits_step - np.max(logits_step)) / temperature_tau)
        probs_step = exp_logits_step / np.sum(exp_logits_step)
        exp_logits_prompt = np.exp((logits_prompt - np.max(logits_prompt)) / temperature_tau)
        probs_prompt = exp_logits_prompt / np.sum(exp_logits_prompt)
        exp_logits_temp = np.exp((logits_temp - np.max(logits_temp)) / temperature_tau)
        probs_temp = exp_logits_temp / np.sum(exp_logits_temp)
        prob_step = probs_step[step_index]
        prob_prompt = probs_prompt[prompt_index]
        prob_temp = probs_temp[temp_index]
        # 组合各部分的概率（假设条件独立）
        chosen_prob = prob_step * prob_prompt * prob_temp

        # 更新奖励归一化（记录最近 100 个 reward）
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        reward_mean = np.mean(self.reward_history)
        reward_std = np.std(self.reward_history) + 1e-8
        norm_reward = (reward - reward_mean) / reward_std

        # 计算各 head 对应的梯度：仅在选中的维度上非零
        dlogits_step = np.zeros_like(logits_step)
        dlogits_step[step_index] = - norm_reward / (prob_step + 1e-8)
        dlogits_prompt = np.zeros_like(logits_prompt)
        dlogits_prompt[prompt_index] = - norm_reward / (prob_prompt + 1e-8)
        dlogits_temp = np.zeros_like(logits_temp)
        dlogits_temp[temp_index] = - norm_reward / (prob_temp + 1e-8)

        # Head 部分梯度
        dW_step = np.outer(h, dlogits_step)
        db_step = dlogits_step
        dW_prompt = np.outer(h, dlogits_prompt)
        db_prompt = dlogits_prompt
        dW_temp = np.outer(h, dlogits_temp)
        db_temp = dlogits_temp

        # 通过各 head 反向传播到共享层（梯度累加）
        dh_step = np.dot(self.W_step, dlogits_step)
        dh_prompt = np.dot(self.W_prompt, dlogits_prompt)
        dh_temp = np.dot(self.W_temp, dlogits_temp)
        dh_total = dh_step + dh_prompt + dh_temp
        # tanh 激活函数的导数：1 - h^2
        dz_shared = dh_total * (1 - h**2)
        dW_shared = np.outer(x, dz_shared)
        db_shared = dz_shared

        # 参数更新（梯度上升以最大化奖励）
        self.W_shared += self.learning_rate * dW_shared
        self.b_shared += self.learning_rate * db_shared
        self.W_step   += self.learning_rate * dW_step
        self.b_step   += self.learning_rate * db_step
        self.W_prompt += self.learning_rate * dW_prompt
        self.b_prompt += self.learning_rate * db_prompt
        self.W_temp   += self.learning_rate * dW_temp
        self.b_temp   += self.learning_rate * db_temp

        # 记录日志：以各 head 的梯度范数作为损失指标（可用于观察收敛）
        loss_step = np.linalg.norm(dW_step)
        loss_prompt = np.linalg.norm(dW_prompt)
        loss_temp = np.linalg.norm(dW_temp)
        self.training_logs["training_step"].append(training_step)
        self.training_logs["reward"].append(reward)
        self.training_logs["loss_step"].append(loss_step)
        self.training_logs["loss_prompt"].append(loss_prompt)
        self.training_logs["loss_temp"].append(loss_temp)

    def plot_convergence(self, save_path=None):
        """
        绘制训练过程中各 head 梯度范数（作为损失）的收敛曲线，
        可用于观察 MLP 网络的训练收敛情况。
        """
        steps = self.training_logs["training_step"]
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.plot(steps, self.training_logs["loss_step"], label="Step Head")
        plt.xlabel("Training Step")
        plt.ylabel("Gradient Norm")
        plt.title("Step Head Convergence")
        plt.legend()

        plt.subplot(1,3,2)
        plt.plot(steps, self.training_logs["loss_prompt"], label="Prompt Head", color='orange')
        plt.xlabel("Training Step")
        plt.ylabel("Gradient Norm")
        plt.title("Prompt Head Convergence")
        plt.legend()

        plt.subplot(1,3,3)
        plt.plot(steps, self.training_logs["loss_temp"], label="Temperature Head", color='green')
        plt.xlabel("Training Step")
        plt.ylabel("Gradient Norm")
        plt.title("Temperature Head Convergence")
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_parameters(self, file_path):
        """
        保存所有参数（共享层和各 head）以及其他辅助变量。
        """
        params = {
            'W_shared': self.W_shared,
            'b_shared': self.b_shared,
            'W_step': self.W_step,
            'b_step': self.b_step,
            'W_prompt': self.W_prompt,
            'b_prompt': self.b_prompt,
            'W_temp': self.W_temp,
            'b_temp': self.b_temp,
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
        """
        从文件中加载参数，恢复网络状态。
        """
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        self.W_shared = params['W_shared']
        self.b_shared = params['b_shared']
        self.W_step = params['W_step']
        self.b_step = params['b_step']
        self.W_prompt = params['W_prompt']
        self.b_prompt = params['b_prompt']
        self.W_temp = params['W_temp']
        self.b_temp = params['b_temp']
        self.step_lengths = params['step_lengths']
        self.prompts = params['prompts']
        self.temperature_options = params['temperature_options']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.learning_rate = params['learning_rate']
        self.token_limit = params['token_limit']
        self.embedding_memory = params['embedding_memory']

