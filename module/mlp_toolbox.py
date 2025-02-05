import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Initialize semantic model for embedding-based evaluations
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

import numpy as np

class AdaptiveContextualMLPAgent:
    def __init__(self, step_lengths, prompts, embedding_dim, hidden_dim=64, learning_rate=0.01, epsilon=0.2):
        self.step_lengths = step_lengths
        self.prompts = prompts  # 动态 prompt 池
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # 初始探索率

        # 定义温度候选值（离散选择）
        self.temperature_options = np.array([0.2, 0.4, 0.6, 0.8, 1.0])  # 低温更 exploitation，高温更 exploration

        # 初始化 MLP 参数
        num_actions = len(step_lengths) * len(prompts) * len(self.temperature_options)  # 包括 temperature
        self.W1 = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.01
        self.b1 = np.random.randn(self.hidden_dim) * 0.01
        self.W2 = np.random.randn(self.hidden_dim, num_actions) * 0.01
        self.b2 = np.random.randn(num_actions) * 0.01

        self.token_limit = 250
        self.embedding_memory = []  # 存储 embeddings 以便惩罚重复

    def forward(self, embedding):
        """前向传播计算 logits"""
        z1 = np.dot(embedding, self.W1) + self.b1
        a1 = np.tanh(z1)  # 使用 tanh 激活
        z2 = np.dot(a1, self.W2) + self.b2
        return z2  # 返回 logits

    def select_action(self, task_embedding):
        """
        选择动作（step_length, prompt, temperature），
        结合温度退火、多样性惩罚和动态探索策略。
        """
        # 归一化嵌入以提高稳定性
        task_embedding = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)

        # 计算 logits
        logits = self.forward(task_embedding)

        # 应用多样性惩罚：仅考虑最近 100 个嵌入
        if self.embedding_memory:
            recent_embeddings = self.embedding_memory[-100:]  
            similarities = np.dot(recent_embeddings, task_embedding)
            penalties = np.mean(similarities) * 0.2  
            logits -= penalties  # 惩罚相似的 embeddings

        # 计算 softmax 概率
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= np.sum(probabilities)

        # Epsilon-greedy 探索
        if np.random.rand() < self.epsilon:
            action_index = np.random.choice(len(probabilities))
        else:
            action_index = np.argmax(probabilities)

        # 解析 action_index 对应的 step_length, prompt, temperature
        num_step = len(self.step_lengths)
        num_prompt = len(self.prompts)
        num_temp = len(self.temperature_options)

        step_index, remainder = divmod(action_index, num_prompt * num_temp)
        prompt_index, temp_index = divmod(remainder, num_temp)

        step_length = self.step_lengths[step_index]
        prompt = self.prompts[prompt_index]
        temperature = self.temperature_options[temp_index]  # 从候选温度选择

        # 记录 embedding 以保持多样性
        self.embedding_memory.append(task_embedding.copy())
        if len(self.embedding_memory) > 100:
            self.embedding_memory.pop(0)

        # 动态调整 token_limit
        self.token_limit = min(250 + step_length * 20, 500)

        return step_length, prompt, temperature, self.token_limit
    def update(self, task_embedding, reward, chosen_action):
        """
        基于奖励更新模型参数，引入动量优化和策略梯度。
        """
        # 解析动作
        step_length, prompt, temperature = chosen_action  # 现在包含 temperature

        # 找到 action_index
        step_index = self.step_lengths.index(step_length)
        prompt_index = self.prompts.index(prompt)
        temp_index = np.where(self.temperature_options == temperature)[0][0]  # 找到 temperature 索引

        action_index = step_index * len(self.prompts) * len(self.temperature_options) + \
                    prompt_index * len(self.temperature_options) + temp_index

        # 计算动作概率（策略梯度需要）
        logits = self.forward(task_embedding)
        probabilities = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        chosen_prob = probabilities[action_index]

        # 策略梯度损失：-log(prob) * reward
        loss_grad = - (reward / (chosen_prob + 1e-8))  # 避免除以零

        # 反向传播计算梯度
        z1 = np.dot(task_embedding, self.W1) + self.b1
        a1 = np.tanh(z1)

        # 梯度计算（链式法则）
        d_logits = np.zeros_like(logits)
        d_logits[action_index] = loss_grad  # 仅更新选中动作的梯度

        # 第二层梯度
        dW2 = np.outer(a1, d_logits)
        db2 = d_logits

        # 第一层梯度
        da1 = np.dot(self.W2, d_logits)
        dz1 = da1 * (1 - a1**2)  # tanh导数
        dW1 = np.outer(task_embedding, dz1)
        db1 = dz1

        # 使用带动量的参数更新（模拟 Adam 优化器）
        self.W1 += self.learning_rate * dW1
        self.b1 += self.learning_rate * db1
        self.W2 += self.learning_rate * dW2
        self.b2 += self.learning_rate * db2

        # **动态调整 epsilon（探索率）**
        self.epsilon = max(self.epsilon * 0.995, 0.05)  # 平滑衰减探索

        # **动态调整 temperature**
        # 让 `temperature` 的更新受 reward 影响
        new_temperature_index = np.clip(temp_index + (1 if reward > 0 else -1), 0, len(self.temperature_options) - 1)
        self.temperature = self.temperature_options[new_temperature_index]


    def diversity_penalty(self, logits, current_embedding):
        """
        Apply a penalty to logits based on similarity to previously selected embeddings.
        """
        penalties = np.zeros_like(logits)
        for prev_embedding in self.embedding_memory:
            similarity = np.dot(prev_embedding, current_embedding) / (np.linalg.norm(prev_embedding) * np.linalg.norm(current_embedding) + 1e-8)
            penalties += similarity * 0.1  # Adjust penalty strength
        return logits - penalties  # Encourage diverse actions

    def save_parameters(self, file_path):
        """Save model parameters to a file."""
        params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'step_lengths': self.step_lengths,
            'prompts': self.prompts,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'temperature': self.temperature,
            'token_limit': self.token_limit,
            'embedding_memory': self.embedding_memory,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)

    def load_parameters(self, file_path):
        """Load model parameters from a file."""
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        
        # Restore parameters
        self.W1 = params['W1']
        self.b1 = params['b1']
        self.W2 = params['W2']
        self.b2 = params['b2']
        self.step_lengths = params['step_lengths']
        self.prompts = params['prompts']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.learning_rate = params['learning_rate']
        self.epsilon = params['epsilon']
        self.temperature = params['temperature']
        self.token_limit = params['token_limit']
        self.embedding_memory = params['embedding_memory']
