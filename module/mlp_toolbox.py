import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Initialize semantic model for embedding-based evaluations
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

class AdaptiveContextualMLPAgent:
    def __init__(self, step_lengths, prompts, embedding_dim, hidden_dim=64, learning_rate=0.01, epsilon=0.2):
        self.step_lengths = step_lengths
        self.prompts = prompts  # Dynamic prompt pool
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Initial exploration rate

        # Initialize MLP parameters
        self.W1 = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.01
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, len(step_lengths) * len(prompts)) * 0.01
        self.b2 = np.zeros(len(step_lengths) * len(prompts))

        self.temperature = 1.0  # Higher temperature encourages exploration
        self.token_limit = 250
        self.embedding_memory = []  # Store embeddings for diversity

    def forward(self, embedding):
        """Forward pass through the MLP."""
        z1 = np.dot(embedding, self.W1) + self.b1
        a1 = np.tanh(z1)  # Using tanh activation
        z2 = np.dot(a1, self.W2) + self.b2
        return z2  # Return logits for each action

    def select_action(self, task_embedding):
        """
        基于任务嵌入和模型学习选择动作（step_length 和 prompt），
        结合温度退火、多样性惩罚和动态探索策略。
        """
        # 归一化嵌入以提高稳定性
        task_embedding = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)

        # 前向传播计算 logits
        logits = self.forward(task_embedding)

        # 应用多样性惩罚：仅考虑最近 20 个嵌入的相似性
        if self.embedding_memory:
            recent_embeddings = self.embedding_memory[-20:]  # 保留最近 20 个
            similarities = np.dot(recent_embeddings, task_embedding)
            penalties = np.mean(similarities) * 0.2  # 平均相似性惩罚
            logits -= penalties

        # 温度退火：温度随训练步数降低
        scaled_logits = logits / self.temperature

        # 使用稳定的 softmax 计算概率
        probabilities = np.exp(scaled_logits - np.max(scaled_logits))
        probabilities /= np.sum(probabilities)

        # Epsilon-greedy 探索：以 epsilon 概率随机选择动作
        if np.random.rand() < self.epsilon:
            action_index = np.random.choice(len(probabilities))
        else:
            action_index = np.argmax(probabilities)

        # 映射到 step_length 和 prompt
        step_index, prompt_index = divmod(action_index, len(self.prompts))
        step_length = self.step_lengths[step_index]
        prompt = self.prompts[prompt_index]

        # 记录当前嵌入（限制内存大小）
        self.embedding_memory.append(task_embedding.copy())
        if len(self.embedding_memory) > 100:
            self.embedding_memory.pop(0)

        # 动态调整 token_limit（根据 step_length 自适应）
        self.token_limit = min(250 + step_length * 20, 500)  # 动态调整上限

        return step_length, prompt, self.temperature, self.token_limit

    def update(self, task_embedding, reward, chosen_action):
        """
        基于奖励更新模型参数，引入动量优化和策略梯度。
        """
        # 将动作映射到索引
        step_length, prompt = chosen_action[0], chosen_action[1]
        action_index = self.step_lengths.index(step_length) * len(self.prompts) \
                    + self.prompts.index(prompt)

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

        # 动态调整探索参数
        self.epsilon = max(self.epsilon * 0.995, 0.05)  # 更平滑的衰减
        self.temperature = max(self.temperature * 0.99, 0.3)

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
