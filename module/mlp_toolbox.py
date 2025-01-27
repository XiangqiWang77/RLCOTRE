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
        Select an action using temperature-scaled softmax sampling to encourage exploration.
        """
        # Normalize the embedding to ensure sensitivity
        task_embedding = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)

        # Compute logits
        logits = self.forward(task_embedding)

        # Apply diversity penalty
        if self.embedding_memory:
            logits = self.diversity_penalty(logits, task_embedding)

        # Apply temperature scaling
        logits = logits / self.temperature

        # Softmax sampling to encourage exploration
        probabilities = np.exp(logits - np.max(logits))  # Stability adjustment
        probabilities /= np.sum(probabilities)

        # Sample an action based on probabilities
        action_index = np.random.choice(len(probabilities), p=probabilities)

        # Map index to step_length and prompt
        step_index, prompt_index = divmod(action_index, len(self.prompts))
        step_length = self.step_lengths[step_index]
        prompt = self.prompts[prompt_index]

        # Save embedding to memory
        self.embedding_memory.append(task_embedding)
        if len(self.embedding_memory) > 100:  # Limit memory size
            self.embedding_memory.pop(0)

        return step_length, prompt, self.temperature, self.token_limit

    def update(self, task_embedding, reward, chosen_action):
        """
        Update the MLP model based on the reward received (single action).
        """
        # Map action to index
        action_index = self.step_lengths.index(chosen_action[0]) * len(self.prompts) \
                    + self.prompts.index(chosen_action[1])

        # Forward pass
        logits = self.forward(task_embedding)
        prediction = logits[action_index]
        error = reward - prediction

        # Backpropagation
        z1 = np.dot(task_embedding, self.W1) + self.b1
        a1 = np.tanh(z1)

        dz2 = np.zeros_like(logits)
        dz2[action_index] = error

        dW2 = np.outer(a1, dz2)
        db2 = dz2

        da1 = np.dot(self.W2, dz2)
        dz1 = da1 * (1 - a1**2)

        dW1 = np.outer(task_embedding, dz1)
        db1 = dz1

        # Update parameters
        self.W1 += self.learning_rate * dW1
        self.b1 += self.learning_rate * db1
        self.W2 += self.learning_rate * dW2
        self.b2 += self.learning_rate * db2

        # Adjust exploration rate dynamically
        self.epsilon = max(self.epsilon * 0.99, 0.05)
        self.temperature = max(self.temperature * 0.995, 0.5)  # Gradually reduce exploration

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
