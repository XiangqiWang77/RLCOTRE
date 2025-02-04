import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from module.utils import load_questions, send_openai_prompt
from module.tools import get_response  # Added import for get_response
import concurrent.futures

# Load the dataset
all_task = json.load(open("./data/shuffled_combined_HARDMATH.json"))
embedding_dim = 768

# Initialize BERT-based model for reasoning evaluation
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
hf_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)

# Define a custom pruner model for shot pruning
class CustomPrunerModel(torch.nn.Module):
    def __init__(self):
        super(CustomPrunerModel, self).__init__()
        self.hidden_layer = torch.nn.Linear(embedding_dim, 512)
        self.output_layer = torch.nn.Linear(512, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, features):
        x = self.activation(self.hidden_layer(features))
        scores = self.output_layer(x)
        return scores


# Define a token-level policy network for token pruning
class TokenPrunerModel(torch.nn.Module):
    def __init__(self):
        super(TokenPrunerModel, self).__init__()
        self.token_hidden_layer = torch.nn.Linear(embedding_dim, 512)
        self.token_output_layer = torch.nn.Linear(512, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, token_features):
        x = self.activation(self.token_hidden_layer(token_features))
        token_scores = self.token_output_layer(x)
        return token_scores

def reward_function(prediction, reasoning_steps, llm, max_steps=5, token_count=None, max_tokens=100):
    """Comprehensive reward function leveraging LLM to evaluate logical reasoning."""
    # Use LLM to assess logical reasoning quality
    reasoning_prompt = (
        "Evaluate the following reasoning steps based on logical coherence, accuracy, and completeness:"
        f"Reasoning Steps: {reasoning_steps}"
        "Rate the reasoning quality on a scale from 0 to 1, where 1 is perfectly logical and 0 is completely illogical."
    )
    try:
        logical_reward_response = send_openai_prompt(reasoning_prompt, model_name=llm, temperature=0.5, token_limit=100)
        logical_reward = float(logical_reward_response.strip())
    except Exception:
        logical_reward = 0.0  # Default to 0 if LLM evaluation fails

    # Length regularization: Penalize excessive token usage
    token_penalty = 0.0
    if token_count is not None:
        token_penalty = max(0, 1 - (token_count / max_tokens))  # Penalize if tokens exceed max_tokens

    # Combine rewards
    total_reward = logical_reward + token_penalty
    return total_reward


def train_pruner(task, pruner_model, tokenizer, model, optimizer, llm="gpt-4o", epochs=5):
    """Train the pruner model with a reinforcement learning approach."""
    pruner_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for example in tqdm(task):
            tokenized_text = tokenizer(example['question'], return_tensors="pt", padding=True, truncation=True)
            input_ids = tokenized_text["input_ids"]
            with torch.no_grad():
                outputs = model(input_ids)
                hidden_states = outputs.hidden_states[-1]  # Use the last hidden state

            # Forward pass through the pruner model
            scores = pruner_model(hidden_states.mean(dim=1))

            # Add small epsilon to avoid log(0)
            scores = scores.clamp(min=1e-8)

            # Generate reasoning steps using LLM
            prompt = (
                "Please think step by step to solve the question:\n"
                f"Question: {example['question']}\n\nAnswer:"
            )
            response = send_openai_prompt(prompt, model_name=llm, temperature=0.7, token_limit=256)
            reasoning_steps = response.split("\n")

            # Compute reward and normalize
            token_count = input_ids.size(1)
            reward = reward_function(
                prediction=response.strip(),
                #ground_truth=example['answer'],
                reasoning_steps=reasoning_steps,
                llm=llm,
                max_steps=5,
                token_count=token_count,
                max_tokens=256
            )
            reward = max(0, min(1, reward))  # Normalize reward to [0, 1]

            # Compute loss
            loss = -torch.log(scores.squeeze()) * reward

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(pruner_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    # Save trained model
    torch.save(pruner_model.state_dict(), "pruner_model.pth")


def prune_shots_and_tokens(task, shot_pruner_model, token_pruner_model, tokenizer, model, num_shots):
    """Prune the few-shot examples and tokens using custom pruner models"""
    pruned_shots = []
    for example in task[:num_shots]:
        tokenized_text = tokenizer(example['question'], return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenized_text["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids)
            hidden_states = outputs.hidden_states[-1]  # Use the last hidden state layer

            # Prune tokens
            token_scores = token_pruner_model(hidden_states)
            important_tokens = (token_scores.squeeze() > 0).nonzero(as_tuple=True)[0]
            reduced_features = hidden_states[:, important_tokens, :].mean(dim=1)  # Aggregate selected tokens

            # Prune shots
            shot_scores = shot_pruner_model(reduced_features)
            if shot_scores.item() > 0:  # Threshold for pruning
                pruned_shots.append(example)
    return pruned_shots


prompt_template = (
    "Please think step by step to solve the question:\n"
    "Question: {question}\n\nAnswer:"
)

def evaluate_simple_cot_influx(task, shots, model, temperature=1.0, shot_pruner_model=None, token_pruner_model=None, tokenizer=None, model_obj=None, mode="whole_dataset"):
    """Evaluate the model with Simple CoT input/output format and save the results."""
    import os
    out_dir = "results/cot_influx"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file = f"{out_dir}/{model}.json"
    
    if mode == "few_shot":
        if shot_pruner_model and token_pruner_model and tokenizer and model_obj:
            shots = prune_shots_and_tokens(task, shot_pruner_model, token_pruner_model, tokenizer, model_obj, shots)
        evaluation_task = shots
        print(f"Retained Shots (Few-shot Mode): {len(shots)}/{len(task)}")
    elif mode == "whole_dataset":
        print(f"Evaluating on the whole dataset (Size: {len(task)})")
        evaluation_task = task
    else:
        raise ValueError("Invalid mode. Choose 'few_shot' or 'whole_dataset'.")
    
    def process_question(question_data):
        prompt = prompt_template.format(question=question_data['question'])
        response = get_response(model=model, prompt=prompt, temperature=temperature)  # Use get_response instead
        if not response:
            predicted_reasoning = None
        else:
            predicted_reasoning = response.split("\n")
        return {
            "question": question_data['question'],
            "correct_option": question_data['answer'],
            "predicted_reasoning": predicted_reasoning,
        }
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_question, evaluation_task), total=len(evaluation_task), desc="Processing With Simple CoT"))
        
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    return 0

# Initialize and train pruner models
shot_pruner_model = CustomPrunerModel()
token_pruner_model = TokenPrunerModel()
shot_optimizer = optim.Adam(shot_pruner_model.parameters(), lr=1e-3)

# Train the shot pruner model
train_pruner(all_task[:20], shot_pruner_model, tokenizer, hf_model, shot_optimizer, epochs=3)

# Load trained models
shot_pruner_model.load_state_dict(torch.load("pruner_model.pth"))

# Few-shot evaluation using Simple CoT I/O format
evaluate_simple_cot_influx(
    task=all_task,
    shots=20,
    model="gpt-4o",
    temperature=0.7,
    shot_pruner_model=shot_pruner_model,
    token_pruner_model=token_pruner_model,
    tokenizer=tokenizer,
    model_obj=hf_model
)
