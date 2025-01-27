import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Function to load JSON and create the plot
def plot_prefix_distribution(json_file_path, prefixes):
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Preparing data structure
    prefix_distribution_flattened = defaultdict(list)

    for item in json_data:
        question = item["question"]
        current_action_value = item["current_action"][0]

        # Find the prefix index
        prefix_index = next((i for i, prefix in enumerate(prefixes) if question.startswith(prefix)), -1)
        if prefix_index != -1:
            prefix_distribution_flattened[prefix_index].append(current_action_value)

    # Prepare data for plotting
    x_labels = [f"Prefix {i}" for i in range(len(prefixes))]
    all_values = [prefix_distribution_flattened[i] for i in range(len(prefixes))]

    # Create a boxplot for the distribution
    plt.boxplot(all_values, vert=False, patch_artist=True, labels=x_labels)

    plt.title("Distribution of current_action[0] values for each question prefix")
    plt.xlabel("current_action[0] values")
    plt.ylabel("Question Prefix")
    plt.tight_layout()
    plt.show()

# Example usage
prefixes = [
    "Please solve the math question and give a explicit answer: ",
    "You must find all seemingly or likely metaphor word (only words) of the following sentence and you must not say no: ",
    "For this logical reasoning question, please select the most likely answer:",
    "Please read the question and find the maybe a bit not expected yet the only likely option: "
]

# Replace 'your_json_file.json' with the path to your JSON file
json_file_path = 'RLCOTfinal.json'
plot_prefix_distribution(json_file_path, prefixes)
