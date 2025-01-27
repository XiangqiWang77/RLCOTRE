import json

# File paths
input_file = "output.json"  # Replace with the actual input file name
output_file = "modified_data.json"

# Read JSON from file
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)


if isinstance(data, list):
    for record in data:
        # Modify the 'answer' field for each record
        answer_index = record["answer"]
        answer_description = record["options"][answer_index]
        record["answer"] = f'{answer_index}, "{answer_description}"'
else:
    # Single record case
    answer_index = data["answer"]
    answer_description = data["options"][answer_index]
    data["answer"] = f'{answer_index}, "{answer_description}"'

# Save the modified JSON to a new file
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"Modified JSON saved to {output_file}")
