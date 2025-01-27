import json
import random

# 四个JSON文件的路径
file_paths = [
    'HARDMath/formatted_question_answer_list.json',
    'metaphor/metaphornew.json',
    'logiQA/modified_data.json',
    'humor/shuffled_questions.json'
]
output_file_path = 'Newcombined.json'

# 加载JSON文件
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 保存JSON文件
def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# 为特定JSON数据的question字段添加前缀
def add_prefix_to_questions(data, prefix):
    for item in data:
        if 'question' in item:
            # 如果存在text字段，将其添加到问题前
            if 'text' in item:
                item['question'] = prefix + item['text'] + "\n\n" + item['question']
            else:
                item['question'] = prefix + item['question']
            
        # 如果存在选项，将选项合并到问题下
        if 'options' in item:
            if isinstance(item['options'], dict):
                options_text = "\nOptions:\n" + "\n".join([f"{key}: {value}" for key, value in item['options'].items()])
            elif isinstance(item['options'], list):
                options_text = "\nOptions:\n" + "\n".join([f"{idx + 1}. {option}" for idx, option in enumerate(item['options'])])
            else:
                options_text = ""
            item['question'] += options_text
    return data

# 从每个文件中随机选取100个item并混合
def combine_json_files_randomly(file_paths, output_file_path):
    combined_data = []
    
    # 为每个文件定义不同的前缀
    prefixes = [
        "Please solve the math question and give a explicit answer: ",
        "You must find all seemingly or likely metaphor word (only words) of the following sentence and you must not say no: ",
        "For this logical reasoning question, please select the most likely answer:",
        "Please read the question and find the maybe a bit not expected yet the only likely option: "
    ]
    
    # 遍历四个文件
    for i, file_path in enumerate(file_paths):
        data = load_json_file(file_path)
        
        # 确保每个文件至少有100个item
        if len(data) < 100:
            raise ValueError(f"File {file_path} must contain at least 100 items.")
        
        # 添加前缀
        data = add_prefix_to_questions(data, prefixes[i])
        
        # 随机抽取100个item
        sampled_data = random.sample(data, 100)
        combined_data.extend(sampled_data)
    
    # 打乱顺序
    random.shuffle(combined_data)
    
    # 保存为新的JSON文件
    save_json_file(combined_data, output_file_path)
    print(f"Combined JSON file saved to {output_file_path}")

# 执行合并函数
combine_json_files_randomly(file_paths, output_file_path)
