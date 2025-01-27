import json
import random

def shuffle_json(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 检查数据是否为列表
    if isinstance(data, list):
        random.shuffle(data)  # 随机打乱列表
    else:
        raise ValueError("The JSON file must contain a list of items.")
    
    # 写入新的JSON文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Shuffled JSON has been saved to {output_file}")

# 示例用法
shuffle_json("combined.json", "shuffled_combined1.json")
