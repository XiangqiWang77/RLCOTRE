import json

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 计算包含 'yes' 的 evaluation 数量
def count_positive_evaluations(data):
    count = 0
    for item in data:
        if 'evaluation' in item and 'yes' in item['evaluation'].lower():
            count += 1
    return count

# 主函数
if __name__ == "__main__":
    input_file = './LLM_as_Judge/RLCOT_Judge.json'
    data = load_json(input_file)
    
    positive_count = count_positive_evaluations(data)
    print(f"Number of evaluations containing 'yes': {positive_count}")
