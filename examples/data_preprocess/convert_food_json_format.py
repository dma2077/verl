import json
import os
import re

# 需要转换的数据集和shot类型
datasets = ["food101", "food172", "foodx251", "veg200", "fru92", "food2k"]
shots = ["4shot", "8shot"]

input_dir = "/llm_reco/dehua/data/food_finetune_data"
output_dir = "/llm_reco/dehua/data/food_finetune_data/converted"
os.makedirs(output_dir, exist_ok=True)

def convert_item(item):
    if "messages" in item and "images" in item:
        user_msg = item["messages"][0]
        assistant_msg = item["messages"][1]
        # 提取 <answer>...</answer> 部分
        match = re.search(r"<answer>(.*?)</answer>", assistant_msg["content"])
        answer = match.group(1).strip() if match else assistant_msg["content"]
        return {
            "images": item["images"],
            "conversations": [
                {"role": "user", "content": user_msg["content"]},
                {"value": answer}
            ]
        }
    else:
        return None

def convert_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data = []
    for item in data:
        new_item = convert_item(item)
        if new_item:
            new_data.append(new_item)
        else:
            print(f"跳过格式不符的数据项: {item}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"转换完成: {input_path} -> {output_path}, 共 {len(new_data)} 条")

if __name__ == "__main__":
    for dataset in datasets:
        for shot in shots:
            input_file = os.path.join(input_dir, f"{dataset}_cold_sft_{shot}.json")
            output_file = os.path.join(output_dir, f"{dataset}_cold_sft_{shot}.json")
            if os.path.exists(input_file):
                convert_file(input_file, output_file)
            else:
                print(f"文件不存在: {input_file}") 