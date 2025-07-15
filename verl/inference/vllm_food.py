import torch
import PIL
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
import argparse
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig
import io
import base64
from PIL import Image
import random
from qwen2_5 import Qwen2_5VL
import json
from nltk.metrics.distance import edit_distance
import re

def load_jsonl(filename):
    results = []
    with open(filename, 'r') as f:
        for line in f:
            line = json.loads(line)
            results.append(line)
    return results

def save_jsonl(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def add_jsonl(filename, data):
    with open(filename, 'a+') as f:
        f.write(json.dumps(data) + '\n')

def find_most_similar_word(target, candidates):
    """在候选列表中找到与 target 编辑距离最小的词及其距离。"""
    best_word, best_dist = None, float('inf')
    for w in candidates:
        d = edit_distance(target, w)
        if d < best_dist:
            best_dist, best_word = d, w
    return best_word, best_dist

def get_id2category(save_filename):
    if "food101" in save_filename:
        from get_acc import build_food101_id2category
        id2category_function = build_food101_id2category
    elif "food172" in save_filename:
        from get_acc import build_food172_id2category
        id2category_function = build_food172_id2category
    elif "fru92" in save_filename:
        from get_acc import build_fru92_id2category
        id2category_function = build_fru92_id2category
    elif "veg200" in save_filename:
        from get_acc import build_veg200_id2category
        id2category_function = build_veg200_id2category
    elif "food2k" in save_filename:
        from get_acc import build_food2k_id2category
        id2category_function = build_food2k_id2category
    elif "foodx251" in save_filename:
        from get_acc import build_foodx251_id2category
        id2category_function = build_foodx251_id2category
    return id2category_function

def image_to_base64(image_path):
    """
    Convert image to base64 string
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def get_prompts(filename):
    test_data = load_jsonl(filename)
    messages = []
    image_path_list = []
    category_list = []
    for idx, data in tqdm(enumerate(test_data), total=len(test_data), desc="Loading test data"):
        image_path = data["image"].replace("/map-vepfs/dehua/data/data", "/llm_reco/dehua/data/food_data").replace("vegfru-dataset/", "")
        base64_image = image_to_base64(image_path)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {'url': f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": "Please analyze these food attributes in the image: shape, texture, composition, color, and cooking style. Then identify the food category."},
                ],
            }
        ]
        messages.append(conversation)
        image_path_list.append(data["image"])
        category_list.append(data["category"])
    return messages, image_path_list, category_list

def get_model(model_path, max_model_len=4096):
    import re
    def extract_number(text):
        match = re.search(r'-(\d+)B', text)
        return int(match.group(1)) if match else None
    model_size = extract_number(model_path)
    if model_size:
        if model_size <= 10:
            tensor_parallel_size = 1
        elif model_size > 10 and model_size < 30:
            tensor_parallel_size = 2
        elif model_size >=30:
            tensor_parallel_size = 4
    else:
        tensor_parallel_size = 1
    print(f"tensor_parallel_size is {tensor_parallel_size}")
    model = Qwen2_5VL(model_path=model_path, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size)
    return model

def main(args):
    os.environ['NCCL_SHM_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['NCCL_IB_TIMEOUT'] = '22'

    print("Running with the following parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Construct absolute paths using ROOT_DIR
    root_dir = os.environ.get('ROOT_DIR', '')
    save_filename = os.path.join(root_dir, args.save_filename)
    question_file = os.path.join(root_dir, args.question_file)
    
    model = get_model(args.model_path)
    messages, image_path_list, category_list = get_prompts(filename=question_file)
    outputs = model.generate_until(messages)
    id2category_function = get_id2category(save_filename)
    id2cat = id2category_function()
    all_categories = list(id2cat.values())
    # Save the results
    formmatted_datas = []
    for idx, o in enumerate(outputs):
        generated_text = o.outputs[0].text
        formmatted_data = {
            "question_id": idx,
            "image": image_path_list[idx],
            "text": generated_text,
            "category": category_list[idx],
        }
        formmatted_datas.append(formmatted_data)
    save_jsonl(save_filename, formmatted_datas)
    formmatted_datas = load_jsonl(save_filename)
    truth = 0
    for idx, o in enumerate(formmatted_datas):
        generated_text = o["text"]
        m = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        if not m:
            continue
        pred = m.group(1).strip().lower()
        if pred == category_list[idx].lower():
            truth += 1
        else:
            closest, dist = find_most_similar_word(pred, [c.lower() for c in all_categories])
            if closest == category_list[idx].lower():
                truth += 1
    total_samples = len(formmatted_datas)
    accuracy = (truth / total_samples) if total_samples > 0 else 0.0
    summary_data = {
            "final_summary": True,
            "accuracy": f"{accuracy:.4f}",
            "correct_count": truth,
            "total_count": total_samples
        }
    add_jsonl(save_filename, summary_data)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM inference with specified parameters.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--save_filename', type=str, required=True, help='Filename to save the results.')
    parser.add_argument('--question_file', type=str, required=True, help='Path to the question file.')
    args = parser.parse_args()
    main(args)
