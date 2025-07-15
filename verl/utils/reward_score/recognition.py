import re
from mathruler.grader import extract_boxed_content, grade_answer
from typing import Dict, Tuple
from difflib import SequenceMatcher
import json

def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def get_category_attribute(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        category_attribute = json.load(file)
    return category_attribute

category_attribute = get_category_attribute('/llm_reco/dehua/data/food_data/food_category_attribute.json')

def acc_reward(predict_str: str, ground_truth: str) -> float:
    match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if match:
        answer = match.group(1)
    else:
        answer = predict_str
    answer_cleaned = answer.lower().strip()
    ground_truth_cleaned = ground_truth.lower().strip()
    return 1.0 if answer_cleaned == ground_truth_cleaned else 0.0

def parse_attributes(text: str) -> Dict[str, str]:
    """
    从文本中抽取五个属性及其值。
    """
    attrs = {}
    keys = ["Shape", "Texture", "Composition", "Color", "Cooking Style"]
    for key in keys:
        pattern = rf"{key}\s*(?:is|:)\s*(.*?)(?:\.|$)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            attrs[key] = m.group(1).strip()
    return attrs

def str_similarity(a: str, b: str) -> float:
    """
    使用 difflib.SequenceMatcher 计算 a, b 的相似度得分，返回 [0,1] 之间的分值。
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def att_reward(
    predict_str: str,
    truth_attrs: Dict[str, str]
) -> Tuple[Dict[str, bool], Dict[str, float], float]:
    """
    返回三部分：
      1. format_align: 每个属性是否有回答（bool）
      2. content_sim:  每个属性值与 ground-truth 的相似度（float ∈ [0,1]）
      3. total_reward: 总分，answered 部分每个 +0.2，相似度部分 *0.2
    """
    # 1) 抽取 <think>…</think> 内容
    m = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL | re.IGNORECASE)
    think_text = m.group(1) if m else ""
    pred_attrs = parse_attributes(think_text)
    format_align: Dict[str, bool] = {}
    content_sim: Dict[str, float] = {}
    total_reward = 0.0
    for key in ["Shape", "Texture", "Composition", "Color", "Cooking Style"]:
        answered = key in pred_attrs
        format_align[key] = answered
        if not answered:
            content_sim[key] = 0.0
            continue
        pred_val = pred_attrs[key]
        truth_val = truth_attrs.get(key, "")
        if pred_val.lower() == truth_val.lower():
            sim_score = 1.0
        else:
            sim_score = str_similarity(pred_val, truth_val)
        content_sim[key] = sim_score
        total_reward += sim_score * 0.2

    return format_align, content_sim, total_reward

def compute_score(predict_str: str, ground_truth: str) -> float:
    format_r = format_reward(predict_str)
    acc_r = acc_reward(predict_str, ground_truth)
    _, _, att_r = att_reward(predict_str, category_attribute[ground_truth])
    reward = 0.1 * format_r + 0.5 * acc_r + 0.4 * att_r
    if acc_reward(predict_str, ground_truth) == 1:
        acc = 1
    else:
        acc = 0
    return {
        "score": reward,
        "acc": acc,
        "pred": predict_str,
    }