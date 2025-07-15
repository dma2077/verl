#!/usr/bin/env python3
"""
处理Food101 JSONL文件，从图像路径中提取category并保存到新文件
"""

import json
import os
from pathlib import Path

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


def process_jsonl_file():
    # 文件路径
    input_file = "/llm_reco/dehua/code/visual-memory/questions/food2k/dinov2_large_test_5_softmax_old.jsonl"
    output_file = "/llm_reco/dehua/code/verl/verl/inference/question/food2k_question.jsonl"
    id2category_function = get_id2category(output_file)
    id2cat = id2category_function()
    all_categories = list(id2cat.values())
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}")
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 尝试作为JSON数组读取
            try:
                data = json.load(f)
                print(f"Loaded JSON array with {len(data)} items")
                is_json_array = True
            except json.JSONDecodeError:
                # 如果失败，尝试作为JSONL读取
                f.seek(0)
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num}: {e}")
                            continue
                print(f"Loaded JSONL with {len(data)} items")
                is_json_array = False
                
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    if not data:
        print("No data found in input file")
        return
    
    # 处理数据
    processed_count = 0
    error_count = 0
    category_stats = {}
    
    print(f"\nProcessing {len(data)} items...")
    
    # 打开输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i, item in enumerate(data):
                try:
                    # 检查是否有image键
                    if 'image' not in item:
                        print(f"Warning: Item {i} missing 'image' key")
                        error_count += 1
                        # 仍然写入，但不更新category
                        out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        continue
                    
                    # 创建新的item副本
                    new_item = item.copy()
                    
                    image_path = item['image']
                    
                    # 使用pathlib解析路径
                    path = Path(image_path)
                    
                    # 获取倒数第二个目录（即文件名的父目录）
                    if len(path.parts) >= 2:
                        category_dir = path.parts[-2]  # 倒数第二个部分
                    else:
                        category_dir = "unknown"
                    
                    # 将下划线替换为空格
                    
                    if "food172" in output_file or "food2k" in output_file:
                        category_dir = int(category_dir)
                        new_category = all_categories[category_dir]
                    else:
                        new_category = category_dir.replace("_", " ")
                    # 更新category
                    old_category = item.get('category', 'N/A')
                    new_item['category'] = new_category
                    processed_count += 1
                    
                    # 统计category分布
                    if new_category not in category_stats:
                        category_stats[new_category] = 0
                    category_stats[new_category] += 1
                    
                    # 显示前几个例子
                    if i < 10:
                        print(f"  {i+1:2d}: {image_path}")
                        print(f"      -> Category: '{new_category}' (was: '{old_category}')")
                    
                    # 写入到输出文件（JSONL格式）
                    out_f.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    error_count += 1
                    # 写入原始item
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    except Exception as e:
        print(f"Error writing output file: {e}")
        return
    
    # 输出统计信息
    print(f"\nProcessing completed:")
    print(f"  - Total items: {len(data)}")
    print(f"  - Successfully processed: {processed_count}")
    print(f"  - Errors: {error_count}")
    print(f"  - Unique categories: {len(category_stats)}")
    
    # 显示category分布（前20个）
    if category_stats:
        print(f"\nTop 20 categories by count:")
        sorted_categories = sorted(category_stats.items(), key=lambda x: x[1], reverse=True)
        for i, (category, count) in enumerate(sorted_categories[:20], 1):
            print(f"  {i:2d}. '{category}': {count} items")
    
    print(f"\nOutput saved to: {output_file}")
    print("Format: JSONL (one JSON object per line)")


def main():
    print("Food101 JSONL Category Converter")
    print("=" * 40)
    process_jsonl_file()


if __name__ == "__main__":
    main()