import json
import argparse
from pathlib import Path

def process_dataset_question(input_file: str, output_file: str, old_path: str = None, new_path: str = None):
    """
    处理数据集的 question.jsonl 文件，提取指定字段并替换图片路径。
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        old_path (str): 要替换的旧路径
        new_path (str): 替换后的新路径
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # 默认路径替换
    if old_path is None:
        old_path = "/map-vepfs/dehua/data/data"
    if new_path is None:
        new_path = "/llm_reco/dehua/data/food_data"
    
    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return False
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    print(f"开始处理文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"路径替换: {old_path} -> {new_path}")
    print("-" * 50)
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析JSON
                data = json.loads(line)
                
                # 提取指定字段
                processed_data = {
                    "question_id": data.get("question_id", ""),
                    "image": data.get("image", ""),
                    "category": data.get("category", "")
                }
                
                # 替换图片路径
                if processed_data["image"]:
                    processed_data["image"] = processed_data["image"].replace(old_path, new_path)
                
                # 写入输出文件
                outfile.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
                processed_count += 1
                
                # 每处理1000行打印一次进度
                if processed_count % 1000 == 0:
                    print(f"已处理 {processed_count} 行...")
                    
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行JSON解析错误: {e}")
                error_count += 1
            except Exception as e:
                print(f"第 {line_num} 行处理错误: {e}")
                error_count += 1
    
    print("-" * 50)
    print(f"处理完成!")
    print(f"成功处理: {processed_count} 行")
    print(f"错误行数: {error_count} 行")
    print(f"输出文件: {output_path}")
    
    return True

def process_all_datasets(base_dir: str, old_path: str = None, new_path: str = None):
    """
    批量处理所有数据集文件。
    
    Args:
        base_dir (str): 基础目录路径
        old_path (str): 要替换的旧路径
        new_path (str): 替换后的新路径
    """
    # 定义要处理的数据集
    datasets = [
        "food101",
        "food172", 
        "foodx251",
        "fru92",
        "veg200"
    ]
    
    base_path = Path(base_dir)
    
    print("=" * 60)
    print("开始批量处理数据集文件")
    print("=" * 60)
    
    success_count = 0
    total_count = len(datasets)
    
    for dataset in datasets:
        print(f"\n处理数据集: {dataset}")
        print("-" * 40)
        
        input_file = base_path / f"{dataset}_question.jsonl"
        output_file = base_path / f"{dataset}_question_processed.jsonl"
        
        if process_dataset_question(str(input_file), str(output_file), old_path, new_path):
            success_count += 1
        else:
            print(f"数据集 {dataset} 处理失败")
    
    print("\n" + "=" * 60)
    print(f"批量处理完成!")
    print(f"成功处理: {success_count}/{total_count} 个数据集")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="批量处理多个数据集的 question.jsonl 文件")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/llm_reco/dehua/code/verl/verl/inference/question",
        help="包含数据集文件的基础目录"
    )
    parser.add_argument(
        "--old-path",
        type=str,
        default="/map-vepfs/dehua/data/data",
        help="要替换的旧路径"
    )
    parser.add_argument(
        "--new-path",
        type=str,
        default="/llm_reco/dehua/data/food_data",
        help="替换后的新路径"
    )
    parser.add_argument(
        "--single-dataset",
        type=str,
        help="只处理指定的单个数据集（例如：food101）"
    )
    
    args = parser.parse_args()
    
    if args.single_dataset:
        # 处理单个数据集
        input_file = Path(args.base_dir) / f"{args.single_dataset}_question.jsonl"
        output_file = Path(args.base_dir) / f"{args.single_dataset}_question_processed.jsonl"
        
        print(f"处理单个数据集: {args.single_dataset}")
        process_dataset_question(
            str(input_file),
            str(output_file),
            args.old_path,
            args.new_path
        )
    else:
        # 批量处理所有数据集
        process_all_datasets(args.base_dir, args.old_path, args.new_path)

if __name__ == "__main__":
    main() 