import os
import subprocess
import argparse
from pathlib import Path
import sys

def find_and_sort_models(base_dir: Path):
    """
    在基础目录中查找所有符合格式的模型检查点，并按step降序排序。

    Args:
        base_dir (Path): 搜索的根目录，例如 '.../food101_nofreeze_1e-7'

    Returns:
        list: 一个包含字典的列表，每个字典包含'step'和'path'。
              例如: [{'step': 100, 'path': Path(...) }, {'step': 50, 'path': Path(...)}]
    """
    # 查找所有符合 .../global_step_*/food* 格式的目录
    # 使用 glob 可以轻松匹配这种模式
    model_paths = list(base_dir.glob("global_step_*/**"))

    if not model_paths:
        print(f"错误：在目录 '{base_dir}' 中没有找到任何匹配 'global_step_*/food*' 格式的子目录。")
        sys.exit(1)

    print(f"找到了 {len(model_paths)} 个匹配的模型检查点。")

    models_with_steps = []
    for path in model_paths:
        try:
            # path.parent.name 应该是 'global_step_50' 这样的格式
            step_str = path.parent.name.split('_')[-1]
            step = int(step_str)
            models_with_steps.append({'step': step, 'path': path})
        except (ValueError, IndexError):
            print(f"警告：无法从目录名 '{path.parent.name}' 中解析 'step'，已跳过。")
            continue

    # 按 'step' 降序排序
    sorted_models = sorted(models_with_steps, key=lambda x: x['step'], reverse=True)
    
    print("模型按 'step' 降序排序完成。")
    return sorted_models

def run_evaluation(base_dir_str: str, gpu_ids_str: str):
    """
    主执行函数：查找、排序并分批运行评估。

    Args:
        base_dir_str (str): 模型所在的根目录路径。
        gpu_ids_str (str): 以逗号分隔的GPU ID字符串, 例如 "7,6,5,4,3,2,1,0"。
    """
    base_dir = Path(base_dir_str)
    if not base_dir.is_dir():
        print(f"错误：提供的目录不存在: {base_dir}")
        sys.exit(1)

    # 1. 提取数据集名称
    # 假设目录名格式为 'dataset_...' (例如 'food101_nofreeze_1e-7')
    dataset_name = base_dir.name.split('_')[0]
    print(f"从目录名中提取的数据集为: {dataset_name}")

    # 2. 查找并排序模型
    sorted_models = find_and_sort_models(base_dir)
    if not sorted_models:
        print("没有可评估的模型，程序退出。")
        return

    # 3. 准备GPU列表
    gpu_ids = [gpu.strip() for gpu in gpu_ids_str.split(',')]
    num_gpus = len(gpu_ids)
    print(f"将使用 {num_gpus} 个GPU进行评估: {gpu_ids}")

    # 4. 分批执行评估
    total_models = len(sorted_models)
    for i in range(0, total_models, num_gpus):
        batch = sorted_models[i : i + num_gpus]
        processes = []
        
        batch_num = (i // num_gpus) + 1
        print(f"\n{'='*20} 开始第 {batch_num} 批评估 {'='*20}")

        for j, model_info in enumerate(batch):
            model_path = model_info['path']
            gpu_id = gpu_ids[j] # 从GPU池中分配一个GPU
            step = model_info['step']

            # 构建命令
            command = [
                "python", "verl/inference/eval_model.py",
                "--model-path", str(model_path),
                "--dataset", dataset_name,
                "--gpu-id", gpu_id
            ]

            print(f"  [批次 {batch_num}, 任务 {j+1}] step={step}, GPU={gpu_id}")
            print(f"  > 命令: {' '.join(command)}")

            # 异步启动子进程
            # stdout=None 和 stderr=None 表示子进程的输出会直接打印到当前终端
            p = subprocess.Popen(command, stdout=None, stderr=None)
            processes.append(p)
        
        print(f"\n--- 第 {batch_num} 批任务已全部启动，等待其完成... (共 {len(processes)} 个任务) ---")

        # 等待当前批次的所有进程结束 (通过PID阻塞)
        for p in processes:
            p.wait()

        print(f"--- 第 {batch_num} 批任务已全部完成 ---")
    
    print(f"\n{'='*20} 所有评估任务已完成 {'='*20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="自动查找、排序并分批并行评估模型。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="包含所有模型检查点的根目录。\n例如: /mmu_mllm_hdd_2/madehua/model/CKPT/verl/DAPO/food172_nofreeze_3e-7/"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="7,6,5,4,3,2,1,0",
        help="用于评估的GPU ID列表，以逗号分隔，按降序排列。\n默认为: \"7,6,5,4,3,2,1,0\""
    )

    args = parser.parse_args()

    run_evaluation(args.base_dir, args.gpu_ids)