import os
import subprocess
import argparse
from pathlib import Path

ROOT_DIR = Path("/llm_reco/dehua/code/verl")
PYTHON_EXECUTABLE = "/llm_reco/dehua/anaconda3/envs/vllm/bin/python"
INFERENCE_SCRIPT = ROOT_DIR / "verl/inference/vllm_food.py"

def run_evaluation(model_path_str: str, dataset_name: str, gpu_id: int):
    """
    使用 vLLM 在指定的 GPU 上运行模型评估。

    Args:
        model_path_str (str): 要评估的 checkpoint 的完整路径。
        dataset_name (str): 要使用的数据集名称 (例如 "food101")。
        gpu_id (int): 要使用的 GPU 的 ID。
    """
    model_path = Path(model_path_str)

    # 1. 检查输入路径和文件是否存在
    if not model_path.is_dir():
        print(f"错误: 模型路径不存在: {model_path}")
        return
    if not INFERENCE_SCRIPT.is_file():
        print(f"错误: 推理脚本不存在: {INFERENCE_SCRIPT}")
        return

    print("--- 开始模型评估 ---")
    print(f"模型路径: {model_path}")
    print(f"数据集:    {dataset_name}")
    print(f"GPU ID:    {gpu_id}")
    print("-" * 20)

    model_name = model_path.name

    save_dir = ROOT_DIR / "verl/inference/answer" / dataset_name
    log_dir = ROOT_DIR / "verl/inference/logs"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    save_filename = save_dir / f"{model_name}_{dataset_name}.jsonl"
    
    log_filename = log_dir / f"{model_name}_{dataset_name}.out"
    
    question_file = ROOT_DIR / f"verl/inference/question/{dataset_name}_question.jsonl"
    print(f"日志文件:  {log_filename}")
    print(f"答案文件:  {save_filename}")
    print("-" * 20)

    command = [
        PYTHON_EXECUTABLE,
        str(INFERENCE_SCRIPT),
        "--model_path", str(model_path),
        "--save_filename", str(save_filename),
        "--question_file", str(question_file),
    ]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["ROOT_DIR"] = str(ROOT_DIR)
    os.environ["PYTHONPATH"] = str(ROOT_DIR)

    try:
        print("正在启动评估进程...")
        with open(log_filename, 'w') as log_file:
            result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True, text=True)
        print("您现在可以安全地关闭此终端，评估将在后台继续进行。")

    except FileNotFoundError:
        print(f"错误: Python 可执行文件未找到: {PYTHON_EXECUTABLE}")
    except Exception as e:
        print(f"启动评估进程时发生错误: {e}")

    print("--- 脚本执行完毕 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在指定 GPU 上运行 VLM 模型评估。")
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True, 
        help="要评估的模型的完整路径。"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help='要评估的数据集名称, 例如 "food101"。'
    )
    parser.add_argument(
        "--gpu-id", 
        type=int, 
        required=True, 
        help="要用于评估的 GPU ID (例如 0, 1, 5)。"
    )

    args = parser.parse_args()

    run_evaluation(
        model_path_str=args.model_path,
        dataset_name=args.dataset,
        gpu_id=args.gpu_id
    )