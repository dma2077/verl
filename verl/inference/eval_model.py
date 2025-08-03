import os
import subprocess
import argparse
from pathlib import Path

ROOT_DIR = Path("/llm_reco/dehua/code/verl")
PYTHON_EXECUTABLE = "/llm_reco/dehua/anaconda3/envs/vllm/bin/python"
INFERENCE_SCRIPT = ROOT_DIR / "verl/inference/vllm_food.py"

def run_evaluation(model_path_str: str, dataset_name: str, gpu_id: int, with_category: bool):
    """
    使用 vLLM 在指定的 GPU 上运行模型评估（后台运行）。

    Args:
        model_path_str (str): 要评估的 checkpoint 的完整路径。
        dataset_name (str): 要使用的数据集名称 (例如 "food101")。
        gpu_id (int): 要使用的 GPU 的 ID。
        with_category (bool): 是否在推理时包含类别列表。
    """
    model_path = Path(model_path_str)

    # 1. 检查输入路径和文件是否存在
    if not model_path.is_dir():
        print(f"错误: 模型路径不存在: {model_path}")
        return
    if not INFERENCE_SCRIPT.is_file():
        print(f"错误: 推理脚本不存在: {INFERENCE_SCRIPT}")
        return

    print("--- 开始模型评估（后台运行）---")
    print(f"模型路径: {model_path}")
    print(f"数据集:    {dataset_name}")
    print(f"GPU ID:    {gpu_id}")
    print(f"Include categories: {with_category}")
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

    # 构建完整的命令字符串
    command_parts = [
        str(PYTHON_EXECUTABLE),
        str(INFERENCE_SCRIPT),
        "--model_path", str(model_path),
        "--save_filename", str(save_filename),
        "--question_file", str(question_file),
    ]

    # 传递 with_category 参数
    if with_category:
        command_parts += ["--with_category"]

    # 使用 nohup 命令实现后台运行
    nohup_command = [
        "nohup",
        "bash", "-c",
        f"export CUDA_VISIBLE_DEVICES={gpu_id}; "
        f"export ROOT_DIR={ROOT_DIR}; "
        f"export PYTHONPATH={ROOT_DIR}; "
        f"{' '.join(command_parts)} > {log_filename} 2>&1 &"
    ]

    try:
        print("正在启动后台评估进程...")
        
        # 执行 nohup 命令
        result = subprocess.run(
            nohup_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("评估进程已在后台启动")
        print(f"您可以通过以下命令查看日志:")
        print(f"  tail -f {log_filename}")
        print(f"您可以通过以下命令查看后台进程:")
        print(f"  ps aux | grep {INFERENCE_SCRIPT.name}")
        print("您现在可以安全地关闭此终端，评估将在后台继续进行。")
        
    except FileNotFoundError:
        print(f"错误: Python 可执行文件未找到: {PYTHON_EXECUTABLE}")
    except subprocess.CalledProcessError as e:
        print(f"启动评估进程时发生错误: {e}")
        print(f"错误输出: {e.stderr}")
    except Exception as e:
        print(f"启动评估进程时发生错误: {e}")

    print("--- 脚本执行完毕 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在指定 GPU 上运行 VLM 模型评估（后台运行）。")
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
    parser.add_argument(
        "--with_category",
        action='store_true',
        help="如果设置，将在推理时包含类别列表。"
    )

    args = parser.parse_args()

    run_evaluation(
        model_path_str=args.model_path,
        dataset_name=args.dataset,
        gpu_id=args.gpu_id,
        with_category=args.with_category    
    ) 