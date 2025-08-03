"""
同时在多个数据集上并行评估一个模型，使用不同的 GPU 卡。
调用 eval_model.py 进行评估。
"""
import subprocess
import os
from pathlib import Path

# 根目录与脚本路径
ROOT_DIR = Path("/llm_reco/dehua/code/verl")
WRAPPER_SCRIPT = ROOT_DIR / "verl" / "inference" / "eval_model.py"
PYTHON_EXEC = "/llm_reco/dehua/anaconda3/envs/vllm/bin/python"

# 模型与数据集配置
MODEL_PATH = "/llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct"
DATASETS = ["food101", "food172", "foodx251", "fru92", "veg200", "food2k"]
GPUS     = [0,        1,        2,         3,       4,       5]
WITH_CATEGORY = True  # 若需要包含类别列表，可设为 True

if not WRAPPER_SCRIPT.is_file():
    raise FileNotFoundError(f"评估脚本不存在: {WRAPPER_SCRIPT}")

# 并行启动评估
for dataset, gpu in zip(DATASETS, GPUS):
    cmd = [
        PYTHON_EXEC,
        str(WRAPPER_SCRIPT),
        "--model-path", MODEL_PATH,
        "--dataset", dataset,
        "--gpu-id", str(gpu)
    ]
    if WITH_CATEGORY:
        cmd.append("--with_category")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["ROOT_DIR"] = str(ROOT_DIR)
    env["PYTHONPATH"] = str(ROOT_DIR)

    print(f"启动评估: 数据集={dataset}, GPU={gpu}\n  命令: {' '.join(cmd)}")
    # 后台运行，不阻塞
    subprocess.Popen(cmd, env=env)

print("所有评估任务已启动，日志请查看 verl/inference/logs 目录。")