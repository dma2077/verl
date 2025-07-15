import os
import subprocess
import argparse
from pathlib import Path

def convert_checkpoints(base_path: Path):
    """
    遍历指定目录，查找并转换模型格式。
    """
    # 1. 检查基础目录是否存在
    if not base_path.is_dir():
        print(f"错误: 基础目录不存在: {base_path}")
        return

    print(f"开始处理目录: {base_path}")
    print("-" * 50)

    model_base_name = base_path.name
    
    # 2. 查找所有 global_step_* 目录
    step_dirs = sorted(base_path.glob("global_step_*"), key=lambda p: int(p.name.split('_')[-1]))

    if not step_dirs:
        print("未在目录中找到任何 'global_step_*' 子目录。")
        return

    print(f"发现了 {len(step_dirs)} 个潜在的 checkpoint 目录。")
    print("-" * 50)

    for step_dir in step_dirs:
        # 确保它是一个目录
        if not step_dir.is_dir():
            continue

        try:
            # 提取步数
            step_number = step_dir.name.split('_')[-1]
            print(f"正在处理 Step: {step_number}")

            # 定义需要转换的源模型目录 (actor)
            local_dir = step_dir / "actor"

            # 3. 检查需要转换的源模型是否存在
            if not local_dir.is_dir():
                print(f"  -> 跳过: 未找到源目录 'actor' in {step_dir}")
                continue

            # 4. 构建转换后的目标目录名称和路径
            target_dir_name = f"{model_base_name}_step_{step_number}"
            target_dir = step_dir / target_dir_name

            # 5. 检查目标目录是否已存在
            if target_dir.is_dir():
                print(f"  -> 跳过: 目标目录已存在: {target_dir}")
                continue

            # 准备执行转换命令
            command = [
                "python",
                "-m", "verl.model_merger", "merge",
                "--backend", "fsdp",
                "--local_dir", str(local_dir),
                "--target_dir", str(target_dir)
            ]

            print(f"  -> 正在转换: {local_dir} -> {target_dir}")
            
            # 6. 执行格式转换
            # 使用 subprocess.run 来执行命令，并捕获输出
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=False # 设置为False，手动检查返回码
            )

            # 检查命令是否成功执行
            if result.returncode == 0:
                print(f"  -> 成功: Step {step_number} 转换完成。")
            else:
                print(f"  -> 错误: Step {step_number} 转换失败。")
                print(f"     错误信息: {result.stderr.strip()}")
        
        except Exception as e:
            print(f"  -> 发生意外错误处理 Step {step_dir.name}: {e}")
        
        finally:
            print("-" * 20)


def main():
    """主函数，解析命令行参数并执行转换"""
    parser = argparse.ArgumentParser(
        description="批量转换 checkpoint 模型格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python auto_merge_checkpoints.py /path/to/checkpoint/dir
  python auto_merge_checkpoints.py "/mmu_mllm_hdd_2/madehua/model/CKPT/verl/DAPO/food101_nofreeze_1e-7"
        """
    )
    
    parser.add_argument(
        "base_dir",
        type=str,
        help="包含所有 global_step_* 子目录的基础目录路径"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示要执行的操作，但不实际执行转换"
    )
    
    args = parser.parse_args()
    
    # 转换为 Path 对象
    base_path = Path(args.base_dir)
    
    if args.dry_run:
        print("*** 预览模式 - 不会执行实际转换 ***")
        print(f"将要处理的目录: {base_path}")
        print("-" * 50)
        
        # 只显示会处理的目录
        if base_path.is_dir():
            step_dirs = sorted(base_path.glob("global_step_*"), key=lambda p: int(p.name.split('_')[-1]))
            if step_dirs:
                print(f"发现 {len(step_dirs)} 个 checkpoint 目录:")
                for step_dir in step_dirs:
                    if step_dir.is_dir():
                        local_dir = step_dir / "actor"
                        if local_dir.is_dir():
                            print(f"  -> 将处理: {step_dir.name}")
                        else:
                            print(f"  -> 跳过: {step_dir.name} (缺少 actor 目录)")
            else:
                print("未找到任何 'global_step_*' 目录")
        else:
            print(f"错误: 目录不存在: {base_path}")
        return
    
    # 执行实际转换
    convert_checkpoints(base_path)
    print("所有任务处理完毕。")


if __name__ == "__main__":
    main()