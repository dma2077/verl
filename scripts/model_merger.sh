python3 /llm_reco/dehua/code/verl/scripts/model_merger.py --backend fsdp \
    --local_dir /mmu_mllm_hdd_2/madehua/model/verl/checkpoints/verl_grpo_example_food101/qwen2_5_vl_7b_function_rm/global_step_500/actor \
    --target_dir /mmu_mllm_hdd_2/madehua/model/verl/checkpoints/verl_grpo_example_food101/qwen2_5_vl_7b_function_rm/global_step_500/qwen2_5_vl_7b \
    --hf_model_path /llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct