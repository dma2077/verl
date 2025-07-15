#!/usr/bin/env bash

# 根目录，请根据实际路径调整
BASE_ROOT="/mmu_mllm_hdd_2/madehua/model/CKPT/verl/DAPO"

# 要处理的数据集列表
datasets=(food101 food172 foodx251)

for ds in "${datasets[@]}"; do
  EXP_DIR="${BASE_ROOT}/Qwen2.5-VL-${ds}_cold_start-${ds}-0702"
  echo ">>> Dataset: ${ds}"
  
  for step in $(seq 50 50 250); do
    echo "    - merging global_step_${step}"
    
    LOCAL_DIR="${EXP_DIR}/global_step_${step}/actor"
    TARGET_DIR="${EXP_DIR}/global_step_${step}/qwen2_5_vl_7b"
    
    # # 执行合并
    # python -m verl.model_merger merge \
    #   --backend fsdp \
    #   --local_dir  "${LOCAL_DIR}" \
    #   --target_dir "${TARGET_DIR}"
    
    # 重命名输出目录为 qwen2_5_vl_7b_${step}
    mv "${TARGET_DIR}" "${EXP_DIR}/global_step_${step}/qwen2_5_vl_7b_${step}"
  done
done