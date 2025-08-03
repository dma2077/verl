# python verl/inference/eval_model.py \
#      --model-path "/mmu_mllm_hdd_2/madehua/model/CKPT/food_model/Qwen2.5-VL-fru92_cold_sft_nofreeze"  \
#      --dataset fru92 \
#      --gpu-id "0"

# python verl/inference/eval_model.py \
#      --model-path "/mmu_mllm_hdd_2/madehua/model/CKPT/food_model/Qwen2.5-VL-veg200_cold_sft_nofreeze"  \
#      --dataset veg200 \
#      --gpu-id "1"
python verl/inference/eval_model.py \
     --model-path "/llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct"  \
     --dataset food2k \
     --gpu-id "3" \
     --with_category

# python verl/inference/eval_model.py \
#      --model-path "/llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct"  \
#      --dataset veg200 \
#      --gpu-id "2" \
#      --with_category
