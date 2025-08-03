#!/bin/bash

# 定义要处理的数据集名称数组
datasets=("food101" "food172" "foodx251" "veg200" "fru92" "food2k")

for dataset in "${datasets[@]}"; do
    INPUT_JSON="/llm_reco/dehua/data/food_finetune_data/converted/${dataset}_cold_sft_8shot.json"
    OUTPUT_DIR="/llm_reco/dehua/data/8_shot/${dataset}"

    echo "Processing dataset: $dataset"
    echo "Input JSON: $INPUT_JSON"
    echo "Output Dir: $OUTPUT_DIR"

    python food_train.py \
        --dataset_name "$dataset" \
        --json "$INPUT_JSON" \
        --output_path "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed $dataset"
    else
        echo "❌ Failed to process $dataset"
    fi

    echo "----------------------------------------"
done

echo "All datasets processing completed!" 