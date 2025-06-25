#!/bin/bash

# Array of dataset names
datasets=("food101" "food172" "foodx251" "veg200" "fru92")
dataset_output_root=/llm_reco/dehua/data
# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    # Run the Python script with the dataset name
    python food_train.py --dataset_name "$dataset" -o "$dataset_output_root"
    
    # Check if the previous command was successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed $dataset"
    else
        echo "❌ Failed to process $dataset"
    fi
    
    echo "----------------------------------------"
done

echo "All datasets processing completed!"