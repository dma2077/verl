# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the food recognition dataset to parquet format
"""

import argparse
import json
import os
import io
import multiprocessing
from datasets import Dataset, DatasetDict
from PIL import Image as PILImage
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image
from verl.utils.hdfs_io import copy, makedirs


food172_category_file = "/llm_reco/dehua/data/food_data/VireoFood172/SplitAndIngreLabel/FoodList.txt"

with open(food172_category_file, 'r', encoding='utf-8') as file:
    category_dict = {}
    for idx, line in enumerate(file.readlines()):
        line = line.strip()
        category_dict[idx+1] = line

def convert_image_to_bytes_format(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()  # 直接获取字节值
        return {
            'bytes': img_byte_arr,
            'path': None
        }

def validate_item(item):
    """Validate the input item structure."""
    if "image" not in item:
        raise KeyError("Missing required key: image")


def process_item(args):
    """Process a single data item."""
    item, idx, dataset_name = args
    try:
        # Validate input item
        validate_item(item)

        # Get image path
        image_path = item["image"]
        if not isinstance(image_path, str):
            raise TypeError(f"Image path is not a string: {type(image_path)}")
        
        # adjust path for local storage
        image_path = image_path.replace(
            "/map-vepfs/dehua/data/data/",
            "/llm_reco/dehua/data/food_data/"
        ).replace("vegfru-dataset/", "")

        # derive category from image path
        category = os.path.basename(os.path.dirname(image_path)).replace('_', ' ')
        if "food172" in dataset_name:
            category = category_dict[int(category)]
        instruction_following = (
            r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
        )
        prompt = "<image>Please analyze these food attributes in the image: shape, texture, composition, color, and cooking style. Then identify the food category."

        # Create the message structure
        message = {
            "role": "user",
            "content": prompt
        }
        messages = [message]
        # Convert image
        try:
            image_data = convert_image_to_bytes_format(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to convert image: {str(e)}")

        return {
            "data_source": dataset_name,
            "prompt": messages,
            "images": [image_data],
            "ability": "food_recognition",
            "reward_model": {
                "style": "rule",
                "ground_truth": category
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "answer": category,
                "question": prompt,
            },
        }
    except Exception as e:
        print(f"⚠️ Error processing item {idx}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL file to parquet format"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to process"
    )
    parser.add_argument(
        "--jsonl", "-j",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output_path", "-o",
        help="Output parquet file directory"
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS directory for output"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=64,
        help="Number of processes for parallel processing"
    )
    args = parser.parse_args()

    if args.jsonl is None:
        args.jsonl = f'/llm_reco/dehua/code/visual-memory/questions/{args.dataset_name}/siglip_test_5_softmax.jsonl'
    if args.output_path is None:
        args.output_path = f'/llm_reco/dehua/data/{args.dataset_name}'
    else:
        args.output_path = args.output_path + f"/{args.dataset_name}"

    dataset_name = f"{args.dataset_name}_test"

    print(f"🚩 Reading JSONL: {args.jsonl}")
    raw_data = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))

    raw_data = raw_data[:1024]

    total_items = len(raw_data)
    num_proc = min(args.num_proc, multiprocessing.cpu_count())
    chunksize = max(1, total_items // num_proc // 4)

    print(f"🚩 Processing {total_items} items using {num_proc} processes (chunksize={chunksize})...")
    process_args = [(item, idx, dataset_name) for idx, item in enumerate(raw_data)]

    with Pool(processes=num_proc, maxtasksperchild=1000) as pool:
        processed = []
        for result in tqdm(
            pool.imap_unordered(process_item, process_args, chunksize=chunksize),
            total=total_items,
            desc="Processing items"
        ):
            if result is not None:
                processed.append(result)

    print(f"Successfully processed {len(processed)}/{total_items} items")

    # build dataset without specifying features
    dataset = Dataset.from_list(processed)
    dataset_dict = DatasetDict({"test": dataset})

    # save locally
    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path, "test.parquet")
    print(f"💾 Saving dataset to: {out_file}")
    dataset_dict["test"].to_parquet(out_file)
    print("✅ Save completed")

    # copy to HDFS if needed
    if args.hdfs_dir:
        print(f"💾 Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=args.output_path, dst=args.hdfs_dir)
        print("✅ HDFS copy completed")

if __name__ == "__main__":
    main()