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
    required_keys = ["images", "conversations"]
    for key in required_keys:
        if key not in item:
            raise KeyError(f"Missing required key: {key}")
    if not item["images"]:
        raise ValueError("Empty images list")
    if len(item["conversations"]) < 2:
        raise ValueError("Invalid conversations format")


def process_item(args):
    """Process a single data item."""
    item, idx, dataset_name = args
    try:
        # Validate input item
        validate_item(item)

        # Get image path
        image_path = item["images"][0]
        if not isinstance(image_path, str):
            raise TypeError(f"Image path is not a string: {type(image_path)}")

        # adjust path for local storage
        if "/map-vepfs/dehua/data/data/food-101/" in image_path:
            image_path = image_path.replace(
                "/map-vepfs/dehua/data/data/food-101/",
                "/llm_reco/dehua/data/food_data/food-101/"
            )
        else:
            image_path = image_path.replace(
                "/map-vepfs/dehua/data/data/",
                "/llm_reco/dehua/data/food_data/"
            ).replace("vegfru-dataset/", "")

        # Get answer
        answer = item["conversations"][1]["value"]
        if not isinstance(answer, str):
            raise TypeError(f"Answer is not a string: {type(answer)}")

        instruction_following = (
            r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
        )
        prompt = "<image>What is the dish?"

        # Create the message structure
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

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
                "ground_truth": answer
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "answer": answer,
                "question": prompt,
            },
        }
    except Exception as e:
        print(f"⚠️ Error processing item {idx}: {e}")
        print(f"Item content: {json.dumps(item, indent=2)}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON file to parquet format"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to process"
    )
    parser.add_argument(
        "--json", "-j",
        help="Input JSON file path"
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

    if args.json is None:
        args.json = f'/llm_reco/dehua/data/questions/{args.dataset_name}_question.json'
    if args.output_path is None:
        args.output_path = f'/llm_reco/dehua/data/{args.dataset_name}'

    dataset_name = args.dataset_name
    print(f"🚩 Reading JSON: {args.json}")
    with open(args.json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    # raw_data = raw_data[:1000]
    # Limit data for testing
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
    dataset_dict = DatasetDict({"train": dataset})

    # save locally
    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path, "train.parquet")
    print(f"💾 Saving dataset to: {out_file}")
    dataset_dict["train"].to_parquet(out_file)
    print("✅ Save completed")

    # copy to HDFS if needed
    if args.hdfs_dir:
        print(f"💾 Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=args.output_path, dst=args.hdfs_dir)
        print("✅ HDFS copy completed")

if __name__ == "__main__":
    main()