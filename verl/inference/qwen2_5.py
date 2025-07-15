import torch
import PIL
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import random

class Qwen2_5VL():
    def __init__(
            self,
            model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
            max_model_len: int = 2048,
            tensor_parallel_size: int = 1,
            temperature: float = 0.0
        ):
            self._model = LLM(
                model=model_path, 
                max_model_len=max_model_len, 
                trust_remote_code=True, 
                gpu_memory_utilization=0.9, 
                tensor_parallel_size=tensor_parallel_size,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self._sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=2048)
    def generate_until(self, messages):
        outputs = self._model.chat(messages, sampling_params=self._sampling_params)
        return outputs
