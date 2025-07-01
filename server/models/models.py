from enum import Enum

class Model(str, Enum):
    LLAMA2 = "LLAMA2"
    LLAMA4 = "LLAMA4"

class ModelEngine(str, Enum):
    HUGGING_FACE = "hf"
    VLLM = "vllm"
    LORA = "lora"
    SGLANG = "sglang"