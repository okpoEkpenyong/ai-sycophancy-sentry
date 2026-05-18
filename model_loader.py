import sys
import os
from pathlib import Path
sys.path.append('.')

import torch
from nnsight import LanguageModel, NNsight
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .config import setup_hf_cache

def get_cache_size(path):
    """Calculate the total size of files in a directory in megabytes."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 ** 2)  # MB

def load_model(model_id="Qwen/Qwen2.5-Coder-7B-Instruct", dtype=torch.float16):
    """
    Load a pretrained language model wrapped in nnsight for interpretability.
    
    Args:
        model_id: The model identifier from Hugging Face Hub.
    
    Returns:
        model: The nnsight LanguageModel object.
    """
    cache_dir = setup_hf_cache()
    
    # Fix for the deprecation warning
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )


    # Load via nnsight
    # nnsight passes extra kwargs (like quantization_config) to Hugging Face
    model = LanguageModel(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        cache_dir=cache_dir
    )
    
    return model, cache_dir