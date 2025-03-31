"""
This module contains the ModelRegistry class, which is a singleton for managing
the loading and caching of pre-trained models and tokenizers. It ensures that
models are not loaded multiple times and provides easy access to them.

Usage:
    model = ModelRegistry.get_model('model_name')
    tokenizer = ModelRegistry.get_tokenizer('model_name')
"""

import torch
from transformers import AutoModel, AutoTokenizer


class ModelRegistry:
    _models: dict[str, AutoModel] = {}
    _tokenizers: dict[str, AutoTokenizer] = {}
    _instance = None
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_model(cls, huggingface_name: str) -> AutoModel:
        if huggingface_name not in cls._models:
            model = AutoModel.from_pretrained(huggingface_name).to(cls._device)
            cls._models[huggingface_name] = model
        return cls._models[huggingface_name]

    @classmethod
    def get_tokenizer(cls, huggingface_name: str) -> AutoTokenizer:
        if huggingface_name not in cls._tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(huggingface_name)
            cls._tokenizers[huggingface_name] = tokenizer
        return cls._tokenizers[huggingface_name]
