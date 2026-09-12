from typing import List, Union
from transformers import AutoTokenizer
import torch

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 384

class ClinicalTextTokenizer:
    """
    Standardized tokenizer wrapper for clinical reports using Bio_ClinicalBERT.
    Enforces unified sequence truncation and padding to MAX_LENGTH (384).
    """
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = MAX_LENGTH,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str):
        """Encode a single report into PyTorch tensors."""
        return self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

    def encode_batch(self, texts: Union[List[str], tuple]):
        """Encode a collection of reports into batch PyTorch tensors."""
        return self.tokenizer(
            list(texts),
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
