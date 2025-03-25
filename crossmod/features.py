import datasets
import torch
from datasets import DatasetDict

from crossmod.constants import (
    CACHE_MOD1_KEY,
    CACHE_MOD2_KEY,
    MOD1_ATTN_MASK_NAME,
    MOD1_INPUT_IDS_NAME,
    MOD1_MODEL_NAME,
    MOD1_SEQUENCE_NAME,
    MOD2_ATTN_MASK_NAME,
    MOD2_INPUT_IDS_NAME,
    MOD2_MODEL_NAME,
    MOD2_SEQUENCE_NAME,
    TARGET,
)
from crossmod.model_registry import ModelRegistry


class CustomDataCollator:
    def __init__(self, mod1_collator, mod2_collator, cfg):
        self.mod1_collator = mod1_collator
        self.mod2_collator = mod2_collator
        self.cfg = cfg

    def __call__(self, batch):
        batch_mod1 = [
            {
                "input_ids": b[self.cfg[MOD1_INPUT_IDS_NAME]],
                "attention_mask": b[self.cfg[MOD1_ATTN_MASK_NAME]],
                **(
                    {self.cfg[CACHE_MOD1_KEY]: b[self.cfg[CACHE_MOD1_KEY]]}
                    if self.cfg[CACHE_MOD1_KEY]
                    else {}
                ),
            }
            for b in batch
        ]

        batch_mod2 = [
            {
                "input_ids": b[self.cfg[MOD2_INPUT_IDS_NAME]],
                "attention_mask": b[self.cfg[MOD2_ATTN_MASK_NAME]],
                **(
                    {self.cfg[CACHE_MOD2_KEY]: b[self.cfg[CACHE_MOD2_KEY]]}
                    if self.cfg[CACHE_MOD2_KEY]
                    else {}
                ),
            }
            for b in batch
        ]

        collated_mod1 = self.mod1_collator(batch_mod1)
        collated_mod2 = self.mod2_collator(batch_mod2)

        return {
            **(
                {self.cfg[CACHE_MOD1_KEY]: collated_mod1[self.cfg[CACHE_MOD1_KEY]]}
                if self.cfg[CACHE_MOD1_KEY] in collated_mod1
                else {}
            ),
            **(
                {self.cfg[CACHE_MOD2_KEY]: collated_mod2[self.cfg[CACHE_MOD2_KEY]]}
                if self.cfg[CACHE_MOD2_KEY] in collated_mod2
                else {}
            ),
            self.cfg[MOD1_INPUT_IDS_NAME]: collated_mod1["input_ids"],
            self.cfg[MOD1_ATTN_MASK_NAME]: collated_mod1["attention_mask"],
            self.cfg[MOD2_INPUT_IDS_NAME]: collated_mod2["input_ids"],
            self.cfg[MOD2_ATTN_MASK_NAME]: collated_mod2["attention_mask"],
            self.cfg[TARGET]: torch.tensor(
                [x[self.cfg[TARGET]] for x in batch]
            ),  # this needs to be changed as well to support regression etc.
        }


def train_test_validation_split(dataset: datasets.DatasetDict) -> datasets.DatasetDict:
    """Perform train-test-validation split.

    DatasetDict from huggingface is expected to have train and test split.
    We split 50:50 split the test split to generate test and validation split
    in final DatasetDict.

    Args:
        dataset: DatasetDict from huggingface.

    Returns:
        DatasetDict with train, test and validation splits.
    """
    dataset_test = dataset["test"]
    dataset_test_val = dataset_test.train_test_split(test_size=0.5)

    dataset_dict = {
        "train": dataset["train"],
        "test": dataset_test_val["train"],
        "validation": dataset_test_val["test"],
    }
    dataset = DatasetDict(dataset_dict)
    return dataset


def tokenize_modality(
    examples, tokenizer, input_ids_name: str, attn_mask_name: str, sequence_name: str
):
    """Tokenize data and give columns a specific name related
    to the modality. Since we have 2 modalities, by default
    tokenizer would create input_ids and attention_mask twice
    and overwrite first run. That is the reason for custom
    names.
    """
    tokenized = tokenizer(examples[sequence_name])
    return {
        input_ids_name: tokenized["input_ids"],
        attn_mask_name: tokenized["attention_mask"],
    }


def tokenize_data(dataset: datasets.DatasetDict, cfg):
    """Apply tokenizers for both modalities."""
    modality1_tokenizer = ModelRegistry.get_tokenizer(cfg[MOD1_MODEL_NAME])
    modality2_tokenizer = ModelRegistry.get_tokenizer(cfg[MOD2_MODEL_NAME])
    print(f"Modality 1 tokenizer is fast: {modality1_tokenizer.is_fast}")
    print(f"Modality 2 tokenizer is fast: {modality2_tokenizer.is_fast}")

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_modality(
            examples,
            modality1_tokenizer,
            cfg[MOD1_INPUT_IDS_NAME],
            cfg[MOD1_ATTN_MASK_NAME],
            cfg[MOD1_SEQUENCE_NAME],
        ),
        batched=True,
    )
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: tokenize_modality(
            examples,
            modality2_tokenizer,
            cfg[MOD2_INPUT_IDS_NAME],
            cfg[MOD2_ATTN_MASK_NAME],
            cfg[MOD2_SEQUENCE_NAME],
        ),
        batched=True,
    )
    return tokenized_dataset
