import hashlib

import datasets
from datasets import DatasetDict

from crossmod.constants import (
    CACHE_MOD1_KEY,
    CACHE_MOD2_KEY,
    MOD1_SEQUENCE_NAME,
    MOD2_SEQUENCE_NAME,
)


def get_sequence_id(example, cache_key, sequence_key):
    """Generate id for the target sequence."""

    example[cache_key] = int(
        hashlib.sha256(example[sequence_key].encode()).hexdigest(), 16
    ) % (10**12)
    return example


def get_dataset(dataset_hf_name: str, cfg) -> datasets.DatasetDict:
    """Pull dataset from huggingface and prepare it.

    Optionally generate keys that will be used for caching.

    Args:
        dataset_hf_name: Unique huggingface identifier for the dataset.

    Returns:
        A huggingface dataset dict with 'train' and 'test' splits.
    """

    dataset = datasets.load_dataset(dataset_hf_name)

    if cfg[CACHE_MOD1_KEY] and MOD1_SEQUENCE_NAME in cfg:
        dataset = dataset.map(
            lambda example: get_sequence_id(
                example, cfg[CACHE_MOD1_KEY], cfg[MOD1_SEQUENCE_NAME]
            )
        )
    if cfg[CACHE_MOD2_KEY] and MOD2_SEQUENCE_NAME in cfg:
        dataset = dataset.map(
            lambda example: get_sequence_id(
                example, cfg[CACHE_MOD2_KEY], cfg[MOD2_SEQUENCE_NAME]
            )
        )

    return dataset


def subsample_dataset(
    dataset: datasets.DatasetDict, percentage: float = 0.02
) -> datasets.DatasetDict:
    """Intended for testing purposes to decrease dataset size.

    Args:
        dataset: Huggingface dataset dict.
        percentage: In range [0,1]
    """
    dataset = DatasetDict(
        {
            split: dataset[split]
            .shuffle(seed=42)
            .select(range(int(percentage * len(dataset[split]))))
            for split in dataset
        }
    )
    return dataset
