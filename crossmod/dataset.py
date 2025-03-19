import hashlib

import datasets

from crossmod.constants import (
    CACHE_MOD1_KEY,
    CACHE_MOD2_KEY,
    SEQUENCE_MOD1_KEY,
    SEQUENCE_MOD2_KEY,
)


def get_sequence_id(example, cache_key, sequence_key):
    """Generate id for the target sequence."""

    example[cache_key] = int(
        hashlib.sha256(example[sequence_key].encode()).hexdigest(), 16
    ) % (10**12)
    return example


def prepare_dataset(dataset_hf_name: str, cfg) -> datasets.DatasetDict:
    """Pull dataset from huggingface and prepare it.

    Optionally generate keys that will be used for caching.

    Args:
        dataset_hf_name: Unique huggingface identifier for the dataset.

    Returns:
        A huggingface dataset dict with 'train' and 'test' splits.
    """

    dataset = datasets.load_dataset(dataset_hf_name)

    if CACHE_MOD1_KEY in cfg and SEQUENCE_MOD1_KEY in cfg:
        ds = ds.map(
            lambda example: get_sequence_id(
                example, cfg[CACHE_MOD1_KEY], cfg[SEQUENCE_MOD1_KEY]
            )
        )
    if CACHE_MOD2_KEY in cfg and SEQUENCE_MOD2_KEY in cfg:
        ds = ds.map(
            lambda example: get_sequence_id(
                example, cfg[CACHE_MOD2_KEY], cfg[SEQUENCE_MOD2_KEY]
            )
        )

    return dataset
