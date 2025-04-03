import datetime
import logging
import os

import click
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

import wandb
from crossmod.config import ConfigProvider
from crossmod.constants import (
    BATCH_SIZE,
    DATASET_NAME,
    EPOCHS,
    LEARNING_RATE,
    MOD1_MODEL_NAME,
    MOD2_MODEL_NAME,
    WANDB_NAME,
    WANDB_PROJECT,
    WARMUP_STEPS,
)
from crossmod.data_preprocessing import get_dataset
from crossmod.embedding_cache import EmbeddingCache
from crossmod.features import (
    CustomDataCollator,
    tokenize_data,
    train_test_validation_split,
)
from crossmod.model import BiCrossAttentionModel
from crossmod.model_registry import ModelRegistry
from crossmod.train import evaluate_model_regression, train_model


def init_logger(cfg, timestamp):
    """
    Creates a logger object that store logs in the log_dir
    Args:
        config (dict): Configuration file containing log_dir
        timestamp (string): Timestamp of the current run
    """
    log_dir = cfg.get("log_dir", "./logs/")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"logfile_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the config JSON file.",
)
@click.option("--wandb_key", type=str, default=None, help="WandB API key (optional).")
def main(config_path, wandb_key):
    cfg = ConfigProvider.get_config(config_path)

    if wandb_key is not None:
        wandb.login(key=wandb_key)
    else:
        # wandb.require("offline")
        wandb.login()

    wandb.init(
        project=cfg[WANDB_PROJECT],
        name=cfg[WANDB_NAME],
        config={
            "WARMUP_STEPS": cfg[WARMUP_STEPS],
            "EPOCHS": cfg[EPOCHS],
            "BATCH_SIZE": cfg[BATCH_SIZE],
            "LR": cfg[LEARNING_RATE],
        },
    )

    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    init_logger(cfg, timestamp)

    mod1_model_name = cfg[MOD1_MODEL_NAME]
    mod2_model_name = cfg[MOD2_MODEL_NAME]

    dataset = get_dataset(cfg[DATASET_NAME], cfg)
    dataset = train_test_validation_split(dataset)
    tokenized_dataset = tokenize_data(dataset, cfg)

    mod1_collator = DataCollatorWithPadding(
        tokenizer=ModelRegistry.get_tokenizer(mod1_model_name)
    )
    mod2_collator = DataCollatorWithPadding(
        tokenizer=ModelRegistry.get_tokenizer(mod2_model_name)
    )
    collator = CustomDataCollator(
        mod1_collator=mod1_collator, mod2_collator=mod2_collator, cfg=cfg
    )
    train_dataloader = DataLoader(
        tokenized_dataset["train"], batch_size=cfg[BATCH_SIZE], collate_fn=collator
    )
    test_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=cfg[BATCH_SIZE], collate_fn=collator
    )
    val_dataloader = DataLoader(
        tokenized_dataset["validation"], batch_size=cfg[BATCH_SIZE], collate_fn=collator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dna_cache = EmbeddingCache(
    #     data=tokenized_dataset["train"],
    #     key=cfg[CACHE_MOD2_KEY],
    #     input_ids_name=cfg[MOD2_INPUT_IDS_NAME],
    #     attention_mask_name=cfg[MOD2_ATTN_MASK_NAME],
    #     emb_model_name=cfg[MOD2_MODEL_NAME],
    #     device=device,
    # )
    model = BiCrossAttentionModel(
        modality1_model_name=mod1_model_name,
        modality2_model_name=mod2_model_name,
        # modality2_cache=dna_cache,
    ).to(device)

    logging.info("Starting training...")

    train_model(model, train_dataloader, val_dataloader, cfg, device)
    evaluate_model_regression(model, test_dataloader, cfg, device)

    logging.info("Finished training...")
    # TODO Implement model saving but only trainable part
    # TODO Add logging
    # TODO add support for regression besides classification


if __name__ == "__main__":
    main()
