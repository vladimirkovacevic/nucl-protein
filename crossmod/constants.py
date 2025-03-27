# all these fields expected in config file, this is just to avoid using
# hardcoded strings everywhere
from enum import Enum

DATASET_NAME = "dataset_name"
CACHE_MOD1_KEY = "cache_mod1_key"
SEQUENCE_MOD1_KEY = "sequence_mod1_key"
CACHE_MOD2_KEY = "cache_mod2_key"
SEQUENCE_MOD2_KEY = "sequence_mod2_key"
MOD1_MODEL_NAME = "mod1_model_name"
MOD2_MODEL_NAME = "mod2_model_name"
MOD1_INPUT_IDS_NAME = "mod1_input_ids_name"
MOD2_INPUT_IDS_NAME = "mod2_input_ids_name"
MOD1_ATTN_MASK_NAME = "mod1_attention_mask_name"
MOD2_ATTN_MASK_NAME = "mod2_attention_mask_name"
MOD1_SEQUENCE_NAME = "mod1_sequence_name"
MOD2_SEQUENCE_NAME = "mod2_sequence_name"
WARMUP_STEPS = "warmup_steps"
EPOCHS = "epochs"
BATCH_SIZE = "batch_size"
LEARNING_RATE = "lr"
WANDB_PROJECT = "wandb_project"
WANDB_NAME = "wandb_name"
TARGET = "target"
TASK_TYPE = "task_type"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
