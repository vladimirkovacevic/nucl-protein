"""
bimodal_protflash_nt.py

This script defines a bi-modal transformer architecture that combines:
    * ProtFlash – for encoding protein sequences
    * A HuggingFace nucleotide transformer – for encoding DNA/RNA sequences

It uses the 🤗 Transformers Trainer API for training, and a custom dataset
collator to handle the two-input modality.

Author: OpenAI o3, 2025-06-09
"""

# --- Standard libraries ---
import argparse  # For command-line interface
from dataclasses import dataclass  # For creating simple data structures
from typing import List, Dict, Any, Optional  # For type hints

# --- PyTorch imports ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset  # For building custom dataset loaders

# --- HuggingFace libraries ---
from datasets import load_dataset  # To load datasets from the HuggingFace hub
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForMaskedLM,
)

# --- wandb import ---
import wandb

# --- Device configuration ---
if torch.backends.mps.is_available():
    print("⚠️ Disabling MPS due to int64 issues.")
    torch.backends.mps._enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Metrics for evaluation ---
from transformers import EvalPrediction
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    if logits.ndim == 3:
        # Flatten (batch, seq_len, hidden) => something is wrong
        raise ValueError(f"Expected logits shape [batch_size, num_labels], got {logits.shape}")

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    if probs.shape[1] == 2:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = roc_auc_score(labels, probs, multi_class="ovr")

    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "auc": auc}

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from typing import Dict, Any

# --- Callback to print metrics at the end of each epoch ---
class PrintMetricsCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        train_metrics = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset, metric_key_prefix="train")
        val_metrics = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset, metric_key_prefix="val")

        print(f"\n🔁 Epoch {int(state.epoch)} Summary:")
        print(f"  ✅ Train  - Accuracy: {train_metrics['train_accuracy']:.4f} | AUC: {train_metrics['train_auc']:.4f}")
        print(f"  ✅ Val    - Accuracy: {val_metrics['val_accuracy']:.4f} | AUC: {val_metrics['val_auc']:.4f}")


# --- Import ProtFlash ---
try:
    from ProtFlash.pretrain import load_prot_flash_base  # Pretrained protein encoder
    from ProtFlash.utils import batchConverter  # For batch processing of protein sequences
except ImportError as e:
    raise ImportError(
        "ProtFlash not found. Install with: "
        "pip install git+https://github.com/isyslab-hust/ProtFlash"
    ) from e


# --- Configuration class used with HuggingFace Trainer ---
class BiModalConfig(PretrainedConfig):
    """
    A configuration class to define model dimensions and label counts.
    Necessary for integration with the 🤗 Transformers Trainer API.
    """

    def __init__(
        self,
        prot_hidden: int = 768,  # Dimension of protein encoder output
        nt_hidden: int = 768,    # Dimension of nucleotide encoder output
        num_labels: int = 2,     # Classification output dimension
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prot_hidden = prot_hidden
        self.nt_hidden = nt_hidden
        self.hidden_size = prot_hidden + nt_hidden
        self.num_labels = num_labels


# --- Bi-modal transformer model ---
class BiModalModel(PreTrainedModel):
    """
    Combines a protein encoder (ProtFlash) and a nucleotide transformer.
    Outputs a fused embedding for classification.
    """

    config_class = BiModalConfig

    def __init__(
        self,
        nt_model_name: str = "InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species",
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        cfg = BiModalConfig(num_labels=num_labels)
        super().__init__(cfg)

        # Load pretrained protein encoder (ProtFlash)
        self.prot_model = load_prot_flash_base()
        for param in self.prot_model.parameters():
            param.requires_grad = False  # Freeze ProtFlash

        # Load pretrained nucleotide tokenizer and transformer model
        self.nt_tokenizer = AutoTokenizer.from_pretrained(nt_model_name, trust_remote_code=True)
        self.nt_model = AutoModelForMaskedLM.from_pretrained(nt_model_name, trust_remote_code=True)

        # Set up classifier head (MLP) on top of fused protein + nucleotide embeddings
        prot_dim = 768
        nt_dim = self.nt_model.config.hidden_size
        hidden_dim = prot_dim + nt_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

        # Select loss function based on task type
        self.loss_fn = (
            nn.MSELoss() if num_labels == 1 else nn.CrossEntropyLoss()
        )

    @staticmethod
    def _mean_pool(tensor, mask):
        """Apply mean pooling over unmasked (valid) tokens."""
        masked = tensor * mask.unsqueeze(-1)
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-5)
        return summed / counts.unsqueeze(-1)

    def forward(
        self,
        prot_ids: torch.Tensor,
        prot_lengths: torch.Tensor,
        nt_input_ids: torch.Tensor,
        nt_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        # --- Encode protein sequences using ProtFlash ---
        prot_out = self.prot_model(prot_ids, prot_lengths)
        B = prot_out.size(0)
        seq_repr = []
        for i in range(B):
            # Average over all valid residues (ignore padding at position 0)
            seq_repr.append(prot_out[i, 1 : prot_lengths[i] + 1].mean(dim=0))
        prot_repr = torch.stack(seq_repr, dim=0)  # Shape: (batch_size, 768)

        # --- Encode nucleotide sequences ---
        nt_outputs = self.nt_model(
            input_ids=nt_input_ids,
            attention_mask=nt_attention_mask,
            output_hidden_states=True,
        )
        nt_repr = self._mean_pool(nt_outputs.hidden_states[-1], nt_attention_mask)

        # --- Concatenate both embeddings and classify ---
        fused = torch.cat([prot_repr, nt_repr], dim=-1)  # (B, hidden_size)
        logits = self.classifier(fused)

        # --- Compute loss if labels are provided ---
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss = self.loss_fn(logits.squeeze(-1), labels.float())
            else:
                loss = self.loss_fn(logits, labels.long())
            return (loss, logits)

        return logits


# --- Dataset class for dual-input modality ---
class BiModalDataset(Dataset):
    """
    Custom dataset for loading paired DNA/protein sequences from HuggingFace Hub.
    Assumes fields: 'seq_a', 'seq_b', 'seq_type_a', 'seq_type_b', 'label'.
    """

    def __init__(self, split: str = "train", hf_path: str = "vladak/CentralDogma"):
        self.ds = load_dataset(hf_path, split=split)

        # Normalize and align gene/protein based on type tags
        self.samples: List[Dict[str, Any]] = []
        for row in self.ds:
            if row["seq_type_a"] == "gene":
                gene_seq, prot_seq = row["seq_a"], row["seq_b"]
            else:
                gene_seq, prot_seq = row["seq_b"], row["seq_a"]

            self.samples.append({
                "gene_seq": gene_seq,
                "prot_seq": prot_seq,
                "label": row["label"],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# --- Collator to prepare batches with both protein and DNA sequences ---
@dataclass
class BiModalCollator:
    nt_tokenizer: AutoTokenizer

    def __post_init__(self):
        self.nt_tokenizer.padding_side = "right"

    def __call__(self, examples: List[Dict[str, Any]]):
        gene_seqs = [ex["gene_seq"] for ex in examples]
        prot_seqs = [(f"id{i}", ex["prot_seq"]) for i, ex in enumerate(examples)]
        labels = torch.tensor([ex["label"] for ex in examples])

        # Protein tokenization (using ProtFlash's batchConverter)
        _, prot_ids, lengths = batchConverter(prot_seqs)
        lengths = lengths.to(torch.long)

        # Nucleotide tokenization
        nt_enc = self.nt_tokenizer(
            gene_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )

        # Return full input batch dictionary
        return {
            "prot_ids": prot_ids,
            "prot_lengths": lengths,
            "nt_input_ids": nt_enc.input_ids,
            "nt_attention_mask": nt_enc.attention_mask,
            "labels": labels,
        }


# --- Argument parsing for CLI training interface ---
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


# --- Custom Trainer callback to log loss per step to wandb ---
class WandbLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log all trainer logs (including loss) to wandb
        if logs is not None:
            wandb.log(logs, step=state.global_step)


# --- Main training loop using HuggingFace Trainer ---
def main():
    args = parse_args()

    # Initialize wandb with your token and project name
    wandb.login(key="8650f0776cd5b7c69f0049de35e2da564004f132")
    wandb.init(project="ProtFlash", config=vars(args))

    # Initialize model and tokenizer
    model = BiModalModel()
    collator = BiModalCollator(model.nt_tokenizer)

    # Load train and validation splits
    ds_train = BiModalDataset("train")
    ds_val = BiModalDataset("val")

    # Define training configuration with wandb reporting enabled
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,  # log every 10 steps
        report_to="wandb",  # Enable wandb logging
        disable_tqdm=False,
        remove_unused_columns=False,
        fp16=not args.no_cuda and torch.cuda.is_available(),  # Mixed precision if CUDA
    )

    # Launch training using HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        compute_metrics=compute_metrics,  # Custom metrics function
    )
    # Add your custom callbacks for printing and wandb logging
    trainer.add_callback(PrintMetricsCallback(trainer))
    trainer.add_callback(WandbLogCallback())

    trainer.train()

    ds_test = BiModalDataset("test")
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=ds_test)
    print("Test metrics:", test_metrics)

    # Finish wandb run
    wandb.finish()


# --- Entry point for the script ---
if __name__ == "__main__":
    main()
