"""
bimodal_protflash_nt.py

This script defines a bi-modal transformer architecture that combines:
    * ProtFlash – for encoding protein sequences
    * A HuggingFace nucleotide transformer – for encoding DNA/RNA sequences

It uses the 🤗 Transformers Trainer API for training, and a custom dataset
collator to handle the two-input modality.

"""

# --- Standard libraries ---
import argparse  # For command-line interface
from dataclasses import dataclass  # For creating simple data structures
from typing import List, Dict, Any, Optional  # For type hints
from datetime import datetime
import os, tempfile, pathlib

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score
)
import numpy as np

# --- PyTorch imports ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset  # For building custom dataset loaders
from transformers import EvalPrediction

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
td = tempfile.gettempdir()
fallback = os.path.expanduser("~/tmp")

if not os.access(td, os.W_OK):
    os.makedirs(fallback, exist_ok=True)
    os.environ["TMPDIR"] = fallback
    tempfile.tempdir = fallback


# --- wandb import ---
import wandb

# --- Device configuration ---
if torch.backends.mps.is_available():
    print("⚠️ Disabling MPS due to int64 issues.")
    torch.backends.mps._enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    num_labels = logits.shape[1] if logits.ndim > 1 else 1
    is_regression = num_labels == 1

    if is_regression:
        preds = logits.squeeze()
        labels = labels.squeeze()

        mse = mean_squared_error(labels, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, preds)
        r2 = r2_score(labels, preds)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    else:  # Classification
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")

        # AUC needs probability scores (for binary or multi-class)
        try:
            if num_labels == 2:  # Binary classification
                probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
                auc = roc_auc_score(labels, probs)
            else:  # Multi-class
                probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
                auc = roc_auc_score(labels, probs, multi_class="ovr")
        except Exception as e:
            print(f"⚠️ Could not compute AUC: {e}")
            auc = float("nan")

        return {
            "accuracy": acc,
            "f1": f1,
            "auc": auc
        }
        

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from typing import Dict, Any

# --- Callback to print metrics at the end of each epoch ---
class PrintMetricsCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        train_metrics = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset, metric_key_prefix="train")
        val_metrics = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset, metric_key_prefix="val")
    
        if "train_accuracy" in train_metrics:
            # <---------- Print classification metrics
            print(f"\n🔁 Epoch {int(state.epoch)} Summary:")
            print(f"  ✅ Train  - Accuracy: {train_metrics['train_accuracy']:.4f} | AUC: {train_metrics['train_auc']:.4f}")
            print(f"  ✅ Val    - Accuracy: {val_metrics['val_accuracy']:.4f} | AUC: {val_metrics['val_auc']:.4f}")
        else:
            # <---------- Print regression metrics
            print(f"\n🔁 Epoch {int(state.epoch)} Summary:")
            print(f"  ✅ Train  - MSE: {train_metrics['train_mse']:.4f} | MAE: {train_metrics['train_mae']:.4f}")
            print(f"  ✅ Val    - MSE: {val_metrics['val_mse']:.4f} | MAE: {val_metrics['val_mae']:.4f}")
    
        wandb.log(train_metrics | val_metrics, step=state.global_step)


class GradientCheckCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"⚠️ NaN detected in gradients of {name}")
                    control.should_training_stop = True  # Optionally halt training
                    break
                elif param.grad.abs().sum() == 0:
                    print(f"⚠️ Zero gradients in {name}")


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
        # dropout: float = 0.1,
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
        for param in self.nt_model.parameters():
            param.requires_grad = False  # Freeze nucleotide transformer weights

        # Set up classifier head (MLP) on top of fused protein + nucleotide embeddings
        prot_dim = 768
        nt_dim = self.nt_model.config.hidden_size
        hidden_dim = prot_dim + nt_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),     # compress from 1280 → 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),       # further reduce to 64
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_labels)          # binary classification (or num_classes)
        )
        self._init_classifier_weights()

        # Select loss function based on task type
        self.loss_fn = (
            nn.MSELoss() if num_labels == 1 else nn.CrossEntropyLoss()
        )
    def _init_classifier_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    @staticmethod
    def _mean_pool(tensor, mask):
        """Apply mean pooling over unmasked (valid) tokens.
           Prints a warning if a sequence is completely masked or if the resulting embedding is near zero.
        """
        # Check for any sequences with no valid tokens (sum==0)
        valid_counts = mask.sum(dim=1)
        if (valid_counts == 0).any():
            print("⚠️ Warning: Some sequences have an all-zero attention mask!")
        
        # Avoid division by zero by ensuring at least one token is considered.
        counts = valid_counts.clamp(min=1e-5)
        masked = tensor * mask.unsqueeze(-1)
        summed = masked.sum(dim=1)
        pooled = summed / counts.unsqueeze(-1)
        
        # Optionally check if the pooled embedding is nearly zero (could indicate an issue)
        threshold = 1e-6
        if (pooled.abs().sum(dim=1) < threshold).any():
            print("⚠️ Warning: Pooled embedding is near zero for some sequences.")
        
        return pooled


    def forward(
        self,
        # prot_ids: torch.Tensor,
        # prot_lengths: torch.Tensor,
        # nt_input_ids: torch.Tensor,
        # nt_attention_mask: torch.Tensor,
        # labels: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,          
        protein_input_ids: torch.Tensor,       
        protein_lengths: torch.Tensor,         
        labels: Optional[torch.Tensor] = None
    ):
        # Get the device the model is on
        device = next(self.parameters()).device
    
        # Move all inputs to the model's device
        protein_input_ids = protein_input_ids.to(device)
        protein_lengths = protein_lengths.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        # --- Encode protein sequences using ProtFlash ---
        prot_out = self.prot_model(protein_input_ids, protein_lengths)
        if torch.isnan(prot_out).any():
            print(f"⚠️ NaNs in raw ProtFlash output")
            print(f"protein_input_ids shape: {protein_input_ids.shape}")
            print(f"protein_lengths: {protein_lengths}")
            print('Prot ids: \n', protein_input_ids.cpu())
            print('Prot out: \n', prot_out.cpu())
            with open("nan_prot_tokens.txt", "w") as f:
                for seq in protein_input_ids.cpu().tolist():
                    # seq is list of ints (token IDs)
                    f.write(" ".join(map(str, seq)) + "\n")
            import sys
            sys.exit()

            nan_mask = torch.isnan(prot_out)
            print(f"NaNs at indices: {nan_mask.nonzero(as_tuple=True)}")
            batch_size = prot_out.size(0)
            
            prot_out = torch.nan_to_num(prot_out, nan=0.0, posinf=0.0, neginf=0.0) # NaN guard

        B = prot_out.size(0)
        seq_repr = []
        for i in range(B):
            # Average over all valid residues (ignore padding at position 0)
            seq_repr.append(prot_out[i, 1 : protein_lengths[i] + 1].mean(dim=0))
        prot_repr = torch.stack(seq_repr, dim=0)  # Shape: (batch_size, 768)
    
        # --- Encode nucleotide sequences ---
        nt_outputs = self.nt_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        nt_repr = self._mean_pool(nt_outputs.hidden_states[-1], attention_mask)
    
        # --- CHECK: Warn if zero or NaN embeddings ---
        prot_norms = prot_repr.norm(dim=1)
        nt_norms = nt_repr.norm(dim=1)
    
        if (prot_norms == 0).any():
            print("⚠️ Warning: Zero norm detected in protein embeddings!")
        if (nt_norms == 0).any():
            print("⚠️ Warning: Zero norm detected in nucleotide embeddings!")
        if torch.isnan(prot_repr).any():
            print("⚠️ Warning: NaNs detected in protein embeddings!")
        if torch.isnan(nt_repr).any():
            print("⚠️ Warning: NaNs detected in nucleotide embeddings!")
    
        # --- Concatenate both embeddings and classify ---
        fused = torch.cat([prot_repr, nt_repr], dim=-1)  # (B, hidden_size)
        logits = self.classifier(fused)
    
        # --- Debug: Check if logits are valid ---
        with torch.no_grad():
            if torch.isnan(logits).any():
                print("⚠️ NaNs detected in logits!")
            elif (logits.std(dim=1) < 1e-5).all():
                print("⚠️ All logits are nearly identical (possible dead network).")
                print("Logits sample:", logits[0].tolist())
    
        # --- Compute loss if labels are provided ---
        loss = None
        if labels is not None:
            # --- Sanity checks for loss inputs ---
            if self.config.num_labels > 1:
                # Check shape: logits should be (B, C), labels should be (B,)
                if logits.dim() != 2 or labels.dim() != 1 or logits.size(0) != labels.size(0):
                    raise ValueError(
                        f"❌ Mismatched logits/labels shapes: logits {logits.shape}, labels {labels.shape}"
                    )
        
                # Check label range: must be in [0, num_labels-1]
                if labels.min() < 0 or labels.max() >= self.config.num_labels:
                    raise ValueError(
                        f"❌ Labels out of range: min={labels.min().item()}, max={labels.max().item()}, "
                        f"expected ∈ [0, {self.config.num_labels - 1}]"
                    )
        
                # Check type
                if labels.dtype != torch.long:
                    print(f"⚠️ Warning: Casting labels from {labels.dtype} to long")
                    labels = labels.long()
        
                loss = self.loss_fn(logits, labels)
        
            else:  # Regression
                # Make sure logits and labels have the same shape for loss
                logits = logits.view(-1)  # shape (B,)
                labels = labels.float().view(-1)  # shape (B,)
                loss = self.loss_fn(logits, labels)

        return (loss, logits)

    @property
    def input_names(self):
        return ["input_ids", "attention_mask", "protein_input_ids", "protein_lengths"]

# --- Dataset class for dual-input modality ---
class BiModalDataset(Dataset):
    """
    Custom dataset for loading paired DNA/protein sequences from HuggingFace Hub.
    Assumes fields: 'seq_a', 'seq_b', 'seq_type_a', 'seq_type_b', 'label'.
    """

    def __init__(self, split: str = "train", hf_path: str = "vladak/ncRPI"):
        self.ds = load_dataset(hf_path, split=split) # , download_mode="force_redownload"
        self.samples = []  # <-- initialize this

        def is_nucleotide(seq):
            return all(c in "ACGT" for c in seq.upper())

        for row in self.ds:
            seq_a, seq_b = row["seq_a"], row["seq_b"]
        
            if is_nucleotide(seq_a):
                gene_seq, prot_seq = seq_a, seq_b
            elif is_nucleotide(seq_b):
                gene_seq, prot_seq = seq_b, seq_a
            else:
                raise ValueError(f"❌ Neither sequence looks like a nucleotide sequence:\nseq_a: {seq_a}\nseq_b: {seq_b}")
        
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

        return {
            "input_ids": nt_enc.input_ids,                  # used for nucleotides
            "attention_mask": nt_enc.attention_mask,
            "protein_input_ids": prot_ids,                  # used for proteins
            "protein_lengths": lengths,
            "labels": labels,
        }


# --- Argument parsing for CLI training interface ---
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_path", type=str, default="vladak/ncRPI", help="Path to HuggingFace dataset") # vladak/CentralDogma 
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


# # --- Custom Trainer callback to log loss per step to wandb ---
# class WandbLogCallback(TrainerCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         # Log all trainer logs (including loss) to wandb
#         if logs is not None:
#             wandb.log(logs, step=state.global_step)


# --- Main training loop using HuggingFace Trainer ---
def main():
    args = parse_args()

    # Initialize wandb with your token and project name
    wandb.login(key="8650f0776cd5b7c69f0049de35e2da564004f132")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project="BiModalProtFlashNT",
        config=vars(args),
        name=f"run_with_{args.hf_path.replace('/', '_')}_{timestamp}",  # Optional run name
        notes=f"Using dataset: {args.hf_path}"              # Adds dataset info to Wandb run
    )



    # Load train and validation splits
    ds_train = BiModalDataset("train", hf_path=args.hf_path)
    ds_val = BiModalDataset("val", hf_path=args.hf_path)
    ds_test = BiModalDataset("test", hf_path=args.hf_path)

    # --- Label sanity check ---
    labels = [s["label"] for s in ds_train.samples]
    label_counts = {l: labels.count(l) for l in set(labels)}
    print("🧪 Label distribution in training set:", label_counts)

    # <---------- Check if task is regression (more than 2 unique labels)
    unique_labels = set(labels)
    is_regression = len(unique_labels) > 2
    num_labels = 1 if is_regression else len(unique_labels)
    
    # Initialize model and tokenizer
    # <---------- Pass task-specific label configuration
    model = BiModalModel(num_labels=num_labels)

    collator = BiModalCollator(model.nt_tokenizer)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params:,}")
    wandb.config.update({"trainable_parameters": trainable_params})

    # Define training configuration with wandb reporting enabled
    steps_per_epoch = len(ds_train) // args.per_device_train_batch_size
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        warmup_steps=500,     # gradual warmup
        save_strategy="no",
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,
        report_to="wandb",  # Enable wandb logging
        disable_tqdm=False,
        remove_unused_columns=False,
        bf16=not args.no_cuda and torch.cuda.is_available(),  # Mixed precision if CUDA
        max_grad_norm=1.0,  # 💡 This enables gradient clipping! Prevents exploding
        run_name = f"bimodal-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
    # trainer.add_callback(WandbLogCallback())
    trainer.add_callback(GradientCheckCallback())

    trainer.train()

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=ds_test)
    print("Test metrics:", test_metrics)
    
    # ✅ Log test metrics to wandb manually
    # <---------- Log all test metrics (handles regression or classification)
    test_metrics_prefixed = {f"test_{k[5:]}": v for k, v in test_metrics.items() if k.startswith("eval_")}
    wandb.log(test_metrics_prefixed, step=trainer.state.global_step)



    # Finish wandb run
    wandb.finish()


# --- Entry point for the script ---
if __name__ == "__main__":
    main()
