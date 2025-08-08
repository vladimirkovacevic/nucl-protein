import logging
import os
import uuid

import numpy as np
import torch
import torch.nn as nn

from crossmod.embedding_cache import EmbeddingCache
from crossmod.model_registry import ModelRegistry


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.modality1_to_modality2_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.modality2_to_modality1_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        ffn_hidden_dim = embed_dim * 3
        self.ffn_modality1 = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )
        self.ffn_modality2 = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )

        self.modality1_norm = nn.LayerNorm(embed_dim)
        self.modality2_norm = nn.LayerNorm(embed_dim)
        self.ffn_modality1_norm = nn.LayerNorm(embed_dim)
        self.ffn_modality2_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        modality1_embedding,
        modality2_embedding,
        key_pad_mask_modality1,
        key_pad_mask_modality2,
    ):
        # Modality1 attending to Modality2
        attended_modality1, _ = self.modality1_to_modality2_attention(
            query=modality1_embedding,
            key=modality2_embedding,
            value=modality2_embedding,
            key_padding_mask=key_pad_mask_modality2,
        )
        attended_modality1 = self.modality1_norm(
            modality1_embedding + attended_modality1
        )
        x_modality1 = self.ffn_modality1(attended_modality1)
        x_modality1 = self.ffn_modality1_norm(
            attended_modality1 + self.dropout(x_modality1)
        )

        # Modality2 attending to Modality1
        attended_modality2, _ = self.modality2_to_modality1_attention(
            query=modality2_embedding,
            key=modality1_embedding,
            value=modality1_embedding,
            key_padding_mask=key_pad_mask_modality1,
        )
        attended_modality2 = self.modality2_norm(
            modality2_embedding + attended_modality2
        )
        x_modality2 = self.ffn_modality2(attended_modality2)
        x_modality2 = self.ffn_modality2_norm(
            attended_modality2 + self.dropout(x_modality2)
        )

        return x_modality1, x_modality2


class BiCrossAttentionModel(nn.Module):
    def __init__(
        self,
        modality1_model_name: str,
        modality2_model_name: str,
        num_layers: int = 3,
        hidden_dim: int = 1024,
        modality1_cache: EmbeddingCache | None = None,
        modality2_cache: EmbeddingCache | None = None,
        load_submodels: bool = True,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_submodels:
            self.modality1_model, self.modality1_tokenizer = self._load_model(
                modality1_model_name
            )

        if load_submodels:
            self.modality2_model, self.modality2_tokenizer = self._load_model(
                modality2_model_name
            )

        self.modality1_cache = modality1_cache
        self.modality2_cache = modality2_cache

        for param in self.modality1_model.parameters():
            param.requires_grad = False

        for param in self.modality2_model.parameters():
            param.requires_grad = False

        self.modality1_embedding_dim = self.modality1_model.config.hidden_size
        self.modality2_embedding_dim = self.modality2_model.config.hidden_size

        # Projecting to the size of Modality1 model
        self.project_to_common = nn.Linear(
            self.modality2_embedding_dim, self.modality1_embedding_dim
        )

        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(embed_dim=self.modality1_embedding_dim)
                for _ in range(num_layers)
            ]
        )

        self.ffn_head = nn.Sequential(
            nn.Linear(2 * self.modality1_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _load_model(self, model_name: str):
        model = ModelRegistry.get_model(model_name)
        tokenizer = ModelRegistry.get_tokenizer(model_name)
        model.eval()
        return model, tokenizer

    def forward(
        self,
        modality1_input_ids: torch.Tensor,
        modality1_attention_mask: torch.Tensor,
        modality2_input_ids: torch.Tensor,
        modality2_attention_mask: torch.Tensor,
        modality1_cache_keys: torch.Tensor | None = None,
        modality2_cache_keys: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        file_path: str = "",
        save_emb: bool = False,
    ):
        # Modality 1
        modality1_inputs = {
            "input_ids": modality1_input_ids,
            "attention_mask": modality1_attention_mask,
        }

        if self.modality1_cache:
            modality1_embedding = self.modality1_cache[modality1_cache_keys]
            modality1_embedding = modality1_embedding.to(self.device)
        else:
            # perform in FP16 for lower memory usage (matmuls)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    modality1_outputs = self.modality1_model(**modality1_inputs)
                    modality1_embedding = modality1_outputs.last_hidden_state

        special_tokens_mask_modality1 = (
            (modality1_inputs["input_ids"] == self.modality1_tokenizer.cls_token_id)
            | (modality1_inputs["input_ids"] == self.modality1_tokenizer.eos_token_id)
            | (modality1_inputs["input_ids"] == self.modality1_tokenizer.pad_token_id)
        )

        # Modality 2
        modality2_inputs = {
            "input_ids": modality2_input_ids,
            "attention_mask": modality2_attention_mask,
        }

        if self.modality2_cache:
            modality2_embedding = self.modality2_cache[modality2_cache_keys]
            modality2_embedding = modality2_embedding.to(self.device)
        else:
            # perform in FP16 for lower memory usage (matmuls)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    modality2_outputs = self.modality2_model(**modality2_inputs)
                    modality2_embedding = modality2_outputs.last_hidden_state

        special_tokens_mask_modality2 = (
            (modality2_inputs["input_ids"] == self.modality2_tokenizer.cls_token_id)
            | (modality2_inputs["input_ids"] == self.modality2_tokenizer.eos_token_id)
            | (modality2_inputs["input_ids"] == self.modality2_tokenizer.pad_token_id)
        )

        modality2_embedding = self.project_to_common(modality2_embedding)

        for layer in self.layers:
            residual_modality1 = modality1_embedding
            residual_modality2 = modality2_embedding

            modality1_embedding, modality2_embedding = layer(
                modality1_embedding,
                modality2_embedding,
                special_tokens_mask_modality1,
                special_tokens_mask_modality2,
            )

            # Add skip connection across the entire layer
            modality1_embedding = modality1_embedding + residual_modality1
            modality2_embedding = modality2_embedding + residual_modality2

        # Perform mean pooling
        modality2_embedding = (
            modality2_embedding * ~special_tokens_mask_modality2.unsqueeze(dim=-1)
        ).mean(dim=1)
        modality1_embedding = (
            modality1_embedding * ~special_tokens_mask_modality1.unsqueeze(dim=-1)
        ).mean(dim=1)

        # Combine embeddings
        combined = torch.cat([modality1_embedding, modality2_embedding], dim=1)
        # Save combined embeddings
        if save_emb:
            self._save_embeddings_to_dir(combined, targets, file_path)

        logits = self.ffn_head(combined)
        return logits, modality1_embedding, modality2_embedding

    def _save_embeddings_to_dir(self, embeddings, target, file):
        os.makedirs(file, exist_ok=True)

        for emb, label in zip(embeddings, target):
            emb = emb.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            random_name = f"{uuid.uuid4().hex}_{label}.npy"
            path = os.path.join(file, random_name)
            np.save(path, emb)


def save_model_trainable_part(model, filename="trained_model.pth"):
    model_state_dict = {
        "layers": model.layers.state_dict(),
        "project_to_common": model.project_to_common.state_dict(),
        "ffn_head": model.ffn_head.state_dict(),
    }
    torch.save(model_state_dict, filename)
    logging.info(f"✅ Trained model saved to {filename}")


def load_trained_model(model, filename):
    model_state_dict = torch.load(filename)

    model.layers.load_state_dict(model_state_dict["layers"])
    model.project_to_common.load_state_dict(model_state_dict["project_to_common"])
    model.ffn_class_head.load_state_dict(model_state_dict["ffn_head"])

    logging.info(f"✅ Trained model loaded from {filename}")
    return model
