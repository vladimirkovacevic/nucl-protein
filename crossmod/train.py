import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

import wandb
from crossmod.constants import (
    CACHE_MOD1_KEY,
    CACHE_MOD2_KEY,
    EPOCHS,
    LEARNING_RATE,
    MOD1_ATTN_MASK_NAME,
    MOD1_INPUT_IDS_NAME,
    MOD2_ATTN_MASK_NAME,
    MOD2_INPUT_IDS_NAME,
    TARGET,
    WARMUP_STEPS,
)


def train_model(model, train_dataloader, val_dataloader, cfg, device):
    def lr_lambda(step):
        if step < cfg[WARMUP_STEPS]:
            # Linear warmup
            return step / cfg[WARMUP_STEPS]
        else:
            remaining_steps = total_steps - cfg[WARMUP_STEPS]
            decay_step = step - cfg[WARMUP_STEPS]
            return max(
                0.5 * cfg[LEARNING_RATE], 1.0 - 0.5 * (decay_step / remaining_steps)
            )

    total_steps = cfg[EPOCHS] * len(train_dataloader)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg[LEARNING_RATE])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step = 0
    ACCUMULATION_STEPS = 1  # effectively turned off

    for epoch in range(cfg[EPOCHS]):
        print(f"Epoch: {epoch + 1}/{cfg[EPOCHS]}")
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_dataloader, desc="Training...")

        for batch in train_progress:
            mod1_input_ids = batch[cfg[MOD1_INPUT_IDS_NAME]].to(device)
            mod1_attention_mask = batch[cfg[MOD1_ATTN_MASK_NAME]].to(device)
            mod2_input_ids = batch[cfg[MOD2_INPUT_IDS_NAME]].to(device)
            mod2_attention_mask = batch[cfg[MOD2_ATTN_MASK_NAME]].to(device)
            mod1_cache_keys = (
                batch[cfg[CACHE_MOD1_KEY]] if cfg[CACHE_MOD1_KEY] else None
            )
            mod2_cache_keys = (
                batch[cfg[CACHE_MOD2_KEY]] if cfg[CACHE_MOD2_KEY] else None
            )
            targets = batch[cfg[TARGET]].unsqueeze(dim=-1).to(device)
            preds = model(
                modality1_input_ids=mod1_input_ids,
                modality1_attention_mask=mod1_attention_mask,
                modality2_input_ids=mod2_input_ids,
                modality2_attention_mask=mod2_attention_mask,
                modality1_cache_keys=mod1_cache_keys,
                modality2_cache_keys=mod2_cache_keys,
            )
            loss = criterion(preds, targets.float())
            loss.backward()
            train_loss += loss.item()
            step += 1
            if step % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            if step % 300 == 0:
                wandb.log({"train_loss": loss.item()})
                wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
        train_loss /= len(train_dataloader)

        model.eval()
        val_loss = 0.0
        val_progress = tqdm(val_dataloader, desc="Validation")
        with torch.no_grad():
            for batch in val_progress:
                mod1_input_ids = batch[cfg[MOD1_INPUT_IDS_NAME]].to(device)
                mod1_attention_mask = batch[cfg[MOD1_ATTN_MASK_NAME]].to(device)
                mod2_input_ids = batch[cfg[MOD2_INPUT_IDS_NAME]].to(device)
                mod2_attention_mask = batch[cfg[MOD2_ATTN_MASK_NAME]].to(device)
                mod1_cache_keys = (
                    batch[cfg[CACHE_MOD1_KEY]] if cfg[CACHE_MOD1_KEY] else None
                )
                mod2_cache_keys = (
                    batch[cfg[CACHE_MOD2_KEY]] if cfg[CACHE_MOD2_KEY] else None
                )
                targets = batch[cfg[TARGET]].unsqueeze(dim=-1).to(device)
                preds = model(
                    modality1_input_ids=mod1_input_ids,
                    modality1_attention_mask=mod1_attention_mask,
                    modality2_input_ids=mod2_input_ids,
                    modality2_attention_mask=mod2_attention_mask,
                    modality1_cache_keys=mod1_cache_keys,
                    modality2_cache_keys=mod2_cache_keys,
                )
                loss = criterion(preds, targets.float())
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        wandb.log({"val_loss": val_loss})
        print(f"Epoch: {epoch} - Train loss: {train_loss} - Validation loss: {val_loss}")


def evaluate_model(model, test_dataloader, cfg, device):
    model.eval()
    all_predictions = []
    all_targets = []
    test_progress = tqdm(test_dataloader, desc="Test set")
    with torch.no_grad():
        for batch in test_progress:
            mod1_input_ids = batch[cfg[MOD1_INPUT_IDS_NAME]].to(device)
            mod1_attention_mask = batch[cfg[MOD1_ATTN_MASK_NAME]].to(device)
            mod2_input_ids = batch[cfg[MOD2_INPUT_IDS_NAME]].to(device)
            mod2_attention_mask = batch[cfg[MOD2_ATTN_MASK_NAME]].to(device)
            mod1_cache_keys = (
                batch[cfg[CACHE_MOD1_KEY]] if cfg[CACHE_MOD1_KEY] else None
            )
            mod2_cache_keys = (
                batch[cfg[CACHE_MOD2_KEY]] if cfg[CACHE_MOD2_KEY] else None
            )
            targets = batch[cfg[TARGET]].unsqueeze(dim=-1).to(device)
            preds = model(
                modality1_input_ids=mod1_input_ids,
                modality1_attention_mask=mod1_attention_mask,
                modality2_input_ids=mod2_input_ids,
                modality2_attention_mask=mod2_attention_mask,
                modality1_cache_keys=mod1_cache_keys,
                modality2_cache_keys=mod2_cache_keys,
            )
            # transform preds to 0 - 1
            probs = torch.sigmoid(preds)
            preds = (probs > 0.5).float()
            all_targets.append(targets)
            all_predictions.append(preds)

    all_predictions = torch.cat(all_predictions).cpu()
    all_targets = torch.cat(all_targets).cpu()

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(all_targets, all_predictions))
