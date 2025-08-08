import csv
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
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
    TASK_TYPE,
    WARMUP_STEPS,
    TaskType,
)
from crossmod.utils import (
    classification_plots_plotly,
    coverage_score,
    regression_plots_plotly,
)

from crossmod.model import save_model_trainable_part


def contrastive_loss(embedding1, embedding2, labels, margin=1):
    """
    Computes contrastive loss between two batches of embeddings.

    Args:
        embedding1: [B, D] Tensor (e.g., modality1)
        embedding2: [B, D] Tensor (e.g., modality2)
        labels: [B] Tensor of 0 (negative) or 1 (positive)
        margin: distance margin for negative pairs

    Returns:
        Scalar loss
    """
    # Normalize to unit vectors
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)

    # Cosine similarity becomes L2 distance due to normalization
    distances = (embedding1 - embedding2).pow(2).sum(dim=1)

    # Contrastive loss
    labels = labels.view(-1)
    positive_loss = labels * distances
    negative_loss = (1 - labels) * F.relu(margin - torch.sqrt(distances + 1e-6)).pow(2)

    loss = 0.5 * (positive_loss + negative_loss).mean()
    return loss


def train_model(model, train_dataloader, val_dataloader, cfg, device):
    CONTRASTIVE_WEIGHT = 0.5
    if cfg[TASK_TYPE] not in [task.value for task in TaskType]:
        raise ValueError(
            f"Invalid task type '{cfg[TASK_TYPE]}'. Must be 'classification' or 'regression'."
        )

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

    if cfg[TASK_TYPE] == TaskType.CLASSIFICATION.value:
        criterion = nn.BCEWithLogitsLoss()
    elif cfg[TASK_TYPE] == TaskType.REGRESSION.value:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg[LEARNING_RATE])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step = 0
    ACCUMULATION_STEPS = 1  # effectively turned off

    # register_all_hooks(model, track_forward=False)
    print(
        "Num trainable params: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    for epoch in range(cfg[EPOCHS]):
        logging.info(f"Epoch: {epoch + 1}/{cfg[EPOCHS]}")
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_dataloader, desc="Training...")
        should_save_emb = epoch == (cfg[EPOCHS] - 1)

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
            preds, mod1_embs, mod2_embs = model(
                modality1_input_ids=mod1_input_ids,
                modality1_attention_mask=mod1_attention_mask,
                modality2_input_ids=mod2_input_ids,
                modality2_attention_mask=mod2_attention_mask,
                modality1_cache_keys=mod1_cache_keys,
                modality2_cache_keys=mod2_cache_keys,
                targets=targets,
                file_path=os.path.join(cfg["emb_path"], "training"),
                save_emb=should_save_emb,
            )
            loss_bce = criterion(preds, targets.float())
            loss_contrastive = contrastive_loss(mod1_embs, mod2_embs, targets)
            loss = (
                1 - CONTRASTIVE_WEIGHT
            ) * loss_bce + CONTRASTIVE_WEIGHT * loss_contrastive
            loss.backward()
            train_loss += loss.item()
            step += 1
            if step % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            if step % 50 == 0:
                wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
                wandb.log({"training_loss": loss.item()})
                wandb.log({"contrastive_loss": loss_contrastive.item()})

        train_loss /= len(train_dataloader)

        # save_model_trainable_part(model, f"trained_classification_1M_epoch_{epoch}.pth")

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
                preds, mod1_embs, mod2_embs = model(
                    modality1_input_ids=mod1_input_ids,
                    modality1_attention_mask=mod1_attention_mask,
                    modality2_input_ids=mod2_input_ids,
                    modality2_attention_mask=mod2_attention_mask,
                    modality1_cache_keys=mod1_cache_keys,
                    modality2_cache_keys=mod2_cache_keys,
                    targets=targets,
                    file_path=os.path.join(cfg["emb_path"], "validation"),
                    save_emb=should_save_emb,
                )
                loss_bce = criterion(preds, targets.float())
                loss_contrastive = contrastive_loss(mod1_embs, mod2_embs, targets)
                loss = (
                    1 - CONTRASTIVE_WEIGHT
                ) * loss_bce + CONTRASTIVE_WEIGHT * loss_contrastive
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        wandb.log({"val_loss_epoch": val_loss})
        wandb.log({"train_loss_epoch": train_loss})
        logging.info(
            f"Epoch: {epoch} - Train loss: {train_loss} - Validation loss: {val_loss}"
        )


def evaluate_model_classification(model, test_dataloader, cfg, device):
    model.eval()
    all_proba = []
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
            preds, _, _ = model(
                modality1_input_ids=mod1_input_ids,
                modality1_attention_mask=mod1_attention_mask,
                modality2_input_ids=mod2_input_ids,
                modality2_attention_mask=mod2_attention_mask,
                modality1_cache_keys=mod1_cache_keys,
                modality2_cache_keys=mod2_cache_keys,
                targets=targets,
                file_path=os.path.join(cfg["emb_path"], "test"),
                save_emb=True,
            )
            # transform preds to 0 - 1
            probs = torch.sigmoid(preds)
            preds = (probs > 0.5).float()
            all_targets.append(targets)
            all_predictions.append(preds)
            all_proba.append(probs)

    all_predictions = torch.cat(all_predictions).cpu().numpy().flatten()
    all_targets = torch.cat(all_targets).cpu().numpy().flatten()
    all_proba = torch.cat(all_proba).cpu().numpy().flatten()
    write_to_csv(
        all_targets,
        all_predictions,
        cfg[TARGET],
        "classification_results.csv",
    )
    classification_plots_plotly(all_targets, all_predictions, all_proba)

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_predictions)
    mcc = matthews_corrcoef(all_targets, all_predictions)
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1:.4f}")
    logging.info(f"ROC-AUC score: {auc:.4f}")
    logging.info(f"MCC score: {mcc:.4f}")
    logging.info(classification_report(all_targets, all_predictions))


def evaluate_model_regression(model, test_loader, cfg, device):
    model.eval()
    all_predictions = []
    all_targets = []
    test_progress = tqdm(test_loader, desc="Test set")
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
            all_targets.append(targets)
            all_predictions.append(preds)

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    write_to_csv(
        all_targets.cpu().numpy().flatten(),
        all_predictions.cpu().numpy().flatten(),
        cfg[TARGET],
        "regression_results.csv",
    )
    regression_plots_plotly(
        all_targets.cpu().numpy().flatten(), all_predictions.cpu().numpy().flatten()
    )

    loss = torch.nn.MSELoss()
    logging.info(
        f"Test set mean squared error (MSE): {loss(all_predictions, all_targets)}"
    )

    mse = mean_squared_error(all_targets.cpu(), all_predictions.cpu())
    mae = mean_absolute_error(all_targets.cpu(), all_predictions.cpu())
    r2 = r2_score(all_targets.cpu(), all_predictions.cpu())
    rmse = root_mean_squared_error(all_targets.cpu(), all_predictions.cpu())
    coverage05 = coverage_score(all_targets.cpu(), all_predictions.cpu(), tolerance=0.5)
    coverage1 = coverage_score(all_targets.cpu(), all_predictions.cpu(), tolerance=1)

    logging.info(
        f"MSE: {mse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f} RMSE: {rmse:.3f} Coverage +-0.5 {coverage05:.3f}% Coverage +-1 {coverage1:.3f}"
    )


def write_to_csv(y_true, y_pred, target_name, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([target_name, "predictions"])

        for true, pred in zip(y_true, y_pred):
            writer.writerow([pred, true])

    logging.info(f"Results saved to {filename}")


def register_all_hooks(
    model,
    log_thresh=1e-7,
    dead_thresh=1e-6,
    track_forward=True,
    track_backward=True,
    log_histogram=True,
    log_to_wandb=True,
    to_print=True,
    step_fn=lambda: wandb.run.step,  # Customize step if needed
):
    hooks = []

    def gradient_hook_fn(name):
        def hook(module, grad_input, grad_output):
            grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            if grad is not None:
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                grad_norm = grad.norm().item()
                is_vanishing = abs(grad_mean) < log_thresh and grad_std < log_thresh
                status = "🛑 VANISHING" if is_vanishing else "✅"
                if to_print:
                    print(
                        f"{status} Grad @ {name:<30} | mean={grad_mean:.2e} | std={grad_std:.2e}"
                    )

                if log_to_wandb:
                    wandb.log(
                        {
                            f"{name}/grad/mean": grad_mean,
                            f"{name}/grad/std": grad_std,
                            f"{name}/grad/norm": grad_norm,
                        },
                        step=step_fn(),
                    )

                    if log_histogram:
                        wandb.log(
                            {
                                f"{name}/grad/hist": wandb.Histogram(
                                    grad.detach().cpu().numpy()
                                )
                            },
                            step=step_fn(),
                        )

        return hook

    def forward_hook_fn(name):
        def hook(module, input, output):
            x = output if isinstance(output, torch.Tensor) else output[0]
            mean = x.mean().item()
            std = x.std().item()
            min_ = x.min().item()
            max_ = x.max().item()

            dead_ratio = (x.abs() < dead_thresh).float().mean().item()
            is_mostly_dead = dead_ratio > 0.99
            status = "💀 MOSTLY DEAD" if is_mostly_dead else "✅"

            if to_print:
                print(
                    f"{status} Forward @ {name:<30} | mean={mean:.2e} | std={std:.2e} | min={min_:.2e} | max={max_:.2e} | near_zero={dead_ratio:.2%}"
                )

            if log_to_wandb:
                wandb.log(
                    {
                        f"{name}/forward_mean": mean,
                        f"{name}/forward_std": std,
                        f"{name}/forward_dead_ratio": dead_ratio,
                        f"{name}/forward_activations": wandb.Histogram(
                            x.detach().cpu().numpy()
                        ),
                    }
                )

        return hook

    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            if track_backward:
                print(f"Tracking backward pass for: {name}")
                hooks.append(module.register_full_backward_hook(gradient_hook_fn(name)))
            if track_forward:
                print(f"Tracking forward pass for: {name}")
                hooks.append(module.register_forward_hook(forward_hook_fn(name)))

    return hooks
