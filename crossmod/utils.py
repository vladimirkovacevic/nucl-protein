import subprocess as sp

import numpy as np
import torch

MB = 1024 * 1024


def count_parameters(model) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def torch_gpu_mem_info(device=None):
    if not device:
        device = "cuda:0"
    alloc = torch.cuda.memory_allocated(device) / MB
    res = torch.cuda.memory_reserved(device) / MB
    max_alloc = torch.cuda.max_memory_allocated(device) / MB
    max_res = torch.cuda.max_memory_reserved(device) / MB
    print(
        f"Allocated: {alloc} Reserved: {res} Max_alloc: {max_alloc} Max_res: {max_res}"
    )


def reset_cache_and_peaks():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def count_model_size_mb(model):
    return sum(p.nelement() * p.element_size() for p in model.parameters()) / MB


def count_parameters(model):
    return sum(p.nelement() for p in model.parameters())


def check_gpu_mem(device=None):
    if not device:
        device = "cuda:0"
    free, total = torch.cuda.mem_get_info(device)
    print(f"Free: {free / MB } Total: {total / MB }")


def get_tensor_size_mb(t):
    sz = t.element_size() * t.nelement()
    sz /= MB
    print(f"Size MiB: {sz}")


def is_model_on_gpu(model):
    print(f"Model is on GPU: {next(model.parameters()).is_cuda}")


def check_gpu_used_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    print("[GPUs]: ", memory_used_info)


def coverage_score(y_true, y_pred, tolerance=0.5):
    """
    Computes the percentage of predictions within the given tolerance of the true values.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.
        tolerance (float): Allowed deviation from the true values.

    Returns:
        float: Percentage of predictions within the tolerance.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    within_tolerance = np.abs(y_true - y_pred) <= tolerance
    return np.mean(within_tolerance) * 100
