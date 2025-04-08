import logging
import subprocess as sp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from plotly.subplots import make_subplots
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

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


# OLD CODE
# def regression_plots(y_true, y_pred):
#     # Calculate residuals
#     residuals = np.array(y_pred) - np.array(y_true)

#     plt.figure(figsize=(18, 5))

#     # 1. Predicted vs True
#     # Overall correlation
#     # Whether the model over- or under-predicts
#     # Outliers or systematic biases
#     plt.subplot(1, 3, 1)
#     sns.scatterplot(x=y_true, y=y_pred)
#     plt.plot(
#         [min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--", label="Ideal"
#     )
#     plt.xlabel("True Values")
#     plt.ylabel("Predicted Values")
#     plt.title("Predicted vs. True")
#     plt.legend()

#     # 2. Residual plot
#     # Homoscedasticity (constant variance of residuals)
#     # Systematic errors (e.g. always over-predicting when true values are high)
#     plt.subplot(1, 3, 2)
#     sns.scatterplot(x=y_true, y=residuals)
#     plt.axhline(0, linestyle="--", color="gray")
#     plt.xlabel("True Values")
#     plt.ylabel("Residuals (Pred - True)")
#     plt.title("Residuals vs. True")

#     # 3. Histogram of errors
#     # Whether errors are normally distributed
#     # Skewness or bias
#     # Long tails (outliers)
#     plt.subplot(1, 3, 3)
#     sns.histplot(residuals, kde=True, bins=30)
#     plt.axvline(0, linestyle="--", color="gray")
#     plt.title("Distribution of Errors")
#     plt.xlabel("Prediction Error")

#     plt.tight_layout()
#     plt.savefig("regression_diagnostics.png", dpi=300)
#     plt.close()


def regression_plots_plotly(y_true, y_pred, filename="regression_diagnostics.html"):
    # Calculate residuals
    residuals = np.array(y_pred) - np.array(y_true)

    # 1. Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Predicted vs True",
            "Residuals vs True",
            "Distribution of Errors",
        ),
        column_widths=[0.33, 0.33, 0.33],
        horizontal_spacing=0.15,
    )

    # 2. Predicted vs True
    fig_pred_true = go.Scatter(
        x=y_true, y=y_pred, mode="markers", name="Predicted vs True"
    )
    ideal_line = go.Scatter(
        x=[min(y_true), max(y_true)],
        y=[min(y_true), max(y_true)],
        mode="lines",
        name="Ideal",
        line=dict(dash="dash", color="red"),
    )
    fig.add_trace(fig_pred_true, row=1, col=1)
    fig.add_trace(ideal_line, row=1, col=1)

    # 3. Residuals vs True
    fig_residuals = go.Scatter(
        x=y_true, y=residuals, mode="markers", name="Residuals vs True"
    )
    line_zero = go.Scatter(
        x=[min(y_true), max(y_true)],
        y=[0, 0],
        mode="lines",
        name="Zero Line",
        line=dict(dash="dash", color="gray"),
    )
    fig.add_trace(fig_residuals, row=1, col=2)
    fig.add_trace(line_zero, row=1, col=2)

    # 4. Distribution of Errors (Histogram of residuals)
    fig_hist = px.histogram(
        residuals,
        nbins=30,
        title="Distribution of Errors",
        labels={"value": "Prediction Error"},
    )
    for trace in fig_hist.data:
        fig.add_trace(trace, row=1, col=3)

    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        title_text="Regression Diagnostics",
        showlegend=False,
        xaxis_title="True Values",
        yaxis_title="Predicted Values",
        xaxis2_title="True Values",
        yaxis2_title="Residuals (Pred - True)",
        xaxis3_title="Prediction Error",
        yaxis3_title="Count",
        autosize=True,
    )

    # Save to a single HTML file
    fig.write_html(filename)
    logging.info(f"✅ Regression diagnostics saved to '{filename}'")


def classification_plots_plotly(
    y_true, y_pred, y_prob, filename="classification_results.html"
):
    # 1. Create subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Confusion Matrix",
            "Classification Report",
            "ROC Curve",
            "PR Curve",
            "Probability Distribution",
        ),
        column_widths=[0.5, 0.5],
        row_heights=[
            0.35,
            0.35,
            0.3,
        ],  # Adjust row heights to give more space to first two rows
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
    )

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Negative", "Positive"]
    fig_cm = px.imshow(
        cm, x=labels, y=labels, color_continuous_scale="Blues", text_auto=True
    )
    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        xaxis=dict(
            tickmode="array", tickvals=[0, 1], ticktext=["Negative", "Positive"]
        ),
        yaxis=dict(
            tickmode="array", tickvals=[0, 1], ticktext=["Negative", "Positive"]
        ),
    )
    for trace in fig_cm.data:
        fig.add_trace(trace, row=1, col=1)

    # 3. Classification Report Plot
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :].T  # drop avg/total row
    fig_report = px.imshow(
        df_report.values,
        x=df_report.columns,
        y=df_report.index,
        text_auto=".2f",
        color_continuous_scale="Viridis",
        title="Classification Report",
    )
    for trace in fig_report.data:
        fig.add_trace(trace, row=1, col=2)

    # 4. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {roc_auc:.2f})")
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")
        )
    )
    fig_roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    for trace in fig_roc.data:
        fig.add_trace(trace, row=2, col=1)

    # 5. Precision-Recall Curve Plot
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    fig_pr = go.Figure()
    fig_pr.add_trace(
        go.Scatter(
            x=recall, y=precision, mode="lines", name=f"PR Curve (AP = {avg_prec:.2f})"
        )
    )
    fig_pr.update_layout(
        title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision"
    )
    for trace in fig_pr.data:
        fig.add_trace(trace, row=2, col=2)

    # 6. Probability Distribution Plot
    fig_hist = px.histogram(x=y_prob, nbins=50, title="Predicted Probabilities")
    fig_hist.update_layout(xaxis_title="Predicted Probability", yaxis_title="Count")
    for trace in fig_hist.data:
        fig.add_trace(trace, row=3, col=1)

    # Update layout for all subplots (adjusting size)
    fig.update_layout(
        height=1200,  # Increase height to better fit the plots
        width=1200,  # Increase width to utilize more space
        title_text="Classification plots",
        showlegend=False,
    )

    # Save to a single HTML file
    fig.write_html(filename)
    logging.info(f"✅ All plots saved to '{filename}'")
