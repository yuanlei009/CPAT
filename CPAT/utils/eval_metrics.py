import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

"""eval_metrics.py
Evaluation utilities for CPAT binary classification.
====================================================
Functions
---------
* **evaluate_model** — run inference on a *DataLoader* and return a dict of
  common metrics (Accuracy, Precision, Recall, F1, ROC‑AUC, PR‑AUC).
* **Confusion matrix** is printed but not plotted; adapt as needed.

Usage example
-------------
```python
from eval_metrics import evaluate_model
from dataset_cpa import CPADataset
from cpat_full_model import CPATBinary

# Assume model trained & loaded
val_ds = CPADataset(root="/data/CPA", text_subdir="ehr_text", csv_file="meta_val.csv")
val_loader = DataLoader(val_ds, batch_size=8, num_workers=4)
metrics = evaluate_model(model, val_loader, device="cuda")
print(metrics)
```
"""

@torch.no_grad()
def evaluate_model(model, loader: DataLoader, device: str = "cpu") -> dict:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []
    for ct, text, lab, prior, label in tqdm(loader, desc="Evaluating"):
        ct, lab, prior = ct.to(device), lab.to(device), prior.to(device)
        logits, _ = model(ct, text, lab, prior)  # CPATBinary forward (label=None)
        all_logits.append(logits.cpu())
        all_labels.append(label.squeeze().cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)

    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")  # only one class present
    pr_auc = average_precision_score(labels, probs)
    cm = confusion_matrix(labels, preds)

    print("Confusion Matrix:\n", cm)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }

if __name__ == "__main__":
    print("This file defines evaluate_model(); import and call it from your training/evaluation script.")