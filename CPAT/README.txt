# CPAT: Clinical Prior-guided Attention Transformer for Binary Prognostic Prediction

This repository contains the official PyTorch implementation of **CPAT**, a novel multimodal AI framework for predicting binary survival outcomes in critically ill patients, using CT images, clinical notes (EHR), and laboratory test data. The model introduces optimal transport alignment, Tucker decomposition, and prior-guided attention to perform effective modality fusion.
---
##  Citation
> **CPAT: Clinical Prior-guided Attention Transformer for Binary Prognostic Prediction**  


---

##  Method Overview

CPAT integrates three key innovations:

1. **Multimodal Feature Extraction**
    - Swin Transformer for CT slices
    - ClinicalBERT for EHR notes
    - Transformer encoder for lab data

2. **Alignment & Decomposition**
    - Optimal Transport (OT) alignment across modalities
    - Tucker decomposition for semantic disentanglement

3. **Prior-guided Attention & Fusion**
    - Clinical priors (e.g. APACHE, age) as attention queries
    - Learnable or adaptive modality fusion


## Project Structure

```
├── cpat_full_model.py         # Main training pipeline (binary classification)
├── dataset_cpa.py             # Real multimodal dataset loader
├── swin_encoder.py            # Swin Transformer CT encoder
├── clinical_bert_encoder.py   # ClinicalBERT wrapper
├── lab_transformer_encoder.py # Transformer encoder for labs
├── ot_alignment.py            # Sinkhorn-based OT alignment
├── tucker_decomposition.py    # Shared/Private Tucker module
├── prior_guided_attention.py  # Clinical prior-guided attention
├── modality_fusion.py         # Final fusion block
├── eval_metrics.py            # Evaluation: Accuracy, F1, ROC-AUC, PR-AUC
├── requirements.txt
└── README.md
```

---

##  Getting Started

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Structure
```
/data/CPA/
├── meta.csv                 # with columns: id, label, prior_*, lab_*, text (or text dir)
├── ct/                      # PNG slices: <id>.png
├── ehr_text/                # Optional: <id>.txt for each patient
```

### 3. Train the Model
Edit `cpat_full_model.py`:
```python
cfg = {
    'data': {'root': '/data/CPA', 'text_subdir': 'ehr_text'},
    ...
}
```
Then run:
```bash
python cpat_full_model.py
```

### 4. Evaluate
```python
from eval_metrics import evaluate_model
metrics = evaluate_model(model, val_loader, device="cuda")
print(metrics)
```

---

## 📈 Metrics
- **Accuracy**
- **Precision / Recall / F1**
- **ROC-AUC** / **PR-AUC**
- **Confusion Matrix**

---

## 📄 License
MIT License (c) 2025. For academic use only.

---


