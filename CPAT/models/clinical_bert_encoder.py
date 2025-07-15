import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

"""clinical_bert_encoder.py
ClinicalBERTEncoder: Extract contextual text representations from clinical notes
using Bio_ClinicalBERT (Alsentzer et al.).

Key features
------------
* **Plug‑and‑play** Returns either the [CLS] embedding (sentence‑level) or the full
  token sequence.
* **Tokenizer inside** Accepts a *list[str]* (batch of raw sentences) and handles
  padding/truncation automatically.
* **Freezing option** `freeze=True` disables gradient updates for inference‑only use.

Example
-------
```python
from clinical_bert_encoder import ClinicalBERTEncoder
model = ClinicalBERTEncoder(output_cls=True)
emb = model(["Patient complains of abdominal pain."])
print(emb.shape)  # (1, 768)
```
"""

__all__ = ["ClinicalBERTEncoder"]

class ClinicalBERTEncoder(nn.Module):
    """Wrapper around Bio_ClinicalBERT for feature extraction."""

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        output_cls: bool = True,
        max_length: int = 128,
        freeze: bool = False,
    ) -> None:
        """Parameters
        ----------
        model_name : str
            Hugging Face model hub ID for ClinicalBERT.
        output_cls : bool, default ``True``
            If ``True`` return the ``[CLS]`` pooled embedding; otherwise return the
            full token‑level hidden states ``[B, L, D]``.
        max_length : int, default ``128``
            Sequence length after tokenization (truncates longer notes).
        freeze : bool, default ``False``
            Disable gradient updates when set to ``True``.
        """
        super().__init__()
        self.output_cls = output_cls
        self.max_length = max_length

        # Tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.hidden_size = self.bert.config.hidden_size

    @torch.no_grad()
    def _prepare_inputs(self, text_batch: list[str], device: torch.device):
        """Tokenize a list of strings and move tensors to target device."""
        enc = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, text_batch: list[str]):  # type: ignore[override]
        """Encode a batch of sentences.

        Parameters
        ----------
        text_batch : list[str]
            Raw clinical sentences/notes.

        Returns
        -------
        torch.Tensor
            * If ``output_cls``: ``[B, D]`` sentence embeddings.
            * Else: ``[B, L, D]`` token embeddings (padded to *max_length*).
        """
        device = next(self.parameters()).device
        inputs = self._prepare_inputs(text_batch, device)
        outputs = self.bert(**inputs)
        if self.output_cls:
            return outputs.last_hidden_state[:, 0, :]  # CLS token
        return outputs.last_hidden_state


if __name__ == "__main__":
    # Smoke test on CPU to avoid large GPU download in sample code.
    model = ClinicalBERTEncoder(output_cls=True, freeze=True)
    sentences = [
        "The patient was admitted to the ICU with severe abdominal pain.",
        "Lab results show elevated serum amylase and lipase levels.",
    ]
    emb = model(sentences)
    print("CLS embedding shape:", emb.shape)
