import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""lab_transformer_encoder.py
LabTransformerEncoder — Encode structured laboratory test results with a lightweight
Transformer backbone.

Design choices
--------------
* **Token‑per‑test** Each laboratory test (e.g. Serum Amylase) is treated as one token.
* **Value + ID embedding** Following TabTransformer (Huang et al., 2020), the final token
  representation is the *sum* of a learnable **feature ID embedding** and a projected
  **numeric value embedding**.
* **Optional [CLS] token** If ``use_cls_token=True`` the first output token holds a
  pooled representation analogous to BERT.
* **Missing values** Provide an optional boolean mask ``missing_mask`` (``True`` where
  value is missing). Missing tokens are replaced with a learnable ``missing_embedding``.

Example
-------
```python
encoder = LabTransformerEncoder(num_features=10)
values = torch.randn(2, 10)  # random lab panel
emb = encoder(values)        # (2, 128)
```
"""

__all__ = ["LabTransformerEncoder"]

class LabTransformerEncoder(nn.Module):
    """Transformer encoder for numeric laboratory features."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ) -> None:
        """Parameters
        ----------
        num_features : int
            Number of laboratory tests (tokens).
        d_model : int, default ``128``
            Embedding dimension inside the Transformer.
        nhead : int, default ``4``
            Number of attention heads.
        depth : int, default ``2``
            Number of TransformerEncoder layers.
        dropout : float, default ``0.1``
            Dropout applied in the Transformer.
        use_cls_token : bool, default ``True``
            Prepend a learnable ``[CLS]`` token and return its embedding.
        """
        super().__init__()
        self.use_cls = use_cls_token
        self.num_features = num_features

        # Learnable feature ID embeddings — one per lab test.
        self.feature_embed = nn.Embedding(num_features, d_model)

        # Linear projection from the scalar value to d_model.
        self.value_fc = nn.Linear(1, d_model)

        # Optional embedding for missing values.
        self.missing_embed = nn.Parameter(torch.zeros(1, 1, d_model))

        # CLS token if requested.
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=depth)

        self._reset_parameters()
        self.output_dim = d_model

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.feature_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.value_fc.weight, std=0.02)
        nn.init.zeros_(self.value_fc.bias)
        nn.init.trunc_normal_(self.missing_embed, std=0.02)
        if self.use_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, values: torch.Tensor, missing_mask: torch.Tensor | None = None):
        """Forward pass.

        Parameters
        ----------
        values : torch.Tensor
            Tensor of shape ``[B, N]`` with raw numeric lab values (float32).
        missing_mask : torch.Tensor | None, optional
            Boolean tensor of shape ``[B, N]``; ``True`` where value is missing.

        Returns
        -------
        torch.Tensor
            Pooled embedding ``[B, d_model]`` (``[CLS]``) if *use_cls_token* else
            mean‑pooled token embeddings.
        """
        B, N = values.shape
        assert N == self.num_features, "Mismatch between provided values and num_features"

        device = values.device
        # Feature ID embeddings [1, N, d_model] broadcast over batch.
        fid_emb = self.feature_embed(torch.arange(N, device=device)).unsqueeze(0).repeat(B, 1, 1)

        # Value embeddings [B, N, d_model]
        val_emb = self.value_fc(values.unsqueeze(-1))

        tokens = fid_emb + val_emb

        # Handle missing values
        if missing_mask is not None:
            missing_emb = self.missing_embed.expand(B, N, -1)
            tokens = torch.where(missing_mask.unsqueeze(-1), missing_emb, tokens)

        # Prepend CLS if required.
        if self.use_cls:
            cls_tok = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tok, tokens], dim=1)  # [B, N+1, d_model]

        encoded = self.encoder(tokens)  # [B, N(+1), d_model]

        if self.use_cls:
            return encoded[:, 0, :]
        # Otherwise mean‑pool over tokens
        return encoded.mean(dim=1)


if __name__ == "__main__":
    # Smoke test
    model = LabTransformerEncoder(num_features=12, use_cls_token=True)
    dummy_vals = torch.randn(4, 12)
    missing = dummy_vals < -1.5  # toy mask
    out = model(dummy_vals, missing_mask=missing)
    print("Output shape:", out.shape)  # [4, 128]
