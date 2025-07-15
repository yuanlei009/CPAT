import torch
import torch.nn as nn
from torch.nn.functional import softmax

"""prior_guided_attention.py
PriorGuidedAttention — Query multimodal features with clinical prior knowledge
==============================================================================
*Clinical prior* (e.g. APACHE II score, risk category, keyword embedding) is
encoded as a query and attends over **three modality representations** obtained
after Tucker reconstruction.

Key design
----------
* **Multi‑Head Attention (MHA)** `nn.MultiheadAttention` with `batch_first=True`.
* **Query** Learned linear projection of the prior vector.
* **Keys/Values** Stacked modality tensors `[z_ct, z_txt, z_lab]`.
* **Temperature scaling** Optional `tau` hyper‑parameter for sharper weights.
* **Return both** attended vector and raw attention map for interpretability.

Example
-------
```python
attn = PriorGuidedAttention(prior_dim=64, embed_dim=128)
vec, w = attn(prior, z_ct, z_txt, z_lab)
```
"""

__all__ = ["PriorGuidedAttention"]

class PriorGuidedAttention(nn.Module):
    """Clinical‑prior‑guided cross‑attention over reconstructed modalities."""

    def __init__(
        self,
        prior_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        tau: float | None = None,
    ) -> None:
        """Parameters
        ----------
        prior_dim : int
            Dimensionality of the prior knowledge vector.
        embed_dim : int, default ``128``
            Dimension of query / key / value embeddings.
        num_heads : int, default ``4``
            Number of attention heads.
        dropout : float, default ``0.1``
            Dropout inside multi‑head attention.
        tau : float | None, optional
            Temperature scaling for attention logits (``softmax(logits / tau)``).
        """
        super().__init__()
        self.query_proj = nn.Linear(prior_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)  # identity if embed_dim unchanged
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.tau = tau

    def forward(
        self,
        prior: torch.Tensor,      # [B, P]
        z_ct: torch.Tensor,       # [B, D]
        z_txt: torch.Tensor,      # [B, D]
        z_lab: torch.Tensor,      # [B, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies cross‑attention.

        Returns
        -------
        attended : torch.Tensor
            Prior‑aware fused embedding ``[B, D]``.
        attn_map : torch.Tensor
            Raw attention weights ``[B, heads, 1, 3]`` (query→modalities).
        """
        B = prior.size(0)
        # Prepare Q (1 token) and K,V (3 tokens)
        q = self.query_proj(prior).unsqueeze(1)  # [B, 1, D]
        kv = torch.stack([z_ct, z_txt, z_lab], dim=1)  # [B, 3, D]
        k = self.key_proj(kv)
        v = self.value_proj(kv)

        # Multi‑head cross‑attention
        if self.tau is not None:
            # Scale queries for temperature; internal attn already scales by 1/sqrt(d)
            q = q / self.tau
        attended, attn_weights = self.attn(q, k, v, need_weights=True, average_attn_weights=False)

        attended = self.out_proj(attended.squeeze(1))  # [B, D]
        return attended, attn_weights  # weights shape [B, heads, 1, 3]


if __name__ == "__main__":
    B, D, P = 2, 128, 64
    mod = PriorGuidedAttention(prior_dim=P, embed_dim=D)
    out, w = mod(torch.randn(B, P), torch.randn(B, D), torch.randn(B, D), torch.randn(B, D))
    print("out", out.shape, "weights", w.shape)
