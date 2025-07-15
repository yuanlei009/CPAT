import torch
import torch.nn as nn
from torch.nn.functional import softmax

"""modality_fusion.py
Learnable modality‑level fusion after prior‑guided attention
===========================================================
This layer collapses the three modality embeddings (CT/Text/Lab) into a single
compact vector, optionally **conditioned on the clinical prior** so that the
fusion weights vary across patients.

Fusion strategies
-----------------
* **StaticLearnable** A set of 3 scalar logits (one per modality) learned
globally and softmax‑normalised.
* **DynamicPrior** Fusion weights = `softmax( W * prior + b )`, providing
patient‑specific weighting based on prior knowledge.

Interface
---------
```python
fusion = ModalityFusion(embed_dim=128, prior_dim=64, dynamic=True)
vec, w = fusion(z_ct, z_txt, z_lab, prior)
```
*Returns* fused vector `[B, D]` and weight tensor `[B, 3]` (for interpretability).
"""

__all__ = ["ModalityFusion"]

class ModalityFusion(nn.Module):
    """Fuse 3 modality embeddings into a single representation."""

    def __init__(
        self,
        embed_dim: int,
        prior_dim: int | None = None,
        dynamic: bool = True,
    ) -> None:
        """Parameters
        ----------
        embed_dim : int
            Dimensionality of each modality embedding.
        prior_dim : int | None, optional
            If set and *dynamic=True*, prior vector dimension used for conditioning.
        dynamic : bool, default ``True``
            Use `DynamicPrior` strategy; otherwise fall back to
            `StaticLearnable` global weights.
        """
        super().__init__()
        self.dynamic = dynamic and prior_dim is not None
        if self.dynamic:
            self.weight_mlp = nn.Linear(prior_dim, 3)
        else:
            self.logits = nn.Parameter(torch.zeros(3))

        # Optional gating MLP to refine fused feature
        self.refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        z_ct: torch.Tensor,      # [B, D]
        z_txt: torch.Tensor,     # [B, D]
        z_lab: torch.Tensor,     # [B, D]
        prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse modalities.

        Returns
        -------
        fused : torch.Tensor  `[B, D]`
        weights : torch.Tensor `[B, 3]`  Softmax weights per modality.
        """
        B = z_ct.size(0)
        Z = torch.stack([z_ct, z_txt, z_lab], dim=1)  # [B, 3, D]

        # Calculate weights
        if self.dynamic:
            assert prior is not None, "Prior vector required for dynamic fusion"
            logits = self.weight_mlp(prior)  # [B, 3]
        else:
            logits = self.logits.unsqueeze(0).expand(B, -1)  # broadcast

        weights = softmax(logits, dim=-1).unsqueeze(-1)  # [B, 3, 1]

        fused = (Z * weights).sum(dim=1)  # [B, D]
        fused = fused + self.refine(fused)  # residual refinement
        return fused, weights.squeeze(-1)


if __name__ == "__main__":
    B, D, P = 4, 128, 64
    mod = ModalityFusion(embed_dim=D, prior_dim=P, dynamic=True)
    out, w = mod(torch.randn(B, D), torch.randn(B, D), torch.randn(B, D), torch.randn(B, P))
    print("out", out.shape, "weights", w.shape)

