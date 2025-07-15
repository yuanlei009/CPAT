import torch
import torch.nn as nn
import torch.nn.functional as F

"""tucker_decomposition.py
TuckerDecomposer — Learnable Tucker decomposition + semantic reconstruction module
================================================================================
This layer decomposes a **modalities × feature** tensor into low‑rank core and
factor matrices, then reconstructs it to enforce shared semantics across
modalities.

Why learnable?  In contrast to classical *offline* Tucker via SVD, we keep the
core *G* and factor matrices *(A, B)* as **trainable parameters**, optimised
jointly with the rest of the network.  Gradients propagate end‑to‑end.

Notation
--------
* *M* — number of modalities (default **3**: CT, Text, Lab)
* *D* — feature dimension after OT alignment
* *R_m*, *R_d* — ranks for modality and feature modes

Forward pass
------------
```
X  = stack([z_ct, z_txt, z_lab])  # [B, M, D]
X̂ =  A  · G · B          #  [M, D]   (shared across batch)
loss_rec = MSE(X̂, X)     # semantic reconstruction loss
fused = mean(X̂, dim=1)   # [B, D]    multimodal representation
```

Example
-------
```python
module = TuckerDecomposer(modalities=3, feat_dim=128, rank_mod=2, rank_feat=64)
rep, loss = module(z_ct, z_txt, z_lab)
```
"""

__all__ = ["TuckerDecomposer"]

class TuckerDecomposer(nn.Module):
    """Learnable Tucker decomposition for three‑modal features."""

    def __init__(
        self,
        modalities: int = 3,
        feat_dim: int = 128,
        rank_mod: int = 2,
        rank_feat: int = 64,
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.feat_dim = feat_dim
        self.rank_mod = rank_mod
        self.rank_feat = rank_feat

        # Factor matrices A (M×R_m) and B (R_d×D)
        self.A = nn.Parameter(torch.randn(modalities, rank_mod))
        self.B = nn.Parameter(torch.randn(rank_feat, feat_dim))
        # Core tensor G (R_m×R_d)
        self.G = nn.Parameter(torch.randn(rank_mod, rank_feat))

        self._init_params()

    def _init_params(self):
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.B)
        nn.init.xavier_uniform_(self.G)

    def forward(
        self,
        z_ct: torch.Tensor,   # [B, D]
        z_txt: torch.Tensor,  # [B, D]
        z_lab: torch.Tensor,  # [B, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction & loss.

        Returns
        -------
        fused : torch.Tensor
            Mean‑pooled reconstructed representation ``[B, D]``.
        loss_rec : torch.Tensor
            MSE reconstruction loss (scalar).
        """
        # Stack modalities → [B, M, D]
        X = torch.stack([z_ct, z_txt, z_lab], dim=1)

        # Shared reconstruction matrix   X̂ = A·G·B  (shape [M,D])
        recon_single = self.A @ self.G @ self.B  # [M, D]
        recon = recon_single.unsqueeze(0).expand(X.size(0), -1, -1)  # broadcast to batch

        # Reconstruction loss
        loss_rec = F.mse_loss(recon, X)

        # Fused multimodal representation (could use attention instead)
        fused = recon.mean(dim=1)  # [B, D]
        return fused, loss_rec


if __name__ == "__main__":
    B, D = 4, 128
    z1 = torch.randn(B, D)
    z2 = torch.randn(B, D)
    z3 = torch.randn(B, D)
    module = TuckerDecomposer(feat_dim=D)
    rep, loss = module(z1, z2, z3)
    print(rep.shape, loss.item())
