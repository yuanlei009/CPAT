import torch
import torch.nn as nn
import timm

"""swin_encoder.py
SwinCTEncoder: A lightweight Swin‑Transformer feature extractor for single‑channel CT slices.

Why a custom wrapper?
---------------------
* **Single‑channel input** Medical CT slices are usually stored as a single channel. Most
  ImageNet‑pretrained Swin models expect three channels. We adapt the first conv layer
  so the pretrained weights can still be leveraged.
* **Feature‑only usage** Downstream multimodal fusion blocks often need a spatial feature
  map instead of classification logits. We build the backbone with `features_only=True`
  (supported by *timm*).

Usage example
-------------
```python
from swin_encoder import SwinCTEncoder
enc = SwinCTEncoder(pretrained=True)
feat = enc(torch.randn(1, 1, 224, 224))  # [1, C, H', W']
print(enc.num_features)                   # channel dimension C
```
"""

__all__ = ["SwinCTEncoder"]

def _adapt_input_conv(weight: torch.Tensor, in_chans: int = 1) -> torch.Tensor:
    """Convert 3‑channel conv weights to *in_chans* by averaging.

    Args:
        weight: pretrained kernel tensor of shape ``[out_channels, 3, k, k]``.
        in_chans: desired number of input channels (default: 1).

    Returns
    -------
    torch.Tensor
        Weight tensor of shape ``[out_channels, in_chans, k, k]``.
    """
    if weight.shape[1] == in_chans:
        return weight
    # Average the RGB kernels to obtain a grayscale kernel.
    return weight.mean(dim=1, keepdim=True)


class SwinCTEncoder(nn.Module):
    """Backbone that wraps a Swin Transformer for CT feature extraction."""

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        out_indices: tuple[int, ...] = (3,),
    ) -> None:
        """Parameters
        ----------
        model_name : str
            Any Swin variant supported by *timm*.
        pretrained : bool, default ``True``
            Load ImageNet weights provided by *timm*.
        out_indices : tuple[int, ...], default ``(3,)``
            Which stages to return (0 = first stage, 3 = last). The final feature map is
            returned by default.
        """
        super().__init__()

        # Build backbone with `features_only=True` so we get a list of feature maps.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=1,
            out_indices=out_indices,
        )

        # If a 3‑to‑1 channel conversion is required, patch the first conv weights.
        if pretrained and self.backbone.default_cfg.get("input_size", (3,))[0] == 3:
            # The first conv can be found in one of several places depending on the model.
            first_conv = None
            stem = getattr(self.backbone, "stem", None)
            if stem is not None and hasattr(stem, "conv1"):
                first_conv = stem.conv1  # Swin v1
            elif hasattr(self.backbone, "patch_embed") and hasattr(self.backbone.patch_embed, "proj"):
                first_conv = self.backbone.patch_embed.proj  # Swin v2

            if first_conv is not None and first_conv.weight.shape[1] != 1:
                first_conv.weight.data = _adapt_input_conv(first_conv.weight.data, in_chans=1)

        # Cache the channel dimension of the deepest feature map for convenience.
        self.num_features = self.backbone.feature_info.channels()[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, 1, H, W]``.

        Returns
        -------
        torch.Tensor
            Feature map from the selected stage, e.g. ``[B, C, H', W']``.
        """
        features = self.backbone(x)  # returns a list of tensors
        return features[-1]


if __name__ == "__main__":
    # Smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwinCTEncoder(pretrained=False).to(device)
    dummy = torch.randn(2, 1, 224, 224).to(device)
    out = model(dummy)
    print("Output shape:", out.shape)
