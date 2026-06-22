"""
models/attention.py — Scaled dot-product self-attention for temporal sequences.

Why attention in SER?
  Speech emotion is not uniformly distributed in time — the most emotionally
  salient frames (vowels, prosodic peaks) carry disproportionate information.
  Self-attention lets the model assign higher weights to those frames and
  reduce noise from emotionally neutral phonemes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Single-head scaled dot-product self-attention.

    Input/Output:
        x  →  (B, T, input_dim)
        out → (B, T, input_dim)   (same shape — drop-in replacement)

    Args:
        input_dim:     Dimension of each input token (BiLSTM output dim).
        attention_dim: Internal projection dimension for Q/K/V.
    """

    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.scale   = attention_dim ** 0.5
        self.W_q     = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k     = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v     = nn.Linear(input_dim, attention_dim, bias=False)
        self.out_proj = nn.Linear(attention_dim, input_dim, bias=False)

        # Weight initialization: Xavier uniform is standard for attention layers
        for module in (self.W_q, self.W_k, self.W_v, self.out_proj):
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple:
        """
        Args:
            x: Input tensor (B, T, D).

        Returns:
            out:     Context-enriched tensor (B, T, D).
            weights: Attention weights (B, T, T) — useful for visualization.
        """
        Q = self.W_q(x)                                    # (B, T, d)
        K = self.W_k(x)                                    # (B, T, d)
        V = self.W_v(x)                                    # (B, T, d)

        # Scaled dot-product scores
        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, T, T)
        weights = F.softmax(scores, dim=-1)                       # (B, T, T)

        # Weighted sum of values
        context = torch.bmm(weights, V)                    # (B, T, d)
        out     = self.out_proj(context)                   # (B, T, D)

        return out, weights