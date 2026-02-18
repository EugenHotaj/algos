import math

import torch
from torch.nn import functional as F


def attention(
    q: torch.Tensor,  # [B, T, Nq, D] 
    k: torch.Tensor,  # [B, T, Nkv, D] 
    v: torch.Tensor,  # [B, T, Nkv, D]
    enable_gqa: bool = False,
) -> torch.Tensor:
    q = q.transpose(1, 2)  # [B, Nq, T, D]
    k = k.transpose(1, 2)  # [B, Nkv, T, D]
    v = v.transpose(1, 2)  # [B, Nkv, T, D]

    if enable_gqa:
        n_repeats = q.shape[1] // k.shape[1]
        k = torch.repeat_interleave(k, n_repeats, dim=1)  # [B, N, T, D]
        v = torch.repeat_interleave(v, n_repeats, dim=1)  # [B, N, T, D]
    
    attn_scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])  # [B, N, T, T]
    attn_scores = torch.tril(attn_scores)
    attn_scores = torch.where(attn_scores == 0, -float("inf"), attn_scores)
    attn_scores = torch.softmax(attn_scores, dim=-1)

    out = (attn_scores @ v) # [B, N, T, D]
    out = out.transpose(1, 2)  # [B, T, N, D]

    return out


if __name__ == '__main__':
    B, T, N, D = 4, 128, 32, 8

    # Multi-head self-attention.
    q = torch.randn(size=(B, T, N, D))
    k = torch.randn(size=(B, T, N, D))
    v = torch.randn(size=(B, T, N, D))
    actual = attention(q, k, v)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), 
        k.transpose(1, 2), 
        v.transpose(1, 2), 
        is_causal=True
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, msg="Full attention failed.")

    # Grouped-query attention.
    q = torch.randn(size=(B, T, N, D))
    k = torch.randn(size=(B, T, N//8, D))
    v = torch.randn(size=(B, T, N//8, D))
    actual = attention(q, k, v, enable_gqa=True)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), 
        k.transpose(1, 2), 
        v.transpose(1, 2), 
        is_causal=True,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, msg="Group-query attention failed.")

    # Multi-query attention.
    q = torch.randn(size=(B, T, N, D))
    k = torch.randn(size=(B, T, 1, D))
    v = torch.randn(size=(B, T, 1, D))
    actual = attention(q, k, v, enable_gqa=True)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), 
        k.transpose(1, 2), 
        v.transpose(1, 2), 
        is_causal=True,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, msg="Multi-query attention failed.")


