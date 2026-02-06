"""
FlightMind Transformer
=======================
A decoder-only transformer language model built from first principles.

This file implements every component explicitly (no nn.TransformerDecoder)
so that each piece is visible and understandable. The architecture follows
modern best practices from LLaMA/Gemma/Mistral:

  Input IDs
      |
  Token Embedding (no positional embedding - RoPE handles position)
      |
  [Transformer Block x N]
  |   |-- RMSNorm
  |   |-- Multi-Head Self-Attention (with RoPE + causal mask)
  |   |-- Residual Add
  |   |-- RMSNorm
  |   |-- SwiGLU MLP
  |   |-- Residual Add
      |
  RMSNorm (final)
      |
  Linear -> logits (weight-tied with embedding)

Each component is a separate nn.Module with clear docstrings explaining
WHY it's designed that way, not just WHAT it does.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FlightMindConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    WHY RMSNorm instead of LayerNorm?
    ---------------------------------
    Standard LayerNorm: y = (x - mean) / std * gamma + beta
    RMSNorm:           y = x / RMS(x) * gamma

    RMSNorm drops the mean-centering and bias terms. Empirically this
    performs identically to LayerNorm for transformers (Zhang & Sennrich,
    2019) while being ~10-15% faster because it avoids computing the mean.
    Used in LLaMA, Gemma, and most modern LLMs.

    The intuition: for transformers, the important thing is normalizing the
    *scale* of activations (preventing them from growing/shrinking through
    layers). The mean-centering of LayerNorm is unnecessary overhead.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale (gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        # Normalize: x / RMS(x) * gamma
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute the complex exponential frequencies for RoPE.

    WHY RoPE instead of learned positional embeddings?
    ---------------------------------------------------
    1. **Relative position**: RoPE encodes *relative* distances between tokens,
       not absolute positions. This means the model naturally learns "token A
       is 5 positions before token B" rather than "token A is at position 42."

    2. **Extrapolation**: With learned embeddings, the model has never seen
       position 2049 during training. With RoPE, the sinusoidal structure
       lets it generalize to unseen positions (with some quality degradation).

    3. **No parameters**: RoPE is purely mathematical - no extra learnable
       weights. Every parameter we save can be used elsewhere.

    4. **Efficiency**: Implemented as element-wise rotation, integrates into
       the attention computation with minimal overhead.

    HOW it works (simplified):
    --------------------------
    For each pair of dimensions (2i, 2i+1) in the head, RoPE rotates the
    vector by an angle proportional to the position. Different dimension
    pairs rotate at different frequencies (like a clock with multiple hands
    spinning at different speeds). When computing dot products in attention,
    these rotations cancel out in a way that depends only on the *distance*
    between two positions, giving us relative position encoding for free.

    Returns: Complex tensor of shape (max_seq_len, head_dim // 2)
    """
    # Frequency for each dimension pair: theta^(-2i/d) for i in [0, d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # Position indices
    positions = torch.arange(max_seq_len, device=device).float()

    # Outer product: (seq_len, head_dim/2) - angle for each position and frequency
    angles = torch.outer(positions, freqs)

    # Convert to complex exponentials: e^(i*angle) = cos(angle) + i*sin(angle)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to queries or keys.

    Args:
        x: (batch, n_head, seq_len, head_dim) - real tensor
        freqs: (seq_len, head_dim // 2) - complex frequencies

    Returns: x with positional information baked in, same shape
    """
    # Reshape x as pairs of adjacent dims -> complex numbers
    # (B, H, T, D) -> (B, H, T, D/2, 2) -> complex (B, H, T, D/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Reshape freqs to broadcast: (1, 1, T, D/2)
    freqs = freqs.unsqueeze(0).unsqueeze(0)

    # Multiply by complex exponential (= rotation in 2D)
    x_rotated = x_complex * freqs

    # Convert back to real pairs -> flatten
    # (B, H, T, D/2) -> (B, H, T, D/2, 2) -> (B, H, T, D)
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention with RoPE and causal masking.

    WHY this specific design?
    -------------------------
    1. **Causal mask**: We're training a *language model* that predicts the
       next token. Each position can only attend to itself and earlier
       positions. Without this, the model would "cheat" by looking ahead.

    2. **Separate Q/K/V projections** (no fused QKV): Clearer code. Fusing
       is an optimization that saves a kernel launch but obscures the logic.
       We prioritize readability for educational purposes.

    3. **No bias in projections**: Following LLaMA. The bias terms in
       attention projections have negligible impact on quality but add
       parameters and complicate the code.

    4. **Flash Attention**: When available (PyTorch >= 2.0), we use
       F.scaled_dot_product_attention which fuses the attention computation
       into a single memory-efficient kernel. This is ~2-4x faster and
       uses O(T) memory instead of O(T^2).
    """

    def __init__(self, config: FlightMindConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.flash = config.flash_attention

        # Q, K, V, O projections (no bias)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_embd)
            rope_freqs: (seq_len, head_dim // 2) complex

        Returns: (batch, seq_len, n_embd)
        """
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, n_embd)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V - position info flows through attention weights)
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Attention
        if self.flash and hasattr(F, "scaled_dot_product_attention"):
            # Flash attention: fused, memory-efficient, handles causal mask internally
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,  # No dropout in attention (modern practice)
                is_causal=True,  # Automatically applies causal mask
            )
        else:
            # Manual attention (for understanding / older PyTorch)
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

            # Causal mask: -inf for future positions so softmax gives 0
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
            attn = attn + causal_mask

            attn = F.softmax(attn, dim=-1)
            if self.dropout is not None:
                attn = self.dropout(attn)

            out = attn @ v  # (B, H, T, D)

        # Reshape back: (B, H, T, D) -> (B, T, H*D) = (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLU_MLP(nn.Module):
    """SwiGLU feed-forward network.

    WHY SwiGLU instead of standard MLP with GELU?
    -----------------------------------------------
    Standard MLP:  y = W2 * GELU(W1 * x)          (2 weight matrices)
    SwiGLU MLP:    y = W_down * (SiLU(W_gate * x) * (W_up * x))  (3 matrices)

    SwiGLU (Shazeer, 2020) consistently outperforms GELU and ReLU MLPs
    in language modeling at the same parameter count. The intuition:

    1. The "gate" (SiLU(W_gate * x)) learns WHICH features to activate
    2. The "up" projection (W_up * x) learns WHAT values to pass through
    3. Element-wise multiply combines these two signals
    4. The "down" projection maps back to model dimension

    This gating mechanism is more expressive than a simple nonlinearity.
    The cost is one extra matrix multiply, but the quality improvement
    is consistent (used in LLaMA, PaLM, Gemma, Mistral, etc.).

    SiLU (Sigmoid Linear Unit) = x * sigmoid(x), also called "swish".
    It's smooth everywhere (unlike ReLU) and slightly outperforms GELU.
    """

    def __init__(self, config: FlightMindConfig):
        super().__init__()
        hidden = config.mlp_hidden

        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate: which features to activate (SiLU activation)
        gate = F.silu(self.gate_proj(x))
        # Up: what values to pass
        up = self.up_proj(x)
        # Combine and project back down
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """One transformer layer: attention + MLP with pre-norm residuals.

    WHY pre-norm instead of post-norm?
    ------------------------------------
    Original transformer (Vaswani 2017):  x + Sublayer(LayerNorm(x))  [post-norm]
    Modern practice:                      x + Sublayer(Norm(x))        [pre-norm]

    Wait, those look the same - the difference is subtle:
    - Post-norm: LayerNorm(x + Sublayer(x))   -- norm AFTER the residual add
    - Pre-norm:  x + Sublayer(Norm(x))         -- norm BEFORE the sublayer

    Pre-norm is strictly better for training stability because:
    1. Gradients flow through the residual connection unimpeded (no norm in the way)
    2. The identity path (skip connection) preserves gradient magnitude
    3. This means we can train deeper models without careful LR tuning

    The tradeoff: pre-norm models sometimes need a final norm before the
    output head (which we do have). All modern LLMs use pre-norm.
    """

    def __init__(self, config: FlightMindConfig):
        super().__init__()
        # Pre-attention norm
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        # Attention
        self.attn = Attention(config)
        # Pre-MLP norm
        self.mlp_norm = RMSNorm(config.n_embd, config.norm_eps)
        # MLP
        self.mlp = SwiGLU_MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
    ) -> torch.Tensor:
        # Attention with residual connection
        # x = x + Attention(Norm(x))
        x = x + self.attn(self.attn_norm(x), rope_freqs)

        # MLP with residual connection
        # x = x + MLP(Norm(x))
        x = x + self.mlp(self.mlp_norm(x))

        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class FlightMind(nn.Module):
    """FlightMind: a decoder-only transformer language model.

    Architecture Summary:
    - Token embeddings (no position embeddings - RoPE handles this)
    - N transformer blocks (pre-norm, causal attention, SwiGLU MLP)
    - Final RMSNorm
    - Linear head to vocab logits (weight-tied with embeddings)

    WHY weight tying?
    ------------------
    The embedding matrix maps token IDs -> vectors (32768 x 1280 = 41.9M params).
    The output head maps vectors -> logits (1280 x 32768 = 41.9M params).
    These two matrices are doing conceptually inverse operations.

    Weight tying shares the same matrix for both, which:
    1. Saves 41.9M parameters (7.4% of our 566M model)
    2. Acts as a regularizer (embedding and output must agree on token geometry)
    3. Empirically works just as well or better than separate matrices
    4. Used in GPT-2, LLaMA, Gemma, and most modern LLMs

    WHY no learned positional embeddings?
    --------------------------------------
    Traditional: add a learned (max_seq_len x n_embd) embedding to token embeddings.
    RoPE: rotate Q and K vectors in attention based on position.

    RoPE is better because:
    - No extra parameters (saves max_seq_len * n_embd = 2048 * 1280 = 2.6M params)
    - Encodes relative position (attention cares about distance, not absolute position)
    - Can extrapolate to longer sequences than trained on
    """

    def __init__(self, config: FlightMindConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final normalization (needed because pre-norm leaves the last
        # layer's output unnormalized)
        self.final_norm = RMSNorm(config.n_embd, config.norm_eps)

        # Output head (projects to vocabulary logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with output head
        if config.weight_tying:
            self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies (not a parameter, just a buffer)
        rope_freqs = precompute_rope_frequencies(
            config.head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights.

        WHY this specific initialization?
        -----------------------------------
        Standard Xavier/Glorot init keeps variance stable through layers for
        vanilla networks. But transformers have residual connections that ADD
        outputs, so variance grows with depth. We counter this:

        1. Normal init (std=0.02) for most weights - standard for transformers
        2. Scale output projections by 1/sqrt(2*n_layer) - this compensates
           for the variance growth from residual additions. Each layer adds
           to the residual stream, and there are 2 additions per layer
           (attention + MLP), so we scale down by sqrt(2*n_layer).

        This is the GPT-2 / nanoGPT initialization scheme.
        """
        # Standard init for all parameters
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

        # Scale residual projections (attention output and MLP down)
        residual_scale = 1.0 / math.sqrt(2 * self.config.n_layer)
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=0.02 * residual_scale)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=0.02 * residual_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) target IDs for loss computation.
                     If None, only returns logits (for inference).

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar cross-entropy loss (if targets provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        # Token embedding
        x = self.tok_emb(input_ids)  # (B, T, n_embd)

        # Get RoPE frequencies for this sequence length
        rope_freqs = self.rope_freqs[:T]  # (T, head_dim // 2)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, rope_freqs)

        # Final norm
        x = self.final_norm(x)

        # Compute logits
        if targets is not None:
            # Training: compute logits for all positions
            logits = self.lm_head(x)  # (B, T, vocab_size)

            # Cross-entropy loss
            # Reshape: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # Allow masking padding tokens with -1
            )
            return logits, loss
        else:
            # Inference: only compute logits for the last position (more efficient)
            logits = self.lm_head(x[:, -1, :])  # (B, vocab_size)
            return logits, None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive text generation.

        Simple top-k sampling with temperature. For each new token:
        1. Forward pass to get logits for the next position
        2. Apply temperature scaling (higher = more random)
        3. Zero out all but top-k logits (prevents sampling rare garbage)
        4. Sample from the resulting distribution
        5. Append to sequence, repeat

        Args:
            input_ids: (batch, seq_len) prompt token IDs
            max_new_tokens: how many tokens to generate
            temperature: sampling temperature (0 = greedy, 1 = standard)
            top_k: only sample from top-k most likely tokens

        Returns: (batch, seq_len + max_new_tokens) generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                else input_ids[:, -self.config.max_seq_len:]

            # Forward pass (inference mode: only last position logits)
            logits, _ = self(idx_cond)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_vals[:, [-1]]] = float("-inf")

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy: take the most likely token
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> dict:
        """Count parameters by component for analysis."""
        counts = {
            "embedding": 0,
            "attention": 0,
            "mlp": 0,
            "norm": 0,
            "lm_head": 0,
            "total": 0,
        }

        for name, p in self.named_parameters():
            n = p.numel()
            counts["total"] += n

            if "tok_emb" in name:
                counts["embedding"] += n
            elif "attn" in name and "norm" not in name:
                counts["attention"] += n
            elif "mlp" in name and "norm" not in name:
                counts["mlp"] += n
            elif "norm" in name:
                counts["norm"] += n
            elif "lm_head" in name:
                if not self.config.weight_tying:
                    counts["lm_head"] += n
                # If weight-tied, already counted in embedding

        return counts
