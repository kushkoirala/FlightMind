"""
FlightMind Model Configuration
================================
Depth-parameterized transformer configuration inspired by Karpathy's nanochat.

The key insight: a single integer `depth` controls the entire model size.
Everything else (embedding dimension, number of heads, number of layers,
MLP hidden size) is derived from depth. This makes it trivial to scale
the model up or down by changing one number.

Depth Scaling Table:
    depth  |  d_model  |  n_layer  |  n_head  |  ~Params
    -------|-----------|-----------|----------|----------
      8    |    512    |     8     |     8    |   ~50M
     12    |    768    |    12     |    12    |   ~138M
     16    |   1024    |    16     |    16    |   ~302M
     20    |   1280    |    20     |    20    |   ~566M
     24    |   1536    |    24     |    24    |   ~956M
     32    |   2048    |    32     |    32    |  ~2.21B

Design Decisions (see ARCHITECTURE.md for full rationale):
- head_dim fixed at 64 (empirically optimal across model scales)
- SwiGLU MLP with 4x expansion (better quality than GELU, worth the 50% more params)
- RoPE positional encoding (extrapolates to longer sequences, no learned params)
- RMSNorm (faster than LayerNorm, equivalent quality)
- Pre-norm residual connections (more stable training)
- Weight tying between embedding and output head (saves params, acts as regularizer)
- No bias terms (marginal quality impact, saves params, simplifies code)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FlightMindConfig:
    """Model configuration with depth-parameterized scaling."""

    # === Core parameter: everything derives from this ===
    depth: int = 20

    # === Vocabulary ===
    vocab_size: int = 32768
    max_seq_len: int = 2048

    # === Derived architecture (computed in __post_init__) ===
    n_layer: int = field(init=False)
    n_head: int = field(init=False)
    n_embd: int = field(init=False)
    head_dim: int = 64       # Fixed across all scales
    mlp_hidden: int = field(init=False)

    # === Training-relevant flags ===
    dropout: float = 0.0      # Set >0 only for small-data fine-tuning
    rope_theta: float = 10000.0  # RoPE base frequency
    norm_eps: float = 1e-6    # RMSNorm epsilon
    weight_tying: bool = True  # Tie embedding and LM head weights

    # === Optional ===
    flash_attention: bool = True  # Use flash attention if available

    def __post_init__(self):
        """Derive all dimensions from depth."""
        self.n_layer = self.depth
        self.n_head = self.depth
        self.n_embd = self.depth * self.head_dim  # depth * 64
        self.mlp_hidden = 4 * self.n_embd  # 4x expansion for SwiGLU

    @property
    def total_params(self) -> int:
        """Estimate total parameter count."""
        d = self.n_embd
        V = self.vocab_size
        L = self.n_layer

        # Embedding (weight-tied with output, so counted once)
        embedding = V * d

        # Per transformer layer:
        #   Attention: Q, K, V, O projections = 4 * d * d
        #   MLP (SwiGLU): gate, up, down = 3 * d * mlp_hidden
        #   Norms: 2 * d (attention_norm + mlp_norm)
        attn_per_layer = 4 * d * d
        mlp_per_layer = 3 * d * self.mlp_hidden
        norm_per_layer = 2 * d
        per_layer = attn_per_layer + mlp_per_layer + norm_per_layer

        # Final norm
        final_norm = d

        return embedding + L * per_layer + final_norm

    def describe(self) -> str:
        """Human-readable description."""
        params = self.total_params
        if params >= 1e9:
            param_str = f"{params / 1e9:.2f}B"
        else:
            param_str = f"{params / 1e6:.0f}M"

        return (
            f"FlightMind-d{self.depth} ({param_str} params)\n"
            f"  Layers: {self.n_layer}\n"
            f"  Heads: {self.n_head}\n"
            f"  Embedding: {self.n_embd}\n"
            f"  Head dim: {self.head_dim}\n"
            f"  MLP hidden: {self.mlp_hidden}\n"
            f"  Vocab: {self.vocab_size:,}\n"
            f"  Max seq len: {self.max_seq_len:,}\n"
            f"  Weight tying: {self.weight_tying}\n"
        )

    @classmethod
    def from_depth(cls, depth: int, **kwargs) -> "FlightMindConfig":
        """Create config from depth with optional overrides."""
        return cls(depth=depth, **kwargs)
