# FlightMind Model Architecture

A decoder-only transformer language model built from scratch, designed for aviation domain expertise while retaining general language capability.

## Design Philosophy

Every architectural choice in FlightMind is made deliberately, with preference for **proven techniques over novelty**. This is an educational project: we want to understand *why* each piece works, not just copy-paste from a paper. Where there are tradeoffs, we document them.

## Depth Parameterization

The entire model is controlled by a single integer: **depth**.

```
depth = 20  ->  FlightMind-d20 (561M parameters)
```

From depth, everything else is derived:

| Parameter | Formula | d=20 Value |
|-----------|---------|------------|
| n_layer | depth | 20 |
| n_head | depth | 20 |
| head_dim | 64 (fixed) | 64 |
| n_embd | depth * head_dim | 1280 |
| mlp_hidden | 4 * n_embd | 5120 |

**Why this parameterization?** Inspired by Karpathy's nanochat, this approach has a key advantage: scaling the model up or down requires changing exactly one number. The ratios between components stay optimal at every scale because they're hardcoded into the scaling law.

**Why head_dim = 64?** This is empirically the sweet spot. Smaller head dimensions (32) lose expressiveness per head. Larger ones (128) work but waste parameters unless you're at very large scale (>7B). At 64, each head can represent complex attention patterns without redundancy. This is what GPT-2, LLaMA, and Gemma use.

## Component-by-Component Design Decisions

### 1. RMSNorm (instead of LayerNorm)

**What it is:** Normalizes by the root-mean-square of activations, without mean-centering.

```
LayerNorm(x) = (x - mean(x)) / std(x) * gamma + beta    # 2 learnable params per dim
RMSNorm(x)   = x / RMS(x) * gamma                        # 1 learnable param per dim
```

**Why:** The mean-centering in LayerNorm is unnecessary for transformers. Zhang & Sennrich (2019) showed that removing it has no quality impact. The benefit is ~10-15% faster normalization. Every modern LLM (LLaMA, Gemma, Mistral) uses RMSNorm.

**Intuition:** Transformers need normalization to prevent activations from exploding through dozens of layers. What matters is controlling the *scale* (magnitude). Centering around zero is a solution looking for a problem that doesn't exist in practice.

### 2. Rotary Position Embeddings (RoPE)

**What it is:** Instead of adding a position vector to the input, RoPE rotates the query and key vectors in attention by position-dependent angles.

**Why not learned positional embeddings?**
- Learned embeddings are absolute: "I am at position 42." The model must learn that position 42 and position 43 are adjacent.
- RoPE is relative: when computing attention between positions i and j, the rotation angles cancel to give a function of (i-j) only. The model inherently understands *distance*.
- RoPE generalizes to longer sequences (no max position limit in principle).
- Saves 2048 * 1280 = 2.6M parameters.

**How it works (simplified):**
Think of each pair of dimensions in a head as a 2D plane. RoPE rotates vectors in this plane by an angle proportional to position. Different dimension pairs rotate at different speeds (like a Fourier decomposition of position). When computing Q*K dot products, the rotation of Q at position i and K at position j combine to give a value that depends on (i-j), achieving relative position encoding through geometry.

**Implementation detail:** We precompute complex exponentials for all positions up to max_seq_len. During forward pass, we multiply Q and K (viewed as complex numbers) by these precomputed values. This is mathematically equivalent to the rotation matrix formulation but more efficient.

### 3. SwiGLU MLP (instead of GELU MLP)

**Standard MLP:**
```
hidden = GELU(x @ W1)      # Activate
output = hidden @ W2        # Project back
```
2 weight matrices, 1 nonlinearity.

**SwiGLU MLP:**
```
gate = SiLU(x @ W_gate)    # Which features to activate
up   = x @ W_up            # What values to pass
output = (gate * up) @ W_down  # Combine and project back
```
3 weight matrices, gated activation.

**Why:** Shazeer (2020) tested 8 different activation/gating schemes. SwiGLU consistently won on language modeling benchmarks. The gating mechanism lets the network learn a more complex feature selection than a pointwise nonlinearity can achieve. The cost is 50% more parameters in the MLP, but the quality gain more than compensates.

**About SiLU:** SiLU(x) = x * sigmoid(x), also called "swish". It's smooth everywhere (unlike ReLU's sharp corner at 0), has non-zero gradients for negative inputs (unlike ReLU), and slightly outperforms GELU empirically.

### 4. Pre-Norm Residual Connections

**Post-norm (original transformer):**
```python
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + MLP(x))
```

**Pre-norm (our approach):**
```python
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

**Why pre-norm?** Training stability. In post-norm, gradients must flow through the normalization layer on the residual path. In pre-norm, the residual connection is a clean identity mapping - gradients flow directly from the output back to any layer. This is crucial for deep models (20+ layers). Without pre-norm, you need careful learning rate warmup and often hit training instabilities.

**The tradeoff:** Pre-norm can make the last layer's output unnormalized (the residual stream hasn't been normed since the last sublayer). Solution: add a final RMSNorm before the output head. This is standard practice.

### 5. Causal (Autoregressive) Masking

**What:** Each token can only attend to tokens at the same or earlier positions.

**Why:** We're training a language model that predicts the next token: P(token_t | token_1, ..., token_{t-1}). If position t could attend to position t+1, it would trivially learn to copy the answer. The causal mask prevents this "cheating."

**Implementation:** We use a triangular mask filled with -infinity for future positions. After adding this to the attention logits, softmax converts -infinity to 0 probability, effectively blocking attention to future tokens. With Flash Attention, this is handled internally via the `is_causal=True` flag.

### 6. Weight Tying

**What:** The token embedding matrix (vocab_size x n_embd) is shared with the output projection (n_embd x vocab_size).

**Why:**
- Saves 41.9M parameters (7.5% of the model)
- The embedding maps tokens to vectors; the output head maps vectors to token probabilities. These are conceptually inverse operations on the same "token space"
- Acts as a regularizer: forces the model's input and output representations to be compatible
- Empirically neutral or slightly positive on quality

**The math:** If embedding row i is vector e_i, then the logit for token i at the output is dot(hidden, e_i). This means tokens with similar embeddings will have similar output probabilities - a sensible inductive bias.

### 7. No Bias Terms

**What:** All Linear layers use `bias=False`.

**Why:** Following LLaMA's finding that bias terms in transformers contribute negligibly to quality but add parameters and implementation complexity. For our 561M model, removing all biases saves about 0.2M parameters - small but nonzero, and the code is cleaner.

**Exception:** If we were using LayerNorm (which we're not), we'd want a bias (beta) term. RMSNorm only has a scale parameter (gamma).

### 8. Weight Initialization

```python
# All weights: N(0, 0.02)
# Residual output projections: N(0, 0.02 / sqrt(2 * n_layer))
```

**Why 0.02?** This is the GPT-2 initialization. The standard deviation 0.02 keeps activations in a reasonable range at the start of training. Too large (0.1) and gradients explode in early steps. Too small (0.001) and the model takes forever to "wake up."

**Why scale residual projections?** Each transformer layer adds its output to the residual stream. With 20 layers and 2 additions each (attention + MLP), that's 40 additions. If each addition has variance sigma^2, the total variance grows to 40 * sigma^2. We counteract this by initializing the output projections (attention o_proj and MLP down_proj) with std = 0.02 / sqrt(40) ≈ 0.0032. This keeps the residual stream's variance stable from layer 1 to layer 20.

## Parameter Budget

For FlightMind-d20 (depth=20, n_embd=1280):

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Token Embedding | 41.9M | 7.5% |
| Attention (Q,K,V,O) x 20 | 131.1M | 23.3% |
| MLP (gate,up,down) x 20 | 393.2M | 70.0% |
| RMSNorm x 41 | 52.5K | <0.01% |
| **Total (with weight tying)** | **~561M** | **100%** |

Note: The MLP dominates at 70% of parameters. This is typical for SwiGLU models and is by design - the MLP is where the model stores "knowledge" (factual associations), while attention handles "routing" (which tokens to combine).

## Scaling Considerations

### Why 561M?

Chinchilla scaling laws (Hoffmann et al., 2022) suggest that for compute-optimal training:
- **Optimal tokens ≈ 20 * parameters**
- For 561M params: ~11.2B tokens optimal

Our training budget:
- Aviation data: ~200M tokens (collected)
- General data: ~50B tokens (FineWeb-EDU, streamed)
- Target: 50B total tokens

This puts us in an *over-trained* regime (50B tokens / 561M params = 89x ratio). This is intentional and follows the LLaMA philosophy: slightly smaller models trained on more data perform better at inference time than larger models trained on less data. The extra training cost is paid once; the smaller model size saves on every inference.

### Depth vs Width

With our depth parameterization, making the model wider also makes it deeper (and vice versa). In general:
- **Deeper models** (more layers) are better at complex reasoning and multi-step processing
- **Wider models** (larger embedding) are better at storing knowledge/facts
- Our scaling keeps these balanced, which is appropriate for a model that needs both aviation knowledge and reasoning ability

## What We Chose NOT to Do (and Why)

| Technique | Why we skipped it |
|-----------|-------------------|
| Grouped-Query Attention (GQA) | GQA saves memory at inference by sharing K/V heads. At 561M params, our KV cache is small enough that this isn't needed. GQA shines at 7B+. |
| Mixture of Experts (MoE) | MoE is complex to implement and debug. Not worth it for our scale. |
| Sliding window attention | Our max_seq_len is 2048, well within full attention's O(T^2) budget. Sliding window helps at 8K+ context. |
| ALiBi / other position schemes | RoPE is the most widely validated position encoding. ALiBi trades some quality for simpler implementation. |
| Parallel attention+MLP | Some models compute attention and MLP in parallel (GPT-J style). Saves one sequential step but hurts quality slightly. Not worth the complexity. |
| μP (maximal update parameterization) | μP makes hyperparameters transfer across scales. Valuable but adds implementation complexity. We'll tune hyperparameters directly. |

## File Structure

```
model/
  __init__.py          # Exports FlightMindConfig and FlightMind
  config.py            # Configuration dataclass with depth parameterization
  flightmind.py        # Full model implementation (attention, MLP, blocks, model)
  ARCHITECTURE.md      # This document
```

## References

- Vaswani et al. (2017) - "Attention Is All You Need" (original transformer)
- Su et al. (2021) - "RoFormer" (Rotary Position Embeddings)
- Zhang & Sennrich (2019) - "Root Mean Square Layer Normalization"
- Shazeer (2020) - "GLU Variants Improve Transformer"
- Touvron et al. (2023) - "LLaMA: Open and Efficient Foundation Language Models"
- Hoffmann et al. (2022) - "Chinchilla: Training Compute-Optimal Large Language Models"
- Karpathy (2024) - nanochat (depth parameterization inspiration)
