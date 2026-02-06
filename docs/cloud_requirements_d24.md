# FlightMind-d24 Cloud Training Requirements

**Model:** FlightMind-d24 (956M parameters)
**Training objective:** Pretrain a 956M-parameter decoder-only transformer on 50B tokens
**Date:** February 2026

---

## 1. Model Specification

| Property | Value |
|---|---|
| Architecture | Decoder-only transformer (GPT-style) |
| Parameters | 956,376,576 (~956M) |
| Layers | 24 |
| Attention heads | 24 |
| Embedding dimension | 1536 |
| Head dimension | 64 |
| MLP hidden dimension | 6144 (SwiGLU, 3 matrices) |
| Vocabulary | 32,768 (BPE) |
| Max sequence length | 2,048 |
| Precision | bfloat16 (mixed precision) |
| Positional encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| Weight tying | Yes (embedding = output head) |

## 2. Training Configuration

| Property | Value |
|---|---|
| Total training tokens | 50,000,000,000 (50B) |
| Effective batch size | ~524,288 tokens/step (~512K) |
| Total optimizer steps | ~95,400 |
| Optimizer | AdamW (lr=6e-4, betas=0.9/0.95, wd=0.1) |
| LR schedule | Linear warmup (500 steps) + cosine decay to 6e-5 |
| Gradient clipping | 1.0 (L2 norm) |
| Precision | bfloat16 forward/backward, fp32 optimizer states |
| Checkpoint interval | Every 1,000 steps |
| Eval interval | Every 500 steps |

**Data mix:** 70% general web text (FineWeb-EDU, streamed from HuggingFace) + 30% aviation domain data (~192M tokens, uploaded).

## 3. Minimum GPU Requirements

### VRAM

| Component | Size |
|---|---|
| Model parameters (bf16) | 1.9 GB |
| Gradients (bf16) | 1.9 GB |
| Optimizer states (fp32 params + m + v) | 11.5 GB |
| **Total model state** | **15.3 GB** |
| Activations (per sequence, bf16) | ~1.8 GB |
| **Minimum VRAM needed** | **~20 GB** (batch_size=1 + overhead) |
| **Recommended VRAM** | **40+ GB** (batch_size=8-16 for efficiency) |

### Max micro-batch size by GPU

| GPU | VRAM | Max micro-batch | Viable? |
|---|---|---|---|
| H100 SXM 80GB | 80 GB | ~31 | Best option |
| A100 SXM 80GB | 80 GB | ~31 | Great option |
| L40S 48GB | 48 GB | ~15 | Good option |
| A6000 48GB | 48 GB | ~15 | Budget option |
| A100 PCIe 40GB | 40 GB | ~11 | Tight but works |
| L4 / A5000 24GB | 24 GB | ~3 | Too small, inefficient |

**Minimum viable GPU: 40 GB VRAM.** Anything below 40 GB will require very small micro-batches, reducing throughput significantly.

## 4. Compute Estimates

**Total compute:** 2.87 x 10^20 FLOPs (287 exaFLOPs)

Estimates assume **30% Model FLOP Utilization (MFU)**, which is realistic for sub-1B models with FlashAttention. Actual MFU depends on batch size, data loading, and multi-GPU communication overhead.

### Time and GPU-hours by configuration

| Configuration | Throughput | Wall-clock time | GPU-hours |
|---|---|---|---|
| **1x H100 SXM** | 52K tok/s | 268h (11.2 days) | 268 |
| **2x H100 SXM** | 104K tok/s | 134h (5.6 days) | 268 |
| **4x H100 SXM** | 207K tok/s | 67h (2.8 days) | 268 |
| **8x H100 SXM** | 414K tok/s | 34h (1.4 days) | 268 |
| 1x H100 PCIe | 40K tok/s | 351h (14.6 days) | 351 |
| 4x H100 PCIe | 158K tok/s | 88h (3.7 days) | 351 |
| **1x A100 SXM 80GB** | 16K tok/s | 851h (35.5 days) | 851 |
| **4x A100 SXM 80GB** | 65K tok/s | 213h (8.9 days) | 851 |
| **8x A100 SXM 80GB** | 130K tok/s | 106h (4.4 days) | 851 |
| 4x L40S | 76K tok/s | 183h (7.6 days) | 734 |
| 8x L40S | 151K tok/s | 92h (3.8 days) | 734 |

### Recommended configurations (best value)

| Priority | Configuration | Est. wall time | Est. cost range |
|---|---|---|---|
| **Best value** | 4x H100 SXM | ~67 hours | $430-720 |
| **Fastest** | 8x H100 SXM | ~34 hours | $430-720 |
| **Budget** | 4x A100 SXM 80GB | ~213 hours | $850-1,180 |
| **Middle ground** | 8x L40S | ~92 hours | $550-900 |

## 5. Storage Requirements

| Item | Size | Notes |
|---|---|---|
| Aviation training data (tokenized) | 209 MB | Uploaded once, reused across epochs |
| Aviation training data (cleaned text) | 433 MB | Optional backup / re-tokenization |
| Code + tokenizer | ~3 MB | model/, train/, tokenizer/ |
| FineWeb-EDU (streamed) | 0 GB on disk | Streamed via HuggingFace datasets |
| Model checkpoints (each) | ~11.5 GB | model + optimizer + config |
| Checkpoints (20 saved) | ~230 GB | At every 1,000 steps over ~95K steps |
| Training logs / metrics | <1 GB | |
| **Total upload to cloud** | **~212 MB** | Essential files only |
| **Total disk needed** | **~250-300 GB** | Conservative estimate |

**Request at least 500 GB** to allow headroom for checkpoint accumulation. NVMe/SSD strongly preferred — spinning disk will bottleneck data loading.

## 6. Network Requirements

| Requirement | Details |
|---|---|
| Internet access | Required to stream FineWeb-EDU from HuggingFace Hub |
| Download bandwidth | 100+ Mbps sustained (streaming ~35B tokens over training) |
| Multi-GPU interconnect | NVLink/NVSwitch preferred for 4+ GPU nodes |
| PCIe interconnect | Acceptable for 2 GPUs, but 10-20% slower |

## 7. Software Stack

| Component | Version / Requirement |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.2+ (CUDA 12.1+, bfloat16, fused AdamW) |
| CUDA | 12.1+ |
| FlashAttention | 2.5+ (for flash_attention=True) |
| HuggingFace datasets | 2.16+ (FineWeb-EDU streaming) |
| tokenizers | 0.15+ |
| numpy | 1.26+ |
| (Optional) wandb | For experiment tracking |

Docker image recommendation: `nvcr.io/nvidia/pytorch:24.01-py3` or later.

## 8. What to Ask Cloud Providers

When shopping for GPU instances, verify:

1. **GPU model and VRAM** — need 40+ GB per GPU (80 GB preferred)
2. **GPU interconnect** — NVLink for multi-GPU? What bandwidth?
3. **Disk space and type** — need 500+ GB SSD/NVMe, not HDD
4. **Internet egress** — can you stream from HuggingFace?
5. **Billing granularity** — per-second or per-hour?
6. **Spot/interruptible pricing** — available? Checkpointing handles interruptions
7. **Multi-day rental** — any discounts for 3-7 day continuous use?
8. **Data persistence** — does disk persist if instance stops?

## 9. Cost Benchmarks (reference pricing, Feb 2026)

| Provider | GPU | Per GPU/hr | 4x GPU cost (268 GPU-hrs) |
|---|---|---|---|
| Vast.ai | H100 SXM | ~$1.50-2.00 | ~$400-540 |
| RunPod | H100 SXM | ~$2.39-2.69 | ~$640-720 |
| Lambda | H100 SXM | ~$2.99 | ~$800 |
| Vast.ai | A100 80GB | ~$0.80-1.00 | ~$680-850 |
| RunPod | A100 80GB | ~$1.19-1.39 | ~$1,010-1,180 |

**Target total budget: $400-800 on H100s, $700-1,200 on A100s.**

---

*Generated for FlightMind project. All estimates assume 30% MFU and may vary with actual hardware, data loading speed, and multi-GPU scaling efficiency.*
