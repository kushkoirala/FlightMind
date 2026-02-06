# FlightMind-d32 Cloud Training Requirements

**Model:** FlightMind-d32 (2.21B parameters)
**Training objective:** Pretrain a 2.21B-parameter decoder-only transformer on 50B tokens
**Date:** February 2026

---

## 1. Model Specification

| Property | Value |
|---|---|
| Architecture | Decoder-only transformer (GPT-style) |
| Parameters | 2,214,725,632 (~2.21B) |
| Layers | 32 |
| Attention heads | 32 |
| Embedding dimension | 2048 |
| Head dimension | 64 |
| MLP hidden dimension | 8192 (SwiGLU, 3 matrices) |
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

**Note on LR:** The learning rate (6e-4) was tuned for d8/d20 scale. For d32 (2.21B), a lower peak LR of 3e-4 may be more stable. This will be validated during the first few hundred steps and adjusted if needed.

## 3. Minimum GPU Requirements

### VRAM

| Component | Size |
|---|---|
| Model parameters (bf16) | 4.4 GB |
| Gradients (bf16) | 4.4 GB |
| Optimizer states (fp32 params + m + v) | 26.6 GB |
| **Total model state** | **35.4 GB** |
| Activations (per sequence, bf16) | ~3.2 GB |
| **Minimum VRAM needed** | **~42 GB** (batch_size=1 + overhead) |
| **Recommended VRAM** | **80 GB** (batch_size=8+ for efficiency) |

### Max micro-batch size by GPU

| GPU | VRAM | Max micro-batch | Viable? |
|---|---|---|---|
| H100 SXM 80GB | 80 GB | ~11 | Best option |
| A100 SXM 80GB | 80 GB | ~11 | Great option |
| L40S 48GB | 48 GB | ~2 | Marginal, very inefficient |
| A6000 48GB | 48 GB | ~2 | Marginal, very inefficient |
| A100 PCIe 40GB | 40 GB | 0 | Does NOT fit |
| L4 / A5000 24GB | 24 GB | 0 | Does NOT fit |

**Minimum viable GPU: 80 GB VRAM.** The d32 model's optimizer state alone is 26.6 GB. GPUs with 48 GB can technically fit the model but with batch_size=2, throughput is severely degraded. **80 GB GPUs (H100 SXM or A100 SXM/PCIe 80GB) are required for practical training.**

## 4. Compute Estimates

**Total compute:** 6.64 x 10^20 FLOPs (664 exaFLOPs)

Estimates assume **30% Model FLOP Utilization (MFU)**, which is realistic for ~2B models with FlashAttention. Actual MFU depends on batch size, data loading, and multi-GPU communication overhead.

### Time and GPU-hours by configuration

| Configuration | Throughput | Wall-clock time | GPU-hours |
|---|---|---|---|
| **1x H100 SXM** | 22K tok/s | 621h (25.9 days) | 621 |
| **2x H100 SXM** | 45K tok/s | 311h (13.0 days) | 621 |
| **4x H100 SXM** | 89K tok/s | 155h (6.5 days) | 621 |
| **8x H100 SXM** | 179K tok/s | 78h (3.2 days) | 621 |
| 1x H100 PCIe | 17K tok/s | 814h (33.9 days) | 814 |
| 4x H100 PCIe | 68K tok/s | 203h (8.5 days) | 814 |
| **1x A100 SXM 80GB** | 7K tok/s | 1,972h (82.2 days) | 1,972 |
| **4x A100 SXM 80GB** | 28K tok/s | 493h (20.5 days) | 1,972 |
| **8x A100 SXM 80GB** | 56K tok/s | 246h (10.3 days) | 1,972 |

### Recommended configurations (best value)

| Priority | Configuration | Est. wall time | Est. cost range |
|---|---|---|---|
| **Best value** | 4x H100 SXM | ~155 hours | $930-1,670 |
| **Fastest** | 8x H100 SXM | ~78 hours | $930-1,670 |
| **Budget** | 8x A100 SXM 80GB | ~246 hours | $1,580-2,740 |

**Note:** A100 PCIe 40GB does NOT have enough VRAM for d32. Only 80GB variants work.

## 5. Storage Requirements

| Item | Size | Notes |
|---|---|---|
| Aviation training data (tokenized) | 209 MB | Uploaded once, reused across epochs |
| Aviation training data (cleaned text) | 433 MB | Optional backup / re-tokenization |
| Code + tokenizer | ~3 MB | model/, train/, tokenizer/ |
| FineWeb-EDU (streamed) | 0 GB on disk | Streamed via HuggingFace datasets |
| Model checkpoints (each) | ~26.6 GB | model + optimizer + config |
| Checkpoints (20 saved) | ~530 GB | At every 1,000 steps over ~95K steps |
| Training logs / metrics | <1 GB | |
| **Total upload to cloud** | **~212 MB** | Essential files only |
| **Total disk needed** | **~600-700 GB** | Conservative estimate |

**Request at least 1 TB** to allow headroom for checkpoint accumulation. NVMe/SSD strongly preferred — spinning disk will bottleneck data loading. Consider a checkpoint rotation policy (keep last 5 + best) to cap disk usage at ~200 GB.

## 6. Network Requirements

| Requirement | Details |
|---|---|
| Internet access | Required to stream FineWeb-EDU from HuggingFace Hub |
| Download bandwidth | 100+ Mbps sustained (streaming ~35B tokens over training) |
| Multi-GPU interconnect | **NVLink/NVSwitch required** for 4+ GPU nodes at this model size |
| PCIe interconnect | Acceptable for 2 GPUs only, expect 15-25% slower |

**Important:** At 2.21B params, gradient synchronization across GPUs is 4.4 GB per step. Without NVLink (900 GB/s), PCIe (64 GB/s) becomes a serious bottleneck for 4+ GPUs. **Insist on NVLink/NVSwitch for multi-GPU nodes.**

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
| (Optional) FSDP / DDP | PyTorch DistributedDataParallel for multi-GPU |

Docker image recommendation: `nvcr.io/nvidia/pytorch:24.01-py3` or later.

**Multi-GPU note:** The training script currently uses single-GPU with gradient accumulation. For 4-8 GPU training, PyTorch DDP (DistributedDataParallel) must be enabled. The codebase has a `--n-gpu` flag for gradient accumulation calculation but will need a DDP wrapper for actual multi-GPU execution.

## 8. What to Ask Cloud Providers

When shopping for GPU instances, verify:

1. **GPU model and VRAM** — **must be 80 GB** (H100 SXM, A100 SXM 80GB, or A100 PCIe 80GB)
2. **GPU interconnect** — **NVLink required** for 4+ GPUs. What generation? What bandwidth?
3. **Disk space and type** — need 1+ TB SSD/NVMe, not HDD
4. **Internet egress** — can you stream from HuggingFace? Any bandwidth caps?
5. **Billing granularity** — per-second or per-hour?
6. **Spot/interruptible pricing** — available? Checkpointing handles interruptions, but at 26.6 GB per checkpoint, save time is non-trivial
7. **Multi-day rental** — any discounts for 7-14 day continuous use?
8. **Data persistence** — does disk persist if instance stops?
9. **Instance preemption policy** — how much warning before spot eviction?

## 9. Cost Benchmarks (reference pricing, Feb 2026)

| Provider | GPU | Per GPU/hr | 4x GPU cost (621 GPU-hrs) | 8x GPU cost (621 GPU-hrs) |
|---|---|---|---|---|
| Vast.ai | H100 SXM | ~$1.50-2.00 | ~$930-1,240 | ~$930-1,240 |
| RunPod | H100 SXM | ~$2.39-2.69 | ~$1,480-1,670 | ~$1,480-1,670 |
| Lambda | H100 SXM | ~$2.99 | ~$1,860 | ~$1,860 |
| Vast.ai | A100 80GB | ~$0.80-1.00 | ~$1,580-1,970 | ~$1,580-1,970 |
| RunPod | A100 80GB | ~$1.19-1.39 | ~$2,350-2,740 | ~$2,350-2,740 |

**Target total budget: $900-1,700 on H100s, $1,500-2,800 on A100s.**

## 10. Risk Considerations for d32

| Risk | Mitigation |
|---|---|
| Higher cost (~2.5x of d24) | Start with d24, scale up only if d24 quality is insufficient |
| Training instability at 2.2B | Lower peak LR to 3e-4, monitor grad norms in first 500 steps |
| Aviation data over-repetition (192M tokens repeated ~78x at 30% mix) | Consider reducing aviation ratio to 10-15%, or total tokens to 20-30B |
| Large checkpoints (26.6 GB each) | Implement checkpoint rotation (keep last 5 + best) |
| Longer wall-clock (3-26 days depending on config) | Use spot instances with auto-resume from checkpoints |

---

*Generated for FlightMind project. All estimates assume 30% MFU and may vary with actual hardware, data loading speed, and multi-GPU scaling efficiency.*
