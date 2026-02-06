# Vast.ai Training Setup Guide — FlightMind

Step-by-step guide for launching a pretraining run on Vast.ai.

## Prerequisites

1. Create a Vast.ai account at https://cloud.vast.ai/
2. Add credits to your account (target: $500-800 for d24, $1000-1700 for d32)
3. Upload your SSH public key at https://cloud.vast.ai/manage-keys/
4. Install the CLI: `pip install vastai`
5. Authenticate: `vastai set api-key YOUR_API_KEY`

## Step 1: Find an Instance

### For d24 (956M) — Recommended

```bash
# 4x H100 SXM with NVLink, on-demand, sorted by total $/hr
vastai search offers \
  'gpu_name=H100_SXM num_gpus=4 bw_nvlink>200 reliability>0.95 direct_port_count>0 disk_space>=500 inet_down>500 cuda_max_good>=12.0' \
  -t on-demand -o 'dph_total'

# Budget alternative: 4x A100 80GB SXM
vastai search offers \
  'gpu_name=A100_SXM4 num_gpus=4 gpu_ram>=80000 bw_nvlink>200 reliability>0.95 direct_port_count>0 disk_space>=500' \
  -t on-demand -o 'dph_total'
```

### For d32 (2.21B) — 80GB VRAM Required

```bash
# 4x H100 SXM (6.5 days) or 8x H100 SXM (3.2 days)
vastai search offers \
  'gpu_name=H100_SXM num_gpus=4 bw_nvlink>200 reliability>0.95 direct_port_count>0 disk_space>=1000 inet_down>500 cuda_max_good>=12.0' \
  -t on-demand -o 'dph_total'

# 8-GPU for faster wall time
vastai search offers \
  'gpu_name=H100_SXM num_gpus=8 bw_nvlink>200 reliability>0.95 direct_port_count>0 disk_space>=1000' \
  -t on-demand -o 'dph_total'
```

### What to Check Before Renting

- **Max duration**: Verify the host allows rentals long enough for your training
  (d24 on 4x H100: ~67h = 3 days; d32 on 4x H100: ~155h = 6.5 days)
- **Reliability score**: > 0.95 (ideally > 0.98)
- **NVLink bandwidth**: > 200 GB/s (H100 SXM should show ~900 GB/s)
- **Internet speed**: > 500 Mbps download (needed for FineWeb-EDU streaming)

## Step 2: Launch the Instance

```bash
# Replace OFFER_ID with the ID from your search results
vastai create instance OFFER_ID \
  --image vastai/pytorch \
  --disk 500 \
  --ssh \
  --direct
```

Wait for the instance to start (1-3 minutes), then get the SSH command:

```bash
vastai show instances
vastai ssh-url INSTANCE_ID
```

## Step 3: Connect and Verify GPUs

```bash
ssh -p PORT root@IP_ADDRESS

# Inside the instance:
nvidia-smi                    # Verify all GPUs visible
nvidia-smi topo -m            # Verify NVLink topology
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, CUDA: {torch.version.cuda}')"
```

## Step 4: Install Dependencies

```bash
# Activate the venv (if using vastai/pytorch image)
source /venv/main/bin/activate 2>/dev/null || true

# Install training dependencies
pip install flash-attn --no-build-isolation
pip install datasets tokenizers wandb sentencepiece
pip install huggingface-hub

# Verify flash attention
python -c "import flash_attn; print(f'FlashAttention {flash_attn.__version__}')"
```

## Step 5: Upload Training Data

### Option A: Direct SCP (simplest, ~212 MB takes < 1 minute)

From your local machine (Git Bash or PowerShell):

```bash
# Create a tarball of essential files
cd D:/FlightMind
tar czf /tmp/flightmind_upload.tar.gz \
  model/ train/ tokenizer/ config.yaml requirements.txt \
  data/tokenized/

# Upload (~212 MB)
scp -P PORT /tmp/flightmind_upload.tar.gz root@IP_ADDRESS:/workspace/
```

On the instance:

```bash
cd /workspace
tar xzf flightmind_upload.tar.gz
ls -la data/tokenized/   # Verify train.bin and val.bin are there
```

### Option B: Cloud Storage (recommended for production)

Upload to Backblaze B2 or S3 first (one-time), then download on any instance:

```bash
# On instance:
pip install awscli
aws s3 cp s3://your-bucket/flightmind_upload.tar.gz /workspace/ \
  --endpoint-url https://s3.us-west-001.backblazeb2.com
tar xzf flightmind_upload.tar.gz
```

## Step 5.5: NCCL Environment Variables (Required)

Vast.ai containers don't support NCCL's NVLink Sharp (NVLS) feature, which causes
DDP to hang at model initialization. Disable it before any `torchrun` command:

```bash
export NCCL_NVLS_ENABLE=0
```

NVLink P2P still works — only the NVLS multicast optimization is disabled.
No measurable performance impact for sub-2B models.

## Step 6: Verify Everything Works (Dry Run)

Run a quick 100-step test using all GPUs before committing to the full run:

```bash
cd /workspace

# Multi-GPU dry run (replace 4 with your GPU count)
NCCL_NVLS_ENABLE=0 \
torchrun --nproc_per_node=4 train/pretrain.py \
  --depth 24 \
  --max-steps 100 \
  --batch-size 8 \
  --log-every 10 \
  --eval-every 50 \
  --checkpoint-every 50
```

Check:
- All GPUs are utilized (`nvidia-smi` in another terminal)
- Log output comes from rank 0 only (single stream, not duplicated)
- Loss is decreasing
- Throughput (tok/s) is ~4x single-GPU (~200K+ tok/s on 4x H100)
- Checkpoints save successfully to /workspace/checkpoints/

## Step 7: Launch Full Training in tmux

```bash
# Start a tmux session (persists if SSH disconnects)
tmux new -s train

cd /workspace

# d24 on 4 GPUs (~67 hours)
NCCL_NVLS_ENABLE=0 \
torchrun --nproc_per_node=4 train/pretrain.py \
  --depth 24 \
  --max-steps 95000 \
  --batch-size 8 \
  --tokens-per-step 524288 \
  --lr 6e-4 \
  --min-lr 6e-5 \
  --warmup-steps 500 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --log-every 10 \
  --eval-every 500 \
  --checkpoint-every 1000 \
  --fineweb

# Detach tmux: Ctrl+B then D
# Reattach later: tmux attach -t train
```

### For d32, use lower learning rate:

```bash
# d32 on 4 GPUs (~155 hours)
NCCL_NVLS_ENABLE=0 \
torchrun --nproc_per_node=4 train/pretrain.py \
  --depth 32 \
  --max-steps 95000 \
  --batch-size 4 \
  --tokens-per-step 524288 \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --warmup-steps 500 \
  --log-every 10 \
  --eval-every 500 \
  --checkpoint-every 1000 \
  --fineweb
```

### Resume after interruption:

```bash
# Resume from latest checkpoint (works with any GPU count)
NCCL_NVLS_ENABLE=0 \
torchrun --nproc_per_node=4 train/pretrain.py \
  --depth 24 \
  --max-steps 95000 \
  --resume checkpoints/step_5000.pt
```

## Step 8: Monitor Training

### From your local machine:

```bash
# SSH with port forwarding for TensorBoard
ssh -p PORT -L 6006:localhost:6006 root@IP_ADDRESS

# On the instance:
pip install tensorboard
tensorboard --logdir /workspace/checkpoints --port 6006
# Then open http://localhost:6006 in your browser
```

### Or just tail the log:

```bash
ssh -p PORT root@IP_ADDRESS "tail -f /workspace/train.log"
```

## Step 9: Download Results

When training completes, download the final and best checkpoints:

```bash
# From your local machine:
scp -P PORT root@IP_ADDRESS:/workspace/checkpoints/best.pt D:/FlightMind/checkpoints/cloud_best.pt
scp -P PORT root@IP_ADDRESS:/workspace/checkpoints/final.pt D:/FlightMind/checkpoints/cloud_final.pt

# Or the whole checkpoints directory:
scp -r -P PORT root@IP_ADDRESS:/workspace/checkpoints/ D:/FlightMind/checkpoints_cloud/
```

**Important**: Download your checkpoints BEFORE destroying the instance!

## Step 10: Destroy the Instance

```bash
# Stop billing
vastai destroy instance INSTANCE_ID
```

---

## Checklist Before Launching

- [ ] Vast.ai account funded with sufficient credits
- [ ] SSH key uploaded to Vast.ai
- [ ] Training data tokenized locally (train.bin + val.bin)
- [ ] Training script tested locally (d8 proof-of-concept already done)
- [ ] Checkpoint/resume logic verified
- [ ] Cloud storage bucket ready (optional, for checkpoint backup)
- [ ] wandb account ready (optional, for remote monitoring)

## Multi-GPU Support

`pretrain.py` has built-in **DDP (DistributedDataParallel)** support. It auto-detects
whether it's launched via `torchrun` or plain `python`:

- **`torchrun --nproc_per_node=N`** — Uses all N GPUs with DDP, near-linear scaling
- **`python`** — Falls back to single-GPU mode (backward compatible)

No flags needed — the script reads `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` from the
environment (set automatically by torchrun). Each GPU gets a different random seed
to ensure different data batches, and `no_sync()` is used during gradient accumulation
to minimize inter-GPU communication.

## Cost Summary

| Config | Est. GPU-hours | On-demand H100 (~$2/GPU/hr) | Reserved H100 (~$1.40/GPU/hr) |
|---|---|---|---|
| d24, 4x H100 SXM | 268 | ~$540 | ~$375 |
| d24, 8x H100 SXM | 268 | ~$540 | ~$375 |
| d32, 4x H100 SXM | 621 | ~$1,240 | ~$870 |
| d32, 8x H100 SXM | 621 | ~$1,240 | ~$870 |

Wall-clock time is faster with more GPUs; total GPU-hours (and cost) stays the same.
